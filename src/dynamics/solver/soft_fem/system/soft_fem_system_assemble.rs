//! The per-substep work: reading the particle state, assembling `A` and the elastic force, the predictor solve and the writeback to the soft body.

use super::super::soft_fem_sparse::Preconditioner;
use super::SoftFemSystem;
use crate::alloc_prelude::*;
use crate::dynamics::{SoftBody, SoftBodyMaterial};
use crate::dynamics::soft_body::soft_body_shape_matching::extract_rotation;
use crate::dynamics::solver::soft_constraint::soft_constraints_set::{
    plastic_flow, total_tensile_strain,
};
#[cfg(feature = "dim3")]
use crate::dynamics::solver::soft_constraint::soft_element_constraint::dihedral_gradients;
use crate::dynamics::solver::soft_constraint::soft_element_constraint::{
    MAX_CONSTRAINT_PARTICLES, NeoHookeanConstraint, STIFFNESS_UPDATE_STRAIN, SoftElasticConstraint,
    StrainJacobian, outer_product, strain_rows_of,
};
use crate::dynamics::solver::solver_body::SolverBodies;
use crate::math::{Matrix, Real, Vector};
use crate::utils::RotationOps;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use simba::scalar::{ComplexField as _, RealField as _};

impl SoftFemSystem {
    /// Reads the particles' state from their solver bodies.
    pub(crate) fn refresh_state(&mut self, bodies: &SolverBodies) {
        self.read_state(bodies);
    }

    /// Reads the particles' state from their solver bodies.
    fn read_state(&mut self, bodies: &SolverBodies) {
        for (k, &slot) in self.slots.iter().enumerate() {
            if slot != u32::MAX {
                self.position[k] = bodies.get_pose(slot).translation;
                self.velocity[k] = bodies.get_vel(slot).linear;
            }
        }
    }

    /// Assembles `A = M + (h² + h β) K` and the elastic force at the current positions; pinned
    /// degrees of freedom become Dirichlet rows, with their coupling to the free rows
    /// (`Σ_{q pinned} A_pq v_q`) kept for the right-hand side; a step's first substep reuses it.
    pub(super) fn assemble_operator(&mut self, dt: Real) {
        self.matrix.clear_values();
        for (k, m) in self.mass.iter().enumerate() {
            let idx = self.matrix.diagonal_index(k as u32) as usize;
            self.matrix.blocks[idx] = Matrix::IDENTITY * *m;
        }
        self.force.fill(Vector::ZERO);

        for (ci, cell) in self.cells.iter_mut().enumerate() {
            if cell.mu <= 0.0 {
                continue;
            }
            let x: [Vector; MAX_CONSTRAINT_PARTICLES] =
                core::array::from_fn(|k| self.position[cell.vertices[k] as usize]);
            let f = SoftBody::cell_edge_matrix(x) * cell.inv_rest_matrix;
            cell.inverted = f.determinant() < 0.0;
            cell.rotation = extract_rotation(f, cell.rotation);
            let s = cell.rotation.inverse().to_mat() * f;
            cell.strain = strain_rows_of(&(s - Matrix::IDENTITY));

            let gradient = if cell.neo_hookean {
                // Stable Neo-Hookean (Smith et al. 2018): the energy is defined and bounded
                // below for inverted cells, and its tangent is floored mode-wise at the rest
                // curvature so one implicit step per substep needs no line search.
                let lambda = cell.lambda + cell.mu;
                let (s, kappa, gradient) =
                    NeoHookeanConstraint::gradient(cell.mu, lambda, &cell.strain);
                let stale = (cell.strain - cell.tangent_strain)
                    .iter()
                    .any(|d| d.abs() > STIFFNESS_UPDATE_STRAIN);
                if stale {
                    cell.tangent = NeoHookeanConstraint::hessian(cell.mu, lambda, &s, kappa);
                    cell.tangent_strain = cell.strain;
                }
                gradient
            } else {
                // Linear elasticity in the cell's rotation-free frame: a constant tangent.
                cell.tangent * cell.strain
            };

            // World-frame strain gradients: column `r` of `jacobians[p]` is `R gᵣ[p]`.
            let rotation = cell.rotation.to_mat();
            let jacobians: [StrainJacobian; MAX_CONSTRAINT_PARTICLES] = core::array::from_fn(|p| {
                SoftElasticConstraint::strain_jacobian(&cell.coeffs, p, &rotation)
            });

            // Elastic force `f_p = −Σᵣ g̃ᵣ[p] gradᵣ`.
            for p in 0..MAX_CONSTRAINT_PARTICLES {
                let fp: Vector = (jacobians[p] * gradient).into();
                self.force[cell.vertices[p] as usize] -= fp;
            }

            // Tangent `K_pq = Σᵣₛ Hᵣₛ g̃ᵣ[p] g̃ₛ[q]ᵀ = J_p H J_qᵀ`, scaled by `h² + h β` (the
            // stiffness and its Rayleigh damping share the same matrix); `K_qp = K_pqᵀ`.
            let scale = dt * dt + dt * cell.beta;
            let blocks = &self.cell_blocks[ci];
            for q in 0..MAX_CONSTRAINT_PARTICLES {
                let hq = cell.tangent * jacobians[q].transpose();
                for p in 0..=q {
                    let slot = blocks[p * MAX_CONSTRAINT_PARTICLES + q];
                    if slot == u32::MAX {
                        continue;
                    }
                    let block: Matrix = (jacobians[p] * hq).into();
                    let block = block * scale;
                    self.matrix.blocks[slot as usize] += block;
                    if p != q {
                        let mirror = blocks[q * MAX_CONSTRAINT_PARTICLES + p];
                        self.matrix.blocks[mirror as usize] += block.transpose();
                    }
                }
            }
        }

        self.assemble_springs(dt);
        #[cfg(feature = "dim3")]
        self.assemble_dihedrals(dt);
        self.assemble_volume_cells(dt);

        // The pinned particles' coupling to the free rows (`A_pq = A_qpᵀ`), then their Dirichlet
        // rows.
        self.pinned_coupling.clear();
        self.pinned_coupling.resize(self.mass.len(), Vector::ZERO);
        for (k, &pinned) in self.pinned.iter().enumerate() {
            if pinned {
                let v = self.velocity[k];
                let coupling = &mut self.pinned_coupling;
                self.matrix.for_each_block_of_row(k as u32, |col, block| {
                    if col as usize != k {
                        coupling[col as usize] += block.transpose() * v;
                    }
                });
                self.matrix.make_dirichlet(k as u32);
            }
        }
        self.cg.update_preconditioner(&self.matrix);
    }

    /// The predictor's right-hand side `b = h f − (A − M) v` from the assembled operator and
    /// the current velocities.
    fn assemble_rhs(&mut self, dt: Real) {
        self.matrix.mul(&self.velocity, &mut self.rhs);
        for k in 0..self.rhs.len() {
            self.rhs[k] = self.force[k] * dt - (self.rhs[k] + self.pinned_coupling[k])
                + self.velocity[k] * self.mass[k];
        }
        for (k, &pinned) in self.pinned.iter().enumerate() {
            if pinned {
                self.rhs[k] = Vector::ZERO;
            }
        }
    }

    /// Accumulates the volume elements' force and Gauss-Newton tangent.
    fn assemble_volume_cells(&mut self, dt: Real) {
        for element in &self.volume_cells {
            if element.stiffness <= 0.0 {
                continue;
            }
            let cell = &self.cells[element.cell as usize];
            let x: [Vector; MAX_CONSTRAINT_PARTICLES] =
                core::array::from_fn(|k| self.position[cell.vertices[k] as usize]);
            let c = SoftBody::cell_volume(x) - element.rest_volume;
            let grad = SoftBody::cell_volume_gradients(x);
            for k in 0..MAX_CONSTRAINT_PARTICLES {
                self.force[cell.vertices[k] as usize] -= grad[k] * (element.stiffness * c);
            }
            let scale = element.stiffness * (dt * dt + dt * element.beta);
            let blocks = &self.cell_blocks[element.cell as usize];
            for p in 0..MAX_CONSTRAINT_PARTICLES {
                for q in 0..MAX_CONSTRAINT_PARTICLES {
                    let slot = blocks[p * MAX_CONSTRAINT_PARTICLES + q];
                    if slot != u32::MAX {
                        self.matrix.blocks[slot as usize] +=
                            outer_product(grad[p], grad[q]) * scale;
                    }
                }
            }
        }
    }

    /// Accumulates the distance elements' force and tangent: the Gauss-Newton term along each
    /// element, and across a pulled one the tension's own stiffness.
    fn assemble_springs(&mut self, dt: Real) {
        let Self {
            springs,
            position,
            force: forces,
            matrix,
            ..
        } = self;
        for spring in springs.iter_mut() {
            if spring.stiffness <= 0.0 {
                continue;
            }
            let a = spring.vertices[0] as usize;
            let b = spring.vertices[1] as usize;
            let d = position[b] - position[a];
            let length = d.length();
            if length < 1.0e-9 {
                continue;
            }
            // A yielding element gives way at its yield force: its rest length follows the
            // load within the substep, so it never carries past it. It keeps its elastic
            // stiffness in the solve, which holds the substep's motion against it.
            spring.return_map(length);
            let c = length - spring.rest_length;
            let (magnitude, stiff) = spring.force(c);
            let n = d / length;
            let force = n * magnitude;
            forces[a] += force;
            forces[b] -= force;
            if !stiff {
                // Slack in the direction it does not load: no force, no stiffness.
                continue;
            }

            let mut block = outer_product(n, n) * (spring.stiffness * (dt * dt + dt * spring.beta));
            // A pulled element also resists turning: its pull, over its length, across it.
            // This is the tensile part of the element's geometric stiffness (`f / |d|` times
            // `I − nnᵀ`), positive semi-definite while the element pulls; it is what holds a
            // taut rope or cable sideways, so a load pressed across it (a body resting on a
            // rope) sees the stiffness the cable's tension gives it.
            if magnitude > 0.0 {
                block += (Matrix::IDENTITY - outer_product(n, n)) * (magnitude / length * dt * dt);
            }
            for (k, &slot) in spring.blocks.iter().enumerate() {
                if slot != u32::MAX {
                    // Diagonal blocks (k = 0, 3) get `+k nnᵀ`, off-diagonal ones `−k nnᵀ`.
                    let sign = if k == 0 || k == 3 { 1.0 } else { -1.0 };
                    matrix.blocks[slot as usize] += block * sign;
                }
            }
        }
    }

    /// Accumulates the dihedral elements' force and Gauss-Newton tangent.
    #[cfg(feature = "dim3")]
    fn assemble_dihedrals(&mut self, dt: Real) {
        for dihedral in &self.dihedrals {
            if dihedral.stiffness <= 0.0 {
                continue;
            }
            let pos: [Vector; 4] =
                core::array::from_fn(|k| self.position[dihedral.vertices[k] as usize]);
            let mut grad = [Vector::ZERO; 4];
            let c = dihedral_gradients(&pos, dihedral.rest_angle, &mut grad);
            for k in 0..4 {
                self.force[dihedral.vertices[k] as usize] -= grad[k] * (dihedral.stiffness * c);
            }
            let scale = dihedral.stiffness * (dt * dt + dt * dihedral.beta);
            for p in 0..4 {
                for q in 0..4 {
                    let slot = dihedral.blocks[p * 4 + q];
                    if slot != u32::MAX {
                        self.matrix.blocks[slot as usize] +=
                            outer_product(grad[p], grad[q]) * scale;
                    }
                }
            }
        }
    }

    /// The implicit elastic step of one substep: assemble, solve `A Δv = b`, and add `Δv` to the
    /// particles' solver velocities. Returns the conjugate-gradient iteration count.
    pub fn predict(
        &mut self,
        bodies: &mut SolverBodies,
        dt: Real,
        tolerance: Real,
        max_iterations: usize,
    ) -> usize {
        self.read_state(bodies);
        // The step-start factorization assembled this operator at these same positions (the
        // substep only changed the velocities): reused by the first substep.
        if !core::mem::take(&mut self.operator_fresh) {
            self.assemble_operator(dt);
        }
        self.assemble_rhs(dt);
        // Started from zero, never warm-started: a truncated CG converges the smooth modes first,
        // so a partial solve is an under-relaxed step, while a stale start leaves high-frequency
        // content the truncation never removes and pumps energy into a stiff body.
        self.delta.fill(Vector::ZERO);
        // A body factorized directly at the step start is preconditioned by that factor: it is
        // this operator on the first substep and close to it on the others, so a stiff body's
        // step converges in a few iterations where the block-Jacobi preconditioner needs
        // hundreds, and the truncated solve would leave the body softer than its material.
        let preconditioner = match (self.direct_valid, self.direct.as_mut()) {
            (true, Some(factor)) => Preconditioner::Factor(factor),
            _ => Preconditioner::BlockJacobi,
        };
        let iterations = self.cg.solve(
            &self.matrix,
            &self.rhs,
            &mut self.delta,
            tolerance,
            max_iterations,
            preconditioner,
        );
        for (k, &slot) in self.slots.iter().enumerate() {
            if slot != u32::MAX && !self.pinned[k] {
                bodies.vels[slot as usize].linear += self.delta[k];
            }
        }
        iterations
    }

    /// Writes back what the soft body keeps across steps: each cell's warm-started polar rotation,
    /// the step's plastic flow, and the tearing marks of elements strained past `tear_strain`
    /// (the FEM counterpart of `SoftConstraintsSet::writeback_constraints`).
    pub fn writeback(&self, sb: &mut SoftBody, step_dt: Real) {
        let mut flowing = false;
        let mut torn = false;
        // The interior strength reads the particles' flags: taken before the cells are borrowed.
        let cell_resistance: Vec<Real> = sb
            .cells
            .iter()
            .map(|c| {
                if sb.element_material(c.material).tear_strain.is_some() {
                    sb.effective_tear_resistance(c.tear_resistance, &c.vertices)
                } else {
                    1.0
                }
            })
            .collect();
        let edge_resistance: Vec<Real> = sb
            .edges
            .iter()
            .map(|e| {
                if sb.element_material(e.material).tears() {
                    sb.effective_tear_resistance(e.tear_resistance, &e.vertices)
                } else {
                    1.0
                }
            })
            .collect();
        let cell_materials: Vec<SoftBodyMaterial> = sb
            .cells
            .iter()
            .map(|c| *sb.element_material(c.material))
            .collect();
        for ((cell, out), ci) in self.cells.iter().zip(sb.cells.iter_mut()).zip(0..) {
            out.rotation = cell.rotation;
            if cell.mu <= 0.0 {
                continue;
            }
            let material = &cell_materials[ci];
            // The tear strain counts the plastic stretch, as it stood before this step's flow.
            let tensile = total_tensile_strain(&cell.strain, &out.plastic_stretch);
            if material.plastic_yield > 0.0 && material.plastic_creep > 0.0 {
                let rest0: [Vector; MAX_CONSTRAINT_PARTICLES] = core::array::from_fn(|k| {
                    sb.particles[out.vertices[k] as usize].initial_rest_position
                });
                if plastic_flow(out, &cell.strain, cell.inverted, &rest0, material, step_dt) {
                    flowing = true;
                }
            }
            if material.tear_strain.is_some() {
                let load =
                    material.cell_tear_load(tensile) * crate::utils::inv(cell_resistance[ci]);
                out.stress = material.smooth_stress(out.stress, load, step_dt);
                if out.stress > 1.0 {
                    out.torn = true;
                    torn = true;
                }
            }
        }
        #[cfg(feature = "dim3")]
        {
            let dihedral_materials: Vec<SoftBodyMaterial> = sb
                .dihedrals
                .iter()
                .map(|d| *sb.element_material(d.material))
                .collect();
            for ((dihedral, d), material) in self
                .dihedrals
                .iter()
                .zip(sb.dihedrals.iter_mut())
                .zip(&dihedral_materials)
            {
                if material.edge_plastic_yield <= 0.0 || material.edge_plastic_creep <= 0.0 {
                    continue;
                }
                let pos: [Vector; 4] =
                    core::array::from_fn(|k| self.position[dihedral.vertices[k] as usize]);
                let mut grad = [Vector::ZERO; 4];
                let angle = d.rest_angle + dihedral_gradients(&pos, d.rest_angle, &mut grad);
                if d.plastic_flow(angle, material, step_dt) {
                    flowing = true;
                }
            }
        }
        let edge_materials: Vec<SoftBodyMaterial> = sb
            .edges
            .iter()
            .map(|e| *sb.element_material(e.material))
            .collect();
        for ((spring, edge), ei) in self.springs.iter().zip(sb.edges.iter_mut()).zip(0..) {
            let d = self.position[spring.vertices[1] as usize]
                - self.position[spring.vertices[0] as usize];
            let length = d.length();
            // The elastic force of the implicit step's end state, within the element's bounds:
            // positive resisting a stretch, negative resisting a squeeze.
            edge.force = if spring.rest_length > 0.0 && length >= 1.0e-9 {
                spring.force(length - spring.rest_length).0
            } else {
                0.0
            };
            if spring.rest_length <= 0.0 {
                continue;
            }
            let material = &edge_materials[ei];
            // The rest length the substeps' flow at the yield force left.
            if spring.rest_length != edge.rest_length {
                let initial = edge.initial_rest_length();
                if (spring.rest_length - edge.rest_length).abs() > 1.0e-4 * initial {
                    flowing = true;
                }
                edge.rest_length = spring.rest_length;
                edge.plastic_strain = spring.rest_length / initial - 1.0;
            }
            if edge.plastic_flow(length, material, step_dt) {
                flowing = true;
            }
            if !material.tears() {
                continue;
            }
            // The stretch is measured against the initial rest length (plastic flow counts).
            let load = material.edge_tear_load(length, edge.initial_rest_length(), edge.force)
                * crate::utils::inv(edge_resistance[ei]);
            edge.stress = material.smooth_stress(edge.stress, load, step_dt);
            if edge.stress > 1.0 {
                edge.torn = true;
                torn = true;
            }
        }
        sb.plastic_flowing |= flowing;
        sb.tearing_pending |= torn;
    }
}
