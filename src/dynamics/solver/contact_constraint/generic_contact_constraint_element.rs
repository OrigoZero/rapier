use crate::dynamics::solver::GenericJointConstraint;
use crate::dynamics::solver::SolverVel;
use crate::dynamics::solver::contact_constraint::GenericContactConstraint;
use crate::dynamics::solver::coupled_multibody_rows::{
    CoupledRows, RowSide, SideJacobian, SideVelocities, coupling,
};
use crate::dynamics::solver::solver_body::SolverBodies;
use crate::dynamics::solver::{ContactConstraintNormalPart, ContactConstraintTangentPart};
use crate::math::{AngVector, DIM, DVector, Real, SPATIAL_DIM, Vector};
use crate::utils::{ComponentMul, DotProduct};
#[cfg(feature = "dim2")]
use {crate::utils::OrthonormalBasis, na::SimdPartialOrd};

pub(crate) enum GenericRhs {
    SolverVel(SolverVel<Real>),
    GenericId(u32),
    Fixed,
}

// Offset between the jacobians of two consecutive constraints.
#[inline]
fn j_step(ndofs1: usize, ndofs2: usize) -> usize {
    (ndofs1 + ndofs2) * 2
}

#[inline]
fn j_id1(j_id: usize, _ndofs1: usize, _ndofs2: usize) -> usize {
    j_id
}

#[inline]
fn j_id2(j_id: usize, ndofs1: usize, _ndofs2: usize) -> usize {
    j_id + ndofs1 * 2
}

#[inline]
fn normal_j_id(j_id: usize, _ndofs1: usize, _ndofs2: usize) -> usize {
    j_id
}

#[inline]
fn tangent_j_id(j_id: usize, ndofs1: usize, ndofs2: usize) -> usize {
    j_id + (ndofs1 + ndofs2) * 2
}

impl GenericRhs {
    /// The side of a contact row this velocity is: a multibody's, its jacobian at `j_id` and its
    /// weighted jacobian right after; a rigid body's (`solver_vel_id`), along `dir` and
    /// `torque_dir`, weighted by `inv_mass` and `ii_torque_dir`.
    #[allow(clippy::too_many_arguments)]
    fn side(
        &self,
        solver_vel_id: u32,
        j_id: usize,
        ndofs: usize,
        dir: Vector,
        torque_dir: AngVector,
        ii_torque_dir: AngVector,
        inv_mass: Vector,
    ) -> Option<RowSide> {
        let (velocities, jacobian) = match self {
            GenericRhs::GenericId(slot) => (
                SideVelocities::Multibody(*slot),
                SideJacobian::Buffer {
                    ndofs,
                    j_id,
                    wj_id: j_id + ndofs,
                },
            ),
            GenericRhs::SolverVel(_) => {
                let mut j = [0.0; SPATIAL_DIM];
                let mut wj = [0.0; SPATIAL_DIM];
                let linear = dir.to_array();
                let weighted = dir.component_mul(&inv_mass).to_array();
                j[..DIM].copy_from_slice(&linear);
                wj[..DIM].copy_from_slice(&weighted);
                #[cfg(feature = "dim3")]
                {
                    j[DIM..].copy_from_slice(&torque_dir.to_array());
                    wj[DIM..].copy_from_slice(&ii_torque_dir.to_array());
                }
                #[cfg(feature = "dim2")]
                {
                    j[DIM] = torque_dir;
                    wj[DIM] = ii_torque_dir;
                }
                (
                    SideVelocities::Body(solver_vel_id),
                    SideJacobian::Inline { j, wj },
                )
            }
            GenericRhs::Fixed => return None,
        };
        Some(RowSide {
            velocities,
            jacobian,
            error_sign: 1.0,
            impulse_sign: 1.0,
        })
    }

    /// Writes a rigid body's velocity held here back to `bodies`.
    fn store(&self, solver_vel_id: u32, bodies: &mut SolverBodies) {
        if let GenericRhs::SolverVel(vel) = self {
            bodies.vels[solver_vel_id as usize] = *vel;
        }
    }

    /// Reads a rigid body's velocity held here again from `bodies`.
    fn load(&mut self, solver_vel_id: u32, bodies: &SolverBodies) {
        if let GenericRhs::SolverVel(vel) = self {
            *vel = bodies.vels[solver_vel_id as usize];
        }
    }

    #[inline]
    fn dvel(
        &self,
        j_id: usize,
        ndofs: usize,
        jacobians: &DVector,
        dir: Vector,
        gcross: AngVector,
        solver_vels: &DVector,
    ) -> Real {
        match self {
            GenericRhs::SolverVel(rhs) => dir.dot(rhs.linear) + gcross.gdot(rhs.angular),
            GenericRhs::GenericId(solver_vel) => {
                let j = jacobians.rows(j_id, ndofs);
                let rhs = solver_vels.rows(*solver_vel as usize, ndofs);
                j.dot(&rhs)
            }
            GenericRhs::Fixed => 0.0,
        }
    }

    #[inline]
    fn apply_impulse(
        &mut self,
        j_id: usize,
        ndofs: usize,
        impulse: Real,
        jacobians: &DVector,
        dir: Vector,
        ii_torque_dir: AngVector,
        solver_vels: &mut DVector,
        inv_mass: Vector,
    ) {
        match self {
            GenericRhs::SolverVel(rhs) => {
                rhs.linear += dir.component_mul(&inv_mass) * impulse;
                #[cfg(feature = "dim2")]
                {
                    // In 2D, angular values are scalars
                    rhs.angular += ii_torque_dir * impulse;
                }
                #[cfg(feature = "dim3")]
                {
                    rhs.angular += ii_torque_dir * impulse;
                }
            }
            GenericRhs::GenericId(solver_vel) => {
                let wj_id = j_id + ndofs;
                let wj = jacobians.rows(wj_id, ndofs);
                let mut rhs = solver_vels.rows_mut(*solver_vel as usize, ndofs);
                rhs.axpy(impulse, &wj, 1.0);
            }
            GenericRhs::Fixed => {}
        }
    }
}

impl ContactConstraintTangentPart<Real> {
    #[inline]
    pub fn generic_warmstart(
        &mut self,
        j_id: usize,
        jacobians: &DVector,
        tangents1: [Vector; DIM - 1],
        im1: Vector,
        im2: Vector,
        ndofs1: usize,
        ndofs2: usize,
        solver_vel1: &mut GenericRhs,
        solver_vel2: &mut GenericRhs,
        solver_vels: &mut DVector,
    ) {
        let j_id1 = j_id1(j_id, ndofs1, ndofs2);
        let j_id2 = j_id2(j_id, ndofs1, ndofs2);
        #[cfg(feature = "dim3")]
        let j_step = j_step(ndofs1, ndofs2);

        #[cfg(feature = "dim2")]
        {
            solver_vel1.apply_impulse(
                j_id1,
                ndofs1,
                self.impulse[0],
                jacobians,
                tangents1[0],
                self.ii_torque_dir1[0],
                solver_vels,
                im1,
            );
            solver_vel2.apply_impulse(
                j_id2,
                ndofs2,
                self.impulse[0],
                jacobians,
                -tangents1[0],
                self.ii_torque_dir2[0],
                solver_vels,
                im2,
            );
        }

        #[cfg(feature = "dim3")]
        {
            solver_vel1.apply_impulse(
                j_id1,
                ndofs1,
                self.impulse[0],
                jacobians,
                tangents1[0],
                self.ii_torque_dir1[0],
                solver_vels,
                im1,
            );
            solver_vel1.apply_impulse(
                j_id1 + j_step,
                ndofs1,
                self.impulse[1],
                jacobians,
                tangents1[1],
                self.ii_torque_dir1[1],
                solver_vels,
                im1,
            );

            solver_vel2.apply_impulse(
                j_id2,
                ndofs2,
                self.impulse[0],
                jacobians,
                -tangents1[0],
                self.ii_torque_dir2[0],
                solver_vels,
                im2,
            );
            solver_vel2.apply_impulse(
                j_id2 + j_step,
                ndofs2,
                self.impulse[1],
                jacobians,
                -tangents1[1],
                self.ii_torque_dir2[1],
                solver_vels,
                im2,
            );
        }
    }

    #[inline]
    pub fn generic_solve(
        &mut self,
        j_id: usize,
        jacobians: &DVector,
        tangents1: [Vector; DIM - 1],
        im1: Vector,
        im2: Vector,
        ndofs1: usize,
        ndofs2: usize,
        limit: Real,
        solver_vel1: &mut GenericRhs,
        solver_vel2: &mut GenericRhs,
        solver_vels: &mut DVector,
    ) {
        let j_id1 = j_id1(j_id, ndofs1, ndofs2);
        let j_id2 = j_id2(j_id, ndofs1, ndofs2);
        #[cfg(feature = "dim3")]
        let j_step = j_step(ndofs1, ndofs2);

        #[cfg(feature = "dim2")]
        {
            let dvel_0 = solver_vel1.dvel(
                j_id1,
                ndofs1,
                jacobians,
                tangents1[0],
                self.torque_dir1[0],
                solver_vels,
            ) + solver_vel2.dvel(
                j_id2,
                ndofs2,
                jacobians,
                -tangents1[0],
                self.torque_dir2[0],
                solver_vels,
            ) + self.rhs[0];

            let new_impulse = (self.impulse[0] - self.r[0] * dvel_0).simd_clamp(-limit, limit);
            let dlambda = new_impulse - self.impulse[0];
            self.impulse[0] = new_impulse;

            solver_vel1.apply_impulse(
                j_id1,
                ndofs1,
                dlambda,
                jacobians,
                tangents1[0],
                self.ii_torque_dir1[0],
                solver_vels,
                im1,
            );
            solver_vel2.apply_impulse(
                j_id2,
                ndofs2,
                dlambda,
                jacobians,
                -tangents1[0],
                self.ii_torque_dir2[0],
                solver_vels,
                im2,
            );
        }

        #[cfg(feature = "dim3")]
        {
            let dvel_0 = solver_vel1.dvel(
                j_id1,
                ndofs1,
                jacobians,
                tangents1[0],
                self.torque_dir1[0],
                solver_vels,
            ) + solver_vel2.dvel(
                j_id2,
                ndofs2,
                jacobians,
                -tangents1[0],
                self.torque_dir2[0],
                solver_vels,
            ) + self.rhs[0];
            let dvel_1 = solver_vel1.dvel(
                j_id1 + j_step,
                ndofs1,
                jacobians,
                tangents1[1],
                self.torque_dir1[1],
                solver_vels,
            ) + solver_vel2.dvel(
                j_id2 + j_step,
                ndofs2,
                jacobians,
                -tangents1[1],
                self.torque_dir2[1],
                solver_vels,
            ) + self.rhs[1];

            let new_impulse = na::Vector2::new(
                self.impulse[0] - self.r[0] * dvel_0,
                self.impulse[1] - self.r[1] * dvel_1,
            );
            let new_impulse = new_impulse.cap_magnitude(limit);

            let dlambda = new_impulse - self.impulse;
            self.impulse = new_impulse;

            solver_vel1.apply_impulse(
                j_id1,
                ndofs1,
                dlambda[0],
                jacobians,
                tangents1[0],
                self.ii_torque_dir1[0],
                solver_vels,
                im1,
            );
            solver_vel1.apply_impulse(
                j_id1 + j_step,
                ndofs1,
                dlambda[1],
                jacobians,
                tangents1[1],
                self.ii_torque_dir1[1],
                solver_vels,
                im1,
            );

            solver_vel2.apply_impulse(
                j_id2,
                ndofs2,
                dlambda[0],
                jacobians,
                -tangents1[0],
                self.ii_torque_dir2[0],
                solver_vels,
                im2,
            );
            solver_vel2.apply_impulse(
                j_id2 + j_step,
                ndofs2,
                dlambda[1],
                jacobians,
                -tangents1[1],
                self.ii_torque_dir2[1],
                solver_vels,
                im2,
            );
        }
    }
}

impl ContactConstraintNormalPart<Real> {
    #[inline]
    pub fn generic_warmstart(
        &mut self,
        j_id: usize,
        jacobians: &DVector,
        dir1: Vector,
        im1: Vector,
        im2: Vector,
        ndofs1: usize,
        ndofs2: usize,
        solver_vel1: &mut GenericRhs,
        solver_vel2: &mut GenericRhs,
        solver_vels: &mut DVector,
    ) {
        let j_id1 = j_id1(j_id, ndofs1, ndofs2);
        let j_id2 = j_id2(j_id, ndofs1, ndofs2);

        solver_vel1.apply_impulse(
            j_id1,
            ndofs1,
            self.impulse,
            jacobians,
            dir1,
            self.ii_torque_dir1,
            solver_vels,
            im1,
        );
        solver_vel2.apply_impulse(
            j_id2,
            ndofs2,
            self.impulse,
            jacobians,
            -dir1,
            self.ii_torque_dir2,
            solver_vels,
            im2,
        );
    }

    #[inline]
    pub fn generic_solve(
        &mut self,
        cfm_factor: Real,
        j_id: usize,
        jacobians: &DVector,
        dir1: Vector,
        im1: Vector,
        im2: Vector,
        ndofs1: usize,
        ndofs2: usize,
        solver_vel1: &mut GenericRhs,
        solver_vel2: &mut GenericRhs,
        solver_vels: &mut DVector,
    ) {
        let j_id1 = j_id1(j_id, ndofs1, ndofs2);
        let j_id2 = j_id2(j_id, ndofs1, ndofs2);

        let dvel = solver_vel1.dvel(
            j_id1,
            ndofs1,
            jacobians,
            dir1,
            self.torque_dir1,
            solver_vels,
        ) + solver_vel2.dvel(
            j_id2,
            ndofs2,
            jacobians,
            -dir1,
            self.torque_dir2,
            solver_vels,
        ) + self.rhs;

        let new_impulse = cfm_factor * (self.impulse - self.r * dvel).max(0.0);
        let dlambda = new_impulse - self.impulse;
        self.impulse = new_impulse;

        solver_vel1.apply_impulse(
            j_id1,
            ndofs1,
            dlambda,
            jacobians,
            dir1,
            self.ii_torque_dir1,
            solver_vels,
            im1,
        );
        solver_vel2.apply_impulse(
            j_id2,
            ndofs2,
            dlambda,
            jacobians,
            -dir1,
            self.ii_torque_dir2,
            solver_vels,
            im2,
        );
    }
}

impl GenericContactConstraint {
    #[inline]
    pub fn generic_warmstart_group(
        normal_parts: &mut [ContactConstraintNormalPart<Real>],
        tangent_parts: &mut [ContactConstraintTangentPart<Real>],
        jacobians: &DVector,
        dir1: Vector,
        #[cfg(feature = "dim3")] tangent1: Vector,
        im1: Vector,
        im2: Vector,
        // ndofs is 0 for a non-multibody body, or a multibody with zero
        // degrees of freedom.
        ndofs1: usize,
        ndofs2: usize,
        // Jacobian index of the first constraint.
        j_id: usize,
        solver_vel1: &mut GenericRhs,
        solver_vel2: &mut GenericRhs,
        solver_vels: &mut DVector,
    ) {
        let j_step = j_step(ndofs1, ndofs2) * DIM;

        // Solve penetration.
        {
            let mut nrm_j_id = normal_j_id(j_id, ndofs1, ndofs2);

            for normal_part in normal_parts {
                normal_part.generic_warmstart(
                    nrm_j_id,
                    jacobians,
                    dir1,
                    im1,
                    im2,
                    ndofs1,
                    ndofs2,
                    solver_vel1,
                    solver_vel2,
                    solver_vels,
                );
                nrm_j_id += j_step;
            }
        }

        // Solve friction.
        {
            #[cfg(feature = "dim3")]
            let tangents1 = [tangent1, dir1.cross(tangent1)];
            #[cfg(feature = "dim2")]
            let tangents1 = [dir1.orthonormal_vector()];
            let mut tng_j_id = tangent_j_id(j_id, ndofs1, ndofs2);

            for tangent_part in tangent_parts {
                tangent_part.generic_warmstart(
                    tng_j_id,
                    jacobians,
                    tangents1,
                    im1,
                    im2,
                    ndofs1,
                    ndofs2,
                    solver_vel1,
                    solver_vel2,
                    solver_vels,
                );
                tng_j_id += j_step;
            }
        }
    }

    #[inline]
    pub fn generic_solve_group(
        cfm_factor: Real,
        normal_parts: &mut [ContactConstraintNormalPart<Real>],
        tangent_parts: &mut [ContactConstraintTangentPart<Real>],
        jacobians: &DVector,
        dir1: Vector,
        #[cfg(feature = "dim3")] tangent1: Vector,
        im1: Vector,
        im2: Vector,
        limit: Real,
        // ndofs is 0 for a non-multibody body, or a multibody with zero
        // degrees of freedom.
        ndofs1: usize,
        ndofs2: usize,
        // Jacobian index of the first constraint.
        j_id: usize,
        solver_vel1: &mut GenericRhs,
        solver_vel2: &mut GenericRhs,
        solver_vels: &mut DVector,
        solve_restitution: bool,
        solve_friction: bool,
    ) {
        let j_step = j_step(ndofs1, ndofs2) * DIM;

        // Solve penetration.
        if solve_restitution {
            let mut nrm_j_id = normal_j_id(j_id, ndofs1, ndofs2);

            for normal_part in &mut *normal_parts {
                normal_part.generic_solve(
                    cfm_factor,
                    nrm_j_id,
                    jacobians,
                    dir1,
                    im1,
                    im2,
                    ndofs1,
                    ndofs2,
                    solver_vel1,
                    solver_vel2,
                    solver_vels,
                );
                nrm_j_id += j_step;
            }
        }

        // Solve friction.
        if solve_friction {
            #[cfg(feature = "dim3")]
            let tangents1 = [tangent1, dir1.cross(tangent1)];
            #[cfg(feature = "dim2")]
            let tangents1 = [dir1.orthonormal_vector()];
            let mut tng_j_id = tangent_j_id(j_id, ndofs1, ndofs2);

            for (normal_part, tangent_part) in normal_parts.iter().zip(tangent_parts.iter_mut()) {
                let limit = limit * normal_part.impulse;
                tangent_part.generic_solve(
                    tng_j_id,
                    jacobians,
                    tangents1,
                    im1,
                    im2,
                    ndofs1,
                    ndofs2,
                    limit,
                    solver_vel1,
                    solver_vel2,
                    solver_vels,
                );
                tng_j_id += j_step;
            }
        }
    }
}

impl GenericContactConstraint {
    /// The normal rows of [`Self::generic_solve_group`], each solved as one block with the
    /// engaged joint rows of `rows` that act on a multibody it acts on (see
    /// [`crate::dynamics::solver::coupled_multibody_rows`]). A row with no engaged joint row on
    /// its multibodies is solved alone, as `generic_solve_group` solves it. `solver_vel1_id` and
    /// `solver_vel2_id` name the solver bodies a rigid side's velocity is held for, so the joint
    /// rows read and move the velocity the contact does, and a joint row acting on the same
    /// rigid body couples with the contact through it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generic_solve_normals_with_joint_rows(
        cfm_factor: Real,
        normal_parts: &mut [ContactConstraintNormalPart<Real>],
        jacobians: &DVector,
        dir1: Vector,
        im1: Vector,
        im2: Vector,
        ndofs1: usize,
        ndofs2: usize,
        j_id: usize,
        solver_vel1: &mut GenericRhs,
        solver_vel1_id: u32,
        solver_vel2: &mut GenericRhs,
        solver_vel2_id: u32,
        bodies: &mut SolverBodies,
        solver_vels: &mut DVector,
        joints: &mut [GenericJointConstraint],
        joint_jacobians: &DVector,
        rows: &mut CoupledRows,
    ) {
        let j_step = j_step(ndofs1, ndofs2) * DIM;
        let mut nrm_j_id = normal_j_id(j_id, ndofs1, ndofs2);

        for normal_part in normal_parts {
            let j_id1 = j_id1(nrm_j_id, ndofs1, ndofs2);
            let j_id2 = j_id2(nrm_j_id, ndofs1, ndofs2);
            let sides = [
                solver_vel1.side(
                    solver_vel1_id,
                    j_id1,
                    ndofs1,
                    dir1,
                    normal_part.torque_dir1,
                    normal_part.ii_torque_dir1,
                    im1,
                ),
                solver_vel2.side(
                    solver_vel2_id,
                    j_id2,
                    ndofs2,
                    -dir1,
                    normal_part.torque_dir2,
                    normal_part.ii_torque_dir2,
                    im2,
                ),
            ];
            let r = normal_part.r;
            let dvel = solver_vel1.dvel(
                j_id1,
                ndofs1,
                jacobians,
                dir1,
                normal_part.torque_dir1,
                solver_vels,
            ) + solver_vel2.dvel(
                j_id2,
                ndofs2,
                jacobians,
                -dir1,
                normal_part.torque_dir2,
                solver_vels,
            ) + normal_part.rhs;
            // A point that bears no impulse and separates takes none: the block would only
            // solve again the joint rows their own solve just left.
            let takes_impulse = normal_part.impulse > 0.0 || dvel < 0.0;
            if takes_impulse && r > 0.0 && cfm_factor > 0.0 {
                rows.select(&[sides], jacobians, joint_jacobians);
            } else {
                rows.block.clear();
            }

            let solved = !rows.block.is_empty() && {
                let m = rows.block.len();
                let n = m + 1;
                rows.begin_block(n);
                solver_vel1.store(solver_vel1_id, bodies);
                solver_vel2.store(solver_vel2_id, bodies);
                for i in 0..m {
                    let p = rows.block[i];
                    let row = &joints[rows.engaged[p]];
                    for k in 0..m {
                        rows.matrix[i * n + k] = rows.engaged_coupling(p, rows.block[k]);
                    }
                    rows.matrix[i * n + m] = rows.block_coupling[i].0;
                    rows.matrix[m * n + i] = rows.block_coupling[i].1;
                    rows.error[i] = row.cfm_gain * row.impulse
                        - row.velocity_error(joint_jacobians, bodies, solver_vels);
                    rows.impulse[i] = row.impulse;
                    rows.bounds[i] = row.impulse_bounds;
                }
                // The contact's own step is `cfm_factor * (impulse - r * dvel)`: its error is
                // `dvel` plus the softness that step settles at, and `1 / (cfm_factor * r)` is
                // the rate an impulse on it changes that error.
                let softness = (1.0 - cfm_factor) / (cfm_factor * r);
                rows.matrix[m * n + m] = 1.0 / (cfm_factor * r);
                rows.error[m] = dvel + softness * normal_part.impulse;
                rows.impulse[m] = normal_part.impulse;
                rows.bounds[m] = [0.0, Real::MAX];
                rows.solve_block()
            };

            if solved {
                let m = rows.block.len();
                let dlambda = rows.delta[m];
                normal_part.impulse += dlambda;
                solver_vel1.apply_impulse(
                    j_id1,
                    ndofs1,
                    dlambda,
                    jacobians,
                    dir1,
                    normal_part.ii_torque_dir1,
                    solver_vels,
                    im1,
                );
                solver_vel2.apply_impulse(
                    j_id2,
                    ndofs2,
                    dlambda,
                    jacobians,
                    -dir1,
                    normal_part.ii_torque_dir2,
                    solver_vels,
                    im2,
                );
                solver_vel1.store(solver_vel1_id, bodies);
                solver_vel2.store(solver_vel2_id, bodies);
                for i in 0..m {
                    let row = rows.engaged[rows.block[i]];
                    joints[row].apply_impulse_delta(
                        rows.delta[i],
                        joint_jacobians,
                        bodies,
                        solver_vels,
                    );
                }
                solver_vel1.load(solver_vel1_id, bodies);
                solver_vel2.load(solver_vel2_id, bodies);
            } else {
                normal_part.generic_solve(
                    cfm_factor,
                    nrm_j_id,
                    jacobians,
                    dir1,
                    im1,
                    im2,
                    ndofs1,
                    ndofs2,
                    solver_vel1,
                    solver_vel2,
                    solver_vels,
                );
            }
            nrm_j_id += j_step;
        }
    }
}

impl GenericContactConstraint {
    /// The friction rows of [`Self::generic_solve_group`], each point's tangent rows solved as
    /// one block with the engaged joint rows of `rows` that act on a multibody the contact acts
    /// on, as [`Self::generic_solve_normals_with_joint_rows`] solves the normal rows. Each tangent
    /// impulse is bounded by `limit` times the point's normal impulse; in 3D a pair the block
    /// carries off its disc is put back on its edge, and the joint rows solved again around it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generic_solve_tangents_with_joint_rows(
        normal_parts: &[ContactConstraintNormalPart<Real>],
        tangent_parts: &mut [ContactConstraintTangentPart<Real>],
        jacobians: &DVector,
        dir1: Vector,
        #[cfg(feature = "dim3")] tangent1: Vector,
        im1: Vector,
        im2: Vector,
        limit: Real,
        ndofs1: usize,
        ndofs2: usize,
        j_id: usize,
        solver_vel1: &mut GenericRhs,
        solver_vel1_id: u32,
        solver_vel2: &mut GenericRhs,
        solver_vel2_id: u32,
        bodies: &mut SolverBodies,
        solver_vels: &mut DVector,
        joints: &mut [GenericJointConstraint],
        joint_jacobians: &DVector,
        rows: &mut CoupledRows,
    ) {
        #[cfg(feature = "dim3")]
        let tangents1 = [tangent1, dir1.cross(tangent1)];
        #[cfg(feature = "dim2")]
        let tangents1 = [dir1.orthonormal_vector()];
        const NT: usize = DIM - 1;
        let row_step = j_step(ndofs1, ndofs2);
        let point_step = row_step * DIM;
        let mut tng_j_id = tangent_j_id(j_id, ndofs1, ndofs2);

        for (normal_part, tangent_part) in normal_parts.iter().zip(tangent_parts.iter_mut()) {
            let bound = limit * normal_part.impulse;
            let mut sides = [[None, None]; NT];
            let mut ids = [(0, 0); NT];
            for t in 0..NT {
                let row_j_id = tng_j_id + t * row_step;
                let j_id1 = j_id1(row_j_id, ndofs1, ndofs2);
                let j_id2 = j_id2(row_j_id, ndofs1, ndofs2);
                ids[t] = (j_id1, j_id2);
                sides[t] = [
                    solver_vel1.side(
                        solver_vel1_id,
                        j_id1,
                        ndofs1,
                        tangents1[t],
                        tangent_part.torque_dir1[t],
                        tangent_part.ii_torque_dir1[t],
                        im1,
                    ),
                    solver_vel2.side(
                        solver_vel2_id,
                        j_id2,
                        ndofs2,
                        -tangents1[t],
                        tangent_part.torque_dir2[t],
                        tangent_part.ii_torque_dir2[t],
                        im2,
                    ),
                ];
            }
            // With no normal impulse the point's friction is held at zero, which a tangent row
            // already at zero is: the block would only solve the joint rows again.
            let takes_impulse = bound > 0.0 || (0..NT).any(|t| tangent_part.impulse[t] != 0.0);
            if takes_impulse && (0..NT).all(|t| tangent_part.r[t] > 0.0) {
                rows.select(&sides, jacobians, joint_jacobians);
            } else {
                rows.block.clear();
            }

            let solved = !rows.block.is_empty() && {
                let m = rows.block.len();
                let n = m + NT;
                rows.begin_block(n);
                solver_vel1.store(solver_vel1_id, bodies);
                solver_vel2.store(solver_vel2_id, bodies);
                for i in 0..m {
                    let p = rows.block[i];
                    let row = &joints[rows.engaged[p]];
                    for k in 0..m {
                        rows.matrix[i * n + k] = rows.engaged_coupling(p, rows.block[k]);
                    }
                    for t in 0..NT {
                        let (to_row, to_contact) = rows.block_coupling[i * NT + t];
                        rows.matrix[i * n + m + t] = to_row;
                        rows.matrix[(m + t) * n + i] = to_contact;
                    }
                    rows.error[i] = row.cfm_gain * row.impulse
                        - row.velocity_error(joint_jacobians, bodies, solver_vels);
                    rows.impulse[i] = row.impulse;
                    rows.bounds[i] = row.impulse_bounds;
                }
                for t in 0..NT {
                    for u in 0..NT {
                        rows.matrix[(m + t) * n + m + u] = if t == u {
                            1.0 / tangent_part.r[t]
                        } else {
                            coupling(&sides[t], jacobians, &sides[u], jacobians)
                        };
                    }
                    let (j_id1, j_id2) = ids[t];
                    rows.error[m + t] = solver_vel1.dvel(
                        j_id1,
                        ndofs1,
                        jacobians,
                        tangents1[t],
                        tangent_part.torque_dir1[t],
                        solver_vels,
                    ) + solver_vel2.dvel(
                        j_id2,
                        ndofs2,
                        jacobians,
                        -tangents1[t],
                        tangent_part.torque_dir2[t],
                        solver_vels,
                    ) + tangent_part.rhs[t];
                    rows.impulse[m + t] = tangent_part.impulse[t];
                    rows.bounds[m + t] = [-bound, bound];
                }
                rows.solve_block() && {
                    #[cfg(feature = "dim3")]
                    {
                        let new = na::Vector2::new(
                            rows.impulse[m] + rows.delta[m],
                            rows.impulse[m + 1] + rows.delta[m + 1],
                        );
                        if new.norm() > bound {
                            let on_edge = new.cap_magnitude(bound);
                            for t in 0..NT {
                                rows.bounds[m + t] = [on_edge[t], on_edge[t]];
                            }
                            rows.release();
                            rows.solve_block()
                        } else {
                            true
                        }
                    }
                    #[cfg(feature = "dim2")]
                    {
                        true
                    }
                }
            };

            if solved {
                let m = rows.block.len();
                for t in 0..NT {
                    let (j_id1, j_id2) = ids[t];
                    let dlambda = rows.delta[m + t];
                    tangent_part.impulse[t] += dlambda;
                    solver_vel1.apply_impulse(
                        j_id1,
                        ndofs1,
                        dlambda,
                        jacobians,
                        tangents1[t],
                        tangent_part.ii_torque_dir1[t],
                        solver_vels,
                        im1,
                    );
                    solver_vel2.apply_impulse(
                        j_id2,
                        ndofs2,
                        dlambda,
                        jacobians,
                        -tangents1[t],
                        tangent_part.ii_torque_dir2[t],
                        solver_vels,
                        im2,
                    );
                }
                solver_vel1.store(solver_vel1_id, bodies);
                solver_vel2.store(solver_vel2_id, bodies);
                for i in 0..m {
                    let row = rows.engaged[rows.block[i]];
                    joints[row].apply_impulse_delta(
                        rows.delta[i],
                        joint_jacobians,
                        bodies,
                        solver_vels,
                    );
                }
                solver_vel1.load(solver_vel1_id, bodies);
                solver_vel2.load(solver_vel2_id, bodies);
            } else {
                tangent_part.generic_solve(
                    tng_j_id,
                    jacobians,
                    tangents1,
                    im1,
                    im2,
                    ndofs1,
                    ndofs2,
                    bound,
                    solver_vel1,
                    solver_vel2,
                    solver_vels,
                );
            }
            tng_j_id += point_step;
        }
    }
}
