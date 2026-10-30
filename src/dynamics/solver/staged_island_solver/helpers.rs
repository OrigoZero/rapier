//! Staged-solver entry points on [`ContactConstraintsSet`] and
//! [`VelocitySolver`]: generic (multibody) constraint init and the
//! multibody-only halves of buffer init and position integration.

use crate::alloc_prelude::*;
use crate::dynamics::solver::SolverVel;
use crate::dynamics::solver::VelocitySolver;
use crate::dynamics::solver::contact_constraint::{
    ContactConstraintsSet, GenericContactConstraint, GenericContactConstraintBuilder,
};
use crate::dynamics::solver::manifold_store::ManifoldStore;
use crate::dynamics::solver::solver_contact_graph::SolverContactGraph;
#[cfg(feature = "dim3")]
use crate::dynamics::solver::velocity_solver::GyroParams;
use crate::dynamics::{IntegrationParameters, MultibodyJointSet, RigidBodyHandle, RigidBodySet};
use crate::math::DVector;
use parry::math::SIMD_WIDTH;

impl ContactConstraintsSet {
    /// Builds the generic (multibody) contact constraints for the staged solver.
    pub(super) fn compute_generic_constraints(
        &mut self,
        bodies: &RigidBodySet,
        multibody_joints: &MultibodyJointSet,
        graph: &SolverContactGraph,
        store: &ManifoldStore,
        jacobian_id: &mut usize,
    ) {
        // The generic list holds its manifolds in the order they qualified, which an
        // incremental update reorders (it appends and swap-removes), and its constraints
        // share the multibodies they touch, so the order they are solved in decides the
        // result. They are solved in ascending (edge, manifold), the layout a full rebuild
        // gives them, so the result depends on the contacts alone and not on the history
        // of updates that brought the list to them.
        let mut generic = core::mem::take(&mut self.generic_order);
        generic.clear();
        generic.extend_from_slice(graph.generic());
        generic.sort_unstable_by_key(|r| (r.edge, r.manifold));
        let total_num_constraints = generic.len();
        self.generic_velocity_constraints_builder.resize(
            total_num_constraints,
            GenericContactConstraintBuilder::invalid(),
        );
        self.generic_velocity_constraints
            .resize(total_num_constraints, GenericContactConstraint::invalid());

        for (curr_id, r) in generic.iter().enumerate() {
            let manifold = store.get(*r);
            GenericContactConstraintBuilder::generate(
                *r,
                manifold,
                bodies,
                multibody_joints,
                &mut self.generic_velocity_constraints_builder[curr_id],
                &mut self.generic_velocity_constraints[curr_id],
                &mut self.generic_jacobians,
                jacobian_id,
            );
        }
        self.generic_order = generic;
    }
}

impl VelocitySolver {
    /// Initializes the solver buffers and multibody solver state; the plain rigid-body
    /// initialization is excluded, the staged solver performs it in parallel in its first stage.
    pub(super) fn init_solver_buffers_and_multibodies(
        &mut self,
        params: &IntegrationParameters,
        island_bodies: &[RigidBodyHandle],
        num_awake_soft_mass_points: usize,
        bodies: &mut RigidBodySet,
        multibodies: &mut MultibodyJointSet,
    ) {
        self.multibody_roots.clear();
        self.solver_bodies.clear();

        // The island's rigid bodies, then the awake soft bodies' particles.
        let aligned_solver_bodies_len =
            (island_bodies.len() + num_awake_soft_mass_points).div_ceil(SIMD_WIDTH) * SIMD_WIDTH;
        self.solver_bodies.resize(aligned_solver_bodies_len);

        self.solver_vels_increment.clear();
        self.solver_vels_increment
            .resize(aligned_solver_bodies_len, SolverVel::zero());

        // Reset every step so multibody/padding slots (which the body-copy stage
        // skips) read as gyro-disabled.
        #[cfg(feature = "dim3")]
        {
            self.solver_gyro.clear();
            self.solver_gyro
                .resize(aligned_solver_bodies_len, GyroParams::default());
        }

        // Grown, never shrunk: a slot is read only once the stage owning it flagged and wrote it.
        if self.solver_gravity.len() < aligned_solver_bodies_len {
            self.solver_gravity
                .resize(aligned_solver_bodies_len, Default::default());
        }

        // Grown, never shrunk: the slots of bodies that record their substeps keep their
        // allocations from one step to the next, and a slot no recording body holds is never read.
        if self.substep_vels.len() < aligned_solver_bodies_len {
            self.substep_vels.resize_with(aligned_solver_bodies_len, Vec::new);
        }

        // Assign solver ids to multibodies, and collect the relevant roots.
        let mut multibody_solver_id = 0;
        for handle in island_bodies {
            if let Some(link) = multibodies.rigid_body_link(*handle).copied() {
                let multibody = multibodies
                    .get_multibody_mut_internal(link.multibody)
                    .unwrap();

                if link.id == 0 || link.id == 1 && !multibody.root_is_dynamic {
                    multibody.solver_id = multibody_solver_id;
                    multibody_solver_id += multibody.ndofs() as u32;
                    self.multibody_roots.push(link);
                }
            }
        }

        self.generic_solver_vels_increment = DVector::zeros(multibody_solver_id as usize);
        self.generic_solver_vels = DVector::zeros(multibody_solver_id as usize);

        for link in &self.multibody_roots {
            let multibody = multibodies
                .get_multibody_mut_internal(link.multibody)
                .unwrap();
            multibody.update_velocities(bodies);
            multibody.update_mass_matrix(params.dt, bodies);
            multibody.update_acceleration(params.dt, bodies);

            let mut solver_vels_incr = self
                .generic_solver_vels_increment
                .rows_mut(multibody.solver_id as usize, multibody.ndofs());
            let mut solver_vels = self
                .generic_solver_vels
                .rows_mut(multibody.solver_id as usize, multibody.ndofs());

            solver_vels_incr.axpy(params.dt, &multibody.accelerations, 0.0);
            solver_vels.copy_from(&multibody.velocities);
        }
    }

    /// The multibody part of `integrate_positions`, used by the staged solver
    /// (regular solver bodies are integrated in parallel by the workers).
    pub(super) fn integrate_multibody_positions(
        &mut self,
        params: &IntegrationParameters,
        is_last_substep: bool,
        bodies: &mut RigidBodySet,
        multibodies: &mut MultibodyJointSet,
    ) {
        for link in &self.multibody_roots {
            let multibody = multibodies
                .get_multibody_mut_internal(link.multibody)
                .unwrap();
            let solver_vels = self
                .generic_solver_vels
                .rows(multibody.solver_id as usize, multibody.ndofs());
            multibody.velocities.copy_from(&solver_vels);
            multibody.integrate(params.dt);
            multibody.forward_kinematics(bodies, false);
            multibody.update_rigid_bodies_internal(bodies, !is_last_substep, true, false);

            if !is_last_substep {
                multibody.update_velocities(bodies);
                multibody.update_mass_matrix(params.dt, bodies);
                multibody.update_acceleration(params.dt, bodies);

                let mut solver_vels_incr = self
                    .generic_solver_vels_increment
                    .rows_mut(multibody.solver_id as usize, multibody.ndofs());
                solver_vels_incr.axpy(params.dt, &multibody.accelerations, 0.0);
            }
        }
    }
}
