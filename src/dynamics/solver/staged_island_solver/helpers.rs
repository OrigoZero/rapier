//! Staged-solver entry points on [`ContactConstraintsSet`] and
//! [`VelocitySolver`]: generic (multibody) constraint init and the
//! multibody-only halves of buffer init and position integration.

use crate::alloc_prelude::*;
use crate::dynamics::solver::GenericJointConstraint;
use crate::dynamics::solver::SolverVel;
use crate::dynamics::solver::VelocitySolver;
use crate::dynamics::solver::contact_constraint::{
    ContactConstraintsSet, GenericContactConstraint, GenericContactConstraintBuilder,
};
use crate::dynamics::solver::manifold_store::ManifoldStore;
use crate::dynamics::solver::solver_contact_graph::SolverContactGraph;
#[cfg(feature = "dim3")]
use crate::dynamics::solver::velocity_solver::GyroParams;
use crate::dynamics::{
    IntegrationParameters, Multibody, MultibodyForces, MultibodyJointSet, RigidBodyHandle,
    RigidBodySet,
};
use crate::math::{AngVector, DVector, Vector};
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
        // result. They are solved in the order of the colliders each pair joins (the lower
        // collider index, then the higher, then the manifold's ordinal in its pair): the
        // contact graph's edge indices move as edges are removed, and collider indices
        // depend only on which slots of the collider set are occupied, so the result
        // depends on the contacts alone and not on the history that brought the list or
        // the graph to them.
        let mut generic = core::mem::take(&mut self.generic_order);
        generic.clear();
        generic.extend_from_slice(graph.generic());
        generic.sort_unstable_by_key(|r| {
            let (a, b) = store.collider_indices(*r);
            (a, b, r.manifold)
        });
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
        if self.ended_vels.len() < aligned_solver_bodies_len {
            self.ended_vels
                .resize(aligned_solver_bodies_len, crate::math::Vector::ZERO);
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
            multibody.update_acceleration(params.dt, bodies, MultibodyForces::Applied);

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

    /// The multibody part of `integrate_positions`, used by the staged solver (regular solver
    /// bodies are integrated in parallel by the workers): each multibody moves through the
    /// substep at the velocity its biased solve left, and its inertial forces then turn that
    /// velocity for the configuration the move reached. The kinetic energy the substep is held
    /// to, when its energy is held, is kept for [`Self::update_multibody_dynamics`]: the energy
    /// it moves with, or, for a multibody no row of the substep ties to something that moves or
    /// drives it ([`Self::mark_driven_multibodies`]), the energy it started the substep with
    /// before any row was solved if that is less.
    ///
    /// The rows that hold a multibody against fixed bodies and bodies at rest, and its own
    /// joints' limits, give it no energy, but their biased pass pushes it out of what it sank
    /// into and that push stood in the energy the substep was held to: a chain hanging from its
    /// anchor with its second link knocking on the anchor's box took tens of kilojoules from a
    /// knock, read back as a ball flung at 14 m/s, and at 951 m/s once the continuous pass
    /// swept it. A row that ties it to something that moves, or a motor, gives a struck or
    /// driven multibody energy it keeps.
    ///
    /// A move carries each joint's velocity unchanged in its parent's frame, so a link that
    /// turns over the move turns the velocity of every link below it with it, and velocities
    /// that cancelled out at the configuration the move began from no longer do: a chain of
    /// light links whipping under a heavy load stands, after a move that turns a joint a few
    /// hundredths of a radian, at many times the kinetic energy it moved with. The Coriolis and
    /// centrifugal forces at the configuration reached take that back, read here at the velocity
    /// the multibody moved with. The stabilization pass that follows solves the substep's
    /// contacts and limits at the velocity they leave: solved at the velocity the move left, it
    /// took a share of that excess out as though the links moved with it, the forces read
    /// afterwards took out the rest again, and a chain whose link slid on its anchor's body
    /// gained energy at every swing. The mass matrix built here is the one the applied forces
    /// and the next substep's rows solve through.
    pub(super) fn integrate_multibody_positions(
        &mut self,
        params: &IntegrationParameters,
        bodies: &mut RigidBodySet,
        multibodies: &mut MultibodyJointSet,
        contacts: &[GenericContactConstraint],
        joints: &[GenericJointConstraint],
    ) {
        self.mark_driven_multibodies(multibodies, contacts, joints);
        self.multibody_energy_holds.clear();
        self.multibody_moved_energies.clear();
        for (root, link) in self.multibody_roots.iter().enumerate() {
            let multibody = multibodies
                .get_multibody_mut_internal(link.multibody)
                .unwrap();
            let solver_vels = self
                .generic_solver_vels
                .rows(multibody.solver_id as usize, multibody.ndofs());
            multibody.velocities.copy_from(&solver_vels);
            let mut energy =
                multibody.kinetic_energy_to_hold(multibody.velocities.as_view(), bodies);
            self.multibody_moved_energies.push(energy.unwrap_or(0.0));
            if !self.multibody_driven[root] {
                let free = self
                    .generic_free_vels
                    .rows(multibody.solver_id as usize, multibody.ndofs());
                if let (Some(e), Some(f)) = (energy, multibody.kinetic_energy_to_hold(free, bodies))
                {
                    energy = Some(e.min(f));
                }
            }
            multibody.integrate(params.dt);
            multibody.forward_kinematics(bodies, false);
            multibody.update_rigid_bodies_internal(bodies, true, true, false);
            multibody.update_velocities(bodies);
            multibody.update_mass_matrix(params.dt, bodies);
            multibody.update_acceleration(params.dt, bodies, MultibodyForces::Inertial);
            let Multibody {
                velocities,
                accelerations,
                ..
            } = &mut *multibody;
            velocities.axpy(params.dt, accelerations, 1.0);
            self.generic_solver_vels
                .rows_mut(multibody.solver_id as usize, multibody.ndofs())
                .copy_from(&multibody.velocities);
            multibody.update_velocities(bodies);
            self.multibody_energy_holds.push(energy);
        }
    }

    /// Marks in `multibody_driven` each multibody of `multibody_roots` that something moves or
    /// drives through this substep's rows: a motor of one of its joints, a contact of `contacts`
    /// with a dynamic body, a moving kinematic body or another multibody, or a row of `joints`
    /// tying it to a rigid body or another multibody.
    fn mark_driven_multibodies(
        &mut self,
        multibodies: &MultibodyJointSet,
        contacts: &[GenericContactConstraint],
        joints: &[GenericJointConstraint],
    ) {
        self.multibody_driven.clear();
        self.multibody_driven
            .resize(self.multibody_roots.len(), false);
        // The roots take their solver ids in order, each after the one before it.
        let solver_ids: Vec<u32> = self
            .multibody_roots
            .iter()
            .map(|link| {
                multibodies
                    .get_multibody(link.multibody)
                    .map_or(u32::MAX, |multibody| multibody.solver_id)
            })
            .collect();
        let root_of = |solver_vel: u32| solver_ids.binary_search(&solver_vel).ok();
        for (root, link) in self.multibody_roots.iter().enumerate() {
            let motorized = multibodies
                .get_multibody(link.multibody)
                .is_some_and(|multibody| {
                    multibody
                        .links()
                        .any(|link| !link.joint.data.motor_axes.is_empty())
                });
            self.multibody_driven[root] = motorized;
        }
        for c in contacts {
            let generic1 = c.solver_vel1 != u32::MAX && c.generic_constraint_mask & 0b01 != 0;
            let generic2 = c.solver_vel2 != u32::MAX && c.generic_constraint_mask & 0b10 != 0;
            // Whether the side `solver_vel` of a contact whose other side is the multibody
            // `own` moves: another multibody, a dynamic body, or a kinematic one in motion.
            let moves = |solver_vel: u32, generic: bool, im: Vector, own: u32| {
                if solver_vel == u32::MAX {
                    false
                } else if generic {
                    solver_vel != own
                } else {
                    let vel = &self.solver_bodies.vels[solver_vel as usize];
                    im != Vector::ZERO
                        || vel.linear != Vector::ZERO
                        || vel.angular != AngVector::default()
                }
            };
            if generic1 && moves(c.solver_vel2, generic2, c.im2, c.solver_vel1) {
                if let Some(root) = root_of(c.solver_vel1) {
                    self.multibody_driven[root] = true;
                }
            }
            if generic2 && moves(c.solver_vel1, generic1, c.im1, c.solver_vel2) {
                if let Some(root) = root_of(c.solver_vel2) {
                    self.multibody_driven[root] = true;
                }
            }
        }
        for j in joints {
            let sides = [
                (j.solver_vel1, j.is_rigid_body1),
                (j.solver_vel2, j.is_rigid_body2),
            ];
            for (own, other) in [(sides[0], sides[1]), (sides[1], sides[0])] {
                let (own_vel, own_rigid) = own;
                if own_vel == u32::MAX || own_rigid {
                    continue;
                }
                let (other_vel, other_rigid) = other;
                let tied = other_vel != u32::MAX && (other_rigid || other_vel != own_vel);
                if tied {
                    if let Some(root) = root_of(own_vel) {
                        self.multibody_driven[root] = true;
                    }
                }
            }
        }
    }

    /// The dynamics each multibody starts the next substep with, read once the substep's
    /// stabilization pass has settled the velocity it ends at: the links' velocities, the
    /// accelerations its applied forces give it, the energy hold of
    /// [`Self::integrate_multibody_positions`] and the velocity increment of the next substep.
    /// After the last substep the velocity the step ends with is held as well
    /// (`Multibody::step_end_energy_scale`), to the larger of the energy the substep was held
    /// to and the energy it moved with.
    pub(super) fn update_multibody_dynamics(
        &mut self,
        params: &IntegrationParameters,
        is_last_substep: bool,
        bodies: &mut RigidBodySet,
        multibodies: &mut MultibodyJointSet,
    ) {
        for ((link, energy), moved) in self
            .multibody_roots
            .iter()
            .zip(&self.multibody_energy_holds)
            .zip(&self.multibody_moved_energies)
        {
            if is_last_substep && energy.is_none() {
                continue;
            }
            let multibody = multibodies
                .get_multibody_mut_internal(link.multibody)
                .unwrap();
            let solver_vels = self
                .generic_solver_vels
                .rows(multibody.solver_id as usize, multibody.ndofs());
            multibody.velocities.copy_from(&solver_vels);
            multibody.update_velocities(bodies);
            multibody.update_acceleration(params.dt, bodies, MultibodyForces::Applied);
            if let Some(energy) = *energy {
                let mut scale = multibody.kinetic_energy_scale(energy, params.dt, bodies);
                if is_last_substep {
                    let cap = energy.max(*moved);
                    scale = scale.min(multibody.step_end_energy_scale(cap, params.dt, bodies));
                }
                if scale < 1.0 {
                    multibody.velocities *= scale;
                    multibody.accelerations *= scale;
                    let mut solver_vels = self
                        .generic_solver_vels
                        .rows_mut(multibody.solver_id as usize, multibody.ndofs());
                    solver_vels *= scale;
                }
            }
            if !is_last_substep {
                let mut solver_vels_incr = self
                    .generic_solver_vels_increment
                    .rows_mut(multibody.solver_id as usize, multibody.ndofs());
                solver_vels_incr.axpy(params.dt, &multibody.accelerations, 0.0);
            }
        }
    }
}
