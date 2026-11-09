//! Per-step assembly and launch: substep solve-groups, the SIMD chunk layout
//! over the persistent contact buckets, and the worker spawn/inline run.

use crate::alloc_prelude::*;
use core::ops::Range;
use core::sync::atomic::Ordering;

use crate::counters::Counters;
use crate::dynamics::solver::manifold_store::ManifoldStore;
use crate::dynamics::solver::reset_buffer_reusing;
use crate::dynamics::solver::solver_contact_graph::{ContactRef, SolverContactGraph};
use crate::dynamics::{
    IntegrationParameters, IslandManager, JointGraphEdge, JointIndex, MultibodyJointSet,
    RigidBodySet, SoftBodySet,
};
use crate::math::{AngularInertia, Real, Vector};
use parry::math::SIMD_WIDTH;

#[cfg(feature = "dim3")]
use crate::dynamics::FrictionModel;

use super::sync::StageSync;
use super::worker::run_worker;
use super::{
    CHUNK_BATCH, ChunkSegment, GroupLayout, LAYOUT_REF_WORKERS, SharedCtx, StagedIslandSolver,
};

/// The most a joint of a multibody turns its link in one substep, in radians, as the substep
/// count of an island holding multibodies is chosen ([`multibody_substeps`]).
///
/// A substep moves each joint at a fixed velocity in its parent's frame and lets the Coriolis
/// and centrifugal forces turn that velocity for the configuration reached, which holds to first
/// order in the turn of the substep. A chain of light links under a heavy load whips its links
/// against one another at tens of radians a second as it swings: at a few tenths of a radian a
/// substep the second-order part of each move stood the chain at several times the kinetic
/// energy it moved with, a free chain released at 70 degrees at 30 steps a second swung its 500
/// kg end at 18 m/s where its fall gives 9.6, and the joint limits of a ragdoll landing on stairs,
/// each row linearised where its substep began, let a joint go 14 degrees past its stop.
const MAX_MULTIBODY_JOINT_TURN: Real = 0.1;

/// The most substeps an island holding multibodies is stepped in, as a multiple of the
/// substeps it is set to ([`multibody_substeps`]).
const MAX_MULTIBODY_SUBSTEP_FACTOR: usize = 4;

/// The substeps an island holding multibodies is stepped in: the `substeps` it is set to, or as
/// many more as keep every joint of its multibodies turning at most [`MAX_MULTIBODY_JOINT_TURN`]
/// in a substep at the velocities they start the step with, up to
/// [`MAX_MULTIBODY_SUBSTEP_FACTOR`] times its base substeps.
fn multibody_substeps(
    substeps: usize,
    base_params: &IntegrationParameters,
    island_bodies: &[crate::dynamics::RigidBodyHandle],
    multibodies: &MultibodyJointSet,
) -> usize {
    let mut fastest: Real = 0.0;
    for handle in island_bodies {
        let Some(link) = multibodies.rigid_body_link(*handle) else {
            continue;
        };
        let multibody = &multibodies[link.multibody];
        if link.id == 0 || link.id == 1 && !multibody.root_is_dynamic {
            fastest = fastest.max(multibody.fastest_joint_turn());
        }
    }
    let needed = (fastest * base_params.dt / MAX_MULTIBODY_JOINT_TURN).ceil();
    let most = (base_params.num_solver_iterations * MAX_MULTIBODY_SUBSTEP_FACTOR).max(substeps);
    if needed.is_finite() && needed > substeps as Real {
        (needed as usize).min(most)
    } else {
        substeps
    }
}

/// The most phase, in radians, the fastest tilt the tension of a body's joints gives it advances
/// in one substep, as the substep count of an island is chosen ([`tension_substeps`]).
///
/// A joint pulling a body at an arm from its centre of mass with a force `f` holds the body
/// upright against a tilt as a spring of stiffness `|f| |arm|` would: a light link of a chain
/// under a heavy load sits between two such springs, the tension of everything hanging below
/// it. A substep solves each joint's rows along the directions they stand in at its start, so
/// that stiffness is stepped forward explicitly, and an explicit step of a spring of angular
/// frequency `w` grows without bound once `w dt` passes 2. A chain of 30 links of 1 kg and
/// 0.5 m holding a 50 kg block tilts its top links at about 133 rad/s: in steps of a 30th of a
/// second (`w dt` 4.4) its end stood 7000 times the chain's length from its anchor after 3 s, in
/// steps of a 60th (2.2) it held to 0.07%, and with each link's inertia nine times larger
/// (`w dt` 1.5 at a 30th) it held exactly. Three quarters of the bound leaves room for a load
/// that pulls harder than it did over the step before, and stays above the 1.1 a chain of five
/// 1 kg rods whipping round as it swings reaches at a 120th of a second, which keeps its energy
/// in the substeps it was set to.
const MAX_TENSION_TILT_PHASE: Real = 1.5;

/// The most substeps an island is stepped in for the tension its joints carry
/// ([`tension_substeps`]). At steps of a 30th of a second it keeps a tilt of up to 2880 rad/s
/// within the bound; the top link of a chain of 0.5 m, 1 kg links holding ten tonnes tilts at
/// about 1500.
const MAX_TENSION_SUBSTEPS: usize = 64;

/// The greatest inverse inertia `ii` gives a turn about an axis square to the unit vector
/// `dir`: the largest eigenvalue of `ii` on the plane square to it, the axes a pull along
/// `dir` tilts a body about.
#[cfg(feature = "dim3")]
fn inverse_inertia_square_to(ii: &AngularInertia, dir: Vector) -> Real {
    let (u, v) = dir.any_orthonormal_pair();
    let (iu, iv) = (*ii * u, *ii * v);
    let (p, q, s) = (u.dot(iu), v.dot(iv), u.dot(iv));
    let half = (p - q) * 0.5;
    (p + q) * 0.5 + (half * half + s * s).sqrt()
}

/// The greatest inverse inertia `ii` gives a tilt that a pull along the unit vector `dir` holds
/// against and that a joint whose frame stands at `frame` with `locked` axes leaves free: about
/// the plane square to `dir` when the joint locks none of its angles
/// ([`inverse_inertia_square_to`]), and otherwise the greatest, over each angle it leaves free,
/// of the inverse inertia about that angle's axis times the share of the pull's stiffness about
/// it (`1 - (axis . dir)^2`). A turn about a locked angle is held by the joint's own rows.
#[cfg(feature = "dim3")]
fn free_tilt_inverse_inertia(
    ii: &AngularInertia,
    dir: Vector,
    frame: crate::math::Rotation,
    locked: crate::dynamics::JointAxesMask,
) -> Real {
    use crate::dynamics::JointAxesMask;
    let angles = [
        (JointAxesMask::ANG_X, Vector::X),
        (JointAxesMask::ANG_Y, Vector::Y),
        (JointAxesMask::ANG_Z, Vector::Z),
    ];
    if !locked.intersects(JointAxesMask::ANG_AXES) {
        return inverse_inertia_square_to(ii, dir);
    }
    let mut most: Real = 0.0;
    for (mask, local) in angles {
        if locked.contains(mask) {
            continue;
        }
        let axis = frame * local;
        let along = axis.dot(dir);
        most = most.max((1.0 - along * along) * axis.dot(*ii * axis));
    }
    most
}

/// The 2D form of the 3D `free_tilt_inverse_inertia`: a body turns about one axis, square to
/// every direction of the plane, which the joint leaves free unless it locks its angle.
#[cfg(feature = "dim2")]
fn free_tilt_inverse_inertia(
    ii: &AngularInertia,
    _dir: Vector,
    _frame: crate::math::Rotation,
    locked: crate::dynamics::JointAxesMask,
) -> Real {
    if locked.contains(crate::dynamics::JointAxesMask::ANG_X) { 0.0 } else { *ii }
}

/// The size of the linear impulse a joint's locked translations applied in the last substep that
/// solved it ([`crate::dynamics::ImpulseJoint::impulses`]).
#[cfg(feature = "dim3")]
fn last_substep_linear_impulse(impulses: &crate::math::SpatialVector) -> Real {
    Vector::new(impulses[0], impulses[1], impulses[2]).length()
}

/// The 2D form of the 3D `last_substep_linear_impulse`.
#[cfg(feature = "dim2")]
fn last_substep_linear_impulse(impulses: &crate::math::SpatialVector) -> Real {
    Vector::new(impulses.x, impulses.y).length()
}

/// Writes into `rates` (one entry per island body, by solver slot) the square of the angular
/// frequency at which the tension of its joints tilts each body: the sum, over the joints
/// `joint_indices` names, of the force each pulls with, times its arm from the body's centre of
/// mass, times the body's inverse inertia about the axes square to that arm
/// ([`free_tilt_inverse_inertia`]). That sum bounds the stiffest tilt the joints hold the body
/// against, a spring of stiffness `|f| |arm|` per joint.
///
/// A joint with a side on a body that locks an axis is left out: the locks and the joint's rows
/// can ask for the same motion, and the impulses they then share say nothing of the tension the
/// joint holds. A chain of hinges on bodies held to a plane, falling onto a floor, read tilts of
/// up to 3000 radians a substep off them while it fell to rest in the substeps it was set to.
///
/// A joint's force is the greater of the mean force it applied over its last step and the force
/// of the last substep of that step. The impulses of the substeps of a step add up to the change
/// of momentum they made, and a load the joints throw back up against them cancels its fall's in
/// that sum: a 1000 kg block on a chain of 1 kg links, rebounding off the correction of the
/// stretch it took while it settled, read a mean force that dropped the island from 32 substeps
/// to 16 for the next step, and the chain folded at 1900 rad/s within it.
fn tension_tilt_rates(
    rates: &mut Vec<Real>,
    num_island_bodies: usize,
    dt: Real,
    bodies: &RigidBodySet,
    impulse_joints: &[JointGraphEdge],
    joint_indices: &[JointIndex],
) {
    rates.clear();
    rates.resize(num_island_bodies, 0.0);
    if dt <= 0.0 {
        return;
    }
    for joint_i in joint_indices {
        let joint = &impulse_joints[*joint_i].weight;
        let locks = |handle| bodies.get(handle).is_some_and(|b| !b.locked_axes().is_empty());
        if locks(joint.body1) || locks(joint.body2) {
            continue;
        }
        let mean = joint.reaction.linear.length() / dt;
        let last = last_substep_linear_impulse(&joint.impulses);
        let frames = [&joint.data.local_frame1, &joint.data.local_frame2];
        for (side, handle) in [joint.body1, joint.body2].into_iter().enumerate() {
            let slot = joint.solver_body_ids[side] as usize;
            if slot >= num_island_bodies {
                continue;
            }
            let body = &bodies[handle];
            if !body.is_dynamic() {
                continue;
            }
            // The last substep of the step that solved the joint: this body's substeps of it.
            let substeps = body.solved_substeps.max(1) as Real;
            let force = mean.max(last * substeps / dt);
            if force.is_nan() || force <= 0.0 {
                continue;
            }
            let local_arm = frames[side].translation - body.mprops.local_mprops.local_com;
            let arm = body.pos.position.rotation * local_arm;
            let reach = arm.length();
            if reach > 0.0 {
                let frame = body.pos.position.rotation * frames[side].rotation;
                let ii = free_tilt_inverse_inertia(
                    &body.mprops.effective_world_inv_inertia,
                    arm / reach,
                    frame,
                    joint.data.locked_axes,
                );
                rates[slot] += force * reach * ii;
            }
        }
    }
}

/// The substeps the island bodies `slots` are stepped in: the `substeps` they are set to,
/// doubled as many times as keeps the fastest tilt the tension of their joints gives any of them
/// (`rates`, from [`tension_tilt_rates`]) within [`MAX_TENSION_TILT_PHASE`] a substep, up to
/// [`MAX_TENSION_SUBSTEPS`]. Doubling keeps the substep length a step is cut to a function of
/// the substep length it was set to: a world of 30 steps a second in 4 substeps and one of 120
/// in 1 that need shorter substeps both run them at a 240th of a second.
fn tension_substeps(substeps: usize, dt: Real, rates: &[Real], slots: Range<usize>) -> usize {
    let fastest = rates.get(slots).map_or(0.0, |r| r.iter().copied().fold(0.0, Real::max));
    let turn = fastest.sqrt() * dt;
    if !turn.is_finite() {
        return substeps;
    }
    let most = MAX_TENSION_SUBSTEPS.max(substeps);
    let mut n = substeps.max(1);
    while n < most && turn > MAX_TENSION_TILT_PHASE * n as Real {
        n = (n * 2).min(most);
    }
    n
}

impl StagedIslandSolver {
    #[profiling::function]
    #[allow(clippy::too_many_arguments)]
    pub fn init_and_solve(
        &mut self,
        num_workers: usize,
        island_id: usize,
        counters: &mut Counters,
        gravity: crate::math::Vector,
        base_params: &IntegrationParameters,
        islands: &IslandManager,
        bodies: &mut RigidBodySet,
        // The persistent per-color solver contact buckets (+ the
        // generic list), maintained incrementally by the narrow-phase — consumed
        // directly; nothing is selected or sorted per step.
        graph: &SolverContactGraph,
        store: &ManifoldStore,
        impulse_joints: &mut [JointGraphEdge],
        joint_indices: &[JointIndex],
        joint_assembly_epoch: u32,
        multibodies: &mut MultibodyJointSet,
        soft_bodies: &mut SoftBodySet,
        // The narrow phase and colliders: read-only, for the soft-body surface contacts.
        narrow_phase: &crate::geometry::NarrowPhase,
        colliders: &crate::geometry::ColliderSet,
        // The narrow-phase's per-body masks of persistent contact solver colors,
        // used to color the joints in the same color space as the contacts.
        contact_color_masks: &[u128],
        // Asked for the gravity of the bodies and soft bodies whose gravity comes from the hooks.
        hooks: &dyn crate::pipeline::PhysicsHooks,
    ) {
        counters.solver.velocity_assembly_time.resume();
        // Substep solve-groups: each group (contiguous constraint-closed awake-body range, keyed
        // by max `additional_solver_iterations`) runs `base + extra` substeps at smaller dt —
        // real substeps buy convergence on high mass ratios, unlike the flat PGS sweeps they
        // replaced. Multibody scenes: one group at max extra (generic tables not group-major yet).
        let island_bodies = islands.island(island_id).bodies();
        let multi_group = islands.solve_groups.len() >= 2 && multibodies.iter().next().is_none();
        let max_extra = islands
            .solve_groups
            .first()
            .map(|g| g.extra_iters as usize)
            .unwrap_or(0);
        let max_extra_pgs = islands
            .solve_groups
            .iter()
            .map(|g| g.extra_pgs as usize)
            .max()
            .unwrap_or(0);
        self.groups.clear();
        tension_tilt_rates(
            &mut self.tension_tilt_rates,
            island_bodies.len(),
            base_params.dt,
            bodies,
            impulse_joints,
            joint_indices,
        );
        if multi_group {
            for group in &islands.solve_groups {
                let num_substeps = tension_substeps(
                    base_params.num_solver_iterations + group.extra_iters as usize,
                    base_params.dt,
                    &self.tension_tilt_rates,
                    group.body_range.clone(),
                );
                let num_pgs_iterations =
                    base_params.num_internal_pgs_iterations + group.extra_pgs as usize;
                self.groups.push(GroupLayout {
                    bodies: group.body_range.clone(),
                    soft_slots: 0..0,
                    chunks: 0..0,
                    colors: 0..0,
                    overflow: 0..0,
                    joint_colors: 0..0,
                    joint_chunks: 0..0,
                    joint_builders: 0..0,
                    joint_overflow: 0..0,
                    joint_trees: 0..0,
                    num_substeps,
                    num_pgs_iterations,
                    dt: base_params.dt / num_substeps as Real,
                });
            }
        } else {
            let num_substeps = tension_substeps(
                multibody_substeps(
                    base_params.num_solver_iterations + max_extra,
                    base_params,
                    island_bodies,
                    multibodies,
                ),
                base_params.dt,
                &self.tension_tilt_rates,
                0..island_bodies.len(),
            );
            let num_pgs_iterations = base_params.num_internal_pgs_iterations + max_extra_pgs;
            self.groups.push(GroupLayout {
                bodies: 0..island_bodies.len(),
                soft_slots: 0..0,
                chunks: 0..0,
                colors: 0..0,
                overflow: 0..0,
                joint_colors: 0..0,
                joint_chunks: 0..0,
                joint_builders: 0..0,
                joint_overflow: 0..0,
                joint_trees: 0..0,
                num_substeps,
                num_pgs_iterations,
                dt: base_params.dt / num_substeps as Real,
            });
        }
        // `params` keeps the FIRST group's substep dt (multibody precompute uses it; multibody
        // scenes are always single-group). The worker loop re-derives each group's dt from
        // `GroupLayout`.
        let mut params = *base_params;
        params.dt /= self.groups[0].num_substeps as Real;

        /*
         * Serial pre-phase: soft-body rows and particle slots, solver buffers & multibodies,
         * coloring, joint & generic constraints. (Plain solver bodies are initialized by the
         * workers in the first parallel stage; the soft particles' solver bodies here.)
         */
        // Soft-body constraints: assembled every step from the awake soft bodies' elements (the warm
        // start impulses persist on the elements themselves). Their particles take the solver
        // slots after the island's rigid bodies, group-major.
        {
            let groups = &self.groups;
            let group_of_slot = |slot: u32| -> usize {
                groups
                    .iter()
                    .position(|g| g.bodies.contains(&(slot as usize)))
                    .unwrap_or(0)
            };
            let group_dt = |gi: usize| groups[gi].dt;
            self.soft_constraints.assemble(
                island_id,
                island_bodies.len(),
                groups.len(),
                group_of_slot,
                group_dt,
                base_params.dt,
                params.max_corrective_velocity(),
                bodies,
                soft_bodies,
            );
        }
        for (g, sg) in self
            .groups
            .iter_mut()
            .zip(self.soft_constraints.groups.iter())
        {
            g.soft_slots = sg.slots.clone();
        }
        let num_soft_slots = self.soft_constraints.slots.len();
        self.velocity_solver.init_solver_buffers_and_multibodies(
            &params,
            island_bodies,
            num_soft_slots,
            bodies,
            multibodies,
        );
        if num_soft_slots > 0 {
            let groups = &self.groups;
            let vs = &mut self.velocity_solver;
            self.soft_constraints.init_solver_bodies(
                &mut vs.solver_bodies,
                &mut vs.solver_vels_increment,
                &mut vs.solver_gravity,
                &mut vs.substep_vels,
                gravity,
                base_params.dt,
                |gi| groups[gi].dt,
            );
        }

        // FEM soft bodies: their systems are updated once the particle slots exist (the
        // sparsity pattern and the element tables persist across steps).
        #[cfg(feature = "fem")]
        {
            let groups = &self.groups;
            self.soft_fem.assemble(
                &mut self.soft_constraints,
                soft_bodies,
                groups.len(),
                |gi| groups[gi].dt,
                &params.soft_bodies.fem,
            );
        }

        // The persistent solver contact graph already holds two-body manifolds grouped by
        // (color, contact count) — colors touch pairwise-disjoint bodies, buckets slice straight
        // into SIMD chunks — and multibody manifolds apart; nothing to categorize/sort per step.
        const NUM_COLORS: usize = 129; // 128 parallel colors + the overflow color.
        {
            let set = &mut self.contact_constraints;
            // NOTE: the twist constraint/builder buffers are NOT cleared: they are
            // the persistent constraint cache, preserved (and resized) further
            // below once the chunk layout is known.
            set.generic_jacobians.fill(0.0);
            set.generic_velocity_constraints.clear();
            set.generic_velocity_constraints_builder.clear();
            set.simd_velocity_coulomb_constraints.clear();
            set.simd_velocity_coulomb_constraints_builder.clear();
        }
        // Soft-body contact and attachment constraints (group-major, after the constraint layout above).
        if num_soft_slots > 0 {
            let groups = &self.groups;
            let group_of_slot = |slot: u32| -> usize {
                groups
                    .iter()
                    .position(|g| {
                        g.bodies.contains(&(slot as usize))
                            || g.soft_slots.contains(&(slot as usize))
                    })
                    .unwrap_or(0)
            };
            let group_dt = |gi: usize| groups[gi].dt;
            self.soft_constraints.assemble_contacts(
                island_id,
                &params,
                narrow_phase,
                colliders,
                bodies,
                soft_bodies,
                group_of_slot,
                group_dt,
            );
            self.soft_constraints
                .assemble_attachments(island_id, bodies, group_dt);

            // Active clusters: the proxies with a joint this step (or a pending external
            // impulse) get the per-pass gather/scatter stages.
            let mut jointed_proxies: parry::utils::hashmap::HashMap<u32, ()> = Default::default();
            for ji in joint_indices {
                let joint = &impulse_joints[*ji].weight;
                for handle in [joint.body1, joint.body2] {
                    if let Some(rb) = bodies.get(handle) {
                        if rb.is_soft_frame() && rb.ids.active_set_id != u32::MAX {
                            jointed_proxies.insert(rb.ids.active_set_id, ());
                        }
                    }
                }
            }
            let vs = &self.velocity_solver;
            self.soft_constraints.assemble_clusters(
                bodies,
                colliders,
                &vs.solver_bodies,
                &jointed_proxies,
            );

            // The FEM bodies' answers to the step's constraints, once every constraint exists: planned here,
            // computed by the workers' first stage.
            #[cfg(feature = "fem")]
            self.soft_fem.plan_responses(&mut self.soft_constraints);
            // The attachments the FEM bodies answer are known: lay out the attachment blocks.
            self.soft_constraints.plan_attachment_blocks();
        }

        // Avoid spawning more workers than there is work: soft constraints count like element
        // constraints (their chunks are the contact stages' claims), and each FEM body is a
        // factorization and solves of its own (the response stage claims one body per worker).
        let num_two_body = graph.len() - graph.generic().len();
        #[cfg(feature = "fem")]
        let fem_chunks = self.soft_fem.num_active() * 16;
        #[cfg(not(feature = "fem"))]
        let fem_chunks = 0;
        let soft = &self.soft_constraints;
        let approx_chunks = num_two_body / SIMD_WIDTH
            + joint_indices.len() / 4
            + (soft.scalar_constraints.len() + soft.shape_constraints.len() + soft.contacts.len())
                / 8
            + fem_chunks;
        let num_workers = num_workers.clamp(1, (approx_chunks / 16).max(1));

        // Joints: colored like contacts and laid out color by color (scalar
        // constraints only). Colors too small to parallelize plus the generic
        // (multibody) joints are solved by worker 0 in exclusive stages.
        self.init_joints(
            islands.active_set_epoch,
            joint_assembly_epoch,
            base_params.warmstart_joints,
            island_bodies,
            bodies,
            multibodies,
            impulse_joints,
            joint_indices,
            contact_color_masks,
        );
        // Refill the per-step group layouts from the (freshly built or reused)
        // per-group joint layout slices.
        debug_assert_eq!(self.groups.len(), self.staged_group_joint_layout.len());
        for (g, l) in self
            .groups
            .iter_mut()
            .zip(self.staged_group_joint_layout.iter())
        {
            g.joint_colors = l.colors.clone();
            g.joint_chunks = l.chunks.clone();
            g.joint_builders = l.builders.clone();
            g.joint_overflow = l.overflow.clone();
            g.joint_trees = l.trees.clone();
        }

        let set = &mut self.contact_constraints;

        // Build the SIMD chunk layout, group-major then color by color, straight from the
        // persistent buckets. Colors too small to distribute across workers merge into a
        // per-group serial "overflow" tail. The threshold is calibrated for
        // [`LAYOUT_REF_WORKERS`], NOT the pool size (see that constant's docs).
        let min_color_chunks = CHUNK_BATCH * LAYOUT_REF_WORKERS / 2;
        self.chunk_segments.clear();
        self.overflow_chunk_refs.clear();
        self.grouped_chunk_refs.clear();
        self.color_ranges.clear();
        let mut next_chunk = 0usize;

        // NOTE: the SIMD builders handle per-lane active-contact counts, so a color's whole
        //       bucket slices straight into chunks (no count grouping). Buckets arrive colors
        //       ascending, overflow last.
        let push_segment =
            |segments: &mut Vec<ChunkSegment>, next_chunk: &mut usize, refs: &[ContactRef]| {
                if refs.is_empty() {
                    return;
                }
                segments.push(ChunkSegment {
                    first_chunk: *next_chunk as u32,
                    refs: refs.as_ptr(),
                    len: refs.len() as u32,
                });
                *next_chunk += refs.len().div_ceil(SIMD_WIDTH);
            };

        if self.groups.len() == 1 {
            // Single group (the common case): the historical zero-copy layout,
            // slicing the persistent buckets directly.
            for (color, refs) in graph.buckets() {
                if (color as usize) < NUM_COLORS - 1
                    && refs.len().div_ceil(SIMD_WIDTH) >= min_color_chunks
                {
                    let range_start = next_chunk;
                    push_segment(&mut self.chunk_segments, &mut next_chunk, refs);
                    self.color_ranges.push((color, range_start..next_chunk));
                }
            }

            let overflow_start = next_chunk;
            // The overflow grouper's output lands in `overflow_chunk_refs`; its segments are
            // created only once the vec stops growing (they hold raw pointers into it). The
            // overflow color sorts last, so deferring them preserves the global chunk order.
            let mut overflow_grouped_len = 0usize;
            for (color, refs) in graph.buckets() {
                let is_overflow_color = (color as usize) == NUM_COLORS - 1;
                if is_overflow_color {
                    // Overflow color: uncolorable pairs / 2nd+ manifolds of a pair CAN share both
                    // bodies; linear chunking would alias a dynamic body across SIMD lanes and the
                    // last-writer-wins scatter would drop an impulse. Body-mask grouping keeps each
                    // chunk's lanes disjoint (ungroupable => 1-lane chunks); worker 0 solves serially.
                    self.overflow_workspace.clear();
                    self.overflow_workspace.extend_from_slice(refs);
                    set.interaction_groups.clear_groups();
                    set.interaction_groups.group_manifold_refs(
                        island_bodies.len(),
                        store,
                        &self.overflow_workspace,
                    );
                    self.overflow_chunk_refs
                        .extend_from_slice(&set.interaction_groups.simd_ref_interactions);
                    overflow_grouped_len = self.overflow_chunk_refs.len();
                    self.overflow_chunk_refs
                        .extend_from_slice(&set.interaction_groups.nongrouped_ref_interactions);
                } else if refs.len().div_ceil(SIMD_WIDTH) < min_color_chunks {
                    // A real color too small to parallelize: body-disjoint already.
                    push_segment(&mut self.chunk_segments, &mut next_chunk, refs);
                }
            }
            // The overflow segments: the grouped prefix is a multiple of SIMD_WIDTH
            // (whole body-disjoint chunks); each ungroupable ref is its own 1-lane
            // chunk, so it gets its own single-ref segment.
            debug_assert_eq!(overflow_grouped_len % SIMD_WIDTH, 0);
            push_segment(
                &mut self.chunk_segments,
                &mut next_chunk,
                &self.overflow_chunk_refs[..overflow_grouped_len],
            );
            for i in overflow_grouped_len..self.overflow_chunk_refs.len() {
                push_segment(
                    &mut self.chunk_segments,
                    &mut next_chunk,
                    &self.overflow_chunk_refs[i..i + 1],
                );
            }
            let g = &mut self.groups[0];
            g.chunks = 0..next_chunk;
            g.colors = 0..self.color_ranges.len();
            g.overflow = overflow_start..next_chunk;
        } else {
            // Multi-group: classify every bucket ref by group (a per-ref pass, only on this
            // gated path), then emit the layout group-major so each group's chunks are
            // contiguous in global chunk-id space. Reserve exact upper bounds up front:
            // segments hold raw pointers into these vecs, which must never reallocate below.
            self.grouped_chunk_refs.reserve(num_two_body);
            self.overflow_chunk_refs.reserve(num_two_body);
            let grouped_cap = self.grouped_chunk_refs.capacity();
            let overflow_cap = self.overflow_chunk_refs.capacity();

            // Slot -> group-index table (groups cover the island slots
            // exactly; fixed ids fall outside and are skipped).
            let num_groups = self.groups.len();
            let mut slot_group = alloc::vec![u16::MAX; island_bodies.len()];
            for (gi, g) in self.groups.iter().enumerate() {
                slot_group[g.bodies.clone()].fill(gi as u16);
            }
            // A manifold's group: max group index over its in-island solver bodies. Both sides
            // agree for dynamic-dynamic pairs (constraints never span groups by construction);
            // for kinematic-dynamic the max picks the dynamic side.
            let group_of = |r: &ContactRef| -> usize {
                let mut gi = 0u16;
                let mut found = false;
                for id in store.get(*r).data.solver_body_ids {
                    if let Some(&g) = slot_group.get(id as usize) {
                        if g != u16::MAX {
                            gi = gi.max(g);
                            found = true;
                        }
                    }
                }
                debug_assert!(found, "solver-active pair without an in-island body");
                gi as usize
            };

            // Pass A: copy each color's refs into per-(group, color)
            // contiguous runs of `grouped_chunk_refs` (overflow color kept
            // aside per group for the body-mask grouper).
            let mut runs: Vec<Vec<(u8, u32, u32)>> = alloc::vec![Vec::new(); num_groups];
            let mut overflow_by_group: Vec<Vec<ContactRef>> = alloc::vec![Vec::new(); num_groups];
            let mut split: Vec<Vec<ContactRef>> = alloc::vec![Vec::new(); num_groups];
            for (color, refs) in graph.buckets() {
                let is_overflow_color = (color as usize) == NUM_COLORS - 1;
                for workspace in &mut split {
                    workspace.clear();
                }
                for r in refs {
                    split[group_of(r)].push(*r);
                }
                for (gi, workspace) in split.iter().enumerate() {
                    if workspace.is_empty() {
                        continue;
                    }
                    if is_overflow_color {
                        overflow_by_group[gi].extend_from_slice(workspace);
                    } else {
                        let start = self.grouped_chunk_refs.len() as u32;
                        self.grouped_chunk_refs.extend_from_slice(workspace);
                        runs[gi].push((color, start, workspace.len() as u32));
                    }
                }
            }

            // Pass B: emit the layout group-major. Within a group: parallel
            // colors first (each a `color_ranges` entry), then the serial
            // tail (small colors + the group's grouped/ungroupable overflow).
            for gi in 0..num_groups {
                let chunks_start = next_chunk;
                let colors_start = self.color_ranges.len();
                for (color, start, len) in &runs[gi] {
                    let refs = &self.grouped_chunk_refs[*start as usize..(*start + *len) as usize];
                    if refs.len().div_ceil(SIMD_WIDTH) >= min_color_chunks {
                        let range_start = next_chunk;
                        push_segment(&mut self.chunk_segments, &mut next_chunk, refs);
                        self.color_ranges.push((*color, range_start..next_chunk));
                    }
                }
                let overflow_start = next_chunk;
                for (color, start, len) in &runs[gi] {
                    let _ = color;
                    let refs = &self.grouped_chunk_refs[*start as usize..(*start + *len) as usize];
                    if refs.len().div_ceil(SIMD_WIDTH) < min_color_chunks {
                        push_segment(&mut self.chunk_segments, &mut next_chunk, refs);
                    }
                }
                set.interaction_groups.clear_groups();
                set.interaction_groups.group_manifold_refs(
                    island_bodies.len(),
                    store,
                    &overflow_by_group[gi],
                );
                let grouped_start = self.overflow_chunk_refs.len();
                self.overflow_chunk_refs
                    .extend_from_slice(&set.interaction_groups.simd_ref_interactions);
                let grouped_end = self.overflow_chunk_refs.len();
                self.overflow_chunk_refs
                    .extend_from_slice(&set.interaction_groups.nongrouped_ref_interactions);
                debug_assert_eq!((grouped_end - grouped_start) % SIMD_WIDTH, 0);
                push_segment(
                    &mut self.chunk_segments,
                    &mut next_chunk,
                    &self.overflow_chunk_refs[grouped_start..grouped_end],
                );
                for i in grouped_end..self.overflow_chunk_refs.len() {
                    push_segment(
                        &mut self.chunk_segments,
                        &mut next_chunk,
                        &self.overflow_chunk_refs[i..i + 1],
                    );
                }

                let g = &mut self.groups[gi];
                g.chunks = chunks_start..next_chunk;
                g.colors = colors_start..self.color_ranges.len();
                g.overflow = overflow_start..next_chunk;
            }
            // The reserves above must have covered all growth (pointer
            // stability of the segment refs).
            debug_assert_eq!(self.grouped_chunk_refs.capacity(), grouped_cap);
            debug_assert_eq!(self.overflow_chunk_refs.capacity(), overflow_cap);
        }

        // Every SIMD chunk (parallel colors AND the serial overflow tail) needs lane-disjoint
        // solver bodies: the wide gather/scatter is last-writer-wins, so an aliased slot drops an
        // impulse. Fixed sides (`u32::MAX`, scatter-skipped) may repeat; sleeping sides hold a real
        // appended slot and must not — the coloring keeps pairs sharing any non-fixed body apart.
        #[cfg(debug_assertions)]
        for chunk in self
            .chunk_segments
            .iter()
            .flat_map(|seg| (0..seg.num_chunks()).map(|local| seg.chunk(local)))
        {
            let mut seen = [u32::MAX; SIMD_WIDTH * 2];
            let mut n = 0;
            for &id in &chunk {
                if id.is_padding() {
                    continue;
                }
                for bid in store.get(id).data.solver_body_ids {
                    if bid == u32::MAX {
                        continue;
                    }
                    // A non-dynamic body still holding an active slot (its type
                    // changed to fixed this step: kinematic-like for one step) may
                    // be shared across lanes: zero inverse mass, so every lane
                    // scatters the same unchanged velocity back.
                    if island_bodies
                        .get(bid as usize)
                        .is_some_and(|h| !bodies[*h].is_dynamic())
                    {
                        continue;
                    }
                    debug_assert!(
                        !seen[..n].contains(&bid),
                        "dynamic solver body {bid} shared by two lanes of one SIMD contact chunk"
                    );
                    seen[n] = bid;
                    n += 1;
                }
            }
        }

        let num_chunks = next_chunk;
        #[cfg(feature = "dim3")]
        let use_twist = matches!(params.friction_model, FrictionModel::Simplified);
        #[cfg(feature = "dim2")]
        let use_twist = false;

        // Constraints are regenerated from the manifolds every step (only the
        // warm-start impulses persist, on the manifolds themselves); reset the wide
        // constraint/builder buffers to one entry per chunk.
        #[cfg(feature = "dim3")]
        if use_twist {
            unsafe {
                reset_buffer_reusing(&mut set.simd_velocity_twist_constraints_builder, num_chunks);
                reset_buffer_reusing(&mut set.simd_velocity_twist_constraints, num_chunks);
            }
        }
        if !use_twist {
            unsafe {
                reset_buffer_reusing(
                    &mut set.simd_velocity_coulomb_constraints_builder,
                    num_chunks,
                );
                reset_buffer_reusing(&mut set.simd_velocity_coulomb_constraints, num_chunks);
            }
        }

        // Generic (multibody) contact constraints: serial init, solved by worker 0.
        let mut jacobian_id = 0;
        set.compute_generic_constraints(bodies, multibodies, graph, store, &mut jacobian_id);

        #[cfg(feature = "dim3")]
        self.any_gyroscopic.store(false, Ordering::Relaxed);
        self.any_ccd_active.store(false, Ordering::Relaxed);
        // Seed the restitution-pass gate with the generic constraints' contribution (built
        // serially above); the workers OR in the SIMD chunks' contribution during the
        // constraint-generation stage.
        let generic_bouncy = set
            .generic_velocity_constraints_builder
            .iter()
            .zip(set.generic_velocity_constraints.iter())
            .any(|(b, c)| b.has_bouncy_seed(c.num_contacts));
        self.any_bouncy.store(generic_bouncy, Ordering::Relaxed);

        // Constraint statistics, summed over the CCD substeps.
        if counters.enabled {
            let num_rigid_contacts: usize = graph
                .buckets()
                .flat_map(|(_, refs)| refs)
                .chain(graph.generic())
                .map(|r| store.get(*r).data.solver_contacts.len())
                .sum();
            counters.solver.nconstraints += graph.len() + joint_indices.len();
            counters.solver.ncontacts += num_rigid_contacts + self.soft_constraints.contacts.len();
        }

        counters.solver.velocity_assembly_time.pause();
        counters.solver.velocity_resolution_time.resume();

        /*
         * Parallel phase.
         */
        self.sync = StageSync::new(num_workers);

        let ctx = SharedCtx {
            sync: &self.sync,
            chunk_segments: &self.chunk_segments,
            num_chunks,
            groups: &self.groups,
            color_ranges: &self.color_ranges,
            joint_color_ranges: &self.joint_color_ranges,
            joint_rows: &self.joint_rows,
            joint_trees: &mut self.joint_trees as *mut _,
            joint_chunk_rows: &self.joint_chunk_rows,
            island_bodies,
            has_multibodies: !self.velocity_solver.multibody_roots.is_empty(),
            base_params,
            store,
            joints: impulse_joints.as_mut_ptr(),
            num_joints: impulse_joints.len(),
            velocity_solver: &mut self.velocity_solver as *mut _,
            joint_constraints: &mut self.joint_constraints as *mut _,
            contact_constraints: set as *mut _,
            soft_constraints: &mut self.soft_constraints as *mut _,
            #[cfg(feature = "fem")]
            soft_fem: &self.soft_fem as *const _,
            coulomb_builders: set.simd_velocity_coulomb_constraints_builder.as_mut_ptr(),
            coulomb_constraints: set.simd_velocity_coulomb_constraints.as_mut_ptr(),
            #[cfg(feature = "dim3")]
            twist_builders: set.simd_velocity_twist_constraints_builder.as_mut_ptr(),
            #[cfg(feature = "dim3")]
            twist_constraints: set.simd_velocity_twist_constraints.as_mut_ptr(),
            use_twist,
            bodies: bodies as *mut _,
            multibodies: multibodies as *mut _,
            #[cfg(feature = "dim3")]
            any_gyroscopic: &self.any_gyroscopic as *const _,
            any_ccd_active: &self.any_ccd_active as *const _,
            any_bouncy: &self.any_bouncy as *const _,
            hooks,
            gravity,
        };

        let ctx_ref = &ctx;

        // Parallel build: spawn workers `1..num_workers` into a rayon scope, run worker 0
        // inline. Non-parallel/wasm build: `num_workers == 1`, worker 0 runs inline and
        // `StageSync` advances every stage immediately (no spawning/stealing/spinning).
        #[cfg(feature = "parallel")]
        rayon::in_place_scope(|scope| {
            for worker_id in 1..num_workers {
                scope.spawn(move |_| {
                    // SAFETY: see `SharedCtx` and the per-stage comments in `run_worker`.
                    unsafe {
                        run_worker(ctx_ref, worker_id);
                    }
                });
            }

            // SAFETY: see `SharedCtx` and the per-stage comments in `run_worker`.
            unsafe {
                run_worker(ctx_ref, 0);
            }
        });

        #[cfg(not(feature = "parallel"))]
        {
            debug_assert_eq!(num_workers, 1);
            // SAFETY: single worker, so no concurrent access to the shared context.
            unsafe {
                run_worker(ctx_ref, 0);
            }
        }

        counters.solver.velocity_resolution_time.pause();
        // NOTE: impulse and rigid-body writeback now happen as parallel stages of
        //       `run_worker`, so their time is included in the resolution counter.

        // Publish the fused post-solve CCD activation verdict. Multibody link bodies skip the
        // fused flag computation (separate writeback), so their presence invalidates it and
        // the pipeline falls back to its own pass.
        self.post_solve_ccd_active = self
            .velocity_solver
            .multibody_roots
            .is_empty()
            .then(|| self.any_ccd_active.load(Ordering::Relaxed));
    }
}
