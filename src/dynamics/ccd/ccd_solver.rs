use crate::alloc_prelude::*;
use crate::dynamics::{
    ImpulseJointSet, IntegrationParameters, IslandManager, MultibodyIndex, MultibodyJointSet,
    RigidBodySet, SoftBodySet,
};
use crate::geometry::{
    BroadPhaseBvh, Collider, ColliderHandle, ColliderSet, CollisionEvent, NarrowPhase,
};
use crate::math::Real;
use crate::parry::bounding_volume::Aabb;
use crate::pipeline::{EventHandler, PhysicsHooks, QueryFilter};
use crate::prelude::{ActiveEvents, CollisionEventFlags};
use parry::query::sweep_toi::Sweep;

use super::kinematic_sweeps::{KinematicPairs, KinematicStrike, solve_kinematic_sweeps};
use super::sweeps::{
    BodyContinuousResult, CcdTargets, PseudoHitMode, SoftTies, collect_fixed_targets,
    collect_soft_targets, is_bullet, map_bodies_parallel, sweep_fast_body, swept_collider_aabbs,
};

/// Continuous Collision Detection solver preventing fast objects from tunneling:
/// after the solver, bodies that moved more than half their thinnest extent sweep their colliders
/// and `next_position` is clamped to the earliest impact — velocities untouched, no re-solve; the
/// residual approach resolves next step via speculative contacts.
///
/// Fast dynamic bodies sweep against fixed colliders and soft-body meshes (each where the solve
/// left its particles; all but the meshes of a soft body tied to the body itself, by a particle
/// attachment or a joint); `ccd_enabled` makes a
/// bullet that also sweeps kinematic/dynamic bodies (never other bullets); mesh-like colliders are
/// never the moving shape; compounds sweep per convex child; `max_ccd_substeps = 0` disables CCD.
///
/// Moving kinematic bodies sweep the dynamic bodies in their path before that: a dynamic body a
/// kinematic body reaches within the step is struck at the time of impact
/// ([`CCDSolver::solve_kinematic_sweeps`]), whatever either body's `ccd_enabled` says.
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CCDSolver {
    /// Cached fixed-target list for the non-bullet sweep pass: the AABB loosening it was built
    /// with; `None` past [`FIXED_TARGETS_LIST_MAX`] (sweep queries the full BVH). Invalidated by
    /// the scene-change flag — re-scanning every collider each step dominated CCD on large scenes.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    fixed_targets_cache: Option<FixedTargetsCache>,
    /// The strikes the kinematic sweeps of the last step resolved.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    kinematic_strikes: Vec<KinematicStrike>,
    /// The pairs the kinematic sweeps of the current pass swept, which the bullets' sweeps of
    /// the same pass leave to them.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    kinematic_pairs: KinematicPairs,
}

/// The AABB loosening the cached fixed-target list was built with, paired with the list
/// itself — `None` past [`FIXED_TARGETS_LIST_MAX`], where the sweep queries the full BVH.
type FixedTargetsCache = (Real, Option<Vec<(ColliderHandle, Aabb)>>);

impl CCDSolver {
    /// Initializes a new CCD solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Discards the cached fixed-target list used by the non-bullet continuous pass.
    /// This must be called any time the fixed targets change. If using the physics pipeline,
    /// it is called for you. If not, it must be called manually.
    pub fn invalidate_fixed_targets_cache(&mut self) {
        self.fixed_targets_cache = None;
    }

    /// The dynamic bodies kinematic bodies struck in the last step
    /// ([`Self::solve_kinematic_sweeps`]), in the order they were struck.
    pub fn kinematic_strikes(&self) -> &[KinematicStrike] {
        &self.kinematic_strikes
    }

    /// Forgets the strikes of the step before. The pipeline calls it as each step starts.
    pub fn clear_kinematic_strikes(&mut self) {
        self.kinematic_strikes.clear();
    }

    /// Updates the set of bodies that needs CCD to be resolved.
    ///
    /// Returns `true` if any rigid-body must have CCD resolved.
    pub fn update_ccd_active_flags(
        &self,
        islands: &IslandManager,
        bodies: &mut RigidBodySet,
        dt: Real,
        include_forces: bool,
    ) -> bool {
        let mut ccd_active = false;

        for handle in islands.active_bodies() {
            let rb = bodies.index_mut_internal(handle);

            // Default tier: every fast dynamic body is a CCD origin. `ccd_enabled`
            // no longer gates *activation*, only the sweep *scope* (fixed-only vs all bodies),
            // applied later during pair selection.
            if rb.is_soft_frame() {
                // A cluster proxy's pose is derived from its particles: it is never swept.
                rb.ccd.ccd_active = false;
            } else if rb.is_dynamic() {
                let moving_fast = if include_forces {
                    // Pre-solve (substep splitter): `next_position` isn't solved yet, use the
                    // current velocity with forces, like `find_first_impact`. The last solved
                    // motion (`ccd_vels`) is stale, or zero on a body's first step.
                    rb.ccd.is_moving_fast(
                        dt,
                        &rb.vels,
                        Some((&rb.forces, &rb.mprops)),
                        rb.mprops.max_extent(),
                    )
                } else {
                    // Post-solve: the fast-body criterion on the actual solved motion.
                    rb.ccd.is_moving_fast_with_next_position(
                        dt,
                        &rb.ccd_vels,
                        &rb.pos,
                        rb.mprops.local_mprops.local_com,
                        rb.mprops.max_extent(),
                    )
                };
                rb.ccd.ccd_active = moving_fast;
                ccd_active = ccd_active || moving_fast;
            }
        }

        ccd_active
    }

    /// Sweeps every flagged kinematic body along its motion of the step against the dynamic
    /// bodies in its path, and strikes each one it reaches at the time of impact: the struck
    /// body leaves the contact at the velocity of the kinematic surface there, with the pair's
    /// restitution and friction applied, and ends the step clear of the kinematic body. A
    /// struck body is woken, and its own CCD flag is set from its new motion so the
    /// continuous pass that follows ([`Self::solve_continuous`]) sweeps it. Each strike is
    /// recorded for [`Self::kinematic_strikes`]. A body that touched the kinematic body as the
    /// step started keeps the velocity the contact solver gave it, and is set back in front of
    /// the kinematic face when the face overtakes it within the step. The bullets' sweeps of
    /// the continuous pass that follows leave every pair swept here alone.
    ///
    /// Returns `true` if any body was struck or set back.
    #[profiling::function]
    #[allow(clippy::too_many_arguments)]
    pub fn solve_kinematic_sweeps(
        &mut self,
        params: &IntegrationParameters,
        islands: &mut IslandManager,
        bodies: &mut RigidBodySet,
        colliders: &ColliderSet,
        impulse_joints: &ImpulseJointSet,
        multibody_joints: &MultibodyJointSet,
        broad_phase: &mut BroadPhaseBvh,
        narrow_phase: &NarrowPhase,
        hooks: &dyn PhysicsHooks,
        events: &dyn EventHandler,
    ) -> bool {
        solve_kinematic_sweeps(
            params.dt,
            params.allowed_linear_error(),
            params.prediction_distance(),
            islands,
            bodies,
            colliders,
            impulse_joints,
            multibody_joints,
            broad_phase,
            narrow_phase,
            hooks,
            events,
            &mut self.kinematic_strikes,
            &mut self.kinematic_pairs,
        )
    }

    /// Find the first time a CCD-active body has a non-sensor collider hitting another
    /// non-sensor collider, for the multi-substep splitter.
    ///
    /// Returns the impact time in `[0, dt)` if any.
    #[profiling::function]
    #[allow(clippy::too_many_arguments)]
    pub fn find_first_impact(
        &mut self,
        dt: Real, // NOTE: this doesn’t necessarily match the `params.dt`.
        params: &IntegrationParameters,
        islands: &IslandManager,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        soft_bodies: &SoftBodySet,
        impulse_joints: &ImpulseJointSet,
        multibody_joints: &MultibodyJointSet,
        broad_phase: &mut BroadPhaseBvh,
        narrow_phase: &NarrowPhase,
        hooks: &dyn PhysicsHooks,
    ) -> Option<Real> {
        let ties = SoftTies {
            soft_bodies,
            impulse_joints,
            multibody_joints,
        };
        // NOTE: broad-phase AABBs are NOT enlarged to the swept volumes: only the fast body's
        // query box is swept (per collider, below); targets keep their regular fat AABBs.
        // Swept AABBs written into the tree would leak into the next step (pair explosion).
        let query_pipeline = broad_phase.as_query_pipeline(
            narrow_phase.query_dispatcher(),
            bodies,
            colliders,
            QueryFilter::default(),
        );
        let (bvh, dispatcher) = (query_pipeline.bvh, query_pipeline.dispatcher);

        let linear_slop = params.allowed_linear_error();
        // `next_position` isn't solved yet: each fast body sweeps to the forces/velocities
        // integration.
        let predicted_pose = |rb: &crate::dynamics::RigidBody| {
            rb.pos
                .integrate_forces_and_velocities(dt, &rb.forces, &rb.vels, &rb.mprops)
        };
        let fast_bodies: Vec<_> = islands
            .active_bodies()
            .filter(|h| bodies[*h].ccd.ccd_active && bodies[*h].is_dynamic())
            .collect();
        let swept: Vec<_> = fast_bodies
            .iter()
            .flat_map(|h| {
                let rb = &bodies[*h];
                swept_collider_aabbs(rb, colliders, predicted_pose(rb))
            })
            .collect();
        let soft = collect_soft_targets(soft_bodies, colliders, params.prediction_distance(), &swept);

        let fractions = map_bodies_parallel(&fast_bodies, hooks, |handle, hooks| {
            sweep_fast_body(
                handle,
                bodies,
                colliders,
                ties,
                predicted_pose(&bodies[handle]),
                CcdTargets::FullBvh { bvh, soft: &soft },
                dispatcher,
                hooks,
                dt,
                linear_slop,
                PseudoHitMode::Ignore,
                None,
            )
            .fraction
        });

        let min_fraction = fractions.into_iter().fold(1.0, Real::min);
        (min_fraction < 1.0).then_some(min_fraction * dt)
    }

    /// Runs the continuous-collision pass on all fast bodies and clamps their `next_position`
    /// to their earliest time of impact: non-bullets sweep the automatic targets (fixed
    /// colliders and soft-body collision meshes) first, then bullets sweep every (possibly
    /// already clamped) body, except the kinematic colliders whose contact with them the
    /// kinematic sweeps of this pass resolved; velocities are never modified. Sensor
    /// crossings the narrow phase would miss entirely emit paired `Started`/`Stopped`
    /// intersection events.
    #[profiling::function]
    #[allow(clippy::too_many_arguments)]
    pub fn solve_continuous(
        &mut self,
        params: &IntegrationParameters,
        islands: &IslandManager,
        bodies: &mut RigidBodySet,
        colliders: &ColliderSet,
        soft_bodies: &SoftBodySet,
        impulse_joints: &ImpulseJointSet,
        multibody_joints: &MultibodyJointSet,
        broad_phase: &mut BroadPhaseBvh,
        narrow_phase: &NarrowPhase,
        hooks: &dyn PhysicsHooks,
        events: &dyn EventHandler,
    ) {
        let ties = SoftTies {
            soft_bodies,
            impulse_joints,
            multibody_joints,
        };
        let dt = params.dt;
        let linear_slop = params.allowed_linear_error();

        // NOTE: broad-phase AABBs are NOT enlarged to the swept volumes: only the fast body's
        // query box is swept; stationary targets keep their fat AABBs. Swept AABBs in
        // the tree leak into the next step's broad phase (pair explosion, ~2x narrow-phase cost).
        let (non_bullets, bullets): (Vec<_>, Vec<_>) = islands
            .active_bodies()
            .filter(|h| bodies[*h].ccd.ccd_active && bodies[*h].is_dynamic())
            .partition(|h| !is_bullet(&bodies[*h]));

        let mut all_results = Vec::new();

        // The soft-body meshes deform every step without any scene change: gathered fresh, where
        // the solve left their particles, for both passes.
        let prediction = params.prediction_distance();
        let swept: Vec<_> = non_bullets
            .iter()
            .chain(&bullets)
            .flat_map(|h| {
                let rb = &bodies[*h];
                swept_collider_aabbs(rb, colliders, rb.pos.next_position)
            })
            .collect();
        let soft = collect_soft_targets(soft_bodies, colliders, prediction, &swept);

        // Pass 1: fast non-bullet bodies vs fixed targets and soft-body meshes (all targets
        // stand still through the pass, so the bodies are independent and can run in parallel).
        {
            let query_pipeline = broad_phase.as_query_pipeline(
                narrow_phase.query_dispatcher(),
                bodies,
                colliders,
                QueryFilter::default(),
            );
            let (bvh, dispatcher) = (query_pipeline.bvh, query_pipeline.dispatcher);
            // Non-bullet fast bodies only hit automatic targets: sweep against the (small) cached
            // fixed-collider list instead of the full BVH.
            let cache_valid = self
                .fixed_targets_cache
                .as_ref()
                .is_some_and(|(p, _)| *p == prediction);
            if !cache_valid {
                self.fixed_targets_cache = Some((
                    prediction,
                    collect_fixed_targets(bodies, colliders, prediction),
                ));
            }
            let targets = match &self.fixed_targets_cache.as_ref().unwrap().1 {
                Some(fixed) => CcdTargets::Lists { fixed, soft: &soft },
                None => CcdTargets::FullBvh { bvh, soft: &soft },
            };
            let results = map_bodies_parallel(&non_bullets, hooks, |handle, hooks| {
                sweep_fast_body(
                    handle,
                    bodies,
                    colliders,
                    ties,
                    bodies[handle].pos.next_position,
                    targets,
                    dispatcher,
                    hooks,
                    dt,
                    linear_slop,
                    PseudoHitMode::Record,
                    None,
                )
            });
            all_results.extend(results);
        }
        Self::apply_clamps(bodies, multibody_joints, &mut all_results);

        // Pass 2: bullets vs everything except other bullets. Targets read the (already
        // clamped) `next_position` from pass 1 (deferred bullet stage).
        if !bullets.is_empty() {
            let mut bullet_results = {
                let query_pipeline = broad_phase.as_query_pipeline(
                    narrow_phase.query_dispatcher(),
                    bodies,
                    colliders,
                    QueryFilter::default(),
                );
                let (bvh, dispatcher) = (query_pipeline.bvh, query_pipeline.dispatcher);
                let kinematic_pairs = &self.kinematic_pairs;
                map_bodies_parallel(&bullets, hooks, |handle, hooks| {
                    sweep_fast_body(
                        handle,
                        bodies,
                        colliders,
                        ties,
                        bodies[handle].pos.next_position,
                        CcdTargets::FullBvh { bvh, soft: &soft },
                        dispatcher,
                        hooks,
                        dt,
                        linear_slop,
                        PseudoHitMode::Record,
                        Some(kinematic_pairs),
                    )
                })
            };
            Self::apply_clamps(bodies, multibody_joints, &mut bullet_results);
            all_results.extend(bullet_results);
        }

        // Emit intersection events for sensor crossings that happened strictly before each
        // body's final solid impact and that the narrow phase would never observe (no
        // overlap at either the start or the clamped end pose).
        for result in &all_results {
            for hit in &result.pseudo_hits {
                if hit.fraction >= result.fraction {
                    // The body stops before reaching this sensor.
                    continue;
                }

                let co1 = &colliders[hit.ch1];
                let co2 = &colliders[hit.ch2];

                if !co1.is_sensor() && !co2.is_sensor() {
                    // TODO: this happens if we found a TOI between two non-sensor
                    //       colliders with mismatching solver_flags. It is not clear
                    //       what we should do in this case: we could report a
                    //       contact started/contact stopped event for example. But in
                    //       that case, what contact pair should be pass to these events?
                    // For now we just ignore this special case. Let's wait for an actual
                    // use-case to come up before we determine what we want to do here.
                    continue;
                }

                let next_pose = |co: &Collider| match co.parent.as_ref() {
                    Some(parent) => bodies[parent.handle].pos.next_position * parent.pos_wrt_parent,
                    None => co.pos.0,
                };

                let prev_pos12 = co1.pos.inv_mul(&co2.pos);
                let next_pos12 = next_pose(co1).inv_mul(&next_pose(co2));

                let dispatcher = narrow_phase.query_dispatcher();
                let intersect_before = dispatcher
                    .intersection_test(&prev_pos12, co1.shape.as_ref(), co2.shape.as_ref())
                    .is_ok_and(|hit| hit.intersecting);
                let intersect_after = dispatcher
                    .intersection_test(&next_pos12, co1.shape.as_ref(), co2.shape.as_ref())
                    .is_ok_and(|hit| hit.intersecting);

                if !intersect_before
                    && !intersect_after
                    && (co1.flags.active_events | co2.flags.active_events)
                        .contains(ActiveEvents::COLLISION_EVENTS)
                {
                    // Emit one intersection-started and one intersection-stopped event.
                    events.handle_collision_event(
                        bodies,
                        colliders,
                        CollisionEvent::Started(hit.ch1, hit.ch2, CollisionEventFlags::SENSOR),
                        None,
                    );
                    events.handle_collision_event(
                        bodies,
                        colliders,
                        CollisionEvent::Stopped(hit.ch1, hit.ch2, CollisionEventFlags::SENSOR),
                        None,
                    );
                }
            }
        }
        self.kinematic_pairs.clear();
    }

    /// Clamps each impacted body's `next_position` to the interpolated pose at its impact
    /// fraction. Pose only — velocities are preserved.
    ///
    /// A multibody states the poses of its links from its root's pose and its joints'
    /// coordinates, and the next step reads its root's pose back from the root's body, so a
    /// multibody is clamped as a whole: at the earliest fraction any of its links strikes, its
    /// root's body is clamped to its own pose at that fraction and every link's body moves with
    /// it by the same rigid motion, so the links end the step where the multibody stands. A
    /// multibody whose root is not dynamic has no motion of its own to clamp, and its links end
    /// the step where its coordinates put them. Each link's result then holds the fraction its
    /// multibody stopped at, `1.0` for one that did not stop.
    fn apply_clamps(
        bodies: &mut RigidBodySet,
        multibodies: &MultibodyJointSet,
        results: &mut [BodyContinuousResult],
    ) {
        let mut struck: Vec<(MultibodyIndex, Real)> = Vec::new();
        for result in results.iter() {
            if result.fraction >= 1.0 {
                continue;
            }
            if let Some(link) = multibodies.rigid_body_link(result.handle) {
                match struck
                    .iter_mut()
                    .find(|(index, _)| *index == link.multibody)
                {
                    Some((_, fraction)) => *fraction = fraction.min(result.fraction),
                    None => struck.push((link.multibody, result.fraction)),
                }
                continue;
            }
            let rb = bodies.index_mut_internal(result.handle);
            let sweep = Sweep::from_poses(
                &rb.pos.position,
                &rb.pos.next_position,
                rb.mprops.local_mprops.local_com,
            );
            rb.pos.next_position = sweep.transform_at(result.fraction);
        }
        for (index, fraction) in &mut struck {
            let Some(multibody) = multibodies.get_multibody(*index) else {
                continue;
            };
            let root = bodies.index_mut_internal(multibody.root().rigid_body_handle());
            if !root.is_dynamic() {
                *fraction = 1.0;
                continue;
            }
            let sweep = Sweep::from_poses(
                &root.pos.position,
                &root.pos.next_position,
                root.mprops.local_mprops.local_com,
            );
            let unclamped = root.pos.next_position;
            root.pos.next_position = sweep.transform_at(*fraction);
            let carried = root.pos.next_position * unclamped.inverse();
            for other in multibody.links().skip(1) {
                let rb = bodies.index_mut_internal(other.rigid_body_handle());
                rb.pos.next_position = carried * rb.pos.next_position;
            }
        }
        for result in results.iter_mut() {
            let Some(link) = multibodies.rigid_body_link(result.handle) else {
                continue;
            };
            if let Some((_, fraction)) = struck.iter().find(|(index, _)| *index == link.multibody) {
                result.fraction = *fraction;
            }
        }
    }
}
