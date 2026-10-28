//! Continuous collision of fast kinematic bodies against the dynamic bodies in their path.
//!
//! A kinematic body is moved, not simulated: the step carries it from its pose to the pose it
//! was given, and the contact solver only meets the dynamic bodies the narrow phase found close
//! to it at the start of the step. A thin body that turns or slides farther in one step than the
//! bodies in its way are thick (a flipper, a bat, a rotor, a paddle) reaches a body the narrow
//! phase never paired with it: without this pass the body is either skipped over or found deep
//! inside it at the next step.
//!
//! This pass sweeps every fast kinematic body along its whole motion (its interpolated
//! velocity, rotation included) against the dynamic bodies its swept bounds overlap, each
//! moving along its own solved motion. At the first time of impact of each pair, the dynamic
//! body is struck like a contact with a body of infinite mass: the normal impulse brings the
//! relative normal velocity at the contact point to `-restitution` times its approach speed,
//! and Coulomb friction acts on the tangential velocity. The struck body then travels its new
//! velocity for what remains of the step, and if the kinematic body would still be inside it
//! (or past it) at the end of the step, it is set against the kinematic body's face, moving at
//! least as fast as that face along its normal.

use crate::alloc_prelude::*;
use crate::dynamics::{
    CoefficientCombineRule, ImpulseJointSet, IslandManager, MultibodyJointSet, RigidBody,
    RigidBodyCcd, RigidBodyHandle, RigidBodySet, RigidBodyVelocity,
};
use crate::geometry::{BroadPhaseBvh, ColliderHandle, ColliderSet, CollisionEvent, NarrowPhase};
use crate::math::{Pose, Real, Vector};
use crate::parry::bounding_volume::{Aabb, BoundingVolume};
use crate::pipeline::{
    ActiveEvents, ActiveHooks, EventHandler, PairFilterContext, PhysicsHooks, QueryFilter,
};
use crate::prelude::CollisionEventFlags;
use parry::query::sweep_toi::{Sweep, SweepToiStatus, ToiProxy, sweep_time_of_impact};
use parry::query::{NonlinearRigidMotion, QueryDispatcher, ShapeCastOptions};
use parry::shape::Shape;
use parry::utils::PoseOpt;

/// One dynamic body a kinematic body struck within a step, as the kinematic sweeps resolved it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KinematicStrike {
    /// The kinematic body's collider that struck.
    pub kinematic_collider: ColliderHandle,
    /// The struck dynamic body's collider.
    pub dynamic_collider: ColliderHandle,
    /// World-space point of the strike, at the time of impact.
    pub point: Vector,
    /// World-space contact normal at the time of impact, pointing from the kinematic collider
    /// to the dynamic one.
    pub normal: Vector,
    /// How fast the two surfaces closed along the normal at the time of impact.
    pub closing_speed: Real,
    /// The normal impulse the strike applied to the dynamic body.
    pub normal_impulse: Real,
    /// The size of the friction impulse the strike applied to the dynamic body.
    pub tangent_impulse: Real,
    /// Whether the two colliders ended the step farther apart than the prediction distance: a
    /// strike that leaves them closer is a contact the next step's narrow phase holds.
    pub separated: bool,
}

/// The motion, in one step, of the farthest point of `rb` from its center of mass: the travel of
/// the center of mass plus the arc the rotation sweeps at the body's farthest extent.
fn max_point_motion(rb: &RigidBody, dt: Real) -> Real {
    let linear = rb.vels.linvel.length() * dt;
    #[cfg(feature = "dim3")]
    let angular = rb.vels.angvel.length() * dt;
    #[cfg(feature = "dim2")]
    let angular = rb.vels.angvel.abs() * dt;
    linear + angular * rb.mprops.max_extent()
}

/// Is `rb` a kinematic body moving far enough this step to sweep the bodies in its path? A body
/// moving less than the allowed linear error leaves every contact to the solver.
pub(crate) fn kinematic_sweeps_this_step(rb: &RigidBody, dt: Real, linear_slop: Real) -> bool {
    rb.is_kinematic() && !rb.is_soft_frame() && max_point_motion(rb, dt) > linear_slop
}

/// The constant-velocity motion of `rb` over the step, starting at its current pose.
fn body_motion(rb: &RigidBody, vels: &RigidBodyVelocity<Real>) -> NonlinearRigidMotion {
    NonlinearRigidMotion::new(
        rb.pos.position,
        rb.mprops.local_mprops.local_com,
        vels.linvel,
        vels.angvel,
    )
}

/// The velocity a dynamic body moves at from its pose at the start of the step to the pose the
/// solver left it at.
fn solved_motion_velocity(rb: &RigidBody, inv_dt: Real) -> RigidBodyVelocity<Real> {
    rb.pos
        .interpolate_velocity(inv_dt, rb.mprops.local_mprops.local_com)
}

/// The angle `rb` turns through in one step.
fn turn_per_step(rb: &RigidBody, dt: Real) -> Real {
    #[cfg(feature = "dim3")]
    return rb.vels.angvel.length() * dt;
    #[cfg(feature = "dim2")]
    return rb.vels.angvel.abs() * dt;
}

/// The largest turn one interpolated sweep covers. The sweep holds each body's points on the
/// arcs they travel and is exact at its two ends; a motion turning farther in a step is swept
/// in as many pieces of at most this turn as it takes.
const MAX_SWEEP_TURN: Real = core::f64::consts::FRAC_PI_4 as Real;

/// The most pieces one step's motion is swept in.
const MAX_SWEEP_PIECES: usize = 64;

/// Where and when two moving colliders first come into contact within the step.
struct FirstContact {
    /// Seconds from the start of the step.
    time: Real,
    /// World-space contact point.
    point: Vector,
    /// World-space contact normal, pointing from the first collider to the second.
    normal: Vector,
}

/// How two moving colliders meet in the step.
enum SweepContact {
    /// They first come into contact within the step.
    Within(FirstContact),
    /// They touch at the start of the step: the contact solver resolved their velocities.
    AtStart,
}

/// How the collider `shape1` moving along `motion1` meets `shape2` moving along `motion2` in
/// the step (`local_center*` being each one's center of mass in its own frame): the first
/// contact within the step, or [`SweepContact::AtStart`] for a pair already touching as the
/// step starts.
#[allow(clippy::too_many_arguments)]
fn first_contact(
    dispatcher: &dyn QueryDispatcher,
    shape1: &dyn Shape,
    motion1: &NonlinearRigidMotion,
    local_center1: Vector,
    shape2: &dyn Shape,
    motion2: &NonlinearRigidMotion,
    local_center2: Vector,
    turn: Real,
    dt: Real,
    linear_slop: Real,
) -> Option<SweepContact> {
    if let (Some(proxy1), Some(proxy2)) =
        (ToiProxy::from_shape(shape1), ToiProxy::from_shape(shape2))
    {
        let pieces = ((turn / MAX_SWEEP_TURN).ceil() as usize).clamp(1, MAX_SWEEP_PIECES);
        let piece = dt / pieces as Real;
        for i in 0..pieces {
            let (start, end) = (piece * i as Real, piece * (i + 1) as Real);
            let sweep1 = Sweep::from_poses(
                &motion1.position_at_time(start),
                &motion1.position_at_time(end),
                local_center1,
            );
            let sweep2 = Sweep::from_poses(
                &motion2.position_at_time(start),
                &motion2.position_at_time(end),
                local_center2,
            );
            let output = sweep_time_of_impact(&proxy1, &sweep1, &proxy2, &sweep2, 1.0, linear_slop);
            match output.status {
                SweepToiStatus::Separated => continue,
                SweepToiStatus::Overlapped => return (i == 0).then_some(SweepContact::AtStart),
                SweepToiStatus::Hit | SweepToiStatus::Failed => {
                    if i == 0 && output.fraction <= 0.0 {
                        return Some(SweepContact::AtStart);
                    }
                    let normal = output.normal.normalize_or_zero();
                    return (normal != Vector::ZERO).then_some(SweepContact::Within(
                        FirstContact {
                            time: start + output.fraction * piece,
                            point: output.point,
                            normal,
                        },
                    ));
                }
            }
        }
        return None;
    }

    // Shapes without a point-cloud proxy (cylinders, cones, composite shapes): the
    // conservative-advancement cast along both motions.
    let cast = dispatcher
        .cast_shapes_nonlinear(motion1, shape1, motion2, shape2, 0.0, dt, false)
        .ok()??;
    if cast.time_of_impact <= 0.0 {
        return Some(SweepContact::AtStart);
    }
    if cast.time_of_impact >= dt {
        return None;
    }
    let pose1 = motion1.position_at_time(cast.time_of_impact);
    let normal = (pose1.rotation * cast.normal1).normalize_or_zero();
    (normal != Vector::ZERO).then_some(SweepContact::Within(FirstContact {
        time: cast.time_of_impact,
        point: pose1 * cast.witness1,
        normal,
    }))
}

/// The contact of two colliders touching at the start of the step, at the poses `pose1` and
/// `pose2`: its world-space point on the first collider and its normal, pointing from the first
/// collider to the second.
fn contact_at_start(
    dispatcher: &dyn QueryDispatcher,
    shape1: &dyn Shape,
    pose1: &Pose,
    shape2: &dyn Shape,
    pose2: &Pose,
    prediction: Real,
) -> Option<FirstContact> {
    let contact = dispatcher
        .contact(&pose1.inv_mul(pose2), shape1, shape2, prediction)
        .ok()??;
    let normal = (pose1.rotation * contact.normal1).normalize_or_zero();
    (normal != Vector::ZERO).then_some(FirstContact {
        time: 0.0,
        point: *pose1 * contact.point1,
        normal,
    })
}

/// The contact the narrow phase holds between the colliders `kinematic` and `dynamic` as the step
/// starts, when it holds an active one: its deepest point, on the kinematic collider, and its
/// normal, pointing from the kinematic collider to the dynamic one.
fn contact_held_at_start(
    narrow_phase: &NarrowPhase,
    colliders: &ColliderSet,
    kinematic: ColliderHandle,
    dynamic: ColliderHandle,
) -> Option<FirstContact> {
    // Looked up from the dynamic collider: the lookup walks the pairs of the collider it is
    // given first, and a kinematic collider holds one for every body resting on it.
    let pair = narrow_phase.contact_pair(dynamic, kinematic)?;
    if !pair.has_any_active_contact() {
        return None;
    }
    let (manifold, contact) = pair.find_deepest_contact()?;
    let pose1 = manifold
        .subshape_pos1()
        .prepend_to(colliders.get(pair.collider1)?.position());
    let normal1 = (pose1.rotation * manifold.local_n1).normalize_or_zero();
    let (point, normal) = if pair.collider1 == kinematic {
        (pose1 * contact.local_p1, normal1)
    } else {
        let pose2 = manifold
            .subshape_pos2()
            .prepend_to(colliders.get(pair.collider2)?.position());
        (pose2 * contact.local_p2, -normal1)
    };
    (normal != Vector::ZERO).then_some(FirstContact {
        time: 0.0,
        point,
        normal,
    })
}

/// The first contact of one kinematic collider with one collider of a dynamic body.
struct KinematicHit {
    /// Time of impact in seconds from the start of the step.
    time: Real,
    kinematic_collider: ColliderHandle,
    dynamic_collider: ColliderHandle,
    /// World-space contact point on the kinematic collider at the time of impact.
    point: Vector,
    /// World-space contact normal at the time of impact, pointing from the kinematic body to
    /// the dynamic one.
    normal: Vector,
    /// The two touched at the start of the step, so the contact solver resolved their
    /// velocities and the hit only keeps the dynamic body clear of the kinematic one.
    at_start: bool,
}

/// What resolving one [`KinematicHit`] did to its dynamic body.
enum HitOutcome {
    /// The body keeps the motion the solver gave it.
    Untouched,
    /// The kinematic face overtook a body the solver had already sent off: it is set back in
    /// front of the face, moving at least as fast as the face along its normal.
    Kept,
    /// The kinematic body struck it within the step.
    Struck(StrikeImpulse),
}

/// The pairs of a dynamic body's collider and a kinematic collider whose contact the kinematic
/// sweeps of one pass resolved: every pair they swept, whether it met or not. The bullets' own
/// sweeps leave these pairs alone: they hold the kinematic collider at its pose at the end of the
/// pass, which is the wrong place for a body that moved through the pass.
#[derive(Clone, Default)]
pub(crate) struct KinematicPairs(Vec<(u32, u32, u32, u32)>);

impl KinematicPairs {
    fn key(dynamic: ColliderHandle, kinematic: ColliderHandle) -> (u32, u32, u32, u32) {
        let (d0, d1) = dynamic.into_raw_parts();
        let (k0, k1) = kinematic.into_raw_parts();
        (d0, d1, k0, k1)
    }

    fn insert(&mut self, dynamic: ColliderHandle, kinematic: ColliderHandle) {
        self.0.push(Self::key(dynamic, kinematic));
    }

    fn sort(&mut self) {
        self.0.sort_unstable();
        self.0.dedup();
    }

    /// Did the kinematic sweeps of this pass resolve the contact of the dynamic collider
    /// `dynamic` with the kinematic collider `kinematic`?
    pub(crate) fn contains(&self, dynamic: ColliderHandle, kinematic: ColliderHandle) -> bool {
        self.0.binary_search(&Self::key(dynamic, kinematic)).is_ok()
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}

/// The inverse effective mass of `rb` along `direction` at the offset `r` from its center of mass.
fn inverse_effective_mass(rb: &RigidBody, r: Vector, direction: Vector) -> Real {
    let linear = direction.dot(direction * rb.mprops.effective_inv_mass);
    #[cfg(feature = "dim3")]
    let angular = {
        let rn = r.cross(direction);
        rn.dot(rb.mprops.effective_world_inv_inertia * rn)
    };
    #[cfg(feature = "dim2")]
    let angular = {
        let rn = r.perp_dot(direction);
        rn * rb.mprops.effective_world_inv_inertia * rn
    };
    linear + angular
}

/// Applies `impulse` at the offset `r` from the center of mass of `vels`' body.
fn apply_impulse(rb: &RigidBody, vels: &mut RigidBodyVelocity<Real>, r: Vector, impulse: Vector) {
    vels.linvel += impulse * rb.mprops.effective_inv_mass;
    #[cfg(feature = "dim3")]
    {
        vels.angvel += rb.mprops.effective_world_inv_inertia * r.cross(impulse);
    }
    #[cfg(feature = "dim2")]
    {
        vels.angvel += rb.mprops.effective_world_inv_inertia * r.perp_dot(impulse);
    }
}

/// The velocity of the point at offset `r` from the center of mass of a body moving at `vels`.
fn point_velocity(vels: &RigidBodyVelocity<Real>, r: Vector) -> Vector {
    #[cfg(feature = "dim3")]
    return vels.linvel + vels.angvel.cross(r);
    #[cfg(feature = "dim2")]
    return vels.linvel + Vector::new(-vels.angvel * r.y, vels.angvel * r.x);
}

/// The impulses one strike applied: normal and friction, and the speed the surfaces closed at.
#[derive(Copy, Clone, Debug, Default)]
struct StrikeImpulse {
    closing_speed: Real,
    normal: Real,
    tangent: Real,
}

/// Strikes the dynamic body at the offset `r_dynamic` from its center of mass with the
/// kinematic surface moving at `surface_velocity`: a normal impulse that leaves the bodies
/// separating at `restitution` times their approach speed, and a friction impulse bounded by
/// `friction` times it. Returns the impulses, or `None` when the bodies were not approaching.
#[allow(clippy::too_many_arguments)]
fn strike(
    rb: &RigidBody,
    vels: &mut RigidBodyVelocity<Real>,
    r_dynamic: Vector,
    normal: Vector,
    surface_velocity: Vector,
    restitution: Real,
    friction: Real,
) -> Option<StrikeImpulse> {
    let relative = point_velocity(vels, r_dynamic) - surface_velocity;
    let approach = relative.dot(normal);
    if approach >= 0.0 {
        return None;
    }
    let k_normal = inverse_effective_mass(rb, r_dynamic, normal);
    if k_normal <= 0.0 {
        return None;
    }
    let normal_impulse = -(1.0 + restitution) * approach / k_normal;
    let mut impulse = normal * normal_impulse;
    let mut friction_impulse = 0.0;

    let tangential = relative - normal * approach;
    let slip = tangential.length();
    if slip > Real::EPSILON && friction > 0.0 {
        let tangent = tangential / slip;
        let k_tangent = inverse_effective_mass(rb, r_dynamic, tangent);
        if k_tangent > 0.0 {
            friction_impulse = (slip / k_tangent).min(friction * normal_impulse);
            impulse -= tangent * friction_impulse;
        }
    }
    apply_impulse(rb, vels, r_dynamic, impulse);
    Some(StrikeImpulse {
        closing_speed: -approach,
        normal: normal_impulse,
        tangent: friction_impulse,
    })
}

/// Whether contacts between the bodies `h1` and `h2` are switched off by a joint linking them.
/// The lookup walks the joints of `h1`.
pub(super) fn contacts_disabled_by_joint(
    impulse_joints: &ImpulseJointSet,
    multibody_joints: &MultibodyJointSet,
    h1: RigidBodyHandle,
    h2: RigidBodyHandle,
) -> bool {
    if impulse_joints
        .joints_between(h1, h2)
        .any(|(_, joint)| !joint.data.contacts_enabled)
    {
        return true;
    }
    multibody_joints
        .joint_between(h1, h2)
        .is_some_and(|(_, _, link)| !link.joint.data.contacts_enabled)
}

/// Sweeps every fast kinematic body against the dynamic bodies in its path and strikes each one
/// it reaches, recording each strike in `strikes` and every pair it swept in `pairs`. A body the
/// face overtakes after the contact solver sent it off, from a contact it held at the start of
/// the step, is set back in front of the face. Returns `true` when any body was struck or set
/// back; every such body is awake, its velocity and end-of-step pose hold what the pass did, and
/// its own CCD flag is re-evaluated on that motion. A strike leaves its pair separated when the
/// two end the step farther apart than `prediction_distance`.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_kinematic_sweeps(
    dt: Real,
    linear_slop: Real,
    prediction_distance: Real,
    islands: &mut IslandManager,
    bodies: &mut RigidBodySet,
    colliders: &ColliderSet,
    impulse_joints: &ImpulseJointSet,
    multibody_joints: &MultibodyJointSet,
    broad_phase: &mut BroadPhaseBvh,
    narrow_phase: &NarrowPhase,
    hooks: &dyn PhysicsHooks,
    events: &dyn EventHandler,
    strikes: &mut Vec<KinematicStrike>,
    pairs: &mut KinematicPairs,
) -> bool {
    pairs.clear();
    if dt <= 0.0 {
        return false;
    }
    let inv_dt = 1.0 / dt;
    let fast_kinematics: Vec<RigidBodyHandle> = islands
        .active_bodies()
        .filter(|h| bodies[*h].ccd.ccd_active && bodies[*h].is_kinematic())
        .collect();
    if fast_kinematics.is_empty() {
        return false;
    }

    let mut moved_any = false;
    for kinematic in fast_kinematics {
        // The hits of this kinematic body, gathered against the poses the bodies hold before
        // any of them is struck, then applied in time order.
        let hits = {
            let query_pipeline = broad_phase.as_query_pipeline(
                narrow_phase.query_dispatcher(),
                bodies,
                colliders,
                QueryFilter::default(),
            );
            let dispatcher = query_pipeline.dispatcher;
            let bvh = query_pipeline.bvh;
            find_kinematic_hits(
                kinematic,
                dt,
                inv_dt,
                linear_slop,
                bodies,
                colliders,
                impulse_joints,
                multibody_joints,
                bvh,
                dispatcher,
                narrow_phase,
                hooks,
                pairs,
            )
        };

        for hit in hits {
            let dynamic = match colliders[hit.dynamic_collider].parent() {
                Some(h) => h,
                None => continue,
            };
            let impulse = match apply_kinematic_hit(
                &hit,
                kinematic,
                dynamic,
                dt,
                inv_dt,
                linear_slop,
                bodies,
                colliders,
                narrow_phase.query_dispatcher(),
            ) {
                HitOutcome::Untouched => continue,
                HitOutcome::Kept => {
                    moved_any = true;
                    islands.wake_up(bodies, dynamic, true);
                    continue;
                }
                HitOutcome::Struck(impulse) => impulse,
            };
            moved_any = true;
            islands.wake_up(bodies, dynamic, true);
            let strike = KinematicStrike {
                kinematic_collider: hit.kinematic_collider,
                dynamic_collider: hit.dynamic_collider,
                point: hit.point,
                normal: hit.normal,
                closing_speed: impulse.closing_speed,
                normal_impulse: impulse.normal,
                tangent_impulse: impulse.tangent,
                separated: !touching_at_end(
                    &hit,
                    prediction_distance,
                    bodies,
                    colliders,
                    narrow_phase,
                ),
            };
            report_strike(&strike, bodies, colliders, narrow_phase, events);
            strikes.push(strike);
        }
    }
    pairs.sort();
    moved_any
}

/// The earliest hit of each dynamic body the kinematic body reaches this step, in time order,
/// a body touching it as the step starts included. A pair the narrow phase holds an active
/// contact for is touching as the step starts, and takes that contact without a sweep. Every
/// pair it sweeps goes into `pairs`.
#[allow(clippy::too_many_arguments)]
fn find_kinematic_hits(
    kinematic: RigidBodyHandle,
    dt: Real,
    inv_dt: Real,
    linear_slop: Real,
    bodies: &RigidBodySet,
    colliders: &ColliderSet,
    impulse_joints: &ImpulseJointSet,
    multibody_joints: &MultibodyJointSet,
    bvh: &parry::partitioning::Bvh,
    dispatcher: &dyn QueryDispatcher,
    narrow_phase: &NarrowPhase,
    hooks: &dyn PhysicsHooks,
    pairs: &mut KinematicPairs,
) -> Vec<KinematicHit> {
    let rb1 = &bodies[kinematic];
    let motion1 = body_motion(rb1, &rb1.vels);
    let reach = max_point_motion(rb1, dt);
    let mut found: Vec<(RigidBodyHandle, KinematicHit)> = Vec::new();

    for ch1 in &rb1.colliders.0 {
        let co1 = &colliders[*ch1];
        let Some(parent1) = co1.parent.as_ref() else {
            continue;
        };
        if co1.is_sensor() || !co1.is_enabled() {
            continue;
        }
        let collider_motion1 = motion1.prepend(parent1.pos_wrt_parent);
        let local_center1 = parent1
            .pos_wrt_parent
            .inverse_transform_point(rb1.mprops.local_mprops.local_com);
        let swept = swept_aabb(co1, &collider_motion1, rb1, dt);

        for leaf in bvh.leaves(|node| node.aabb().intersects(&swept)) {
            let Some((co2, ch2)) = colliders.get_unknown_gen(leaf) else {
                continue;
            };
            let Some(parent2) = co2.parent.as_ref() else {
                continue;
            };
            let bh2 = parent2.handle;
            let Some(rb2) = bodies.get(bh2) else {
                continue;
            };
            if bh2 == kinematic || !rb2.is_dynamic() || rb2.is_soft_frame() {
                continue;
            }
            if co2.is_sensor() || !co2.is_enabled() || co2.is_deformable_collider() {
                continue;
            }
            // A multibody link's velocity is its multibody's: it is not struck directly.
            if multibody_joints.rigid_body_link(bh2).is_some() {
                continue;
            }
            // The kinematic body moves less than half of the thinner of the two bodies: the
            // contact solver resolves what it reaches.
            if reach
                <= RigidBodyCcd::FAST_BODY_SAFETY_FACTOR
                    * rb1.ccd.ccd_thickness.min(rb2.ccd.ccd_thickness)
            {
                continue;
            }
            if !co1
                .flags
                .active_collision_types
                .test(rb1.body_type, rb2.body_type)
                && !co2
                    .flags
                    .active_collision_types
                    .test(rb1.body_type, rb2.body_type)
            {
                continue;
            }
            if !co1.flags.collision_groups.test(co2.flags.collision_groups)
                || !co1.flags.solver_groups.test(co2.flags.solver_groups)
            {
                continue;
            }
            if contacts_disabled_by_joint(impulse_joints, multibody_joints, bh2, kinematic) {
                continue;
            }
            let active_hooks = co1.flags.active_hooks | co2.flags.active_hooks;
            if active_hooks.contains(ActiveHooks::FILTER_CONTACT_PAIRS) {
                let context = PairFilterContext {
                    bodies,
                    colliders,
                    rigid_body1: Some(kinematic),
                    rigid_body2: Some(bh2),
                    collider1: *ch1,
                    collider2: ch2,
                };
                if hooks.filter_contact_pair(&context).is_none() {
                    continue;
                }
            }

            pairs.insert(ch2, *ch1);

            let (contact, at_start) =
                if let Some(contact) = contact_held_at_start(narrow_phase, colliders, *ch1, ch2) {
                    (contact, true)
                } else {
                    let motion2 = body_motion(rb2, &solved_motion_velocity(rb2, inv_dt))
                        .prepend(parent2.pos_wrt_parent);
                    let local_center2 = parent2
                        .pos_wrt_parent
                        .inverse_transform_point(rb2.mprops.local_mprops.local_com);
                    match first_contact(
                        dispatcher,
                        co1.shape.as_ref(),
                        &collider_motion1,
                        local_center1,
                        co2.shape.as_ref(),
                        &motion2,
                        local_center2,
                        turn_per_step(rb1, dt).max(turn_per_step(rb2, dt)),
                        dt,
                        linear_slop,
                    ) {
                        Some(SweepContact::Within(contact)) => (contact, false),
                        Some(SweepContact::AtStart) => {
                            let Some(contact) = contact_at_start(
                                dispatcher,
                                co1.shape.as_ref(),
                                &collider_motion1.position_at_time(0.0),
                                co2.shape.as_ref(),
                                &motion2.position_at_time(0.0),
                                2.0 * linear_slop + co1.contact_skin() + co2.contact_skin(),
                            ) else {
                                continue;
                            };
                            (contact, true)
                        }
                        None => continue,
                    }
                };
            let hit = KinematicHit {
                time: contact.time,
                kinematic_collider: *ch1,
                dynamic_collider: ch2,
                point: contact.point,
                normal: contact.normal,
                at_start,
            };
            if hit.normal == Vector::ZERO {
                continue;
            }
            found.push((bh2, hit));
        }
    }

    // Each body's earliest hit, then every body's in time order.
    found.sort_by(|(h1, a), (h2, b)| {
        h1.into_raw_parts()
            .cmp(&h2.into_raw_parts())
            .then(a.time.total_cmp(&b.time))
    });
    found.dedup_by_key(|(h, _)| *h);
    let mut hits: Vec<KinematicHit> = found.into_iter().map(|(_, hit)| hit).collect();
    hits.sort_by(|a, b| a.time.total_cmp(&b.time));
    hits
}

/// Bounds of everything a collider moving along `motion` covers over the step. The collider of a
/// turning body is bounded by its own farthest extent around the body's center of mass at both
/// ends of the step, which holds however far it turns.
fn swept_aabb(
    co: &crate::geometry::Collider,
    motion: &NonlinearRigidMotion,
    rb: &RigidBody,
    dt: Real,
) -> Aabb {
    let start = motion.position_at_time(0.0);
    let end = motion.position_at_time(dt);
    let mut aabb = co
        .shape
        .compute_aabb(&start)
        .merged(&co.shape.compute_aabb(&end));
    #[cfg(feature = "dim3")]
    let turning = rb.vels.angvel.length() > 0.0;
    #[cfg(feature = "dim2")]
    let turning = rb.vels.angvel != 0.0;
    if turning {
        let com = rb.mprops.local_mprops.local_com;
        let pos_wrt_parent = co
            .parent
            .as_ref()
            .map_or(Pose::IDENTITY, |p| p.pos_wrt_parent);
        let sphere = co
            .shape
            .compute_local_bounding_sphere()
            .transform_by(&pos_wrt_parent);
        let extent = Vector::splat((sphere.center - com).length() + sphere.radius);
        for pose in [rb.pos.position, rb.pos.next_position] {
            let center = pose * com;
            aabb.merge(&Aabb::new(center - extent, center + extent));
        }
    }
    aabb
}

/// Resolves the dynamic body of `hit` and sets its velocity and end-of-step pose.
///
/// A body met within the step and approaching the kinematic surface there is struck
/// ([`HitOutcome::Struck`]), travels its new velocity for the rest of the step and ends it clear
/// of the kinematic body. A body touching it at the start of the step, or already leaving it at
/// the contact, keeps the motion the contact solver gave it, unless the face overtakes it by the
/// end of the step (its center of mass ends behind the face): then it is set back in front of
/// the face, moving at least as fast as the face along its normal ([`HitOutcome::Kept`]).
#[allow(clippy::too_many_arguments)]
fn apply_kinematic_hit(
    hit: &KinematicHit,
    kinematic: RigidBodyHandle,
    dynamic: RigidBodyHandle,
    dt: Real,
    inv_dt: Real,
    linear_slop: Real,
    bodies: &mut RigidBodySet,
    colliders: &ColliderSet,
    dispatcher: &dyn QueryDispatcher,
) -> HitOutcome {
    let co1 = &colliders[hit.kinematic_collider];
    let co2 = &colliders[hit.dynamic_collider];
    let restitution = CoefficientCombineRule::combine(
        co1.material.restitution,
        co2.material.restitution,
        co1.material.restitution_combine_rule,
        co2.material.restitution_combine_rule,
    );
    let friction = CoefficientCombineRule::combine(
        co1.material.friction,
        co2.material.friction,
        co1.material.friction_combine_rule,
        co2.material.friction_combine_rule,
    );

    let rb1 = &bodies[kinematic];
    let kinematic_vels = rb1.vels;
    let motion1 = body_motion(rb1, &kinematic_vels);
    let kinematic_com_at =
        |time: Real| motion1.position_at_time(time) * rb1.mprops.local_mprops.local_com;
    let kinematic_end = rb1.pos.next_position;
    let kinematic_rotation_at_hit = motion1.position_at_time(hit.time).rotation;
    let kinematic_com_end = kinematic_com_at(dt);
    let surface_velocity = point_velocity(&kinematic_vels, hit.point - kinematic_com_at(hit.time));

    let rb2 = &bodies[dynamic];
    let local_com2 = rb2.mprops.local_mprops.local_com;
    let solved_vels = solved_motion_velocity(rb2, inv_dt);
    let pose2_at_hit = body_motion(rb2, &solved_vels).position_at_time(hit.time);
    let r2 = hit.point - pose2_at_hit * local_com2;
    let mut vels2 = rb2.vels;
    let mut struck = if hit.at_start {
        None
    } else {
        strike(
            rb2,
            &mut vels2,
            r2,
            hit.normal,
            surface_velocity,
            restitution,
            friction,
        )
    };

    // The rest of the step at the struck velocity, or the pose the solver moved it to.
    let mut end2 = match struck {
        Some(_) => vels2.integrate(dt - hit.time, &pose2_at_hit, &local_com2),
        None => rb2.pos.next_position,
    };

    // The contact, carried with each body to the end of the step: the kinematic face turns with
    // the kinematic body, the struck point with the struck body.
    let turn = kinematic_end.rotation * kinematic_rotation_at_hit.inverse();
    let normal_end = (turn * hit.normal).normalize_or_zero();
    let face_end = kinematic_end * (motion1.position_at_time(hit.time).inverse() * hit.point);
    let struck_point_end = end2 * (pose2_at_hit.inverse() * hit.point);
    let collider_end1 = kinematic_end
        * co1
            .parent
            .as_ref()
            .map_or(Pose::IDENTITY, |p| p.pos_wrt_parent);
    let parent2 = co2
        .parent
        .as_ref()
        .map_or(Pose::IDENTITY, |p| p.pos_wrt_parent);

    let behind_face = (struck_point_end - face_end).dot(normal_end) < 0.0;
    let overtaken = if struck.is_some() {
        behind_face
            || dispatcher
                .intersection_test(
                    &collider_end1.inv_mul(&(end2 * parent2)),
                    co1.shape.as_ref(),
                    co2.shape.as_ref(),
                )
                .is_ok_and(|hit| hit.intersecting)
    } else {
        // A body the solver moved is left touching or slightly inside a face it rests on; one
        // whose center of mass ends behind the face is more than half way through it.
        (end2 * local_com2 - face_end).dot(normal_end) < 0.0
    };
    let mut kept = false;
    if overtaken && normal_end != Vector::ZERO {
        // Set the struck body against the face, approaching it from the side the contact
        // normal points to. When the cast finds no face on that line the body has slid past
        // the kinematic body's edge and keeps the pose it travelled to.
        let collider_end2 = end2 * parent2;
        let reach = co1.shape.compute_local_bounding_sphere().radius
            + co2.shape.compute_local_bounding_sphere().radius;
        let back_off = 2.0 * reach + (struck_point_end - face_end).dot(normal_end).abs();
        let start2 = Pose::from_translation(normal_end * back_off) * collider_end2;
        let pos12 = collider_end1.inv_mul(&start2);
        let local_dir = collider_end1.rotation.inverse() * (-normal_end);
        let options = ShapeCastOptions {
            max_time_of_impact: back_off,
            target_distance: linear_slop,
            stop_at_penetration: false,
            compute_impact_geometry_on_penetration: false,
        };
        if let Ok(Some(cast)) = dispatcher.cast_shapes(
            &pos12,
            local_dir,
            co1.shape.as_ref(),
            co2.shape.as_ref(),
            options,
        ) {
            let offset = normal_end * (back_off - cast.time_of_impact);
            end2 = Pose::from_translation(offset) * end2;
            kept = true;
            // Against the face, the body moves with it at least along its normal.
            let contact_end = collider_end1 * cast.witness1;
            let face_velocity = point_velocity(&kinematic_vels, contact_end - kinematic_com_end);
            let r_end = contact_end - end2 * local_com2;
            if let Some(carry) = strike(
                &bodies[dynamic],
                &mut vels2,
                r_end,
                normal_end,
                face_velocity,
                0.0,
                friction,
            ) && let Some(impulse) = struck.as_mut()
            {
                impulse.normal += carry.normal;
                impulse.tangent += carry.tangent;
            }
        }
    }
    let outcome = match struck {
        Some(impulse) => HitOutcome::Struck(impulse),
        None if kept => HitOutcome::Kept,
        None => return HitOutcome::Untouched,
    };

    let rb2 = bodies.index_mut_internal(dynamic);
    rb2.vels = vels2;
    rb2.pos.next_position = end2;
    rb2.ccd_vels = rb2
        .pos
        .interpolate_velocity(inv_dt, rb2.mprops.local_mprops.local_com);
    rb2.ccd.ccd_active = rb2.ccd.is_moving_fast_with_next_position(
        dt,
        &rb2.ccd_vels,
        &rb2.pos,
        rb2.mprops.local_mprops.local_com,
        rb2.mprops.max_extent(),
    );
    outcome
}

/// Whether the colliders of `hit` end the step within `prediction_distance` of each other (their
/// contact skins added), the distance inside which the narrow phase holds an active contact.
fn touching_at_end(
    hit: &KinematicHit,
    prediction_distance: Real,
    bodies: &RigidBodySet,
    colliders: &ColliderSet,
    narrow_phase: &NarrowPhase,
) -> bool {
    let co1 = &colliders[hit.kinematic_collider];
    let co2 = &colliders[hit.dynamic_collider];
    let next_pose = |co: &crate::geometry::Collider| match co.parent.as_ref() {
        Some(parent) => bodies[parent.handle].pos.next_position * parent.pos_wrt_parent,
        None => co.pos.0,
    };
    narrow_phase
        .query_dispatcher()
        .distance(
            &next_pose(co1).inv_mul(&next_pose(co2)),
            co1.shape.as_ref(),
            co2.shape.as_ref(),
        )
        .is_ok_and(|separation| {
            separation.distance <= prediction_distance + co1.contact_skin() + co2.contact_skin()
        })
}

/// Reports a strike that left the pair apart as a collision that started and stopped within the
/// step, when the narrow phase holds no contact for the pair and either collider asks for
/// collision events. A pair left touching is reported by the next step's narrow phase instead.
fn report_strike(
    strike: &KinematicStrike,
    bodies: &RigidBodySet,
    colliders: &ColliderSet,
    narrow_phase: &NarrowPhase,
    events: &dyn EventHandler,
) {
    if !strike.separated {
        return;
    }
    let co1 = &colliders[strike.kinematic_collider];
    let co2 = &colliders[strike.dynamic_collider];
    if !(co1.flags.active_events | co2.flags.active_events).contains(ActiveEvents::COLLISION_EVENTS)
    {
        return;
    }
    if narrow_phase
        .contact_pair(strike.dynamic_collider, strike.kinematic_collider)
        .is_some_and(|pair| pair.has_any_active_contact())
    {
        return;
    }
    for event in [
        CollisionEvent::Started(
            strike.kinematic_collider,
            strike.dynamic_collider,
            CollisionEventFlags::empty(),
        ),
        CollisionEvent::Stopped(
            strike.kinematic_collider,
            strike.dynamic_collider,
            CollisionEventFlags::empty(),
        ),
    ] {
        events.handle_collision_event(bodies, colliders, event, None);
    }
}
