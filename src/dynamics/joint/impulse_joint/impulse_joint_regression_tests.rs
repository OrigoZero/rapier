//! Regression tests for the energy impulse joints keep.
//!
//! An ideal joint does no work: a body swinging or tumbling on joints with nothing else
//! acting on it but gravity keeps the sum of its kinetic and potential energy. These tests
//! hold the solver to that over seconds of motion at the substep lengths the engine runs.

use crate::alloc_prelude::*;
use crate::prelude::*;

/// The kinetic and potential energy of `links` under `gravity`.
fn links_energy(bodies: &RigidBodySet, links: &[RigidBodyHandle], gravity: Vector) -> Real {
    links
        .iter()
        .map(|link| {
            let body = &bodies[*link];
            let (linvel, angvel) = (body.linvel(), body.angvel());
            let inertia = body.mass_properties().local_mprops;
            let principal_axes = *body.rotation() * inertia.principal_inertia_local_frame;
            let local = principal_axes.inverse() * angvel;
            0.5 * body.mass() * linvel.dot(linvel)
                + 0.5 * angvel.dot(principal_axes * (inertia.principal_inertia() * local))
                - body.mass() * gravity.dot(body.mass_properties().world_com)
        })
        .sum()
}

/// A collider that touches nothing: the links of these tests meet only through their joints.
fn untouching_rod() -> ColliderBuilder {
    ColliderBuilder::cuboid(0.25, 0.04, 0.04)
        .mass(1.0)
        .collision_groups(InteractionGroups::new(
            Group::GROUP_1,
            Group::NONE,
            InteractionTestMode::And,
        ))
}

/// The steps-per-second and substeps the tests run at: substeps of a 120th, a 240th and a
/// 960th of a second.
const RATES: [(Real, usize); 7] = [
    (30.0, 4),
    (60.0, 2),
    (60.0, 4),
    (120.0, 1),
    (120.0, 2),
    (240.0, 1),
    (240.0, 4),
];

/// The angles, in degrees above the horizontal, the swings are released at.
const RELEASES: [Real; 6] = [-60.0, -30.0, -10.0, 0.0, 10.0, 30.0];

/// A chain of `links` rods (0.5 m by 0.08 m by 0.08 m, 1 kg each) laid out straight at
/// `release` degrees above the horizontal from a fixed anchor at (0, 10, 0), each joined to the
/// one above it by `joint` at their touching ends, and let swing for 10 s at `rate` steps a second in
/// `substeps` substeps: the change of its energy over those 10 s, as a share of the energy it
/// stands above hanging straight down at its release.
fn swing(
    links_n: usize,
    rate: Real,
    substeps: usize,
    release: Real,
    joint: impl Fn() -> GenericJoint,
) -> Real {
    swing_from(Vector::new(0.0, 10.0, 0.0), links_n, rate, substeps, release, joint)
}

/// [`swing`] with the anchor at `top`.
fn swing_from(
    top: Vector,
    links_n: usize,
    rate: Real,
    substeps: usize,
    release: Real,
    joint: impl Fn() -> GenericJoint,
) -> Real {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let anchor = world.insert_body(RigidBodyBuilder::fixed().translation(top));
    let along = Vector::new(release.to_radians().cos(), release.to_radians().sin(), 0.0);
    let turn = Rotation::from_rotation_z(release.to_radians()).to_scaled_axis();
    let mut above = anchor;
    let mut links = Vec::new();
    for i in 0..links_n {
        let (link, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(top + along * (0.25 + i as Real * 0.5))
                .rotation(turn)
                .allow_fast_rotation(true),
            untouching_rod(),
        );
        let mut data = joint();
        data.set_local_anchor1(Vector::new(-0.25, 0.0, 0.0));
        data.set_local_anchor2(if above == anchor {
            Vector::ZERO
        } else {
            Vector::new(0.25, 0.0, 0.0)
        });
        world.insert_impulse_joint(link, above, data);
        links.push(link);
        above = link;
    }
    let gravity = world.gravity;
    let start = links_energy(&world.bodies, &links, gravity);
    let hanging: Real = (0..links_n)
        .map(|i| -gravity.y * (top.y - 0.25 - i as Real * 0.5))
        .sum();
    for _ in 0..(10.0 * rate) as usize {
        world.step();
    }
    (links_energy(&world.bodies, &links, gravity) - start) / (start - hanging)
}

/// The releases and rates at which `swing` changed the energy by more than `bound` of what it
/// stands above hanging, each with the change it read.
fn swings_past(links_n: usize, bound: Real, joint: impl Fn() -> GenericJoint) -> Vec<String> {
    let mut past = Vec::new();
    for (rate, substeps) in RATES {
        for release in RELEASES {
            let change = swing(links_n, rate, substeps, release, &joint);
            if change.abs() > bound {
                past.push(alloc::format!(
                    "{release}° at {rate} Hz x {substeps}: {:+.1}%",
                    100.0 * change
                ));
            }
        }
    }
    past
}

/// A rod swinging from a ball joint keeps its energy to within 2% over 10 s at every substep
/// length, released level, above level and below. The stabilization pass that ends each substep
/// took back the velocity that carries the rod's anchor across the chord its turn cuts, a
/// projection of the velocity the rod swings on with at every substep: released level it lost
/// 56% of its energy at 30 steps a second in 4 substeps, 39% at 240 in 1 and 13% at 240 in 4.
#[test]
#[cfg(feature = "dim3")]
fn a_rod_swinging_from_a_ball_joint_keeps_its_energy_at_every_substep_length() {
    let past = swings_past(1, 0.02, || SphericalJointBuilder::new().build().into());
    assert!(past.is_empty(), "the rod's energy changed: {past:?}");
}

/// A rod swinging from a hinge keeps its energy to within 2% over 10 s at every substep
/// length: a hinge's locked axes hold its anchor as a ball joint's do.
#[test]
#[cfg(feature = "dim3")]
fn a_rod_swinging_from_a_hinge_keeps_its_energy_at_every_substep_length() {
    let past = swings_past(1, 0.02, || {
        RevoluteJointBuilder::new(Vector::Z).build().into()
    });
    assert!(past.is_empty(), "the rod's energy changed: {past:?}");
}

/// A chain of five rods on ball joints keeps its energy to within 12% over 10 s at 240 steps a
/// second in 4 substeps, from every release, and to within 6% on average over them. It lost a
/// quarter of its energy on average and half of it from 10 and 30 degrees above level.
#[test]
#[cfg(feature = "dim3")]
fn a_chain_of_five_rods_on_ball_joints_keeps_its_energy_in_short_substeps() {
    let changes: Vec<Real> = RELEASES
        .iter()
        .map(|release| {
            swing(5, 240.0, 4, *release, || {
                SphericalJointBuilder::new().build().into()
            })
        })
        .collect();
    let mean = changes.iter().sum::<Real>() / changes.len() as Real;
    assert!(
        mean.abs() <= 0.06 && changes.iter().all(|change| change.abs() <= 0.12),
        "the chain's energy changed by {:+.1}% on average, from each release {:+.3?}",
        100.0 * mean,
        changes
    );
}

/// Two rods joined end to end by a ball joint, tumbling through empty space, keep their energy
/// and their linear and angular momentum over 10 s at 30 steps a second in 4 substeps: the
/// joint holds their ends together and does no work on them. The joint's rows took back the
/// velocity that carries its anchors across the chords their turns cut at every substep.
#[test]
#[cfg(feature = "dim3")]
fn two_rods_tumbling_on_a_ball_joint_keep_their_energy_and_momentum() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = 1.0 / 30.0;
    world.integration_parameters.num_solver_iterations = 4;
    let (a, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(-0.25, 0.0, 0.0))
            .linvel(Vector::new(0.0, -2.0, 0.5))
            .angvel(Vector::new(1.0, 0.5, 8.0))
            .allow_fast_rotation(true),
        untouching_rod(),
    );
    let (b, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.25, 0.0, 0.0))
            .linvel(Vector::new(0.0, 2.0, -0.5))
            .angvel(Vector::new(-1.0, 0.5, 8.0))
            .allow_fast_rotation(true),
        untouching_rod(),
    );
    world.insert_impulse_joint(
        a,
        b,
        SphericalJointBuilder::new()
            .local_anchor1(Vector::new(0.25, 0.0, 0.0))
            .local_anchor2(Vector::new(-0.25, 0.0, 0.0)),
    );
    // One step lets the joint take the velocities in to what it allows.
    world.step();
    let state = |world: &PhysicsWorld| {
        let links = [a, b];
        let energy = links_energy(&world.bodies, &links, Vector::ZERO);
        let mut linear = Vector::ZERO;
        let mut angular = Vector::ZERO;
        for link in links {
            let body = &world.bodies[link];
            let com = body.mass_properties().world_com;
            linear += body.linvel() * body.mass();
            let inertia = body.mass_properties().local_mprops;
            let principal_axes = *body.rotation() * inertia.principal_inertia_local_frame;
            let local = principal_axes.inverse() * body.angvel();
            angular += com.cross(body.linvel() * body.mass())
                + principal_axes * (inertia.principal_inertia() * local);
        }
        (energy, linear, angular)
    };
    let (energy0, linear0, angular0) = state(&world);
    for _ in 0..300 {
        world.step();
    }
    let (energy, linear, angular) = state(&world);
    assert!(
        (energy - energy0).abs() <= 0.02 * energy0,
        "energy {energy0} J became {energy} J"
    );
    assert!(
        (linear - linear0).length() <= 1.0e-3 * (1.0 + linear0.length()),
        "linear momentum {linear0:?} became {linear:?}"
    );
    assert!(
        (angular - angular0).length() <= 0.02 * angular0.length(),
        "angular momentum {angular0:?} became {angular:?}"
    );
}

/// A 1 kg ball on a rope from a fixed anchor, released `from` the anchor, swung for 10 s at
/// `rate` steps a second in `substeps` substeps: its energy before and after, in joules.
fn rope_swing(rate: Real, substeps: usize, from: Vector) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let top = Vector::new(0.0, 10.0, 0.0);
    let anchor = world.insert_body(RigidBodyBuilder::fixed().translation(top));
    let (ball, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(top + from),
        ColliderBuilder::ball(0.05)
            .mass(1.0)
            .collision_groups(InteractionGroups::new(
                Group::GROUP_1,
                Group::NONE,
                InteractionTestMode::And,
            )),
    );
    world.insert_impulse_joint(anchor, ball, RopeJointBuilder::new(1.0));
    let gravity = world.gravity;
    let start = links_energy(&world.bodies, &[ball], gravity);
    for _ in 0..(10.0 * rate) as usize {
        world.step();
    }
    (start, links_energy(&world.bodies, &[ball], gravity))
}

/// A ball swinging on a taut 1 m rope, released level, keeps its energy to within 2% over 10 s
/// at every substep length. The stabilization pass took back the velocity the rope's turn
/// carries its length out by over a substep: the ball lost 36% of its energy at 30 steps a
/// second in 4 substeps and 22% at 240 in 1.
#[test]
#[cfg(feature = "dim3")]
fn a_ball_swinging_on_a_taut_rope_keeps_its_energy_at_every_substep_length() {
    let mut past = Vec::new();
    for (rate, substeps) in RATES {
        let (start, end) = rope_swing(rate, substeps, Vector::new(1.0, 0.0, 0.0));
        // The energy it can give up: from level to hanging 1 m below the anchor.
        let change = (end - start) / 9.81;
        if change.abs() > 0.02 {
            past.push(alloc::format!(
                "{rate} Hz x {substeps}: {:+.1}%",
                100.0 * change
            ));
        }
    }
    assert!(past.is_empty(), "the ball's energy changed: {past:?}");
}

/// A ball dropped beside its anchor, 0.5 m out on a 1 m rope, falls slack until the rope
/// catches it and swings on: it ends with no more energy than it was dropped with, at every
/// substep length.
#[test]
#[cfg(feature = "dim3")]
fn a_ball_caught_by_a_slack_rope_gains_no_energy() {
    let mut past = Vec::new();
    for (rate, substeps) in RATES {
        let (start, end) = rope_swing(rate, substeps, Vector::new(0.5, 0.0, 0.0));
        if end > start + 0.01 * 9.81 {
            past.push(alloc::format!(
                "{rate} Hz x {substeps}: {start} J became {end} J"
            ));
        }
    }
    assert!(past.is_empty(), "the ball gained energy: {past:?}");
}

/// A torso, thigh and calf of 5 kg boxes stacked on a floor at `x` along X, joined by hinges
/// about `axis` (with `spring`, each driven toward the straight chain by a stiffness of 100 and
/// a damping of 10), every one held from moving along Z and the torso (or, with `every`, each
/// of them) from turning about X and Y, falls for 3 s at `dt` in `substeps` substeps: the
/// highest any of them stood and the widest any hinge's anchors came apart.
fn planar_hinge_chain(
    x: Real,
    axis: Vector,
    every: bool,
    spring: bool,
    dt: Real,
    substeps: usize,
) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = dt;
    world.integration_parameters.num_solver_iterations = substeps;
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(x, -0.5, 0.0)),
        ColliderBuilder::cuboid(3.0, 0.5, 3.0),
    );
    let links = [
        (1.30, Vector::new(0.15, 0.2, 0.1), true),
        (0.85, Vector::new(0.06, 0.2, 0.06), every),
        (0.40, Vector::new(0.05, 0.2, 0.05), every),
    ];
    let mut bodies = Vec::new();
    for (y, half, locked) in links {
        let mut body = RigidBodyBuilder::dynamic()
            .translation(Vector::new(x, y, 0.0))
            .enabled_translations(true, true, false);
        if locked {
            body = body.enabled_rotations(false, false, true);
        }
        let (handle, _) = world.insert(body, ColliderBuilder::cuboid(half.x, half.y, half.z).mass(5.0));
        bodies.push(handle);
    }
    let mut hinges = Vec::new();
    for i in 1..bodies.len() {
        let mut hinge = RevoluteJointBuilder::new(axis.normalize())
            .local_anchor1(Vector::new(0.0, 0.22, 0.0))
            .local_anchor2(Vector::new(0.0, -0.23, 0.0));
        if spring {
            hinge = hinge.motor_position(0.0, 100.0, 10.0);
        }
        hinges.push(world.insert_impulse_joint(bodies[i], bodies[i - 1], hinge));
    }
    let mut peak = Real::MIN;
    for _ in 0..(3.0 / dt) as usize {
        world.step();
        for body in &bodies {
            peak = peak.max(world.bodies[*body].translation().y);
        }
    }
    let gap = hinges
        .iter()
        .map(|hinge| {
            let joint = &world.impulse_joints.get(*hinge).expect("a hinge").data;
            let (body1, body2) = (world.impulse_joints.get(*hinge).unwrap().body1, world.impulse_joints.get(*hinge).unwrap().body2);
            let anchor1 = *world.bodies[body1].position() * joint.local_frame1.translation;
            let anchor2 = *world.bodies[body2].position() * joint.local_frame2.translation;
            (anchor1 - anchor2).length()
        })
        .fold(0.0, Real::max);
    (peak, gap)
}

/// A hinge chain held to a plane, falling onto a floor, folds (hinges about Z) or falls as
/// one piece (hinges about the Y-Z diagonal, which the locks leave no turn about) and comes to
/// rest: no body climbs above the 1.30 m the torso starts at and every hinge holds its anchors
/// within 2 cm, with the torso's turning locked or every link's, its hinges free or sprung, at
/// 30 steps a second in 4 substeps and 60 in 2, at the origin and 704 to 720 m out along X.
/// A joint row that carried its points' turning along the plane's normal, which only turning
/// the links can meet, or read that turning off the velocities the substep started with rather
/// than the ones its biased pass left, spun the chain up wherever it could only hold its points
/// by turning the bodies: its calf reached 1e5 rad/s and the chain climbed to 6 m.
#[test]
#[cfg(feature = "dim3")]
fn a_hinge_chain_held_to_a_plane_falls_to_rest_without_climbing() {
    let mut failed = Vec::new();
    for x in [0.0, 704.0, 712.0, 720.0] {
        for axis in [Vector::Z, Vector::new(0.0, 1.0, 1.0)] {
            for every in [false, true] {
                for spring in [false, true] {
                    for (dt, substeps) in [(1.0 / 30.0, 4), (1.0 / 60.0, 2)] {
                        let (peak, gap) = planar_hinge_chain(x, axis, every, spring, dt, substeps);
                        if peak > 1.31 || gap > 0.02 {
                            failed.push(alloc::format!(
                                "at x {x}, hinges about {axis:?}, every link locked {every}, \
                                 sprung {spring}, dt {dt} x {substeps}: peak {peak} m, gap {gap} m"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(failed.is_empty(), "the chain climbed or came apart: {failed:?}");
}

/// A rod swinging from a ball joint or a hinge keeps its energy to within 2% over 10 s far from
/// the origin, at 30 steps a second in 4 substeps and 240 in 1, as it does at the origin: 930 m
/// out, where positions are held to 61 µm, a substep's chord is a few of those steps. A carry
/// bounded by the position correction read the chord at that resolution, and the rod lost up
/// to 4.5% of its energy at (-930, 300, -930) and 3.6% at (562, 20, 140).
#[test]
#[cfg(feature = "dim3")]
fn a_rod_swinging_far_from_the_origin_keeps_its_energy() {
    let mut past = Vec::new();
    for top in [Vector::new(562.0, 20.0, 140.0), Vector::new(-930.0, 300.0, -930.0)] {
        for (rate, substeps) in [(30.0, 4), (240.0, 1)] {
            for hinge in [false, true] {
                let change = swing_from(top, 1, rate, substeps, 0.0, || {
                    if hinge {
                        RevoluteJointBuilder::new(Vector::Z).build().into()
                    } else {
                        SphericalJointBuilder::new().build().into()
                    }
                });
                if change.abs() > 0.02 {
                    past.push(alloc::format!(
                        "hinge {hinge} at {top:?}, {rate} Hz x {substeps}: {:+.1}%",
                        100.0 * change
                    ));
                }
            }
        }
    }
    assert!(past.is_empty(), "the rod's energy changed: {past:?}");
}

/// A 100 kg ball held on a vertical circle of 2 m by a distance held at its radius (a point on
/// a rail's bend), started at the bottom fast enough to loop, loops for a minute at `rate`
/// steps a second in `substeps` substeps: the change of its energy over that minute, as a
/// share of its energy at the start, and the farthest it stood off the circle.
#[cfg(feature = "dim3")]
fn circle_loop(rate: Real, substeps: usize) -> (Real, Real) {
    let radius = 2.0;
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let centre = world.insert_body(RigidBodyBuilder::fixed());
    let (ball, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, -radius, 0.0))
            .linvel(Vector::new((7.0 * 9.81 * radius).sqrt(), 0.0, 0.0))
            .can_sleep(false),
        ColliderBuilder::ball(0.2).mass(100.0).collision_groups(InteractionGroups::new(
            Group::GROUP_1,
            Group::NONE,
            InteractionTestMode::And,
        )),
    );
    let mut circle = GenericJointBuilder::new(JointAxesMask::LIN_Z)
        .coupled_axes(JointAxesMask::LIN_X | JointAxesMask::LIN_Y)
        .build();
    circle.set_limits(JointAxis::LinX, [radius, radius]);
    world.insert_impulse_joint(centre, ball, circle);
    let energy = |world: &PhysicsWorld| {
        let body = &world.bodies[ball];
        0.5 * 100.0 * body.linvel().length_squared() + 100.0 * 9.81 * (body.translation().y + radius)
    };
    let start = energy(&world);
    let mut off: Real = 0.0;
    for _ in 0..(60.0 * rate) as usize {
        world.step();
        off = off.max((world.bodies[ball].translation().length() - radius).abs());
    }
    ((energy(&world) - start) / start, off)
}

/// A ball held on a vertical 2 m circle loops for a minute within 1% of its energy, at 30 steps
/// a second in 4 substeps, 60 in 4 and 240 in 1, never more than 5 mm off the circle. The
/// chord of the ball's path read off the velocities each substep started with, before its
/// biased pass, lost 1.5% at 30 Hz x 4.
#[test]
#[cfg(feature = "dim3")]
fn a_ball_held_on_a_vertical_circle_loops_for_a_minute_keeping_its_energy() {
    let mut past = Vec::new();
    for (rate, substeps) in [(30.0, 4), (60.0, 4), (240.0, 1)] {
        let (change, off) = circle_loop(rate, substeps);
        if change.abs() > 0.01 || off > 0.005 {
            past.push(alloc::format!(
                "{rate} Hz x {substeps}: {:+.2}%, {off} m off the circle",
                100.0 * change
            ));
        }
    }
    assert!(past.is_empty(), "the loop changed its energy: {past:?}");
}
