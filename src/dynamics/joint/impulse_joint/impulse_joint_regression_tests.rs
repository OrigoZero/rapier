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

/// The chain of [`planar_hinge_chain`] with the torso's turning locked alone, hinged about Z,
/// free or sprung, falls to rest at 120 steps a second in 1 substep at the origin and 704 to
/// 720 m out along X: no body climbs above the 1.30 m the torso starts at and every hinge holds
/// its anchors within 2 cm. Solved exactly as a tree, the hinges on its free thigh and calf came
/// apart by 6 to 24 cm 704 to 720 m out, a row the others nearly held taking an impulse of up to
/// 23000 N s on its 5 kg bodies.
#[test]
#[cfg(feature = "dim3")]
fn a_hinge_chain_locked_at_the_torso_holds_together_far_from_the_origin_in_one_substep() {
    let mut failed = Vec::new();
    for x in [0.0, 704.0, 712.0, 720.0] {
        for spring in [false, true] {
            let (peak, gap) = planar_hinge_chain(x, Vector::Z, false, spring, 1.0 / 120.0, 1);
            if peak > 1.31 || gap > 0.02 {
                failed.push(alloc::format!(
                    "at x {x}, sprung {spring}: peak {peak} m, gap {gap} m"
                ));
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

/// A free-swinging chain of five rods on ball joints, released level, keeps 65% of the energy
/// it can give up over 10 s at 120 substeps a second (30 Hz x 4, 60 x 2, 120 x 1), 95% of it at
/// 240 (60 x 4, 120 x 2, 240 x 1) and 99% at 960 (240 x 4). Its whipping end makes its motion
/// chaotic, so at 120 substeps a second where it ends depends on the last bits of where it
/// starts: the engine reads 13% and 29% lost from one release at 30 Hz x 4 and 60 x 2. Solved by
/// Gauss-Seidel, one sweep a pass, the chain's joints left most of the tension each substep
/// needs unsolved, and the stabilization pass took the velocity it would have turned out of the
/// links' motion: the chain lost 62% at 120 substeps a second and 63% at 240.
#[test]
#[cfg(feature = "dim3")]
fn a_chain_of_five_rods_on_ball_joints_keeps_most_of_its_energy_at_every_substep_length() {
    let mut past = Vec::new();
    let mut read = Vec::new();
    for ((rate, substeps), bound) in [
        ((30.0, 4), 0.35),
        ((60.0, 2), 0.35),
        ((120.0, 1), 0.35),
        ((60.0, 4), 0.05),
        ((120.0, 2), 0.05),
        ((240.0, 1), 0.05),
        ((240.0, 4), 0.01),
    ] {
        let change = swing(5, rate, substeps, 0.0, || {
            SphericalJointBuilder::new().build().into()
        });
        read.push(alloc::format!("{rate} Hz x {substeps}: {:+.1}%", 100.0 * change));
        if change.abs() > bound {
            past.push(alloc::format!("{rate} Hz x {substeps}: {:+.1}%", 100.0 * change));
        }
    }
    std::println!("five-rod chain, released level, 10 s: {read:?}");
    assert!(past.is_empty(), "the chain's energy changed: {past:?}");
}

/// A chain of twenty rods on ball joints hanging straight down from a fixed anchor holds its
/// length to within 1 mm over 2 s at 60 Hz x 2: the joints of a chain are solved together, so
/// each one carries the weight of the links below it in the same substep. Solved by
/// Gauss-Seidel, one sweep a pass, an impulse travelled one joint along the chain per sweep.
#[test]
#[cfg(feature = "dim3")]
fn a_hanging_chain_of_twenty_rods_holds_its_length() {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / 60.0;
    world.integration_parameters.num_solver_iterations = 2;
    let top = Vector::new(0.0, 20.0, 0.0);
    let anchor = world.insert_body(RigidBodyBuilder::fixed().translation(top));
    let down = Rotation::from_rotation_z(-(core::f64::consts::FRAC_PI_2 as Real)).to_scaled_axis();
    let mut above = anchor;
    let mut links = Vec::new();
    for i in 0..20 {
        let (link, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(top - Vector::Y * (0.25 + i as Real * 0.5))
                .rotation(down),
            untouching_rod(),
        );
        let mut data: GenericJoint = SphericalJointBuilder::new().build().into();
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
    let mut stretch: Real = 0.0;
    for _ in 0..120 {
        world.step();
        let tip = world.bodies[links[19]].position() * Vector::new(0.25, 0.0, 0.0);
        stretch = stretch.max((top - tip).length() - 10.0);
    }
    std::println!("twenty-rod hanging chain: stretched by {stretch} m at most");
    assert!(stretch <= 0.001, "the chain stretched by {stretch} m");
}

/// A door held to its frame by a hinge and by a catch fixed to the frame, both joints of the
/// world, shoved along its normal at 6000 N for a quarter second at 60 Hz x 2, puts at least a
/// third of the shove through the catch: enough for a catch that breaks at 2000 N to break.
/// The second joint to the world closes a loop through it and is solved with Gauss-Seidel
/// beside the first. Solved exactly with the first, the two joints' rows were singular and the
/// joint eliminated later, the catch, took 0.0009 N s of the 1500.
#[test]
#[cfg(feature = "dim3")]
fn a_door_held_by_a_hinge_and_a_catch_puts_a_shove_through_the_catch() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = 1.0 / 60.0;
    world.integration_parameters.num_solver_iterations = 2;
    let frame = world.insert_body(RigidBodyBuilder::fixed());
    let (door, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(0.65, 0.0, 0.0)),
        ColliderBuilder::cuboid(0.5, 1.0, 0.05).mass(20.0).collision_groups(InteractionGroups::new(
            Group::GROUP_1,
            Group::NONE,
            InteractionTestMode::And,
        )),
    );
    let hinge = world.insert_impulse_joint(
        door,
        frame,
        RevoluteJointBuilder::new(Vector::Y)
            .local_anchor1(Vector::new(-0.55, 0.0, 0.0))
            .local_anchor2(Vector::new(0.1, 0.0, 0.0)),
    );
    let catch = world.insert_impulse_joint(
        door,
        frame,
        FixedJointBuilder::new()
            .local_anchor1(Vector::new(0.5, 0.0, 0.0))
            .local_anchor2(Vector::new(1.15, 0.0, 0.0)),
    );
    let mut taken = [0.0 as Real; 2];
    for _ in 0..15 {
        world.bodies[door].reset_forces(false);
        world.bodies[door].add_force(Vector::new(0.0, 0.0, -6000.0), true);
        world.step();
        for (k, joint) in [hinge, catch].into_iter().enumerate() {
            taken[k] += world.impulse_joints.get(joint).unwrap().reaction.linear.z.abs();
        }
    }
    let shove = 6000.0 * 15.0 / 60.0;
    std::println!("door shove {shove} N s: hinge {} N s, catch {} N s", taken[0], taken[1]);
    assert!(
        taken[1] >= shove / 3.0,
        "the shove of {shove} N s was not shared: hinge {} N s, catch {} N s",
        taken[0],
        taken[1]
    );
}

/// The chain of [`swing`], five rods on ball joints laid out level from a fixed anchor at
/// (0, 10, 0), released turning at `spin` rad/s about the vertical through the anchor, swung for
/// 10 s at `rate` steps a second in `substeps` substeps: the change of its energy as a share of
/// what it stands above hanging straight down at rest, and the farthest any rod's centre of mass
/// went from the vertical plane it was released in.
#[cfg(feature = "dim3")]
fn spun_chain(rate: Real, substeps: usize, spin: Real) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let top = Vector::new(0.0, 10.0, 0.0);
    let anchor = world.insert_body(RigidBodyBuilder::fixed().translation(top));
    let turning = Vector::new(0.0, spin, 0.0);
    let mut above = anchor;
    let mut links = Vec::new();
    for i in 0..5 {
        let at = top + Vector::X * (0.25 + i as Real * 0.5);
        let (link, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(at)
                .linvel(turning.cross(at - top))
                .angvel(turning)
                .allow_fast_rotation(true),
            untouching_rod(),
        );
        let mut data: GenericJoint = SphericalJointBuilder::new().build().into();
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
    let hanging: Real = (0..5)
        .map(|i| -gravity.y * (top.y - 0.25 - i as Real * 0.5))
        .sum();
    let mut out_of_plane: Real = 0.0;
    for _ in 0..(10.0 * rate) as usize {
        world.step();
        for link in &links {
            out_of_plane = out_of_plane.max(world.bodies[*link].translation().z.abs());
        }
    }
    let change = (links_energy(&world.bodies, &links, gravity) - start) / (start - hanging);
    (change, out_of_plane)
}

/// The chain of [`spun_chain`], released turning at 1 rad/s about the vertical, swings out of
/// its plane and keeps the same share of its energy over 10 s however many substeps a step is
/// split into: within half a point of the steps of one substep of the same length, and within
/// 10% of what it can give up at substeps of a 120th of a second (30 Hz x 4, 60 x 2, 120 x 1)
/// and 1% at a 240th (60 x 4, 120 x 2, 240 x 1) and a 480th (120 x 4, 240 x 2, 480 x 1). A
/// substep after the first solved the rods' joints with the inertia the step began with: the
/// chain lost 36% at 60 Hz x 2 against 6% at 120 x 1, and 23% at 240 x 2 against 0.2% at 480 x 1.
#[test]
#[cfg(feature = "dim3")]
fn a_chain_swinging_out_of_its_plane_keeps_its_energy_however_its_steps_are_split() {
    let mut past = Vec::new();
    let mut read = Vec::new();
    for (single, split, bound) in [
        (120.0, [(30.0, 4), (60.0, 2)], 0.10),
        (240.0, [(60.0, 4), (120.0, 2)], 0.01),
        (480.0, [(120.0, 4), (240.0, 2)], 0.01),
    ] {
        let (whole, reach) = spun_chain(single, 1, 1.0);
        assert!(reach > 0.5, "the chain stayed within {reach} m of its plane");
        read.push(alloc::format!("{single} Hz x 1: {:+.2}%", 100.0 * whole));
        if whole.abs() > bound {
            past.push(alloc::format!("{single} Hz x 1: {:+.2}%", 100.0 * whole));
        }
        for (rate, substeps) in split {
            let (change, _) = spun_chain(rate, substeps, 1.0);
            read.push(alloc::format!("{rate} Hz x {substeps}: {:+.2}%", 100.0 * change));
            if change.abs() > bound || (change - whole).abs() > 0.005 {
                past.push(alloc::format!(
                    "{rate} Hz x {substeps}: {:+.2}% against {:+.2}% at {single} Hz x 1",
                    100.0 * change,
                    100.0 * whole
                ));
            }
        }
    }
    std::println!("five-rod chain turning out of its plane, 10 s: {read:?}");
    assert!(past.is_empty(), "the chain's energy changed: {past:?}");
}

/// An 80 kg ball held, at a point 1 m above its centre, to a line sloping 2 degrees down
/// toward an end stop (a joint locking the point's motion across the line, with a limit along
/// it at the end), hung under the stop with the point at rest there and the ball swinging
/// about it at 0.05 rad/s, too gently for the swing to pull the point back up the slope, for
/// 10 s at `rate` steps a second in `substeps` substeps: the most the point stood past the
/// stop or short of it, and the fastest it moved along the line, as any step ended. With
/// `link`, a 1 kg weight fixed 0.5 m under the ball by a multibody joint makes the ball the root
/// link of a multibody, so the joint holding it is solved as a joint on a multibody link.
fn held_at_a_stop(rate: Real, substeps: usize, link: bool) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let end = Vector::new(0.0, 10.0, 0.0);
    let slope = Real::to_radians(-2.0);
    let outward = Vector::new(slope.cos(), slope.sin(), 0.0);
    let path = world.insert_body(RigidBodyBuilder::fixed());
    let (ball, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(end - Vector::Y)
            .linvel(Vector::new(0.05, 0.0, 0.0))
            .angvel(Vector::new(0.0, 0.0, 0.05))
            .can_sleep(false),
        ColliderBuilder::ball(0.2)
            .mass(80.0)
            .collision_groups(InteractionGroups::new(
                Group::GROUP_1,
                Group::NONE,
                InteractionTestMode::And,
            )),
    );
    if link {
        let (weight, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(end - Vector::Y * 1.5)
                .can_sleep(false),
            ColliderBuilder::ball(0.05)
                .mass(1.0)
                .collision_groups(InteractionGroups::new(
                    Group::GROUP_1,
                    Group::NONE,
                    InteractionTestMode::And,
                )),
        );
        let fixed = FixedJointBuilder::new().local_anchor1(-Vector::Y * 0.5);
        world.insert_multibody_joint(ball, weight, fixed).unwrap();
    }
    let frame = Pose::from_parts(end, Rotation::from_rotation_z(slope));
    let mut data = GenericJointBuilder::new(JointAxesMask::LIN_Y | JointAxesMask::LIN_Z)
        .local_frame1(frame)
        .local_frame2(Pose::from_translation(Vector::Y))
        .build();
    data.set_limits(JointAxis::LinX, [-Real::MAX, 0.0]);
    world.insert_impulse_joint(path, ball, data);
    let (mut off, mut moving) = (0.0 as Real, 0.0 as Real);
    for _ in 0..(10.0 * rate) as usize {
        world.step();
        let body = &world.bodies[ball];
        let held = body.position().transform_point(Vector::Y);
        off = off.max((held - end).dot(outward).abs());
        moving = moving.max(body.velocity_at_point(held).dot(outward).abs());
    }
    (off, moving)
}

/// A point resting against a joint's linear limit stays there at every substep length, on a
/// lone body and on a multibody link: every step ends with it within 0.01 mm of the bound and
/// moving along the line at under 1 mm/s.
/// The limit's row took hold only in a substep that began with the point at or past the bound,
/// and its bias leaves a point resting on the bound a rounding short of it, so in every few
/// substeps the row held nothing and the rows carrying the ball's weight across the sloping
/// line turned it and drove the point through the bound for a whole substep: steps ended with
/// the point up to 0.7 mm past the stop and moving into it at 0.15 m/s.
#[test]
#[cfg(feature = "dim3")]
fn a_point_resting_against_a_linear_limit_stays_at_it_at_every_substep_length() {
    let mut failed = Vec::new();
    for link in [false, true] {
        for (rate, substeps) in RATES {
            let (off, moving) = held_at_a_stop(rate, substeps, link);
            if off > 1.0e-5 || moving > 1.0e-3 {
                failed.push(alloc::format!(
                    "{} at {rate} Hz x {substeps}: {:.4} mm off the stop, moving at {moving:.4} m/s",
                    if link { "a multibody link" } else { "a body" },
                    off * 1000.0
                ));
            }
        }
    }
    assert!(failed.is_empty(), "the point left its limit: {failed:?}");
}
