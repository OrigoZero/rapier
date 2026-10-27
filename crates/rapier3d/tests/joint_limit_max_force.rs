//! A joint limit given a `max_force` holds up to that force and no further: a load
//! below it is held at the limit, a load above it pushes past at the excess while the
//! limit keeps pushing back with exactly its `max_force`.

use rapier3d::prelude::*;

const GRAVITY: Real = 10.0;
const MASS: Real = 10.0;

/// A fixed anchor and a dynamic body of `MASS` hanging `length` below it on a rope
/// whose limit applies at most `max_force`. Returns the world, the joint and the body.
fn hung_on_a_rope(
    length: Real,
    max_force: Real,
) -> (PhysicsWorld, ImpulseJointHandle, RigidBodyHandle) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -GRAVITY, 0.0);
    let anchor = world.bodies.insert(RigidBodyBuilder::fixed());
    let body = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, -length, 0.0))
            .can_sleep(false),
    );
    let collider = ColliderBuilder::ball(0.1).mass(MASS);
    world
        .colliders
        .insert_with_parent(collider, body, &mut world.bodies);
    let mut rope = RopeJointBuilder::new(length).build();
    rope.data.set_limit_max_force(JointAxis::LinX, max_force);
    let joint = world.impulse_joints.insert(anchor, body, rope, true);
    (world, joint, body)
}

fn depth(world: &PhysicsWorld, body: RigidBodyHandle) -> Real {
    -world.bodies[body].translation().y
}

/// The force the rope's limit applied over the last substep.
fn rope_force(world: &PhysicsWorld, joint: ImpulseJointHandle) -> Real {
    let substep_dt = world.integration_parameters.dt
        / world.integration_parameters.num_solver_iterations as Real;
    world.impulse_joints.get(joint).unwrap().data.limits[JointAxis::LinX as usize].impulse
        / substep_dt
}

#[test]
fn a_rope_whose_load_is_below_its_limit_force_holds() {
    // 100 N of weight on a rope that holds up to 150 N.
    let (mut world, joint, body) = hung_on_a_rope(2.0, 150.0);
    for _ in 0..120 {
        world.step();
    }
    let at = depth(&world, body);
    assert!(
        (at - 2.0).abs() < 1e-3,
        "a 100 N load on a rope holding 150 N hangs {at} m down, the rope is 2 m"
    );
    let force = rope_force(&world, joint);
    assert!(
        (force - MASS * GRAVITY).abs() < 1.0,
        "the rope carries {force} N of a 100 N load"
    );
}

#[test]
fn a_rope_whose_load_exceeds_its_limit_force_gives_way_at_that_force() {
    // 100 N of weight on a rope that holds up to 60 N: the body falls at (100 - 60) / 10 m/s².
    let (mut world, joint, body) = hung_on_a_rope(2.0, 60.0);
    let steps = 60;
    for _ in 0..steps {
        world.step();
    }
    let t = steps as Real * world.integration_parameters.dt;
    let expected = 2.0 + 0.5 * (MASS * GRAVITY - 60.0) / MASS * t * t;
    let at = depth(&world, body);
    assert!(
        (at - expected).abs() < 0.05,
        "the body hangs {at} m down after {t} s, giving way at the excess puts it at {expected}"
    );
    let force = rope_force(&world, joint);
    assert!(
        (force - 60.0).abs() < 0.5,
        "the rope applies {force} N while it gives way, its limit force is 60"
    );
}

#[test]
fn a_limit_left_without_a_force_bound_stays_a_hard_stop() {
    let (mut world, _joint, body) = hung_on_a_rope(2.0, Real::MAX);
    for _ in 0..120 {
        world.step();
    }
    let at = depth(&world, body);
    assert!(
        (at - 2.0).abs() < 1e-3,
        "a rope with no limit force hangs its body {at} m down, the rope is 2 m"
    );
}

#[test]
fn a_prismatic_stop_given_a_limit_force_is_pushed_past_at_the_excess() {
    for (push, held) in [(40.0, true), (200.0, false)] {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::ZERO;
        let frame = world.bodies.insert(RigidBodyBuilder::fixed());
        // The slider starts at rest against its stop, so the stop holds the push alone.
        let slider = world.bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.5, 0.0, 0.0))
                .can_sleep(false),
        );
        world.colliders.insert_with_parent(
            ColliderBuilder::cuboid(0.1, 0.1, 0.1).mass(MASS),
            slider,
            &mut world.bodies,
        );
        let mut joint = PrismaticJointBuilder::new(Vector::X)
            .limits([-0.5, 0.5])
            .build();
        joint.data.set_limit_max_force(JointAxis::LinX, 100.0);
        world.impulse_joints.insert(frame, slider, joint, true);
        for _ in 0..60 {
            world.bodies[slider].reset_forces(false);
            world.bodies[slider].add_force(Vector::new(push, 0.0, 0.0), true);
            world.step();
        }
        let x = world.bodies[slider].translation().x.abs();
        if held {
            assert!(
                (x - 0.5).abs() < 1e-2,
                "a {push} N push against a stop holding 100 N left the slider at {x}, the stop is at 0.5"
            );
        } else {
            // Past the stop the slider gains (200 - 100) / 10 m/s².
            assert!(
                x > 1.0,
                "a {push} N push against a stop holding 100 N left the slider at {x}, short of pushing past it"
            );
        }
    }
}
