//! A coupled linear limit bounds the length of the anchors' offset along its coupled
//! axes from below as well as from above: a positive `min` keeps the anchors at least
//! that far apart, and `min == max` holds them at exactly that distance, from either
//! side. A coupled pair in a plane whose third axis is locked holds a point on a
//! circle; a rope (`min` zero) stays a maximum distance only.

use rapier3d::prelude::*;

const GRAVITY: Real = 9.81;
const RADIUS: Real = 2.0;

/// A fixed anchor and a ball of `mass` starting `RADIUS` below it in the XY plane,
/// held by `joint`. Returns the world, the joint and the ball.
fn ball_held_by(
    joint: GenericJoint,
    mass: Real,
    start: Vector,
    linvel: Vector,
) -> (PhysicsWorld, ImpulseJointHandle, RigidBodyHandle) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::new(0.0, -GRAVITY, 0.0);
    let anchor = world.bodies.insert(RigidBodyBuilder::fixed());
    let body = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(start)
            .linvel(linvel)
            .can_sleep(false),
    );
    let collider = ColliderBuilder::ball(0.1).mass(mass);
    world
        .colliders
        .insert_with_parent(collider, body, &mut world.bodies);
    let handle = world.impulse_joints.insert(anchor, body, joint, true);
    (world, handle, body)
}

/// A point held on the circle of `RADIUS` about the anchor in the anchor's XY plane:
/// linear Z locked, linear X and Y coupled and limited to `[min, max]`.
fn circle_joint(min: Real, max: Real) -> GenericJoint {
    let mut joint = GenericJointBuilder::new(JointAxesMask::LIN_Z)
        .coupled_axes(JointAxesMask::LIN_X | JointAxesMask::LIN_Y)
        .build();
    joint.set_limits(JointAxis::LinX, [min, max]);
    joint
}

fn distance(world: &PhysicsWorld, body: RigidBodyHandle) -> Real {
    world.bodies[body].translation().length()
}

#[test]
fn a_point_held_on_a_circle_goes_round_a_vertical_loop_at_its_radius() {
    // Released at the bottom at the speed that carries it over the top with
    // centripetal force to spare: v_top^2 = v^2 - 4 g r > g r.
    let mass = 2.0;
    let speed = (6.0 * GRAVITY * RADIUS).sqrt();
    let (mut world, _joint, body) = ball_held_by(
        circle_joint(RADIUS, RADIUS),
        mass,
        Vector::new(0.0, -RADIUS, 0.0),
        Vector::new(speed, 0.0, 0.0),
    );
    let mut worst: Real = 0.0;
    let mut highest = Real::MIN;
    for _ in 0..240 {
        world.step();
        worst = worst.max((distance(&world, body) - RADIUS).abs());
        highest = highest.max(world.bodies[body].translation().y);
        assert!(world.bodies[body].translation().z.abs() < 1e-3);
    }
    assert!(
        highest > RADIUS - 0.01,
        "the ball reaches {highest} m, the loop's top is {RADIUS}"
    );
    assert!(
        worst < 0.01,
        "the ball stands {worst} m off the circle's radius"
    );
}

#[test]
fn a_point_held_on_a_circle_too_slow_for_its_top_stays_on_the_circle() {
    // Too slow to pass the top: a rope goes slack and the ball falls inside the
    // circle; held at the radius from both sides, it rocks back on the circle.
    let mass = 1.0;
    let speed = (3.0 * GRAVITY * RADIUS).sqrt();
    let (mut world, _joint, body) = ball_held_by(
        circle_joint(RADIUS, RADIUS),
        mass,
        Vector::new(0.0, -RADIUS, 0.0),
        Vector::new(speed, 0.0, 0.0),
    );
    let mut worst: Real = 0.0;
    for _ in 0..300 {
        world.step();
        worst = worst.max((distance(&world, body) - RADIUS).abs());
    }
    assert!(
        worst < 0.01,
        "the ball stands {worst} m off the circle's radius"
    );
}

#[test]
fn a_positive_lower_bound_keeps_the_anchors_apart() {
    // A ball pushed toward the anchor stops at the lower bound, a ring of radius 1
    // to 2 about it.
    let (mut world, _joint, body) = ball_held_by(
        circle_joint(1.0, RADIUS),
        1.0,
        Vector::new(1.5, 0.0, 0.0),
        Vector::new(-5.0, 0.0, 0.0),
    );
    world.gravity = Vector::ZERO;
    let mut nearest = Real::MAX;
    for _ in 0..60 {
        world.step();
        nearest = nearest.min(distance(&world, body));
    }
    assert!(
        nearest > 1.0 - 0.01,
        "the ball comes {nearest} m from the anchor, the lower bound is 1"
    );
}

#[test]
fn a_rope_lets_its_anchors_close() {
    // A rope's lower bound is zero: the ball passes through its anchor freely.
    let rope: GenericJoint = RopeJointBuilder::new(RADIUS).build().into();
    let (mut world, _joint, body) = ball_held_by(
        rope,
        1.0,
        Vector::new(1.5, 0.0, 0.0),
        Vector::new(-5.0, 0.0, 0.0),
    );
    world.gravity = Vector::ZERO;
    let mut nearest = Real::MAX;
    for _ in 0..30 {
        world.step();
        nearest = nearest.min(distance(&world, body));
    }
    assert!(
        nearest < 0.1,
        "the ball comes no nearer than {nearest} m of the anchor on a slack rope"
    );
}
