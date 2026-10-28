//! The continuous pass sweeps a fast dynamic body against fixed geometry, and a joint whose
//! contacts are switched off (`contacts_enabled(false)`) switches the sweep off between its two
//! bodies as it does their contacts: a body sliding fast along a joint through the collider of
//! the fixed body it is jointed to passes through it, and is stopped there while their contacts
//! stay on.

use rapier3d::prelude::*;

/// A fixed body holding a 1 m box at the origin, and a ball 3 m to its left sliding right at
/// `speed` along a prismatic joint to it; returns the ball's x after `steps` steps at 60 Hz.
fn slide_through_the_box(contacts: bool, speed: Real, steps: usize) -> Real {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let frame = world.bodies.insert(RigidBodyBuilder::fixed());
    world
        .colliders
        .insert_with_parent(ColliderBuilder::cuboid(0.5, 0.5, 0.5), frame, &mut world.bodies);
    let ball = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(-3.0, 0.0, 0.0))
            .linvel(Vector::new(speed, 0.0, 0.0))
            .can_sleep(false),
    );
    world
        .colliders
        .insert_with_parent(ColliderBuilder::ball(0.2), ball, &mut world.bodies);
    let joint = PrismaticJointBuilder::new(Vector::X)
        .local_anchor1(Vector::new(-3.0, 0.0, 0.0))
        .contacts_enabled(contacts);
    world.impulse_joints.insert(frame, ball, joint, true);
    for _ in 0..steps {
        world.step();
    }
    world.bodies[ball].translation().x
}

#[test]
fn a_fast_body_passes_through_the_fixed_body_its_joint_switches_contacts_off_with() {
    // 30 m/s for 12 steps of 1/60 s: 6 m, from x = -3 to x = 3, through the box.
    let x = slide_through_the_box(false, 30.0, 12);
    assert!((x - 3.0).abs() < 1.0e-3, "the ball slid to x = {x}; with contacts off it passes the box to x = 3");
}

#[test]
fn a_fast_body_is_stopped_at_the_fixed_body_its_joint_keeps_contacts_with() {
    let x = slide_through_the_box(true, 30.0, 12);
    assert!(x < -0.69, "the ball slid to x = {x}; with contacts on it stops at the box's face");
}
