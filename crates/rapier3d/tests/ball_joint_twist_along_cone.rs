//! A ball joint with a swing cone and a twist range keeps its twist inside the range while
//! the bone it holds is driven around the rim of its cone.

use rapier3d::prelude::*;

/// The twist between the joint's two frames, in degrees.
fn twist_of(world: &PhysicsWorld, arm: RigidBodyHandle, hook: RigidBodyHandle, joint: &GenericJoint) -> f32 {
    let frame1 = world.bodies[arm].position().rotation * joint.local_frame1.rotation;
    let frame2 = world.bodies[hook].position().rotation * joint.local_frame2.rotation;
    let mut relative = frame1.inverse() * frame2;
    if relative.w < 0.0 {
        relative = -relative;
    }
    (2.0 * relative.x.atan2(relative.w)).to_degrees()
}

#[test]
fn a_ball_joint_driven_around_its_cone_keeps_its_twist_range() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = 1.0 / 30.0;
    world.integration_parameters.num_solver_iterations = 4;
    let hook = world.bodies.insert(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 5.0, 0.0)));
    let arm = world
        .bodies
        .insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 4.6, 0.0)).can_sleep(false));
    world.colliders.insert_with_parent(ColliderBuilder::capsule_y(0.15, 0.03).mass(1.0), arm, &mut world.bodies);
    // The joint's X axis runs along the arm: a 40 degree cone, a twist of 15 either way.
    let along_arm = Rotation::from_rotation_arc(Vector::X, Vector::Y);
    let mut joint = GenericJointBuilder::new(JointAxesMask::LIN_AXES)
        .local_frame1(Pose::from_parts(Vector::new(0.0, 0.25, 0.0), along_arm))
        .local_frame2(Pose::from_parts(Vector::new(0.0, -0.15, 0.0), along_arm))
        .contacts_enabled(false)
        .build();
    let swing = 40.0f32.to_radians();
    let twist = 15.0f32.to_radians();
    joint.coupled_axes |= JointAxesMask::ANG_Y | JointAxesMask::ANG_Z;
    joint.set_limits(JointAxis::AngY, [-swing, swing]);
    joint.set_limits(JointAxis::AngZ, [-swing, swing]);
    joint.set_limits(JointAxis::AngX, [-twist, twist]);
    world.impulse_joints.insert(arm, hook, joint, true);

    // A torque of 3 N·m across the arm, turning half a radian each step: the arm sweeps the
    // rim of its cone, which turns it about its own length as it goes.
    let mut widest: f32 = 0.0;
    for step in 0..120 {
        let heading = step as f32 * 0.5;
        world.bodies[arm].reset_torques(false);
        world.bodies[arm].add_torque(Vector::new(heading.cos() * 3.0, 0.0, heading.sin() * 3.0), true);
        world.step();
        widest = widest.max(twist_of(&world, arm, hook, &joint).abs());
    }
    assert!(widest < 35.0, "the twist reached {widest} degrees against its 15 degree range");
}
