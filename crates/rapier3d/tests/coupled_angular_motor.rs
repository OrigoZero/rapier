//! A ball joint whose two swing angles are coupled into one cone keeps the motors on them.

use rapier3d::prelude::*;

#[test]
fn a_cone_limited_ball_joint_keeps_its_swing_motors() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let a = world.bodies.insert(RigidBodyBuilder::fixed());
    let b = world.bodies.insert(RigidBodyBuilder::dynamic().can_sleep(false));
    world.colliders.insert_with_parent(ColliderBuilder::cuboid(0.5, 0.5, 0.1), b, &mut world.bodies);
    let mut joint = SphericalJointBuilder::new().build();
    let data: &mut GenericJoint = &mut joint.data;
    data.coupled_axes |= JointAxesMask::ANG_Y | JointAxesMask::ANG_Z;
    data.set_limits(JointAxis::AngY, [-1.0, 1.0]);
    data.set_limits(JointAxis::AngZ, [-1.0, 1.0]);
    for axis in [JointAxis::AngX, JointAxis::AngY, JointAxis::AngZ] {
        data.set_motor(axis, 0.0, 0.0, 0.0, 0.0);
    }
    world.impulse_joints.insert(a, b, joint, true);
    world.bodies[b].set_angvel(Vector::new(0.0, 2.0, 2.0), true);
    for _ in 0..3 {
        world.step();
    }
    let w = world.bodies[b].angvel();
    assert!(w.length() < 0.05, "the swing went on at {w:?} against the motors holding it");
}
