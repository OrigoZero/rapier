//! Soft-CCD regression tests: a body launched by a user write is paired, in the step it is
//! launched, with what it reaches within that step.

use crate::dynamics::RigidBodyBuilder;
use crate::geometry::ColliderBuilder;
use crate::math::Vector;
use crate::pipeline::PhysicsWorld;

/// How a resting soft-CCD ball is set moving between two steps.
#[derive(Copy, Clone)]
enum Launch {
    Velocity,
    Impulse,
}

/// A 2 cm ball with soft-CCD `7` m stands `start` metres in front of a free 5 cm plate, rests
/// for a step, is launched toward it at 400 m/s (6.67 m a step at 60 Hz) and runs ten steps.
/// Returns how far in front of the plate's face the ball ends (negative: it passed through).
fn shot(start: f32, launch: Launch) -> f32 {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let plate_z = 20.0;
    let half_thickness = 0.025;
    let _ = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.0, plate_z))
            .lock_rotations()
            .can_sleep(false),
        ColliderBuilder::cuboid(1.0, 1.0, half_thickness).mass(20.0),
    );
    let radius = 0.01;
    let (ball, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.0, plate_z + start))
            .soft_ccd_prediction(7.0)
            .can_sleep(false),
        ColliderBuilder::ball(radius).mass(0.01),
    );
    world.step();
    let speed = 400.0;
    let rb = &mut world.bodies[ball];
    match launch {
        Launch::Velocity => rb.set_linvel(Vector::new(0.0, 0.0, -speed), true),
        Launch::Impulse => rb.apply_impulse(Vector::new(0.0, 0.0, -speed * rb.mass()), true),
    }
    for _ in 0..10 {
        world.step();
    }
    world.bodies[ball].translation().z - (plate_z + half_thickness + radius)
}

#[test]
fn a_soft_ccd_body_launched_at_rest_meets_a_plate_inside_its_first_step() {
    for start in [2.0, 4.0, 6.0, 8.0, 10.0] {
        let ends = shot(start, Launch::Velocity);
        assert!(
            ends > -0.05,
            "launched {start} m in front of the plate, the ball ends {ends} m past its face"
        );
    }
}

#[test]
fn a_soft_ccd_body_launched_by_an_impulse_meets_a_plate_inside_its_first_step() {
    for start in [2.0, 4.0, 6.0] {
        let ends = shot(start, Launch::Impulse);
        assert!(
            ends > -0.05,
            "launched {start} m in front of the plate, the ball ends {ends} m past its face"
        );
    }
}
