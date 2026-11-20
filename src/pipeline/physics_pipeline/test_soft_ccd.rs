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
    /// A force applied for the launch step alone, taking the ball to the launch speed over it.
    Force,
}

/// What the plate the ball is shot at is.
#[derive(Copy, Clone, Debug)]
enum Plate {
    /// A free 20 kg plate.
    Free,
    /// A fixed plate.
    Fixed,
}

/// Where a shot leaves the ball and the plate.
struct Shot {
    /// How far in front of the plate's face the ball ends (negative: it passed through).
    ends: f32,
    /// The ball's velocity along the shot's axis at the end (negative: toward the plate).
    ball_vz: f32,
    /// The plate's velocity along the shot's axis at the end.
    plate_vz: f32,
}

/// A 2 cm ball with soft-CCD `7` m stands `start` metres in front of a 5 cm plate, both with
/// the given restitution, rests for a step, is launched toward it at 400 m/s (6.67 m a step at
/// 60 Hz) and runs ten steps.
fn shot(start: f32, launch: Launch, plate: Plate, restitution: f32) -> Shot {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    let plate_z = 20.0;
    let half_thickness = 0.025;
    let plate_body = match plate {
        Plate::Free => RigidBodyBuilder::dynamic()
            .lock_rotations()
            .can_sleep(false),
        Plate::Fixed => RigidBodyBuilder::fixed(),
    };
    let (plate, _) = world.insert(
        plate_body.translation(Vector::new(0.0, 0.0, plate_z)),
        ColliderBuilder::cuboid(1.0, 1.0, half_thickness)
            .mass(20.0)
            .restitution(restitution),
    );
    let radius = 0.01;
    let (ball, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.0, plate_z + start))
            .soft_ccd_prediction(7.0)
            .can_sleep(false),
        ColliderBuilder::ball(radius)
            .mass(0.01)
            .restitution(restitution),
    );
    world.step();
    let speed = 400.0;
    let rb = &mut world.bodies[ball];
    match launch {
        Launch::Velocity => rb.set_linvel(Vector::new(0.0, 0.0, -speed), true),
        Launch::Impulse => rb.apply_impulse(Vector::new(0.0, 0.0, -speed * rb.mass()), true),
        Launch::Force => {
            let dt = world.integration_parameters.dt;
            rb.add_force(Vector::new(0.0, 0.0, -speed * rb.mass() / dt), true);
        }
    }
    world.step();
    world.bodies[ball].reset_forces(true);
    for _ in 1..10 {
        world.step();
    }
    Shot {
        ends: world.bodies[ball].translation().z - (plate_z + half_thickness + radius),
        ball_vz: world.bodies[ball].linvel().z,
        plate_vz: world.bodies[plate].linvel().z,
    }
}

/// Shoots the ball from each of `starts` at a free plate with no restitution, and checks it
/// stops at the plate's face and hands the plate its momentum: 0.01 kg at 400 m/s takes the
/// 20 kg plate and the ball with it to 0.2 m/s along the shot.
fn meets_the_plate(launch: Launch, starts: &[f32]) {
    for &start in starts {
        let Shot {
            ends,
            ball_vz,
            plate_vz,
        } = shot(start, launch, Plate::Free, 0.0);
        assert!(
            ends > -0.05 && ends < 0.05,
            "launched {start} m in front of the plate, the ball ends {ends} m in front of its face"
        );
        assert!(
            (plate_vz + 0.2).abs() < 0.01 && (ball_vz - plate_vz).abs() < 0.01,
            "launched {start} m in front of the plate, the plate moves at {plate_vz} m/s and the \
             ball at {ball_vz} m/s"
        );
    }
}

#[test]
fn a_soft_ccd_body_launched_at_rest_meets_a_plate_inside_its_first_step() {
    meets_the_plate(Launch::Velocity, &[2.0, 4.0, 6.0, 8.0, 10.0]);
}

#[test]
fn a_soft_ccd_body_launched_by_an_impulse_meets_a_plate_inside_its_first_step() {
    meets_the_plate(Launch::Impulse, &[2.0, 4.0, 6.0]);
}

#[test]
fn a_soft_ccd_body_launched_by_a_force_meets_a_plate_inside_its_first_step() {
    meets_the_plate(Launch::Force, &[2.0, 4.0, 6.0]);
}

/// With restitution 0.3 the ball rebounds from a free and from a fixed plate at 0.3 of the
/// 400 m/s it closed at, whether a velocity or a force launched it in the step it reaches the
/// plate.
#[test]
fn a_soft_ccd_body_launched_by_a_force_rebounds_like_one_launched_by_a_velocity() {
    for plate in [Plate::Free, Plate::Fixed] {
        for start in [2.0, 4.0, 6.0] {
            let by_velocity = shot(start, Launch::Velocity, plate, 0.3);
            let by_force = shot(start, Launch::Force, plate, 0.3);
            assert!(
                by_velocity.ball_vz > 115.0 && by_velocity.ball_vz < 121.0,
                "launched by a velocity {start} m in front of a {plate:?} plate, the ball \
                 rebounds at {} m/s",
                by_velocity.ball_vz
            );
            assert!(
                (by_force.ball_vz - by_velocity.ball_vz).abs() < 0.5,
                "launched by a force {start} m in front of a {plate:?} plate, the ball rebounds \
                 at {} m/s; launched by a velocity, at {} m/s",
                by_force.ball_vz,
                by_velocity.ball_vz
            );
            assert!(
                by_force.ends > 0.0 && by_force.ends > by_velocity.ends - 7.0,
                "launched by a force {start} m in front of a {plate:?} plate, the ball ends {} m \
                 in front of its face; launched by a velocity, {} m",
                by_force.ends,
                by_velocity.ends
            );
        }
    }
}
