//! The continuous pass sweeps a fast dynamic body against the collision meshes of soft bodies
//! where the solve left their particles: a fast body pushing a soft body ahead of it moves as far
//! through every step as its contacts allow, the same with the continuous pass on as off, and a
//! fast body the contacts did not meet is still stopped at the soft surface in its path.

use rapier3d::prelude::*;

/// A ground, a 0.6 m jelly cube (3x3x3 corotational cells) resting on it at x = 1 (moved 30 m
/// aside when `jelly_in_path` is false), and a 10 kg box sliding at 15 m/s along +x toward it,
/// its rotation and its y/z translation locked, stepped for one second at `hz`. Returns the box's
/// travel over each step.
fn push_a_jelly(hz: Real, ccd: bool, jelly_in_path: bool) -> Vec<Real> {
    let mut world = PhysicsWorld::new();
    world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(50.0, 0.5, 50.0).translation(Vector::new(0.0, -0.5, 0.0)),
    );
    world.integration_parameters.dt = 1.0 / hz;
    if !ccd {
        world.integration_parameters.max_ccd_substeps = 0;
    }
    let z = if jelly_in_path { 0.0 } else { 30.0 };
    world.insert_soft_body(
        SoftBodyBuilder::cuboid(Vector::new(1.0, 0.31, z), Vector::splat(0.6), 3, 3, 3)
            .cell_model(SoftBodyCellModel::Corotational)
            .material(SoftBodyMaterial {
                young_modulus: 5.0e4,
                poisson_ratio: 0.3,
                elastic_damping_ratio: 1.0,
                ..Default::default()
            })
            .particle_mass(0.05)
            .particle_radius(0.02)
            .surface_collider(ColliderBuilder::ball(0.02)),
    );
    let (pusher, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.51, 0.0))
            .linvel(Vector::new(15.0, 0.0, 0.0))
            .locked_axes(
                LockedAxes::ROTATION_LOCKED
                    | LockedAxes::TRANSLATION_LOCKED_Y
                    | LockedAxes::TRANSLATION_LOCKED_Z,
            ),
        ColliderBuilder::cuboid(0.4, 0.5, 0.5).mass(10.0).friction(0.0),
    );
    let mut last = world.bodies[pusher].translation().x;
    let mut travel = Vec::new();
    for _ in 0..(hz as usize) {
        world.step();
        let x = world.bodies[pusher].translation().x;
        travel.push(x - last);
        last = x;
    }
    travel
}

#[test]
fn a_fast_body_pushing_a_soft_body_moves_through_every_step_as_its_contacts_allow() {
    for hz in [60.0, 30.0] {
        let free = push_a_jelly(hz, true, false);
        let on = push_a_jelly(hz, true, true);
        let off = push_a_jelly(hz, false, true);
        // The scene pushes: the jelly takes speed off the box, which a clear path does not.
        let free_total: Real = free.iter().sum();
        let off_total: Real = off.iter().sum();
        assert!(
            off_total < free_total - 1.0,
            "{hz} Hz: the box travelled {off_total} m pushing the jelly, {free_total} m free"
        );
        for (step, (a, b)) in on.iter().zip(&off).enumerate() {
            assert!(
                (a - b).abs() < 1.0e-4,
                "{hz} Hz, step {step}: the box moved {a} m with the continuous pass on, {b} m with it off"
            );
        }
    }
}

/// A 0.05 m ball 2 m left of a 0.6 m jelly cube floating at the origin, shot through it at
/// `speed` along +x for `steps` steps at 60 Hz, no gravity. Returns the ball's final x.
fn shoot_through_a_jelly(speed: Real, ccd: bool, steps: usize) -> Real {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    if !ccd {
        world.integration_parameters.max_ccd_substeps = 0;
    }
    world.insert_soft_body(
        SoftBodyBuilder::cuboid(Vector::ZERO, Vector::splat(0.6), 3, 3, 3)
            .cell_model(SoftBodyCellModel::Corotational)
            .material(SoftBodyMaterial {
                young_modulus: 5.0e4,
                poisson_ratio: 0.3,
                elastic_damping_ratio: 1.0,
                ..Default::default()
            })
            .particle_mass(0.05)
            .particle_radius(0.02)
            .surface_collider(ColliderBuilder::ball(0.02)),
    );
    let (ball, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(-2.0, 0.0, 0.0))
            .linvel(Vector::new(speed, 0.0, 0.0)),
        ColliderBuilder::ball(0.05).density(100.0),
    );
    for _ in 0..steps {
        world.step();
    }
    world.bodies[ball].translation().x
}

#[test]
fn a_fast_body_the_contacts_miss_is_stopped_at_the_soft_surface_in_its_path() {
    // 240 m/s: 4 m a step, from x = -2 to x = 2 in one step, through the whole jelly.
    let through = shoot_through_a_jelly(240.0, false, 1);
    assert!(
        through > 0.3,
        "with the continuous pass off the ball ended at x = {through}: the scene no longer shoots through the jelly"
    );
    let stopped = shoot_through_a_jelly(240.0, true, 1);
    assert!(
        stopped < -0.25,
        "with the continuous pass on the ball ended at x = {stopped}, past the jelly's near face at x = -0.3"
    );
}
