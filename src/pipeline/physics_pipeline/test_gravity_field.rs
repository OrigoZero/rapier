//! A body's gravity taken from the hooks at every substep (`RigidBody::set_gravity_field`,
//! `SoftBody::set_gravity_field`).

use crate::alloc_prelude::*;
use crate::dynamics::{RigidBodyBuilder, RigidBodyHandle, SoftBodyBuilder, SoftBodyHandle};
use crate::geometry::ColliderBuilder;
use crate::math::{Real, Vector};
use crate::pipeline::{GravitySubject, PhysicsHooks, PhysicsWorld};

/// A pull of `gm / r²` toward `centre`, for every subject asked about.
struct PointPull {
    centre: Vector,
    gm: Real,
}

impl PhysicsHooks for PointPull {
    fn gravity(&self, _subject: GravitySubject, point: Vector, _gravity: Vector) -> Vector {
        let d = self.centre - point;
        let r2 = d.length_squared();
        d * (self.gm / (r2 * r2.sqrt()))
    }
}

/// The step's own gravity vector, for every subject asked about.
struct AsTheStep;

impl PhysicsHooks for AsTheStep {}

/// A world with no gravity of its own, stepping at `rate` steps a second.
fn world(rate: Real) -> PhysicsWorld {
    let mut world = PhysicsWorld::default();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = 1.0 / rate;
    world
}

/// A 1 kg ball at `at`, moving at `velocity`.
fn ball(world: &mut PhysicsWorld, at: Vector, velocity: Vector) -> RigidBodyHandle {
    let handle = world.bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(at)
            .linvel(velocity)
            .can_sleep(false),
    );
    world.colliders.insert_with_parent(
        ColliderBuilder::ball(0.25).density(1.0 / (4.0 / 3.0 * core::f32::consts::PI * 0.015625)),
        handle,
        &mut world.bodies,
    );
    handle
}

/// The farthest the orbit's radius strays from `radius`, as a share of it, over `orbits`
/// orbits of a ball on a circular orbit of `radius` and `period` seconds about the origin,
/// at `rate` steps a second, pulled `pull`.
fn orbit_spread(rate: Real, radius: Real, period: Real, orbits: usize, pull: Pull) -> Real {
    let mut world = world(rate);
    let speed = 2.0 * core::f32::consts::PI * radius / period;
    let gm = speed * speed * radius;
    let handle = ball(&mut world, Vector::X * radius, Vector::Z * speed);
    let hooks = PointPull {
        centre: Vector::ZERO,
        gm,
    };
    if pull == Pull::Field {
        world.bodies[handle].set_gravity_field(true, true);
    }
    let steps = (period * rate * orbits as Real).round() as usize;
    let mut worst: Real = 0.0;
    for _ in 0..steps {
        if pull == Pull::HeldForce {
            let body = &mut world.bodies[handle];
            let force = hooks.gravity(
                GravitySubject::RigidBody(handle),
                body.center_of_mass(),
                Vector::ZERO,
            ) * body.mass();
            body.reset_forces(false);
            body.add_force(force, false);
        }
        world.step_with_events(&hooks, &());
        let r = world.bodies[handle].center_of_mass().length();
        worst = worst.max((r - radius).abs() / radius);
    }
    worst
}

#[derive(Copy, Clone, PartialEq)]
enum Pull {
    /// The pull is the body's gravity, asked at every substep.
    Field,
    /// The pull is a force worked out where the body starts each step and held over its
    /// substeps.
    HeldForce,
}

/// A ball on a circular orbit about a point pulled through its gravity holds its radius
/// within 1% over ten orbits at 60 and at 30 steps a second, twice each; the same pull given
/// as a force worked out once a step and held over the substeps strays further.
#[test]
fn a_point_field_holds_a_circular_orbit() {
    for rate in [60.0, 30.0] {
        for pass in 0..2 {
            let spread = orbit_spread(rate, 20.0, 6.0, 10, Pull::Field);
            assert!(
                spread < 0.01,
                "pass {pass} at {rate} Hz: the radius strayed {:.3}% over ten orbits",
                spread * 100.0
            );
        }
        let held = orbit_spread(rate, 20.0, 6.0, 10, Pull::HeldForce);
        assert!(
            held > 0.01,
            "at {rate} Hz the held force kept the radius within {:.3}%",
            held * 100.0
        );
    }
}

/// A body whose gravity the hooks answer as the step's own vector moves exactly as a body
/// taking the step's gravity does, and a force of its weight held against that gravity keeps
/// it where it stands.
#[test]
fn a_field_answering_the_step_vector_moves_a_body_as_the_step_does() {
    let mut world = world(60.0);
    world.gravity = Vector::new(0.0, -9.81, 0.0);
    let plain = ball(
        &mut world,
        Vector::new(-5.0, 0.0, 0.0),
        Vector::new(1.0, 2.0, 0.0),
    );
    let field = ball(
        &mut world,
        Vector::new(5.0, 0.0, 0.0),
        Vector::new(1.0, 2.0, 0.0),
    );
    let held = ball(&mut world, Vector::new(0.0, 0.0, 5.0), Vector::ZERO);
    world.bodies[field].set_gravity_field(true, true);
    world.bodies[held].set_gravity_field(true, true);
    let weight = world.bodies[held].mass() * 9.81;
    world.bodies[held].add_force(Vector::new(0.0, weight, 0.0), true);
    for _ in 0..90 {
        world.step_with_events(&AsTheStep, &());
    }
    let a = world.bodies[plain].translation() + Vector::new(10.0, 0.0, 0.0);
    let b = world.bodies[field].translation();
    assert!((a - b).length() < 1.0e-4, "plain {a:?}, field {b:?}");
    let still = world.bodies[held].translation();
    assert!(
        (still - Vector::new(0.0, 0.0, 5.0)).length() < 1.0e-5,
        "a weight held against the field's gravity moved the body to {still:?}"
    );
}

/// A field pulling along an axis the body's translation is locked on moves it nowhere along
/// that axis, the way the step's gravity does not; the body's gravity scale scales the pull.
#[test]
fn a_field_respects_locked_axes_and_the_gravity_scale() {
    let mut world = world(60.0);
    let locked = ball(&mut world, Vector::new(0.0, 10.0, 0.0), Vector::ZERO);
    let half = ball(&mut world, Vector::new(10.0, 10.0, 0.0), Vector::ZERO);
    let whole = ball(&mut world, Vector::new(-10.0, 10.0, 0.0), Vector::ZERO);
    world.bodies[locked].set_enabled_translations(true, false, true, true);
    world.bodies[half].set_gravity_scale(0.5, true);
    for h in [locked, half, whole] {
        world.bodies[h].set_gravity_field(true, true);
    }
    struct Down;
    impl PhysicsHooks for Down {
        fn gravity(&self, _: GravitySubject, _: Vector, _: Vector) -> Vector {
            Vector::new(0.0, -4.0, 0.0)
        }
    }
    for _ in 0..60 {
        world.step_with_events(&Down, &());
    }
    assert_eq!(world.bodies[locked].translation().y, 10.0);
    let v_half = world.bodies[half].linvel().y;
    let v_whole = world.bodies[whole].linvel().y;
    assert!(
        (v_whole + 4.0).abs() < 1.0e-3,
        "one second of 4 m/s² gave {v_whole}"
    );
    assert!(
        (v_half + 2.0).abs() < 1.0e-3,
        "half the scale gave {v_half}"
    );
}

/// The particles of a soft body whose gravity comes from the hooks fall toward a point the
/// hooks pull them to, each asked about the body it belongs to; a body taking the step's gravity
/// beside it keeps the step's (zero) gravity.
#[test]
fn a_soft_body_falls_toward_a_point_field() {
    let mut world = world(60.0);
    let builder =
        |at: Vector| SoftBodyBuilder::cuboid(at, Vector::splat(0.5), 3, 3, 3).particle_mass(0.1);
    let pulled: SoftBodyHandle = world.soft_bodies.insert(
        builder(Vector::new(10.0, 0.0, 0.0)),
        &mut world.bodies,
        &mut world.colliders,
    );
    let free: SoftBodyHandle = world.soft_bodies.insert(
        builder(Vector::new(0.0, 10.0, 0.0)),
        &mut world.bodies,
        &mut world.colliders,
    );
    world.soft_bodies[pulled].set_gravity_field(true);
    struct Asked {
        seen: std::sync::Mutex<Vec<GravitySubject>>,
    }
    impl PhysicsHooks for Asked {
        fn gravity(&self, subject: GravitySubject, point: Vector, _: Vector) -> Vector {
            self.seen.lock().unwrap().push(subject);
            -point.normalize_or_zero() * 5.0
        }
    }
    let hooks = Asked {
        seen: std::sync::Mutex::new(Vec::new()),
    };
    let start_pulled = world.soft_bodies[pulled].center_of_mass();
    let start_free = world.soft_bodies[free].center_of_mass();
    for _ in 0..30 {
        world.step_with_events(&hooks, &());
    }
    let moved = world.soft_bodies[pulled].center_of_mass() - start_pulled;
    assert!(moved.x < -0.5, "the pulled body moved {moved:?}");
    assert!(
        moved.y.abs() < 1.0e-3 && moved.z.abs() < 1.0e-3,
        "it moved off the line to the point: {moved:?}"
    );
    let still = world.soft_bodies[free].center_of_mass() - start_free;
    assert!(
        still.length() < 1.0e-4,
        "the body taking the step's gravity moved {still:?}"
    );
    let seen = hooks.seen.lock().unwrap();
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|s| *s == GravitySubject::SoftBody(pulled)));
}
