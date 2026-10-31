//! A soft body placed at rest stays at rest, and a body carried as a whole keeps its sleep: the
//! copy of a body another simulation runs (`SoftBodySet::rest_at`, `SoftBodySet::carry`).

use rapier3d::prelude::*;

fn world_with_ground() -> PhysicsWorld {
    let mut world = PhysicsWorld::new();
    world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(50.0, 0.5, 50.0).translation(Vector::new(0.0, -0.5, 0.0)),
    );
    world
}

/// A soft box a little above the ground, turned so it lands on an edge.
fn soft_box() -> SoftBodyBuilder {
    SoftBodyBuilder::cuboid(Vector::new(0.0, 0.8, 0.0), Vector::splat(0.5), 4, 4, 4)
        .softness(SpringCoefficients::new(30.0, 1.0))
}

/// Drops [`soft_box`] on the ground of a fresh world and steps until it sleeps.
fn settled_box() -> (PhysicsWorld, SoftBodyHandle) {
    let mut world = world_with_ground();
    let handle = world.insert_soft_body(soft_box());
    for _ in 0..1200 {
        world.step();
        if world.soft_bodies[handle].is_sleeping() {
            return (world, handle);
        }
    }
    panic!("the soft box did not fall asleep");
}

fn positions(world: &PhysicsWorld, handle: SoftBodyHandle) -> Vec<Vector> {
    world.soft_bodies[handle].particle_positions().collect()
}

fn largest_gap(a: &[Vector], b: &[Vector]) -> Real {
    a.iter()
        .zip(b)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0, Real::max)
}

/// A body built fresh and put at rest where another simulation's copy of it settled sleeps
/// there from the first step on: the contacts its places begin with the ground do not wake it.
/// A ball dropped on it still does.
#[test]
fn a_body_put_at_rest_where_its_copy_settled_sleeps_through_the_contacts_it_begins() {
    let (settled, settled_handle) = settled_box();
    let places: Vec<(usize, Vector)> = positions(&settled, settled_handle)
        .into_iter()
        .enumerate()
        .collect();

    let mut world = world_with_ground();
    let handle = world.insert_soft_body(soft_box());
    assert!(
        world
            .soft_bodies
            .rest_at(handle, &places, &mut world.bodies),
        "the places fit the body"
    );
    let placed = positions(&world, handle);
    for step in 0..60 {
        world.step();
        assert!(
            world.soft_bodies[handle].is_sleeping(),
            "step {step}: the body put at rest woke"
        );
    }
    assert_eq!(
        positions(&world, handle),
        placed,
        "the body at rest moved"
    );

    world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 3.0, 0.0)),
        ColliderBuilder::ball(0.2),
    );
    let woke = (0..200).any(|_| {
        world.step();
        !world.soft_bodies[handle].is_sleeping()
    });
    assert!(woke, "the ball dropped on the body at rest did not wake it");
}

/// A sleeping body carried with no velocity change sleeps on where it was carried, standing
/// there particle for particle with its frame carried along; carried with a velocity change, it
/// wakes and moves at it.
#[test]
fn a_sleeping_body_carried_still_sleeps_where_it_was_carried() {
    let (mut world, handle) = settled_box();
    let before = positions(&world, handle);
    let root = world.soft_bodies[handle].root_body();
    let frame_before = world.bodies[root].translation();
    let shift = Vector::new(0.3, 0.0, -0.2);
    let turn = Rotation::from_rotation_y(0.4);
    let pivot = frame_before;
    assert!(world.soft_bodies.carry(
        handle,
        pivot,
        turn,
        shift,
        Vector::ZERO,
        &mut world.bodies,
        &mut world.colliders,
    ));
    let carried: Vec<Vector> = before
        .iter()
        .map(|p| turn * (*p - pivot) + pivot + shift)
        .collect();
    assert!(largest_gap(&positions(&world, handle), &carried) < 1.0e-5);
    let frame = world.bodies[root].translation();
    assert!(
        (frame - (frame_before + shift)).length() < 1.0e-4,
        "the body's frame stands at {frame:?}, carried from {frame_before:?} by {shift:?}"
    );
    for step in 0..60 {
        world.step();
        assert!(
            world.soft_bodies[handle].is_sleeping(),
            "step {step}: the body carried at rest woke"
        );
    }
    assert!(largest_gap(&positions(&world, handle), &carried) < 1.0e-5);

    let push = Vector::new(0.0, 0.0, 2.0);
    assert!(world.soft_bodies.carry(
        handle,
        pivot,
        Rotation::IDENTITY,
        Vector::ZERO,
        push,
        &mut world.bodies,
        &mut world.colliders,
    ));
    world.step();
    assert!(
        !world.soft_bodies[handle].is_sleeping(),
        "the body given a velocity did not wake"
    );
    let moved = world.soft_bodies[handle].center_of_mass().z
        - carried.iter().map(|p| p.z).sum::<Real>() / carried.len() as Real;
    assert!(moved > 0.01, "the body given a velocity along z moved {moved} m along it");
}

/// An awake body carried back and forth by half a millimetre every third step, as a copy held to
/// another simulation's pose is, falls asleep on its own motion within the time a body left
/// alone takes: the carries count as none of its travel and keep its time toward sleeping.
#[test]
fn an_awake_body_carried_now_and_then_falls_asleep_on_its_own_motion() {
    let (mut world, handle) = settled_box();
    world
        .soft_bodies
        .wake_up(handle, &mut world.bodies, true);
    let mut asleep_at = None;
    for step in 0..120 {
        if step % 3 == 0 {
            let along = if step % 6 == 0 { 5.0e-4 } else { -5.0e-4 };
            let pivot = world.soft_bodies[handle].center_of_mass();
            world.soft_bodies.carry(
                handle,
                pivot,
                Rotation::IDENTITY,
                Vector::new(along, 0.0, 0.0),
                Vector::ZERO,
                &mut world.bodies,
                &mut world.colliders,
            );
        }
        world.step();
        if world.soft_bodies[handle].is_sleeping() {
            asleep_at = Some(step);
            break;
        }
    }
    let asleep_at = asleep_at.expect("the carried body never fell asleep");
    assert!(
        asleep_at < 60,
        "the carried body took {asleep_at} steps to fall asleep"
    );
}
