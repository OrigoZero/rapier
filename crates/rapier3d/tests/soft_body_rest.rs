//! A soft body placed at rest stays at rest: the copy of a body another simulation runs
//! (`SoftBodySet::rest_at`).

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
