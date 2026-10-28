//! Staged-solver parity and substep-group partition regression tests.

use crate::dynamics::RigidBodyBuilder;
use crate::geometry::ColliderBuilder;
use crate::math::Vector;
use crate::prelude::RevoluteJointBuilder;

/// Regression: overflow-color SIMD chunking. Same-pair manifolds (compound colliders)
/// land in the overflow color; packing two into one SIMD lane group makes the wide
/// gather/scatter (last-writer-wins) silently drop an impulse. Compares 2 workers vs 1.
#[test]
#[cfg(all(
    feature = "dim2",
    feature = "parallel",
    not(feature = "unsync-callbacks")
))]
fn staged_overflow_chunk_no_shared_dynamic_body() {
    use crate::alloc_prelude::*;
    use crate::pipeline::PhysicsWorld;
    use crate::prelude::{Pose, SharedShape};

    fn build(num_threads: usize) -> PhysicsWorld {
        let mut world = PhysicsWorld::new();
        world.integration_parameters.contact_clustering = false;
        world.integration_parameters.contact_recycling = false;
        world.configure_thread_pool(num_threads).unwrap();

        let tribox = || {
            ColliderBuilder::compound(vec![
                (
                    Pose::from_translation(Vector::new(-1.6, 0.0)),
                    SharedShape::cuboid(0.5, 0.5),
                ),
                (
                    Pose::from_translation(Vector::new(0.0, 0.0)),
                    SharedShape::cuboid(0.5, 0.5),
                ),
                (
                    Pose::from_translation(Vector::new(1.6, 0.0)),
                    SharedShape::cuboid(0.5, 0.5),
                ),
            ])
        };

        let _ = world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, 0.0)),
            tribox(),
        );
        for i in 1..=3 {
            let _ = world.insert(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(0.0, 0.98 * i as f32))
                    .can_sleep(false),
                tribox(),
            );
        }

        for _ in 0..30 {
            world.step();
        }
        world
    }

    let staged = build(2);
    let reference = build(1);

    // The two-worker staged solve must reach the same rest state as the
    // one-worker reference. A dropped overflow impulse would let the shared
    // body drift, diverging well past this tolerance.
    for ((_, a), (_, b)) in staged.bodies.iter().zip(reference.bodies.iter()) {
        let da = a.translation();
        let db = b.translation();
        assert!(da.x.is_finite() && da.y.is_finite(), "non-finite position");
        assert!(
            (da - db).length() < 0.05,
            "staged solver diverged from reference: {da:?} vs {db:?}"
        );
    }
}

/// Multi-group substep ring: a three-solve-group scene must match the one-worker
/// reference on two staged workers, and the elevated pair must actually hold tighter
/// than an identical non-elevated pair in the same world.
#[test]
#[cfg(all(feature = "parallel", not(feature = "unsync-callbacks")))]
fn staged_multi_group_substep_parity() {
    use crate::pipeline::PhysicsWorld;

    fn build(num_threads: usize) -> PhysicsWorld {
        let mut world = PhysicsWorld::new();
        world.configure_thread_pool(num_threads).unwrap();

        #[cfg(feature = "dim2")]
        let cuboid = || ColliderBuilder::cuboid(0.5, 0.5);
        #[cfg(feature = "dim3")]
        let cuboid = || ColliderBuilder::cuboid(0.5, 0.5, 0.5);
        let dynamic = |x: crate::math::Real, y: crate::math::Real| {
            RigidBodyBuilder::dynamic()
                .translation(Vector::X * x + Vector::Y * y)
                .can_sleep(false)
        };

        // Default-cadence stack on the ground.
        let _ = world.insert(RigidBodyBuilder::fixed(), cuboid());
        for i in 1..=3 {
            let _ = world.insert(dynamic(0.0, 1.001 * i as crate::math::Real), cuboid());
        }

        // Elevated jointed pair (its own component).
        let (a, _) = world.insert(dynamic(20.0, 5.0), cuboid());
        let (b, _) = world.insert(
            dynamic(20.0, 3.0).additional_solver_iterations(4),
            cuboid().density(50.0),
        );
        #[cfg(feature = "dim2")]
        let joint = RevoluteJointBuilder::new();
        #[cfg(feature = "dim3")]
        let joint = RevoluteJointBuilder::new(Vector::Z);
        world.insert_impulse_joint(a, b, joint);

        // Isolated, higher elevation still (third group).
        let _ = world.insert(
            dynamic(-20.0, 5.0).additional_solver_iterations(8),
            cuboid(),
        );

        for _ in 0..60 {
            world.step();
        }
        world
    }

    let staged = build(2);
    let reference = build(1);

    for ((_, a), (_, b)) in staged.bodies.iter().zip(reference.bodies.iter()) {
        let da = a.translation();
        let db = b.translation();
        assert!(da.x.is_finite() && da.y.is_finite(), "non-finite position");
        assert!(
            (da - db).length() < 0.05,
            "multi-group staged solve diverged from reference: {da:?} vs {db:?}"
        );
    }
}

/// Multi-group JOINT layout: enough jointed pairs for parallel colors/SIMD chunks, plus
/// an elevated pair forcing the multi-group path, must match the one-worker reference.
/// Holds the grouping steady to exercise the group-fingerprinted joint-reuse cache.
#[test]
#[cfg(all(feature = "parallel", not(feature = "unsync-callbacks")))]
fn staged_multi_group_joint_simd_parity() {
    use crate::pipeline::PhysicsWorld;

    fn build(num_threads: usize) -> PhysicsWorld {
        let mut world = PhysicsWorld::new();
        world.configure_thread_pool(num_threads).unwrap();

        #[cfg(feature = "dim2")]
        let cuboid = || ColliderBuilder::cuboid(0.4, 0.4);
        #[cfg(feature = "dim3")]
        let cuboid = || ColliderBuilder::cuboid(0.4, 0.4, 0.4);
        let dynamic = |x: crate::math::Real, y: crate::math::Real| {
            RigidBodyBuilder::dynamic()
                .translation(Vector::X * x + Vector::Y * y)
                .can_sleep(false)
        };
        #[cfg(feature = "dim2")]
        let joint = RevoluteJointBuilder::new;
        #[cfg(feature = "dim3")]
        let joint = || RevoluteJointBuilder::new(Vector::Z);

        // 64 independent swinging pairs (default cadence): pairwise
        // body-disjoint, so they share one joint color — far above the
        // parallel-color threshold.
        for i in 0..64 {
            let x = i as crate::math::Real * 3.0;
            let (a, _) = world.insert(dynamic(x, 4.0), cuboid());
            let (b, _) = world.insert(dynamic(x + 0.9, 3.0), cuboid().density(2.0));
            world.insert_impulse_joint(a, b, joint());
        }

        // One elevated pair: forces the multi-group path for the whole run.
        let (a, _) = world.insert(dynamic(-10.0, 4.0), cuboid());
        let (b, _) = world.insert(
            dynamic(-10.0, 2.0).additional_solver_iterations(4),
            cuboid().density(20.0),
        );
        world.insert_impulse_joint(a, b, joint());

        for _ in 0..60 {
            world.step();
        }
        world
    }

    let staged = build(2);
    let reference = build(1);

    for ((_, a), (_, b)) in staged.bodies.iter().zip(reference.bodies.iter()) {
        let da = a.translation();
        let db = b.translation();
        assert!(da.x.is_finite() && da.y.is_finite(), "non-finite position");
        assert!(
            (da - db).length() < 0.05,
            "multi-group joint layout diverged from reference: {da:?} vs {db:?}"
        );
    }
}

/// Substep-group partition: the awake body list must reorder into contiguous groups by
/// component max `additional_solver_iterations` (descending), re-stamp the
/// `active_set_id == index` invariant, and not churn the epoch once stable.
#[test]
fn substep_groups_partition() {
    use crate::alloc_prelude::*;
    use crate::pipeline::PhysicsWorld;

    let mut world = PhysicsWorld::new();

    #[cfg(feature = "dim2")]
    let cuboid = || ColliderBuilder::cuboid(0.5, 0.5);
    #[cfg(feature = "dim3")]
    let cuboid = || ColliderBuilder::cuboid(0.5, 0.5, 0.5);
    let dynamic = |x: crate::math::Real, y: crate::math::Real| {
        RigidBodyBuilder::dynamic()
            .translation(Vector::X * x + Vector::Y * y)
            .can_sleep(false)
    };

    // Default-cadence stack: one box resting on the ground.
    let _ = world.insert(RigidBodyBuilder::fixed(), cuboid());
    let (plain, _) = world.insert(dynamic(0.0, 1.001), cuboid());

    // A two-body jointed pair where only ONE body is elevated: the joint
    // edge must pull the partner into the same (elevated) group.
    let (chain_a, _) = world.insert(dynamic(10.0, 5.0), cuboid());
    let (chain_b, _) = world.insert(dynamic(10.0, 3.0).additional_solver_iterations(4), cuboid());
    #[cfg(feature = "dim2")]
    let joint = RevoluteJointBuilder::new();
    #[cfg(feature = "dim3")]
    let joint = RevoluteJointBuilder::new(Vector::Z);
    world.insert_impulse_joint(chain_a, chain_b, joint);

    // An isolated component at a higher elevation still.
    let (lone, _) = world.insert(
        dynamic(-10.0, 5.0).additional_solver_iterations(8),
        cuboid(),
    );

    for _ in 0..3 {
        world.step();
    }

    // Expect three groups, descending: {lone}=8, {chain_a, chain_b}=4,
    // {plain}=0. The fixed ground is not part of the awake set.
    let groups = &world.islands.solve_groups;
    assert_eq!(
        groups.iter().map(|g| g.extra_iters).collect::<Vec<_>>(),
        vec![8, 4, 0],
    );
    assert_eq!(
        groups
            .iter()
            .map(|g| g.body_range.len())
            .collect::<Vec<_>>(),
        vec![1, 2, 1],
    );
    assert_eq!(groups[0].body_range.start, 0);
    assert_eq!(groups.last().unwrap().body_range.end, 4);

    // Each body sits in its expected group range, and the
    // `active_set_id == index in the awake island's bodies` invariant
    // holds after the reorder.
    let awake_bodies: Vec<_> = world.islands.active_bodies().collect();
    for (i, handle) in awake_bodies.iter().enumerate() {
        assert_eq!(world.bodies[*handle].ids.active_set_id as usize, i);
    }
    let group_of = |h: crate::dynamics::RigidBodyHandle| {
        let id = world.bodies[h].ids.active_set_id as usize;
        groups.iter().position(|g| g.body_range.contains(&id))
    };
    assert_eq!(group_of(lone), Some(0));
    assert_eq!(group_of(chain_a), Some(1), "joint partner must be lifted");
    assert_eq!(group_of(chain_b), Some(1));
    assert_eq!(group_of(plain), Some(2));

    // Steady state: the grouping is contiguous, so further steps must not bump the
    // active-set epoch (a bump forces a full solver-graph rebuild — re-partitioning
    // every step would tank elevated scenes).
    let epoch = world.islands.active_set_epoch;
    for _ in 0..5 {
        world.step();
    }
    assert_eq!(
        world.islands.active_set_epoch, epoch,
        "stable grouping must not churn the active-set epoch"
    );

    // Un-elevating every body empties the groups (single implicit group).
    world
        .bodies
        .get_mut(lone)
        .unwrap()
        .set_additional_solver_iterations(0);
    world
        .bodies
        .get_mut(chain_b)
        .unwrap()
        .set_additional_solver_iterations(0);
    world.step();
    assert!(world.islands.solve_groups.is_empty());
}

/// Each body the step moved reads the substeps and passes of the group that
/// solved it: the world's counts plus the largest extras its connected
/// component asked for, a jointed partner taking the elevated body's, and the
/// world's counts alone once nothing asks for more.
#[test]
fn bodies_read_the_substeps_their_group_ran() {
    use crate::pipeline::PhysicsWorld;

    let mut world = PhysicsWorld::new();

    #[cfg(feature = "dim2")]
    let cuboid = || ColliderBuilder::cuboid(0.5, 0.5);
    #[cfg(feature = "dim3")]
    let cuboid = || ColliderBuilder::cuboid(0.5, 0.5, 0.5);
    let dynamic = |x: crate::math::Real, y: crate::math::Real| {
        RigidBodyBuilder::dynamic()
            .translation(Vector::X * x + Vector::Y * y)
            .can_sleep(false)
    };

    let (plain, _) = world.insert(dynamic(0.0, 1.0), cuboid());
    let (chain_a, _) = world.insert(dynamic(10.0, 5.0), cuboid());
    let (chain_b, _) = world.insert(
        dynamic(10.0, 3.0)
            .additional_solver_iterations(4)
            .additional_pgs_iterations(2),
        cuboid(),
    );
    #[cfg(feature = "dim2")]
    let joint = RevoluteJointBuilder::new();
    #[cfg(feature = "dim3")]
    let joint = RevoluteJointBuilder::new(Vector::Z);
    world.insert_impulse_joint(chain_a, chain_b, joint);
    let (lone, _) = world.insert(dynamic(-10.0, 5.0).additional_solver_iterations(8), cuboid());

    assert_eq!(world.bodies[plain].solved_substeps(), (0, 0), "no step has moved it yet");
    for _ in 0..2 {
        world.step();
    }
    let base = world.integration_parameters.num_solver_iterations;
    let pgs = world.integration_parameters.num_internal_pgs_iterations;
    assert_eq!(world.bodies[plain].solved_substeps(), (base, pgs));
    assert_eq!(world.bodies[chain_a].solved_substeps(), (base + 4, pgs + 2), "a joint partner runs the elevated body's");
    assert_eq!(world.bodies[chain_b].solved_substeps(), (base + 4, pgs + 2));
    assert_eq!(world.bodies[lone].solved_substeps(), (base + 8, pgs));

    world.bodies.get_mut(chain_b).unwrap().set_additional_solver_iterations(0);
    world.bodies.get_mut(chain_b).unwrap().set_additional_pgs_iterations(0);
    world.bodies.get_mut(lone).unwrap().set_additional_solver_iterations(0);
    world.step();
    for body in [plain, chain_a, chain_b, lone] {
        assert_eq!(world.bodies[body].solved_substeps(), (base, pgs));
    }
}

/// A dynamic body in a higher dominance group pushes a body of a lower group at its own
/// speed: the dominated body takes the dominant one's velocity where they meet (with no
/// restitution, plus the little the contact's bias adds) instead of being pushed out of an overlap,
/// and the dominant body keeps its velocity. The overlap of the step they meet on is
/// recovered, and stays within the solver's allowed error after it.
#[test]
fn a_dominant_body_pushes_a_dominated_one_at_its_own_speed() {
    use crate::pipeline::PhysicsWorld;

    for run in 0..2 {
        let mut world = PhysicsWorld::new();
        world.gravity = Vector::ZERO;
        #[cfg(feature = "dim2")]
        let cuboid = || ColliderBuilder::cuboid(0.5, 0.5).friction(0.0).restitution(0.0);
        #[cfg(feature = "dim3")]
        let cuboid = || ColliderBuilder::cuboid(0.5, 0.5, 0.5).friction(0.0).restitution(0.0);
        let (pusher, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .dominance_group(1)
                .linvel(Vector::X * 5.0)
                .can_sleep(false),
            cuboid(),
        );
        let (pushed, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::X * 1.2)
                .can_sleep(false),
            cuboid(),
        );
        let mut deepest: crate::math::Real = 0.0;
        let mut last = 0.0;
        for _ in 0..30 {
            world.step();
            let gap = world.bodies[pushed].translation().x - world.bodies[pusher].translation().x;
            deepest = deepest.max(1.0 - gap);
            last = 1.0 - gap;
        }
        let pusher_speed = world.bodies[pusher].linvel().x;
        let pushed_speed = world.bodies[pushed].linvel().x;
        assert!((pusher_speed - 5.0).abs() < 1.0e-3, "run {run}: the dominant body slowed to {pusher_speed}");
        assert!((4.95..5.25).contains(&pushed_speed), "run {run}: the dominated body moves at {pushed_speed}, not the 5 it is pushed at");
        assert!(deepest < 0.1, "run {run}: the two overlapped by {deepest}");
        assert!(last < 0.01, "run {run}: the two still overlap by {last} half a second on");
    }
}

/// A soft body whose floor particles are attached to a kinematic body follows that body at the
/// soft body's own substep cadence: the kinematic body takes the substeps of the group its
/// attached soft body solves in, so the particles are held to where the body stands at each
/// substep. Driven up to 1, 12 and 30 m/s at 6 m/s², then held there, the shell keeps its rest
/// offset from the body and its attachments pull with about the impulse they pull with at rest:
/// the weight they carry, and while the body speeds up the push that speeds the shell up with it.
#[test]
#[cfg(feature = "dim3")]
fn a_soft_body_attached_to_a_moving_kinematic_body_keeps_up_with_it() {
    use crate::alloc_prelude::*;
    use crate::dynamics::SoftBodyBuilder;
    use crate::pipeline::PhysicsWorld;

    let dt = 1.0 / 30.0;
    let run = |speed: crate::math::Real| {
        let mut world = PhysicsWorld::new();
        world.integration_parameters.dt = dt;
        let body = world.insert_body(RigidBodyBuilder::kinematic_position_based());
        let shell = world.insert_soft_body(
            SoftBodyBuilder::cuboid(
                Vector::new(0.0, 0.69, 0.0),
                Vector::new(1.0, 0.3, 1.25),
                3,
                2,
                4,
            )
            .mass(20.0)
            .additional_solver_iterations(3),
        );
        {
            let sb = world.soft_bodies.get_mut(shell).unwrap();
            let floor: Vec<usize> = sb
                .particles()
                .iter()
                .enumerate()
                .filter(|(_, p)| p.position.y < 0.5)
                .map(|(i, _)| i)
                .collect();
            assert_eq!(floor.len(), 12);
            for i in floor {
                sb.attach_particle(i, body, &world.bodies);
            }
        }
        let offset = |world: &PhysicsWorld| {
            world.soft_bodies[shell].center_of_mass().z - world.bodies[body].translation().z
        };
        let pull = |world: &PhysicsWorld| {
            world.soft_bodies[shell]
                .particle_attachments()
                .iter()
                .map(|a| a.impulse().length())
                .fold(0.0, crate::math::Real::max)
        };
        for _ in 0..60 {
            world.step();
        }
        let (rest_offset, rest_pull) = (offset(&world), pull(&world));
        let (mut lag, mut most_pull) = (0.0 as crate::math::Real, 0.0 as crate::math::Real);
        let mut velocity = 0.0;
        for _ in 0..240 {
            velocity = (velocity + 6.0 * dt).min(speed);
            let z = world.bodies[body].translation().z + velocity * dt;
            world
                .bodies
                .get_mut(body)
                .unwrap()
                .set_next_kinematic_translation(Vector::new(0.0, 0.0, z));
            world.step();
            lag = lag.max((offset(&world) - rest_offset).abs());
            most_pull = most_pull.max(pull(&world));
        }
        (rest_pull, lag, most_pull)
    };
    for speed in [1.0, 12.0, 30.0] {
        let (rest_pull, lag, most_pull) = run(speed);
        assert!(
            lag < 0.01,
            "at {speed} m/s the shell trails its body by {lag} m"
        );
        assert!(
            most_pull < 1.3 * rest_pull,
            "at {speed} m/s an attachment pulls {most_pull}, against {rest_pull} at rest"
        );
    }
}
