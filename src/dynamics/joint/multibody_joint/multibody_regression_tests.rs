//! Regression tests for reported multibody bugs.
//!
//! Each test reproduces a specific issue from the GitHub tracker. The crash
//! tests assert that a known-bad scenario steps (or mutates the joint set)
//! without panicking; the behavior tests assert that the simulation result
//! matches what the multibody solver is supposed to produce.

#[cfg(feature = "dim3")]
use crate::alloc_prelude::*;
use crate::prelude::*;

/// Regression test for <https://github.com/dimforge/rapier/issues/927> (Bug 1).
///
/// Removing a joint that leaves one of its bodies as an isolated single-link
/// multibody must clean up that body's `rb2mb` entry. Otherwise the entry keeps
/// a stale `MultibodyIndex` pointing at a freed arena slot, and the next call
/// to `MultibodyJointSet::iter()` panics with "No element at index".
#[test]
#[cfg(feature = "dim3")]
fn issue_927_remove_isolating_single_link_keeps_iter_valid() {
    let mut world = PhysicsWorld::new();

    let a = world.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 5.0, 0.0))
            .additional_mass(1.0),
    );
    let b = world.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(1.0, 5.0, 0.0))
            .additional_mass(1.0),
    );
    let c = world.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(2.0, 5.0, 0.0))
            .additional_mass(1.0),
    );

    let joint = || RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0));
    world.insert_multibody_joint(a, b, joint());
    let handle_bc = world.insert_multibody_joint(b, c, joint()).unwrap();

    // Removing B->C isolates C as a single-link multibody.
    world.remove_multibody_joint(handle_bc);

    // Used to panic: "No element at index".
    let count = world.multibody_joints.iter().count();
    assert_eq!(count, 1, "only the A->B joint should remain");
}

/// Regression test for <https://github.com/dimforge/rapier/issues/927> (Bug 2).
///
/// Building multi-link sub-chains separately and then connecting them to a
/// common parent forces `Multibody::append()` to merge subtrees that have more
/// than one link. A bad `internal_id` rebase used to make `fill_jacobians`
/// read `body_jacobians` out of bounds during the velocity solve.
#[test]
#[cfg(feature = "dim3")]
fn issue_927_branching_multibody_tree_steps_without_panic() {
    let mut world = PhysicsWorld::new();

    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
        ColliderBuilder::cuboid(50.0, 0.5, 50.0),
    );

    let make_body = |world: &mut PhysicsWorld, pos: Vector| {
        world
            .insert(
                RigidBodyBuilder::dynamic()
                    .translation(pos)
                    .additional_mass(5.0),
                ColliderBuilder::ball(0.3),
            )
            .0
    };

    let (chassis, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 1.5, 0.0))
            .additional_mass(50.0),
        ColliderBuilder::cuboid(2.0, 0.3, 1.0),
    );

    let joint = || RevoluteJointBuilder::new(Vector::new(1.0, 0.0, 0.0));

    // Two front sub-chains with three links each (so `append()` merges a
    // subtree with more than one link), and two rear sub-chains with two links.
    let axle_fl = make_body(&mut world, Vector::new(-1.5, 1.2, 1.2));
    let mangueta_fl = make_body(&mut world, Vector::new(-1.5, 1.0, 1.5));
    let wheel_fl = make_body(&mut world, Vector::new(-1.5, 0.5, 1.8));
    world.insert_multibody_joint(axle_fl, mangueta_fl, joint());
    world.insert_multibody_joint(mangueta_fl, wheel_fl, joint());

    let axle_fr = make_body(&mut world, Vector::new(1.5, 1.2, 1.2));
    let mangueta_fr = make_body(&mut world, Vector::new(1.5, 1.0, 1.5));
    let wheel_fr = make_body(&mut world, Vector::new(1.5, 0.5, 1.8));
    world.insert_multibody_joint(axle_fr, mangueta_fr, joint());
    world.insert_multibody_joint(mangueta_fr, wheel_fr, joint());

    let axle_rl = make_body(&mut world, Vector::new(-1.5, 1.2, -1.2));
    let wheel_rl = make_body(&mut world, Vector::new(-1.5, 0.5, -1.5));
    world.insert_multibody_joint(axle_rl, wheel_rl, joint());

    let axle_rr = make_body(&mut world, Vector::new(1.5, 1.2, -1.2));
    let wheel_rr = make_body(&mut world, Vector::new(1.5, 0.5, -1.5));
    world.insert_multibody_joint(axle_rr, wheel_rr, joint());

    // Connecting each sub-chain to the chassis triggers `append()` on a
    // multi-link subtree.
    world.insert_multibody_joint(chassis, axle_fl, joint());
    world.insert_multibody_joint(chassis, axle_fr, joint());
    world.insert_multibody_joint(chassis, axle_rl, joint());
    world.insert_multibody_joint(chassis, axle_rr, joint());

    for _ in 0..30 {
        world.step();
    }

    assert!(world.bodies[chassis].translation().y.is_finite());
}

/// Regression test for <https://github.com/dimforge/rapier/issues/906>.
///
/// Inserting a multibody joint *after* the pipeline has already stepped at
/// least once used to overflow a subtraction in `Multibody::append()`.
/// Inserting joints only during initialization, or using impulse joints, did
/// not trigger it.
#[test]
#[cfg(feature = "dim3")]
fn issue_906_insert_multibody_joint_between_steps() {
    const SHIFT: Real = 1.15;

    let mut world = PhysicsWorld::new();

    let (root, _) = world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );

    let joint = || {
        RevoluteJointBuilder::new(Vector::new(0.0, 1.0, 0.0))
            .local_anchor1(Vector::new(0.0, -SHIFT, 0.0))
    };

    // Build a short chain before the first step (this path always worked).
    let mut last = root;
    for i in 1..4 {
        let (body, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -SHIFT * i as Real, 0.0)),
            ColliderBuilder::cuboid(0.5, 0.5, 0.5),
        );
        world.insert_multibody_joint(last, body, joint());
        last = body;
    }

    world.step();

    // Now extend the chain dynamically, between steps. This used to panic with
    // "attempt to subtract with overflow" in `Multibody::append()`.
    for i in 4..10 {
        let (body, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -SHIFT * i as Real, 0.0)),
            ColliderBuilder::cuboid(0.5, 0.5, 0.5),
        );
        world.insert_multibody_joint(last, body, joint());
        last = body;

        world.step();
    }

    assert!(world.bodies[last].translation().y.is_finite());
}

/// Regression test for <https://github.com/dimforge/rapier/issues/907>.
///
/// A multibody-jointed body and a free rigid body both falling onto the same
/// ground used to crash the velocity solver's warm-start with a nalgebra
/// "Matrix slicing out of bounds" panic in `GenericRhs::apply_impulse`.
#[test]
#[cfg(feature = "dim3")]
fn issue_907_body_colliding_with_multibody() {
    let mut world = PhysicsWorld::new();

    let (ground, _) = world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(50.0, 0.5, 50.0),
    );

    // A spinning body attached to the ground with a revolute multibody joint.
    let (attached, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 3.0, 0.0))
            .angvel(Vector::new(0.0, 1.0, 0.0)),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );
    world.insert_multibody_joint(
        ground,
        attached,
        RevoluteJointBuilder::new(Vector::new(0.0, 1.0, 0.0))
            .local_anchor1(Vector::new(0.0, 3.0, 0.0)),
    );

    // A free body that falls onto the ground next to the multibody.
    let (free, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(3.0, 3.0, 0.0)),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );

    for _ in 0..400 {
        world.step();
    }

    assert!(world.bodies[free].translation().y.is_finite());
}

/// Regression test for <https://github.com/dimforge/rapier/issues/908>.
///
/// Removing the links of a multibody chain one by one, while a free body rests
/// in contact with the chain's fixed root, used to crash the contact solver
/// with "No element at index" in `GenericContactConstraintBuilder::generate`.
#[test]
#[cfg(feature = "dim3")]
fn issue_908_remove_body_from_multibody_chain() {
    let mut world = PhysicsWorld::new();

    let (root, _) = world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );

    // A chain of spherical-jointed bodies hanging from the root.
    let mut chain = vec![root];
    for i in 1..4 {
        let (body, _) = world.insert(
            RigidBodyBuilder::dynamic().translation(Vector::new(0.0, -2.0 * i as Real, 0.0)),
            ColliderBuilder::cuboid(0.5, 0.5, 0.5),
        );
        world.insert_multibody_joint(
            *chain.last().unwrap(),
            body,
            SphericalJointBuilder::new().local_anchor1(Vector::new(0.0, -2.0, 0.0)),
        );
        chain.push(body);
    }

    // A free body resting on the root. The crash only happened with it present.
    let (free, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 2.0, 0.0)),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );

    // Let everything settle, then peel the chain off one body at a time,
    // stepping in-between. Removing a link used to panic.
    for _ in 0..32 {
        world.step();
    }

    while chain.len() > 1 {
        let body = chain.pop().unwrap();
        world.remove_body(body);

        for _ in 0..32 {
            world.step();
        }
    }

    assert!(world.bodies[free].translation().y.is_finite());
}

/// Regression test for <https://github.com/dimforge/rapier/issues/400>.
///
/// A dynamic body attached to a *fixed* parent through a `MultibodyJoint`,
/// plus a free dynamic body colliding with that fixed parent, used to crash
/// the moment the free body touched the parent.
#[test]
#[cfg(feature = "dim3")]
fn issue_400_multibody_joint_fixed_parent_dynamic_collision() {
    let mut world = PhysicsWorld::new();

    // The fixed "table".
    let (table, _) = world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(1.0, 0.1, 1.0),
    );

    // A "flipper": dynamic body attached to the fixed table by a multibody
    // revolute joint, with a velocity motor.
    let (flipper, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(-0.5, 0.3, -0.5)),
        ColliderBuilder::cuboid(0.1, 0.1, 0.1),
    );
    world.insert_multibody_joint(
        table,
        flipper,
        RevoluteJointBuilder::new(Vector::new(0.0, 1.0, 0.0))
            .local_anchor1(Vector::new(-0.5, 0.3, -0.5))
            .motor_velocity(-1.0, 1.0),
    );

    // The "ball" that falls onto the table.
    let (ball, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 1.0, 0.0)),
        ColliderBuilder::ball(0.01).friction(0.0),
    );

    for _ in 0..200 {
        world.step();
    }

    assert!(world.bodies[ball].translation().y.is_finite());
}

/// Regression test for <https://github.com/dimforge/rapier/issues/656>.
///
/// A free-floating multibody whose root collides off-center with a ledge must
/// pick up angular velocity, just like the same setup built with an impulse
/// joint does. The bug left the multibody root unable to react to the
/// collision torque.
#[test]
#[cfg(feature = "dim2")]
fn issue_656_multibody_root_reacts_to_collision_torque() {
    let impulse_angvel = run_issue_656_scenario(false);
    let multibody_angvel = run_issue_656_scenario(true);

    // Sanity check: the off-center landing must spin the body in the
    // (known-good) impulse-joint case.
    assert!(
        impulse_angvel > 0.1,
        "scenario should induce rotation; impulse-joint baseline = {impulse_angvel}"
    );

    // Regression check: the multibody root must react to the collision torque
    // comparably. With the bug its angular velocity stays near zero.
    assert!(
        multibody_angvel > impulse_angvel * 0.25,
        "multibody root barely rotated: multibody = {multibody_angvel}, impulse = {impulse_angvel}"
    );
}

/// Builds the issue-656 scenario (two co-located dynamic bodies joined at their
/// center, free-falling onto a ledge they only partly overlap) and returns the
/// largest angular velocity magnitude the root body reaches.
#[cfg(feature = "dim2")]
fn run_issue_656_scenario(use_multibody: bool) -> Real {
    let mut world = PhysicsWorld::new();

    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -2.0)),
        ColliderBuilder::cuboid(0.5, 0.5),
    );

    // Both bodies start at the same place; the joint pins them at that point.
    let start = Vector::new(0.70, 0.0);
    let (root, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(start),
        ColliderBuilder::cuboid(0.3, 0.1).mass(0.1),
    );
    let (child, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(start),
        ColliderBuilder::cuboid(0.05, 0.05).mass(0.1),
    );

    let joint = RevoluteJointBuilder::new()
        .local_anchor1(Vector::new(0.0, 0.0))
        .local_anchor2(Vector::new(0.0, 0.0))
        .contacts_enabled(false);
    if use_multibody {
        world.insert_multibody_joint(root, child, joint);
    } else {
        world.insert_impulse_joint(root, child, joint);
    }

    let mut max_angvel: Real = 0.0;
    for _ in 0..600 {
        world.step();
        max_angvel = max_angvel.max(world.bodies[root].angvel().abs());
    }
    max_angvel
}

/// Regression test for <https://github.com/dimforge/rapier/issues/379>.
///
/// Angular limits set on a ball (spherical) joint must be enforced by the
/// multibody constraint solver. The bug ignored them entirely, so the child
/// swung freely as if no limit had been set.
#[test]
#[cfg(feature = "dim3")]
fn issue_379_multibody_ball_joint_respects_angular_limits() {
    const LIMIT: Real = 0.3;

    let mut world = PhysicsWorld::new();

    // Fixed root at the origin.
    let root = world.insert_body(RigidBodyBuilder::fixed());

    // A child whose left end is pinned to the origin, extending along +X. Under
    // gravity it tries to swing down (rotating about the Z axis).
    let (child, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(1.0, 0.0, 0.0))
            .additional_mass(1.0),
        ColliderBuilder::cuboid(1.0, 0.1, 0.1),
    );

    world.insert_multibody_joint(
        root,
        child,
        SphericalJointBuilder::new()
            .local_anchor1(Vector::new(0.0, 0.0, 0.0))
            .local_anchor2(Vector::new(-1.0, 0.0, 0.0))
            .limits(JointAxis::AngZ, [-LIMIT, LIMIT]),
    );

    let mut lowest: Real = 0.0;
    for _ in 0..600 {
        world.step();
        lowest = lowest.min(world.bodies[child].translation().y);
    }

    // Gravity pulls the child down until the `AngZ` limit stops it, so the
    // child center settles at about `y = -sin(LIMIT)` (~ -0.30):
    //  * if the limit is ignored, it keeps swinging down towards y = -1.0;
    //  * if the child somehow never swings, it stays at y = 0.
    // Bracketing `lowest` around the expected stop rejects both failure modes.
    let max_drop = -LIMIT.sin();
    assert!(
        lowest > max_drop - 0.15,
        "ball joint ignored its angular limit: child dropped to y = {lowest}, \
         but the {LIMIT} rad limit should keep it above {max_drop}"
    );
    assert!(
        lowest < max_drop + 0.1,
        "child never reached the joint limit (y = {lowest}); the test is not \
         exercising the limit"
    );
}

/// Companion to [`issue_906_insert_multibody_joint_between_steps`], covering the
/// variant that motivated the question in issue #906's discussion: attaching a
/// body with a **fixed** multibody joint (`joint.ndofs() == 0`) to the
/// **non-dynamic root** of an existing multibody, between steps.
///
/// This is the same `Multibody::append()` underflow as #906 — the old
/// `self.velocities.len() - rhs_root_ndofs` subtraction underflows whenever
/// `self` has a fixed (small-`ndofs`) root, regardless of the connecting
/// joint's dof count. The fix's per-link arithmetic is underflow-free for any
/// `joint.ndofs()`, including 0.
#[test]
#[cfg(feature = "dim3")]
fn issue_906_append_fixed_joint_to_fixed_multibody_root() {
    let mut world = PhysicsWorld::new();

    // A multibody with a fixed (non-dynamic) root and one revolute child.
    let (root, _) = world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );
    let (child, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, -2.0, 0.0))
            .additional_mass(1.0),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );
    world.insert_multibody_joint(
        root,
        child,
        RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
            .local_anchor1(Vector::new(0.0, -2.0, 0.0)),
    );

    // Step once so `update_root_type` recognizes the root as non-dynamic.
    world.step();

    // Attach a new body to the (now non-dynamic) root with a *fixed* joint.
    let (attached, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(2.0, 0.0, 0.0))
            .additional_mass(1.0),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );
    let handle = world.insert_multibody_joint(
        root,
        attached,
        FixedJointBuilder::new().local_anchor1(Vector::new(2.0, 0.0, 0.0)),
    );
    assert!(
        handle.is_some(),
        "attaching a fixed joint to a fixed multibody root should be allowed"
    );

    let start = world.bodies[attached].translation();
    for _ in 0..60 {
        world.step();
    }
    let end = world.bodies[attached].translation();

    // A fixed joint to a fixed root contributes zero dofs: the attached body
    // must stay rigidly in place despite gravity.
    let drift = (end.x - start.x).abs() + (end.y - start.y).abs() + (end.z - start.z).abs();
    assert!(
        drift < 1.0e-3,
        "body fixed-jointed to a fixed root should not move, but drifted by {drift}"
    );
    // The pre-existing revolute branch must still simulate.
    assert!(world.bodies[child].translation().y.is_finite());
    assert_eq!(world.multibody_joints.iter().count(), 2);
}

/// Regression test for an issue #907 variant uncovered while reviewing that
/// fix. The #907 contact-side fix only treats a multibody's *fixed root* as a
/// plain fixed body — but a body attached to that root (here with a fixed
/// joint) is a *non-root link*, so it is still handled as a multibody link.
///
/// Such a link sits in a branch joined to the rest of the multibody only
/// *through* the fixed root. Island traversals never cross fixed bodies, so the
/// branch used to land in a separate island from the multibody's solver
/// representative — and a contact against it hit the same empty
/// `generic_solver_vels` out-of-bounds as #907. The fix keeps every link of a
/// multibody in a single island (island-merge anchoring in `do_insert` +
/// whole-multibody island traversal in `push_linked_bodies`).
#[test]
#[cfg(feature = "dim3")]
fn issue_907_contact_with_branch_off_fixed_root() {
    let mut world = PhysicsWorld::new();

    // A multibody with a fixed root and a revolute child.
    let (root, _) = world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );
    let (child, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, -3.0, 0.0))
            .additional_mass(1.0),
        ColliderBuilder::ball(0.5),
    );
    world.insert_multibody_joint(
        root,
        child,
        RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
            .local_anchor1(Vector::new(0.0, -3.0, 0.0)),
    );

    // Step once so the root is recognized as non-dynamic.
    world.step();

    // A second branch off the (non-dynamic) root, far from `child` — it lands
    // in a different island unless the whole multibody is kept together.
    let (branch, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(5.0, 0.0, 0.0))
            .additional_mass(1.0),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5),
    );
    world.insert_multibody_joint(
        root,
        branch,
        FixedJointBuilder::new().local_anchor1(Vector::new(5.0, 0.0, 0.0)),
    );

    // A free body that falls onto `branch`: the contact used to crash the
    // velocity solver's warm-start with an out-of-bounds matrix slice.
    let (free, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(5.0, 3.0, 0.0))
            .additional_mass(1.0),
        ColliderBuilder::cuboid(0.3, 0.3, 0.3),
    );

    for _ in 0..200 {
        world.step();
    }

    // The free body must come to rest on top of `branch` (`branch` top at
    // y = 0.5, free half-height 0.3 → free center settles around y = 0.8).
    let free_y = world.bodies[free].translation().y;
    assert!(
        (free_y - 0.8).abs() < 0.15,
        "free body should rest on the fixed-jointed branch, but settled at y = {free_y}"
    );
    assert!(world.bodies[child].translation().y.is_finite());
}

/// Builds a one-link pendulum hinged at the origin, its rod along +X so gravity
/// torques the hinge, steps it through the start of its descent, and returns
/// how far the link's centre fell.
///
/// The horizon is deliberately short: a free pendulum reaches the bottom in
/// roughly half a second and swings back up, so drop-at-a-fixed-time only
/// decreases with friction while every case is still descending.
#[cfg(feature = "dim3")]
fn friction_pendulum_drop(friction: Real) -> Real {
    const LINK_LEN: Real = 1.0;
    let mut world = PhysicsWorld::new();

    let root = world.insert_body(RigidBodyBuilder::fixed());
    let link = world.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(LINK_LEN, 0.0, 0.0))
            .additional_mass(1.0),
    );

    let handle = world
        .insert_multibody_joint(
            root,
            link,
            RevoluteJointBuilder::new(Vector::new(0.0, 0.0, 1.0))
                .local_anchor1(Vector::ZERO)
                .local_anchor2(Vector::new(-LINK_LEN, 0.0, 0.0)),
        )
        .unwrap();

    if friction > 0.0 {
        let (multibody, _) = world.multibody_joints.get_mut(handle).unwrap();
        multibody.frictions_mut().fill(friction);
    }

    for _ in 0..15 {
        world.step();
    }
    -world.bodies[link].translation().y
}

/// Joint dry friction (MuJoCo `frictionloss`).
#[test]
#[cfg(feature = "dim3")]
fn joint_friction_bounds_the_joint_force() {
    // Gravity torque about the hinge for a 1 kg rod of length 1 m.
    let gravity_torque = 1.0 * 9.81 * 1.0;

    let free = friction_pendulum_drop(0.0);
    let weak = friction_pendulum_drop(0.25 * gravity_torque);
    let locked = friction_pendulum_drop(4.0 * gravity_torque);
    let extreme = friction_pendulum_drop(1000.0 * gravity_torque);

    assert!(free > 0.05, "frictionless pendulum should fall: {free}");
    assert!(
        weak < free * 0.9,
        "sub-gravity friction should slow the fall: {weak} vs {free}"
    );
    assert!(
        locked.abs() < 1.0e-4,
        "friction above the gravity torque should hold the joint at rest: {locked}"
    );
    assert!(
        extreme.abs() < 1.0e-4,
        "an oversized friction bound must stay inert, not chatter: {extreme}"
    );
}

/// A link hanging off a fixed root by a spherical joint whose swing is held to a cone of
/// `cone_deg` about the joint's X axis and whose twist about it is held to `twist_deg` either
/// way, with gravity pulling it down and to the side so it swings toward the cone's edge
/// between the two coupled axes. Returns the largest swing and the largest twist, in degrees,
/// the link reached over `steps` steps.
#[cfg(feature = "dim3")]
fn spherical_link_extremes(
    cone_deg: Real,
    twist_deg: Real,
    spin: Real,
    steps: usize,
) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    let g = 9.81 / core::f32::consts::SQRT_2 as Real;
    world.gravity = Vector::new(0.0, -g, -g);
    let root = world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 5.0, 0.0)));
    let (link, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(1.0, 5.0, 0.0))
            .angvel(Vector::new(spin, 0.0, 0.0))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.5, 0.1, 0.1),
    );
    let cone = cone_deg.to_radians();
    let twist = twist_deg.to_radians();
    let mut joint: GenericJoint = SphericalJointBuilder::new()
        .local_anchor1(Vector::ZERO)
        .local_anchor2(Vector::new(-1.0, 0.0, 0.0))
        .contacts_enabled(false)
        .build()
        .into();
    joint.coupled_axes |= JointAxesMask::ANG_Y | JointAxesMask::ANG_Z;
    joint.set_limits(JointAxis::AngY, [-cone, cone]);
    joint.set_limits(JointAxis::AngZ, [-cone, cone]);
    joint.set_limits(JointAxis::AngX, [-twist, twist]);
    world.insert_multibody_joint(root, link, joint).unwrap();

    let (mut max_swing, mut max_twist): (Real, Real) = (0.0, 0.0);
    for _ in 0..steps {
        world.step();
        let rot = *world.bodies[link].rotation();
        let x2 = rot * Vector::X;
        max_swing = max_swing.max(Vector::X.dot(x2).clamp(-1.0, 1.0).acos().to_degrees());
        // The twist about the link's own X: its rotation with the shortest arc taking X to
        // where it points taken off.
        let swing_rot = Rotation::from_rotation_arc(Vector::X, x2);
        let mut twist_rot = swing_rot.inverse() * rot;
        if twist_rot.w < 0.0 {
            twist_rot = -twist_rot;
        }
        max_twist = max_twist.max((2.0 * twist_rot.x.atan2(twist_rot.w)).to_degrees().abs());
    }
    (max_swing, max_twist)
}

/// A spherical multibody joint's swing cone holds the angle between its two frames' X axes:
/// a link pulled toward the cone's edge between its two coupled axes stops at the cone, where
/// two separate limits on those axes would let it reach the corner of their square, 41° for a
/// 30° cone.
#[test]
#[cfg(feature = "dim3")]
fn a_spherical_multibody_joint_holds_its_swing_cone() {
    let (swing, _) = spherical_link_extremes(30.0, 180.0, 0.0, 240);
    assert!(swing < 31.5, "the link swings {swing}° on a 30° cone");
    assert!(
        swing > 25.0,
        "the link reaches the cone: it swings {swing}°"
    );
}

/// A spherical multibody joint's twist range holds the turn about its X axis while it swings:
/// a spinning link stops turning at the range's bound.
#[test]
#[cfg(feature = "dim3")]
fn a_spherical_multibody_joint_holds_its_twist_range_while_it_swings() {
    let (swing, twist) = spherical_link_extremes(30.0, 20.0, 6.0, 240);
    assert!(twist < 22.0, "the link twists {twist}° on a 20° range");
    assert!(swing < 31.5, "the link swings {swing}° on a 30° cone");
}

/// A chain of `links` boxes 0.5 m long laid out level from a fixed wall on multibody hinges
/// about Z, each limited to 30 degrees either way, the adjacent links not colliding, the last
/// link `load` kg and the others 1 kg, dropped under gravity for three seconds of `dt` steps.
/// Answers the widest angle, in degrees, any joint stood at, or `None` once a pose stopped
/// being finite.
#[cfg(feature = "dim3")]
fn hinge_chain_widest_angle(links: usize, load: Real, dt: Real) -> Option<Real> {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = dt;
    let wall =
        world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(0.0, 20.0, 0.0)));
    let limit = 30.0 as Real * core::f32::consts::PI as Real / 180.0;
    let mut above = wall;
    let mut chain = Vec::new();
    for i in 0..links {
        let mass = if i + 1 == links { load } else { 1.0 };
        let (link, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.5 * i as Real + 0.25, 20.0, 0.0))
                .can_sleep(false),
            ColliderBuilder::cuboid(0.25, 0.05, 0.05).mass(mass),
        );
        let joint = RevoluteJointBuilder::new(Vector::Z)
            .local_anchor1(Vector::new(if i == 0 { 0.0 } else { 0.25 }, 0.0, 0.0))
            .local_anchor2(Vector::new(-0.25, 0.0, 0.0))
            .limits([-limit, limit])
            .contacts_enabled(false);
        world.insert_multibody_joint(above, link, joint).unwrap();
        chain.push((link, above));
        above = link;
    }
    let mut widest: Real = 0.0;
    for _ in 0..(3.0 / dt) as usize {
        world.step();
        for (link, parent) in &chain {
            let turn = world.bodies[*parent].rotation().inverse() * *world.bodies[*link].rotation();
            let angle =
                turn.angle_between(Rotation::IDENTITY) * 180.0 / core::f32::consts::PI as Real;
            if !angle.is_finite() {
                return None;
            }
            widest = widest.max(angle);
        }
    }
    Some(widest)
}

/// Several joints of a multibody at their limits at once: a level chain of limited hinges
/// whipping down under a heavy end link. The limits used to hold only once a link had
/// passed one, pushing it back by the whole violation each step; with three or more joints
/// pinned at once the pushes fed each other until the chain's velocities ran off to
/// infinity within a second.
#[test]
#[cfg(feature = "dim3")]
fn a_multibody_chain_of_limited_hinges_keeps_its_limits_under_a_heavy_load() {
    for dt in [1.0 / 60.0, 1.0 / 30.0] {
        for load in [1.0, 10.0, 20.0] {
            let widest = hinge_chain_widest_angle(10, load, dt);
            assert!(
                widest.is_some_and(|widest| widest < 33.0),
                "a 30 degree limit in a chain with a {load} kg end at dt {dt} reached {widest:?} degrees"
            );
        }
    }
}

/// One limited hinge swung into its stop by gravity: the link stops at the limit.
#[test]
#[cfg(feature = "dim3")]
fn a_multibody_hinge_swung_into_its_stop_stops_at_it() {
    for dt in [1.0 / 60.0, 1.0 / 30.0] {
        let widest = hinge_chain_widest_angle(1, 5.0, dt).expect("a finite pose");
        assert!(
            widest < 30.05,
            "a 30 degree limit at dt {dt} reached {widest} degrees"
        );
    }
}

/// A chain of `links` boxes 0.5 m long hanging straight down from a fixed anchor on multibody
/// spherical joints, the last link pushed sideways by an impulse of 2 N·s through
/// `apply_link_impulse` before the multibody ever stepped, then stepped once. Answers the
/// pushed link's velocity and the anchor-to-end length over the chain's length.
#[cfg(feature = "dim3")]
fn pushed_chain(links: usize, root: RigidBodyBuilder) -> (Vector, Real) {
    let mut world = PhysicsWorld::new();
    let anchor = world.insert_body(root.translation(Vector::new(0.0, 20.0, 0.0)));
    let mut above = anchor;
    let mut last = anchor;
    for i in 0..links {
        let (link, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 20.0 - 0.5 * i as Real - 0.25, 0.0))
                .can_sleep(false),
            ColliderBuilder::cuboid(0.05, 0.25, 0.05).mass(1.0),
        );
        let joint = SphericalJointBuilder::new()
            .local_anchor1(Vector::new(0.0, if i == 0 { 0.0 } else { -0.25 }, 0.0))
            .local_anchor2(Vector::new(0.0, 0.25, 0.0))
            .contacts_enabled(false);
        world.insert_multibody_joint(above, link, joint).unwrap();
        above = link;
        last = link;
    }
    let link = world
        .multibody_joints
        .rigid_body_link(last)
        .copied()
        .unwrap();
    let multibody = world
        .multibody_joints
        .get_multibody_mut(link.multibody)
        .unwrap();
    multibody.apply_link_impulse(
        &world.bodies,
        link.id,
        Vector::new(2.0, 0.0, 0.0),
        Vector::ZERO,
    );
    world.step();
    let end = world.bodies[last].position() * Vector::new(0.0, -0.25, 0.0);
    let length = (end - world.bodies[anchor].translation()).length() / (0.5 * links as Real);
    (world.bodies[last].linvel(), length)
}

/// An impulse on a link moves the link and the chain it hangs in, before the multibody ever
/// stepped, with a fixed root and with a kinematic one, and the chain does not stretch.
#[test]
#[cfg(feature = "dim3")]
fn an_impulse_on_a_multibody_link_moves_the_link_and_its_chain() {
    for root in [
        RigidBodyBuilder::fixed(),
        RigidBodyBuilder::kinematic_position_based(),
    ] {
        let (velocity, length) = pushed_chain(4, root);
        assert!(velocity.x > 0.3, "the pushed link moves at {velocity:?}");
        assert!(
            length < 1.0 + 1.0e-3,
            "the chain stretched to {length} of its length"
        );
    }
}

/// A multibody whose links strike the ground steps to the same bits however
/// often its links' bodies are borrowed mutably between steps. Borrowing a body
/// mutably marks it modified, and the narrow phase then reconciles its pairs
/// with the solver contact graph, which appends and swap-removes the generic
/// (multibody-involved) manifolds as they qualify: the solve order of those
/// contacts followed that history, so writing a link's own velocity back to it
/// (or only touching it) on some steps and not others parted the result.
#[test]
#[cfg(feature = "dim3")]
fn multibody_contacts_step_the_same_however_often_their_links_are_touched() {
    let run = |every: usize| {
        let mut world = PhysicsWorld::new();
        world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
            ColliderBuilder::cuboid(10.0, 0.5, 10.0),
        );
        world.insert(
            RigidBodyBuilder::fixed()
                .translation(Vector::new(0.0, 0.2, 0.6))
                .rotation(Vector::new(0.3, 0.0, 0.0)),
            ColliderBuilder::cuboid(2.0, 0.2, 0.3),
        );
        let mut links = Vec::new();
        let mut above = None;
        for i in 0..6 {
            let (link, _) = world.insert(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(0.3 * i as Real, 1.5, 0.3 * i as Real))
                    .linvel(Vector::new(0.0, 0.0, 2.0))
                    .angvel(Vector::new(0.4, 1.0, 0.0))
                    .can_sleep(false),
                ColliderBuilder::cuboid(0.15, 0.08, 0.08),
            );
            if let Some(parent) = above {
                let joint = SphericalJointBuilder::new()
                    .local_anchor1(Vector::new(0.15, 0.0, 0.15))
                    .local_anchor2(Vector::new(-0.15, 0.0, -0.15))
                    .limits(JointAxis::AngX, [-0.5, 0.5])
                    .limits(JointAxis::AngY, [-0.5, 0.5])
                    .limits(JointAxis::AngZ, [-0.5, 0.5]);
                world.insert_multibody_joint(parent, link, joint);
            }
            links.push(link);
            above = Some(link);
        }
        for step in 0..150 {
            if every > 0 && step % every == 0 {
                for link in &links {
                    let body = &mut world.bodies[*link];
                    let v = body.linvel();
                    body.set_linvel(v, true);
                }
            }
            world.step();
        }
        links
            .iter()
            .map(|link| {
                let body = &world.bodies[*link];
                let p = body.translation();
                let q = body.rotation();
                [p.x, p.y, p.z, q.x, q.y, q.z, q.w].map(Real::to_bits)
            })
            .collect::<Vec<_>>()
    };
    let untouched = run(0);
    assert_eq!(untouched, run(1), "touched before every step");
    assert_eq!(untouched, run(2), "touched before every second step");
    assert_eq!(untouched, run(3), "touched before every third step");
}

/// A multibody striking the ground steps to the same bits whether or not a body
/// far from it rested on the same ground and was removed while it stepped.
/// Removing that body's contact moved the contact graph's edges, so a solve
/// order read off edge indices changed the multibody's result.
#[test]
#[cfg(feature = "dim3")]
fn multibody_contacts_step_the_same_whatever_contacts_elsewhere_were_removed() {
    let run = |aside: bool| {
        let mut world = PhysicsWorld::new();
        world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
            ColliderBuilder::cuboid(50.0, 0.5, 50.0),
        );
        let others: Vec<_> = (0..3)
            .map(|i| {
                world
                    .insert(
                        RigidBodyBuilder::dynamic()
                            .translation(Vector::new(30.0 + 2.0 * i as Real, 0.3, 30.0))
                            .can_sleep(false),
                        ColliderBuilder::cuboid(0.3, 0.3, 0.3),
                    )
                    .0
            })
            .collect();
        let mut links = Vec::new();
        let mut above = None;
        for i in 0..6 {
            let (link, _) = world.insert(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(0.3 * i as Real, 1.0, 0.3 * i as Real))
                    .linvel(Vector::new(0.0, 0.0, 2.0))
                    .angvel(Vector::new(0.4, 1.0, 0.0))
                    .can_sleep(false),
                ColliderBuilder::cuboid(0.15, 0.08, 0.08),
            );
            if let Some(parent) = above {
                let joint = SphericalJointBuilder::new()
                    .local_anchor1(Vector::new(0.15, 0.0, 0.15))
                    .local_anchor2(Vector::new(-0.15, 0.0, -0.15))
                    .limits(JointAxis::AngX, [-0.5, 0.5])
                    .limits(JointAxis::AngY, [-0.5, 0.5])
                    .limits(JointAxis::AngZ, [-0.5, 0.5]);
                world.insert_multibody_joint(parent, link, joint);
            }
            links.push(link);
            above = Some(link);
        }
        for step in 0..150 {
            if aside && step == 40 {
                world.remove_body(others[0]);
            }
            world.step();
        }
        links
            .iter()
            .map(|link| {
                let body = &world.bodies[*link];
                let p = body.translation();
                let q = body.rotation();
                [p.x, p.y, p.z, q.x, q.y, q.z, q.w].map(Real::to_bits)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(false), run(true), "a body removed elsewhere");
}

/// How a chain of [`level_coned_chain`] fell.
#[cfg(feature = "dim3")]
struct ConedChainFall {
    /// The step a link's pose first read not finite at, if one did.
    not_finite_at: Option<usize>,
    /// The widest any joint swung before that, in degrees.
    widest_swing: Real,
    /// The most the links' kinetic and potential energy stood above where it started, in joules.
    energy_gain: Real,
}

/// The kinetic and potential energy of `links` under `gravity`.
#[cfg(feature = "dim3")]
fn links_energy(bodies: &RigidBodySet, links: &[RigidBodyHandle], gravity: Vector) -> Real {
    links
        .iter()
        .map(|link| {
            let body = &bodies[*link];
            let (linvel, angvel) = (body.linvel(), body.angvel());
            let inertia = body.mass_properties().local_mprops;
            let principal_axes = *body.rotation() * inertia.principal_inertia_local_frame;
            let local = principal_axes.inverse() * angvel;
            0.5 * body.mass() * linvel.dot(linvel)
                + 0.5 * angvel.dot(principal_axes * (inertia.principal_inertia() * local))
                - body.mass() * gravity.dot(body.mass_properties().world_com)
        })
        .sum()
}

/// A level chain of eight thin rods (0.4 m by 0.1 m along X, 1 kg each, 10 kg the last)
/// joined by spherical joints whose swing is held to a 40° cone about the joint frames' X
/// axis, which `frame` turns from the rods' length, hung from a fixed root at `at` and let fall
/// for 5 s at `rate` steps a second under a gravity pulled `side` m/s² along Z.
#[cfg(feature = "dim3")]
fn level_coned_chain(at: Vector, rate: Real, frame: Rotation, side: Real) -> ConedChainFall {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.gravity.z = side;
    let (root, _) = world.insert(
        RigidBodyBuilder::fixed().translation(at),
        ColliderBuilder::cuboid(0.2, 0.2, 0.2),
    );
    let cone = 40.0_f32.to_radians() as Real;
    let mut above = root;
    let mut links = Vec::new();
    for i in 0..8 {
        let mass = if i == 7 { 10.0 } else { 1.0 };
        let (link, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(at + Vector::new(0.45 * (i as Real + 1.0), 0.0, 0.0))
                .allow_fast_rotation(true)
                .can_sleep(false),
            ColliderBuilder::cuboid(0.2, 0.05, 0.05).mass(mass),
        );
        let mut joint: GenericJoint = SphericalJointBuilder::new()
            .local_anchor1(Vector::new(if i == 0 { 0.0 } else { 0.225 }, 0.0, 0.0))
            .local_anchor2(Vector::new(-0.225, 0.0, 0.0))
            .contacts_enabled(false)
            .build()
            .into();
        joint.local_frame1.rotation = frame;
        joint.local_frame2.rotation = frame;
        joint.coupled_axes |= JointAxesMask::ANG_Y | JointAxesMask::ANG_Z;
        joint.set_limits(JointAxis::AngY, [-cone, cone]);
        joint.set_limits(JointAxis::AngZ, [-cone, cone]);
        world.insert_multibody_joint(above, link, joint).unwrap();
        links.push(link);
        above = link;
    }
    let start = links_energy(&world.bodies, &links, world.gravity);
    let mut fall = ConedChainFall {
        not_finite_at: None,
        widest_swing: 0.0,
        energy_gain: 0.0,
    };
    for step in 0..(5.0 * rate) as usize {
        world.step();
        let energy = links_energy(&world.bodies, &links, world.gravity);
        fall.energy_gain = fall.energy_gain.max(energy - start);
        for (k, link) in links.iter().enumerate() {
            let pose = *world.bodies[*link].position();
            if !pose.translation.is_finite() || !pose.rotation.is_finite() {
                fall.not_finite_at = Some(step);
                return fall;
            }
            let parent = if k == 0 { root } else { links[k - 1] };
            let rel = world.bodies[parent].rotation().inverse() * pose.rotation;
            let axis = frame * Vector::X;
            let swing = axis.dot(rel * axis).clamp(-1.0, 1.0).acos().to_degrees();
            fall.widest_swing = fall.widest_swing.max(swing);
        }
    }
    fall
}

/// A chain lying on the swing cones of its joints under a heavy end load stays finite and
/// keeps its cones, at 60 and 30 steps a second, wherever it hangs, with the cones about the
/// rods' length and across it. A cone's bound is curved, so the cone holds a link pressed on
/// it against sliding round it with a stiffness the load sets; met only at the start of each
/// substep, that stiffness swung the light links between two loaded joints round their cones
/// wider each substep, the other way each time, until the chain's state was NaN within a
/// second of falling.
#[test]
#[cfg(feature = "dim3")]
fn a_level_chain_on_its_swing_cones_under_a_heavy_end_stays_finite() {
    let mut failed = Vec::new();
    let mut widest: Real = 0.0;
    let across = Rotation::from_axis_angle(Vector::Z, core::f32::consts::FRAC_PI_2 as Real);
    for (name, frame) in [("along", Rotation::IDENTITY), ("across", across)] {
        for rate in [60.0, 30.0] {
            for i in 0..8 {
                let at = Vector::new(-30.0 + 5.0 * i as Real, 8.0 + (i % 3) as Real, 0.0);
                // A breath of sideways pull, so the chain does not fall in one exact plane.
                let fall = level_coned_chain(at, rate, frame, 0.01);
                widest = widest.max(fall.widest_swing);
                if let Some(step) = fall.not_finite_at {
                    failed.push(alloc::format!(
                        "cone {name} the rods, {rate} Hz at {at:?}: step {step}"
                    ));
                }
            }
        }
    }
    assert!(failed.is_empty(), "the chain went not finite: {failed:?}");
    assert!(widest < 43.0, "a joint swung {widest}° on a 40° cone");
}

/// A chain free to turn about the rods' width whips round its cones at tens of radians a
/// second, and a substep then turns a joint by a large fraction of a radian. The energy of the
/// move of a substep and of the Coriolis accelerations that follow it is balanced over the
/// whole substep, so the chain stays finite and on its cones and within 1000 J of the 1300 J it
/// fell with: left unbalanced, each substep gained on the last, thousands of joules within
/// seconds and NaN in some placements.
#[test]
#[cfg(feature = "dim3")]
fn a_chain_whipping_round_its_swing_cones_holds_its_energy() {
    let across = Rotation::from_axis_angle(Vector::Z, core::f32::consts::FRAC_PI_2 as Real);
    let mut failed = Vec::new();
    let mut widest: Real = 0.0;
    let mut gained: Real = 0.0;
    for side in [0.1, 0.3, 1.0] {
        for rate in [30.0, 60.0] {
            for i in 0..12 {
                let at = Vector::new(-30.0 + 5.0 * i as Real, 8.0, 0.0);
                let fall = level_coned_chain(at, rate, across, side);
                widest = widest.max(fall.widest_swing);
                gained = gained.max(fall.energy_gain);
                if let Some(step) = fall.not_finite_at {
                    failed.push(alloc::format!(
                        "{side} m/s², {rate} Hz at {at:?}: step {step}"
                    ));
                }
            }
        }
    }
    assert!(failed.is_empty(), "the chain went not finite: {failed:?}");
    assert!(widest < 43.0, "a joint swung {widest}° on a 40° cone");
    assert!(gained < 1000.0, "the chain gained {gained} J");
}

/// A ragdoll of eleven boxes on multibody joints, each joint built the way an engine joins a
/// body to its parent: pelvis, chest and head on spherical joints with swing cones and twist
/// ranges, ball shoulders and hips, hinge elbows and knees on 0 to 140° ranges. It stands
/// beside a flight of eight fixed stairs whose foot is at `at`, its pelvis' centre `height` +
/// 0.2 m above that foot, so that for a `height` below 2.3 m its shins start inside the top
/// step, which is 1.6 m high. It is pushed by 60 N·s on its
/// chest one step in and stepped for `seconds` at `rate` steps a second in `substeps` substeps.
/// Answers the step a body's pose first read not finite at, if one did, and how far, in metres,
/// any body rose above the highest point the ragdoll was spawned at.
#[cfg(feature = "dim3")]
fn ragdoll_by_stairs(
    at: Vector,
    height: Real,
    rate: Real,
    substeps: usize,
    seconds: Real,
) -> (Option<usize>, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    world.integration_parameters.normalized_max_linear_velocity = Real::MAX;
    for step in 0..8 {
        let top = 1.6 - 0.2 * step as Real;
        world.insert(
            RigidBodyBuilder::fixed()
                .translation(at + Vector::new(0.0, top * 0.5, 0.3 * step as Real + 0.15)),
            ColliderBuilder::cuboid(1.5, top * 0.5, 0.15),
        );
    }
    let base = at + Vector::new(0.0, height, 0.15);
    let mut made = Vec::new();
    let mut part = |world: &mut PhysicsWorld, offset: Vector, half: Vector, mass: Real| {
        let (body, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(base + offset)
                .allow_fast_rotation(true)
                .can_sleep(false),
            ColliderBuilder::cuboid(half.x, half.y, half.z).mass(mass),
        );
        made.push(body);
        body
    };
    // Both frames' X axes along `axis`, the way the engine completes a joint's frames.
    let frame = |axis: Vector| Rotation::from_rotation_arc(Vector::X, axis);
    let ball = |axis: Vector, on_parent: Vector, on_child: Vector, swing: Real, twist: Real| {
        let mut joint: GenericJoint = SphericalJointBuilder::new()
            .local_anchor1(on_parent)
            .local_anchor2(on_child)
            .contacts_enabled(false)
            .build()
            .into();
        joint.local_frame1.rotation = frame(axis);
        joint.local_frame2.rotation = frame(axis);
        let (swing, twist) = (swing.to_radians(), twist.to_radians());
        joint.coupled_axes |= JointAxesMask::ANG_Y | JointAxesMask::ANG_Z;
        joint.set_limits(JointAxis::AngY, [-swing, swing]);
        joint.set_limits(JointAxis::AngZ, [-swing, swing]);
        joint.set_limits(JointAxis::AngX, [-twist, twist]);
        joint
    };
    let hinge = |axis: Vector, on_parent: Vector, on_child: Vector| {
        let mut joint: GenericJoint = RevoluteJointBuilder::new(Vector::X)
            .local_anchor1(on_parent)
            .local_anchor2(on_child)
            .contacts_enabled(false)
            .build()
            .into();
        joint.local_frame1.rotation = frame(axis);
        joint.local_frame2.rotation = frame(axis);
        joint.set_limits(JointAxis::AngX, [0.0, 140.0_f32.to_radians() as Real]);
        joint
    };
    let v = Vector::new;
    let pelvis = part(&mut world, v(0.0, 0.2, 0.0), v(0.15, 0.1, 0.1), 10.0);
    let chest = part(&mut world, v(0.0, 0.5, 0.0), v(0.15, 0.2, 0.1), 15.0);
    let head = part(&mut world, v(0.0, 0.8, 0.0), v(0.1, 0.1, 0.1), 5.0);
    let joints = [
        (
            pelvis,
            chest,
            ball(Vector::Y, v(0.0, 0.1, 0.0), v(0.0, -0.2, 0.0), 30.0, 20.0),
        ),
        (
            chest,
            head,
            ball(Vector::Y, v(0.0, 0.2, 0.0), v(0.0, -0.1, 0.0), 40.0, 30.0),
        ),
    ];
    for (parent, child, joint) in joints {
        world.insert_multibody_joint(parent, child, joint).unwrap();
    }
    for side in [-1.0 as Real, 1.0] {
        let upper = part(
            &mut world,
            v(0.3 * side, 0.65, 0.0),
            v(0.15, 0.05, 0.05),
            2.0,
        );
        let fore = part(
            &mut world,
            v(0.6 * side, 0.65, 0.0),
            v(0.15, 0.04, 0.04),
            1.5,
        );
        let thigh = part(
            &mut world,
            v(0.1 * side, -0.1, 0.0),
            v(0.06, 0.2, 0.06),
            6.0,
        );
        let shin = part(
            &mut world,
            v(0.1 * side, -0.5, 0.0),
            v(0.05, 0.2, 0.05),
            4.0,
        );
        let shoulder = ball(
            v(side, 0.0, 0.0),
            v(0.15 * side, 0.15, 0.0),
            v(-0.15 * side, 0.0, 0.0),
            70.0,
            40.0,
        );
        let elbow = hinge(
            Vector::Z,
            v(0.15 * side, 0.0, 0.0),
            v(-0.15 * side, 0.0, 0.0),
        );
        let hip = ball(
            -Vector::Y,
            v(0.1 * side, -0.1, 0.0),
            v(0.0, 0.2, 0.0),
            60.0,
            30.0,
        );
        let knee = hinge(Vector::X, v(0.0, -0.2, 0.0), v(0.0, 0.2, 0.0));
        let limbs = [
            (chest, upper, shoulder),
            (upper, fore, elbow),
            (pelvis, thigh, hip),
            (thigh, shin, knee),
        ];
        for (parent, child, joint) in limbs {
            world.insert_multibody_joint(parent, child, joint).unwrap();
        }
    }
    let top = base.y + 0.9;
    let mut rose: Real = 0.0;
    for step in 0..(seconds * rate) as usize {
        if step == 1 {
            let link = *world.multibody_joints.rigid_body_link(chest).unwrap();
            let mb = world
                .multibody_joints
                .get_multibody_mut(link.multibody)
                .unwrap();
            mb.apply_link_impulse(
                &world.bodies,
                link.id,
                Vector::new(0.0, 0.0, 60.0),
                AngVector::ZERO,
            );
        }
        world.step();
        for body in &made {
            let pose = world.bodies[*body].position();
            if !pose.translation.is_finite() || !pose.rotation.is_finite() {
                return (Some(step), rose);
            }
            rose = rose.max(pose.translation.y - top);
        }
    }
    (None, rose)
}

/// A ragdoll on multibody joints spawned with its legs deep inside a fixed step stays finite and
/// rises no more than half a metre above where it was spawned, wherever it stands, at 30, 60 and
/// 120 steps a second and at 30 in 16 substeps. Its thighs and shins are pushed out of the step
/// opposite ways and their contacts held hundreds of N·s against one another through the knees;
/// carried into the next step, where the narrow phase had replaced some of their points, they
/// threw the legs with the impulse the replaced points gave up, and solved with the contact
/// jacobians of the configuration the step began at, the push drove the links along directions
/// the contacts no longer pointed. The ragdoll went NaN within a second at every rate. On impulse
/// joints the same ragdoll rises at most a third of a metre.
#[test]
#[cfg(feature = "dim3")]
fn a_ragdoll_spawned_with_its_legs_inside_a_step_stays_finite_and_in_place() {
    let mut failed = Vec::new();
    let mut highest: Real = 0.0;
    for (rate, substeps) in [(30.0 as Real, 4), (60.0, 4), (120.0, 4), (30.0, 16)] {
        for height in [1.5 as Real, 1.6] {
            for i in 0..4 {
                let at = Vector::new(
                    -20.0 + 5.0 * i as Real,
                    0.2 * i as Real,
                    3.0 * (i % 2) as Real,
                );
                let (nan, rose) = ragdoll_by_stairs(at, height, rate, substeps, 2.0);
                highest = highest.max(rose);
                if let Some(step) = nan {
                    failed.push(alloc::format!(
                        "{rate} Hz in {substeps} substeps, `height` {height}, at {at:?}: step {step}"
                    ));
                }
            }
        }
    }
    assert!(failed.is_empty(), "the ragdoll went not finite: {failed:?}");
    assert!(
        highest < 0.5,
        "a body rose {highest} m above where the ragdoll was spawned"
    );
}

/// Five boxes stacked on a multibody's dynamic root, a slab lying on the ground with a link
/// hinged to its side, stepped for `seconds` at `rate` steps a second in four substeps.
/// Answers how far the top box moved over the last two seconds and how far it stands below
/// where the stack's own heights put it, in metres.
#[cfg(feature = "dim3")]
fn stack_on_a_multibody(rate: Real, seconds: Real) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = 4;
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
        ColliderBuilder::cuboid(20.0, 0.5, 20.0),
    );
    let (slab, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.1, 0.0))
            .can_sleep(false),
        ColliderBuilder::cuboid(1.0, 0.1, 1.0).mass(10.0),
    );
    let (side, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.1, 1.2))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.1, 0.1, 0.1).mass(2.0),
    );
    let hinge: GenericJoint = RevoluteJointBuilder::new(Vector::X)
        .local_anchor1(Vector::new(0.0, 0.0, 1.1))
        .local_anchor2(Vector::new(0.0, 0.0, -0.1))
        .contacts_enabled(false)
        .build()
        .into();
    world.insert_multibody_joint(slab, side, hinge).unwrap();
    let mut top = slab;
    for i in 0..5 {
        let (body, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 0.45 + 0.5 * i as Real, 0.0))
                .can_sleep(false),
            ColliderBuilder::cuboid(0.25, 0.25, 0.25).mass(1.0),
        );
        top = body;
    }
    let steps = (seconds * rate) as usize;
    let mut from = Vector::ZERO;
    for step in 0..steps {
        if step + (2.0 * rate) as usize == steps {
            from = world.bodies[top].translation();
        }
        world.step();
    }
    let at = world.bodies[top].translation();
    ((at - from).length(), 0.45 + 0.5 * 4.0 - at.y)
}

/// A stack of boxes resting on a multibody holds still, at 30 and 60 steps a second: its
/// contacts with the multibody open each step with the impulses their points carry from the
/// last, as the contacts between the boxes do. Started from no impulse at every step, the
/// contacts holding the stack up gave way at the first substep of each one, and the top box
/// crept 21 mm in two seconds and stood 23 mm low at 30 steps a second.
#[test]
#[cfg(feature = "dim3")]
fn a_stack_resting_on_a_multibody_holds_still() {
    for rate in [30.0 as Real, 60.0] {
        let (moved, low) = stack_on_a_multibody(rate, 10.0);
        assert!(
            moved < 0.005,
            "at {rate} Hz the top box moved {moved} m in its last two seconds"
        );
        assert!(
            low < 0.01,
            "at {rate} Hz the top box stands {low} m below the stack's height"
        );
    }
}

/// A heavy box (`root_mass` kg, 0.3 m by 0.2 m by 0.2 m) with a 1.5 kg forearm hinged to its end
/// about Z, the hinge held to bend down only (its range -140° to 0°, the arm straight at its
/// stop), dropped from `height` m onto fixed ground tilted `tilt` degrees forearm first, and
/// stepped for 2 s at `rate` steps a second in `substeps` substeps. Answers how far, in degrees,
/// the forearm was ever bent up past its stop.
#[cfg(feature = "dim3")]
fn arm_landing_past_its_stop(
    height: Real,
    tilt: Real,
    rate: Real,
    substeps: usize,
    root_mass: Real,
) -> Real {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
        ColliderBuilder::cuboid(5.0, 0.5, 5.0).friction(0.7),
    );
    let turn = Rotation::from_axis_angle(Vector::Z, -tilt.to_radians());
    let at = Vector::new(0.0, height, 0.0);
    let (root, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(at)
            .rotation(turn.to_scaled_axis())
            .can_sleep(false),
        ColliderBuilder::cuboid(0.15, 0.1, 0.1)
            .mass(root_mass)
            .friction(0.7),
    );
    let (forearm, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(at + turn * Vector::new(0.3, 0.0, 0.0))
            .rotation(turn.to_scaled_axis())
            .can_sleep(false),
        ColliderBuilder::cuboid(0.15, 0.04, 0.04)
            .mass(1.5)
            .friction(0.7),
    );
    let elbow = RevoluteJointBuilder::new(Vector::Z)
        .local_anchor1(Vector::new(0.15, 0.0, 0.0))
        .local_anchor2(Vector::new(-0.15, 0.0, 0.0))
        .limits([-140.0_f32.to_radians() as Real, 0.0])
        .contacts_enabled(false);
    world.insert_multibody_joint(root, forearm, elbow).unwrap();
    let mut past: Real = 0.0;
    for _ in 0..(2.0 * rate) as usize {
        world.step();
        let link = world.multibody_joints.rigid_body_link(forearm).unwrap();
        let angle = world.multibody_joints[link.multibody]
            .link(link.id)
            .unwrap()
            .joint
            .coords[3];
        past = past.max(angle.to_degrees());
    }
    past
}

/// A heavy multibody landing on its light forearm keeps the forearm's hinge at its stop, from
/// every height, tilt and mass tried, at 30, 60 and 120 steps a second, in 2 substeps as in 4.
/// The ground's contact on the forearm changes every coordinate of the multibody, and it was
/// solved after the hinge's limit in each pass with nothing solving the limit again: the forearm
/// was driven up to 29° past its stop as the box came down on it. Solved against each other row
/// by row, the contact and the limit then passed the error between them on the forearm's share of
/// the mass a round, and in 2 substeps a 200 kg box still drove its forearm up to 7° past.
#[test]
#[cfg(feature = "dim3")]
fn a_multibody_landing_on_its_forearm_keeps_the_forearm_at_its_stop() {
    let mut failed = Vec::new();
    for substeps in [2, 4] {
        for rate in [30.0 as Real, 60.0, 120.0] {
            for height in [2.0 as Real, 4.0, 8.0] {
                for tilt in [10.0 as Real, 20.0, 40.0] {
                    for root_mass in [40.0 as Real, 100.0, 200.0] {
                        let past =
                            arm_landing_past_its_stop(height, tilt, rate, substeps, root_mass);
                        if past > 2.0 {
                            failed.push(alloc::format!(
                                "{rate} Hz x {substeps}, {height} m, {tilt}°, {root_mass} kg: \
                                 {past}° past its stop"
                            ));
                        }
                    }
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "the forearm was bent past its stop: {failed:?}"
    );
}

/// A `root_mass` kg box with a 1.5 kg forearm hinged to it, a multibody lying on fixed ground,
/// and a `light_mass` kg box held above the root by an impulse joint (`fixed`, else a ball joint)
/// whose anchor on the light box is `offset` m from its centre along a diagonal, stepped for 4 s
/// at `rate` steps a second in `substeps` substeps. Answers the largest distance between the
/// joint's two anchors and the fastest the light box moved after the first second.
#[cfg(feature = "dim3")]
fn light_body_held_to_a_resting_multibody(
    fixed: bool,
    root_mass: Real,
    light_mass: Real,
    offset: Real,
    rate: Real,
    substeps: usize,
) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
        ColliderBuilder::cuboid(10.0, 0.5, 10.0).friction(0.7),
    );
    let (root, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.11, 0.0))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.15, 0.1, 0.1)
            .mass(root_mass)
            .friction(0.7),
    );
    let (forearm, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.3, 0.11, 0.0))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.15, 0.04, 0.04)
            .mass(1.5)
            .friction(0.7),
    );
    let elbow = RevoluteJointBuilder::new(Vector::Z)
        .local_anchor1(Vector::new(0.15, 0.0, 0.0))
        .local_anchor2(Vector::new(-0.15, 0.0, 0.0))
        .limits([-140.0_f32.to_radians() as Real, 0.0])
        .contacts_enabled(false);
    world.insert_multibody_joint(root, forearm, elbow).unwrap();
    let anchor1 = Vector::new(0.0, 0.3, 0.0);
    let anchor2 = Vector::new(offset, -offset, offset);
    let (light, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.11, 0.0) + anchor1 - anchor2)
            .can_sleep(false),
        ColliderBuilder::cuboid(0.05, 0.05, 0.05)
            .mass(light_mass)
            .friction(0.7),
    );
    if fixed {
        let joint = FixedJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .contacts_enabled(false);
        world.insert_impulse_joint(root, light, joint);
    } else {
        let joint = SphericalJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(anchor2)
            .contacts_enabled(false);
        world.insert_impulse_joint(root, light, joint);
    }
    let (mut apart, mut fastest): (Real, Real) = (0.0, 0.0);
    for step in 0..(4.0 * rate) as usize {
        world.step();
        let held = world.bodies[light].position() * anchor2;
        let holding = world.bodies[root].position() * anchor1;
        let gap = (held - holding).length();
        let speed = world.bodies[light].linvel().length();
        if !(gap.is_finite() && speed.is_finite()) {
            return (Real::INFINITY, Real::INFINITY);
        }
        apart = apart.max(gap);
        if step as Real >= rate {
            fastest = fastest.max(speed);
        }
    }
    (apart, fastest)
}

/// A light body held by an impulse joint to a multibody resting on the ground stays at its
/// anchor, at 30 and 60 steps a second in 2 substeps and at 60 in 4. Each of the ground's
/// contacts on the multibody is solved as one block with the joint rows of the multibody that
/// bear an impulse, and the joint holding the light body is one of them: solved in that block
/// through the multibody alone, with the light body left out of how its rows push on one another
/// and on the contact, a 0.1 kg box held 0.87 m from its centre to a 50 kg multibody was thrown
/// off at 6.6 m/s, its anchor 3.6 m from where the joint holds it.
#[test]
#[cfg(feature = "dim3")]
fn a_light_body_held_to_a_resting_multibody_stays_at_its_anchor() {
    let mut failed = Vec::new();
    for fixed in [true, false] {
        for (root_mass, light_mass) in [(50.0 as Real, 0.1 as Real), (200.0, 0.05), (10.0, 1.0)] {
            for offset in [0.2 as Real, 0.5] {
                for (rate, substeps) in [(30.0 as Real, 2), (60.0, 2), (60.0, 4)] {
                    let (apart, fastest) = light_body_held_to_a_resting_multibody(
                        fixed, root_mass, light_mass, offset, rate, substeps,
                    );
                    // The ball joint lets the light box swing; the fixed joint holds it still.
                    if apart > 0.05 || (fixed && fastest > 0.2) {
                        failed.push(alloc::format!(
                            "{} joint, {root_mass} kg holding {light_mass} kg {offset} m off its \
                             centre, {rate} Hz x {substeps}: anchors {apart} m apart, the light \
                             box at up to {fastest} m/s",
                            if fixed { "fixed" } else { "ball" }
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "the light body left its anchor: {failed:?}"
    );
}


/// How a chain of [`chain_by_its_anchor`] swung.
#[cfg(feature = "dim3")]
struct AnchoredChainSwing {
    /// The step the ball's pose first read not finite at, if one did.
    not_finite_at: Option<usize>,
    /// The fastest the ball moved, in m/s.
    fastest: Real,
    /// How far, in metres, the ball rose above the height it was released from, after its first
    /// two seconds.
    rose: Real,
    /// The farthest the ball moved in one step, in metres.
    longest_step: Real,
    /// The speed the ball reaches falling from the height it was released at to the lowest the
    /// chain hangs it, in m/s: the fastest its swing can carry it.
    fall_speed: Real,
}

/// Twelve 4 kg capsule links, 0.55 m apart, hung on spherical multibody joints from a fixed
/// anchor whose collider is a 1 m box centred on the first joint, with a 500 kg ball on the
/// end, raised `angle` degrees and let go for 8 s at `rate` steps a second in `substeps`
/// substeps. The joints switch contacts off between the bodies they join, so the second link
/// is the first that meets the anchor's box, near the root. With `anchor_collides` false the
/// box is a sensor and the chain swings free of it. A force of `push` newtons, across the plane
/// the chain is raised in, acts on the ball throughout. Every collider has rapier's default
/// surface.
#[cfg(feature = "dim3")]
fn chain_by_its_anchor(
    angle: Real,
    rate: Real,
    substeps: usize,
    anchor_collides: bool,
    push: Real,
) -> AnchoredChainSwing {
    chain_by_its_anchor_on(
        angle,
        rate,
        substeps,
        anchor_collides,
        push,
        ChainSurfaces::default(),
    )
}

/// The friction and restitution of the colliders of [`chain_by_its_anchor_on`].
#[cfg(feature = "dim3")]
#[derive(Copy, Clone)]
struct ChainSurfaces {
    /// The anchor's box: `(friction, restitution)`.
    anchor: (Real, Real),
    /// Each link and the ball: `(friction, restitution)`.
    chain: (Real, Real),
}

#[cfg(feature = "dim3")]
impl Default for ChainSurfaces {
    fn default() -> Self {
        let collider = ColliderBuilder::ball(1.0);
        let surface = (collider.friction, collider.restitution);
        Self {
            anchor: surface,
            chain: surface,
        }
    }
}

/// The chain of [`chain_by_its_anchor`], its colliders made of `surfaces`.
#[cfg(feature = "dim3")]
fn chain_by_its_anchor_on(
    angle: Real,
    rate: Real,
    substeps: usize,
    anchor_collides: bool,
    push: Real,
    surfaces: ChainSurfaces,
) -> AnchoredChainSwing {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let anchor_at = Vector::new(60.0, 16.0, 60.0);
    let (pitch, links, ball_radius, eye) = (0.55 as Real, 12, 0.42 as Real, 0.12 as Real);
    let angle = angle.to_radians();
    let raised = Rotation::from_axis_angle(Vector::X, angle);
    let down = Vector::new(0.0, -angle.cos(), -angle.sin());
    let (anchor, _) = world.insert(
        RigidBodyBuilder::fixed()
            .translation(anchor_at)
            .rotation(raised.to_scaled_axis()),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5)
            .sensor(!anchor_collides)
            .friction(surfaces.anchor.0)
            .restitution(surfaces.anchor.1),
    );
    let mut bodies = Vec::new();
    for i in 0..links {
        let (link, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(anchor_at + down * (pitch * (i as Real + 0.5)))
                .rotation(raised.to_scaled_axis())
                .can_sleep(false),
            ColliderBuilder::capsule_y(pitch * 0.3, 0.07)
                .mass(4.0)
                .friction(surfaces.chain.0)
                .restitution(surfaces.chain.1),
        );
        bodies.push(link);
    }
    let (ball, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(anchor_at + down * (pitch * links as Real + eye + ball_radius))
            .rotation(raised.to_scaled_axis())
            .can_sleep(false),
        ColliderBuilder::ball(ball_radius)
            .mass(500.0)
            .friction(surfaces.chain.0)
            .restitution(surfaces.chain.1),
    );
    bodies.push(ball);
    world.bodies[ball].add_force(Vector::new(push, 0.0, 0.0), true);
    for (i, body) in bodies.iter().enumerate() {
        let parent = if i == 0 { anchor } else { bodies[i - 1] };
        let on_parent = if i == 0 {
            Vector::ZERO
        } else {
            Vector::new(0.0, -pitch / 2.0, 0.0)
        };
        let on_child = if i == links {
            Vector::new(0.0, ball_radius + eye, 0.0)
        } else {
            Vector::new(0.0, pitch / 2.0, 0.0)
        };
        let joint = SphericalJointBuilder::new()
            .local_anchor1(on_parent)
            .local_anchor2(on_child)
            .contacts_enabled(false)
            .build();
        world.insert_multibody_joint(parent, *body, joint).unwrap();
    }
    let released = world.bodies[ball].translation().y;
    let reach = pitch * links as Real + eye + ball_radius;
    let mut swing = AnchoredChainSwing {
        not_finite_at: None,
        fastest: 0.0,
        rose: Real::MIN,
        longest_step: 0.0,
        fall_speed: (2.0 * 9.81 * reach * (1.0 - angle.cos())).sqrt(),
    };
    let mut was = world.bodies[ball].translation();
    for step in 0..(8.0 * rate) as usize {
        world.step();
        let ball = &world.bodies[ball];
        if !ball.translation().is_finite() || !ball.linvel().is_finite() {
            swing.not_finite_at = Some(step);
            return swing;
        }
        swing.longest_step = swing.longest_step.max((ball.translation() - was).length());
        was = ball.translation();
        swing.fastest = swing.fastest.max(ball.linvel().length());
        if step as Real >= 2.0 * rate {
            swing.rose = swing.rose.max(ball.translation().y - released);
        }
    }
    swing
}

/// A chain of light links with a heavy ball on the end, whose second link slides on its fixed
/// anchor's body as it swings, swings as the same chain clear of that body does: no more than a
/// tenth faster, and never a quarter metre above where it was released, at 30 and 60 steps a
/// second in 2 and 4 substeps and at 60 in 8. The link sliding on the box whips the light links
/// below it against one another. A move that turned them read, at the configuration it reached,
/// several times the kinetic energy the chain moved with until the Coriolis forces of the next
/// substep took it back; the stabilization pass solved the contact at that velocity in between,
/// and the chain gained energy at every swing: released at 85 degrees at 60 steps in 4 substeps
/// it rose 0.32 m above its release, and released at 70 degrees at 30 steps in 4 its ball
/// reached 37 m/s where the free chain's reached 12.5.
#[test]
#[cfg(feature = "dim3")]
fn a_chain_sliding_on_its_anchors_body_swings_as_the_free_chain_does() {
    let mut failed = Vec::new();
    for angle in [40.0, 55.0, 70.0, 85.0] {
        for (rate, substeps) in [(30.0, 2), (30.0, 4), (60.0, 2), (60.0, 4), (60.0, 8)] {
            let free = chain_by_its_anchor(angle, rate, substeps, false, 0.0);
            let swing = chain_by_its_anchor(angle, rate, substeps, true, 0.0);
            let finite = swing.not_finite_at.is_none() && free.not_finite_at.is_none();
            if !finite || swing.fastest > free.fastest * 1.1 || swing.rose > 0.25 {
                failed.push(alloc::format!(
                    "{angle}°, {rate} Hz x {substeps}: not finite at {:?}, fastest {} m/s (free {} m/s), rose {} m",
                    swing.not_finite_at,
                    swing.fastest,
                    free.fastest,
                    swing.rose
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "the chain swung faster or higher: {failed:?}"
    );
}

/// The chain of [`chain_by_its_anchor`] pushed out of the plane it is raised in, by 0.3 to 30 N
/// on its 500 kg ball, against its anchor's body at 60 steps a second in 4 substeps, never moves
/// faster than a tenth over the speed of its fall from where it was released, never rises a
/// quarter metre above it and never leaps a metre in a step. Pushed off its plane, the second
/// link presses on the box with its side and spins about its own length, and the joints below
/// it turn back against it. Two things flung it. The multibody's mass matrix, which spans
/// eight orders of magnitude between that spin and the swing of the ball, was factored in
/// `f32`, which read some of its smallest inertias with the wrong sign: a contact solved through
/// it pushed with a negative effective mass and the chain leapt metres in a step. And the
/// Coriolis update of the joints below the spinning link added a percent of the chain's energy
/// each substep while the fastest of them turned a few hundredths of a radian, below the turn
/// from which a substep's energy was held, and the ball climbed metres over its release.
#[test]
#[cfg(feature = "dim3")]
fn a_chain_pushed_off_its_plane_against_its_anchors_body_never_climbs_nor_leaps() {
    let mut failed = Vec::new();
    for push in [0.3 as Real, 3.0, 30.0] {
        for angle in [50.0 as Real, 60.0, 70.0, 80.0] {
            let swing = chain_by_its_anchor(angle, 60.0, 4, true, push);
            if swing.not_finite_at.is_some()
                || swing.fastest > swing.fall_speed * 1.1
                || swing.rose > 0.25
                || swing.longest_step > 1.0
            {
                failed.push(alloc::format!(
                    "{push} N, {angle}°: not finite at {:?}, fastest {} m/s (its fall {} m/s), \
                     rose {} m, longest step {} m",
                    swing.not_finite_at,
                    swing.fastest,
                    swing.fall_speed,
                    swing.rose,
                    swing.longest_step
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "the chain pushed off its plane climbed or leapt: {failed:?}"
    );
}

/// The chain of [`chain_by_its_anchor`] pushed out of the plane it is raised in, by 1 to 200 N
/// on its 500 kg ball, against its anchor's body, released at 55, 70 and 85 degrees at 30 and 60
/// steps a second in 2 and 4 substeps, never moves its ball faster than a twentieth over the
/// speed of its fall and never lifts it a tenth of a metre above its release. Its second link
/// knocks on the anchor's box as the chain hangs, and the continuous pass takes the chain back
/// to the share of its motion that knock stopped at, keeping its joints' velocities. The joints
/// below the knocking link turn against one another, their shares of the ball's velocity
/// cancelling at the configuration the solve ended at and far less at the one a few hundredths
/// of a radian back: released at 55 degrees at 30 steps in 4 substeps, a 1 N push left the ball
/// at 31.5 m/s for a step where its fall gives it 7.7, the chain 239 kJ over its release.
#[test]
#[cfg(feature = "dim3")]
fn a_chain_the_continuous_pass_takes_back_off_its_anchors_body_keeps_its_energy() {
    let mut failed = Vec::new();
    for push in [1.0 as Real, 20.0, 200.0] {
        for angle in [55.0 as Real, 70.0, 85.0] {
            for (rate, substeps) in [(30.0 as Real, 2), (30.0, 4), (60.0, 2), (60.0, 4)] {
                let swing = chain_by_its_anchor(angle, rate, substeps, true, push);
                if swing.not_finite_at.is_some()
                    || swing.fastest > swing.fall_speed * 1.05
                    || swing.rose > 0.1
                {
                    failed.push(alloc::format!(
                        "{push} N, {angle}°, {rate} Hz x {substeps}: not finite at {:?}, fastest \
                         {} m/s (its fall {} m/s), rose {} m",
                        swing.not_finite_at,
                        swing.fastest,
                        swing.fall_speed,
                        swing.rose
                    ));
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "the chain gained energy through the continuous pass: {failed:?}"
    );
}

/// The chain of [`chain_by_its_anchor`] with bouncy links (restitution 0.3, friction 0.5) on an
/// anchor's box with friction 0.7, pushed out of the plane it is raised in by 1 to 200 N on its
/// 500 kg ball, released at 55, 70 and 85 degrees at 30 and 60 steps a second in 2 and 4
/// substeps, never moves its ball faster than a twentieth over the speed of its fall and never
/// lifts it a tenth of a metre above its release. Its second link knocks on the anchor's box as
/// the chain hangs, and the step's restitution pass bounces it once the substeps have moved the
/// chain. The bounce was solved through the rows the last substep began with, a few hundredths
/// of a radian behind joints that turn against one another, and moved the chain along
/// directions the contact never asked for: released at 70 degrees at 30 steps in 2 substeps, a
/// 1 N push left the ball at 14 m/s where its fall gives it 9.6, the chain 28 kJ over its
/// release, a bounce taking it from 5.5 kJ of kinetic energy to 192 kJ.
#[test]
#[cfg(feature = "dim3")]
fn a_bouncy_chain_knocking_on_its_anchors_body_keeps_its_energy() {
    let surfaces = ChainSurfaces {
        anchor: (0.7, 0.0),
        chain: (0.5, 0.3),
    };
    let mut failed = Vec::new();
    for push in [1.0 as Real, 20.0, 200.0] {
        for angle in [55.0 as Real, 70.0, 85.0] {
            for (rate, substeps) in [(30.0 as Real, 2), (30.0, 4), (60.0, 2), (60.0, 4)] {
                let swing = chain_by_its_anchor_on(angle, rate, substeps, true, push, surfaces);
                if swing.not_finite_at.is_some()
                    || swing.fastest > swing.fall_speed * 1.05
                    || swing.rose > 0.1
                {
                    failed.push(alloc::format!(
                        "{push} N, {angle}°, {rate} Hz x {substeps}: not finite at {:?}, fastest \
                         {} m/s (its fall {} m/s), rose {} m",
                        swing.not_finite_at,
                        swing.fastest,
                        swing.fall_speed,
                        swing.rose
                    ));
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "the bouncy chain gained energy: {failed:?}"
    );
}

/// The chain of [`a_bouncy_chain_knocking_on_its_anchors_body_keeps_its_energy`], pushed out of
/// the plane it is raised in by 3 to 50 N, released at 55, 70 and 85 degrees at 30 and 60 steps
/// a second in 2 and 4 substeps, never moves its ball faster than its fall gives it, to within
/// a hundredth, nor lifts it a tenth of a metre above its release. As it passes the bottom of
/// its first swing its second link knocks on the anchor's box and the links below whip against
/// one another at tens of radians a second. Each substep held the free velocity of the next,
/// whose accelerations through the semi-implicit Coriolis terms turn much of that whip back,
/// and the velocity the step ended with kept the rest: released at 70 degrees at 30 steps in 4
/// substeps with a 10 N push, a step ended at 25.5 kJ of kinetic energy where its last substep
/// was held to 16.2 kJ, and the ball left it 4% faster than its fall gives it.
#[test]
#[cfg(feature = "dim3")]
fn a_whipping_chain_ends_each_step_no_faster_than_its_fall() {
    let surfaces = ChainSurfaces {
        anchor: (0.7, 0.0),
        chain: (0.5, 0.3),
    };
    let mut failed = Vec::new();
    for push in [3.0 as Real, 10.0, 50.0] {
        for angle in [55.0 as Real, 70.0, 85.0] {
            for (rate, substeps) in [(30.0 as Real, 2), (30.0, 4), (60.0, 2), (60.0, 4)] {
                let swing = chain_by_its_anchor_on(angle, rate, substeps, true, push, surfaces);
                if swing.not_finite_at.is_some()
                    || swing.fastest > swing.fall_speed * 1.01
                    || swing.rose > 0.1
                {
                    failed.push(alloc::format!(
                        "{push} N, {angle}°, {rate} Hz x {substeps}: not finite at {:?}, fastest \
                         {} m/s (its fall {} m/s), rose {} m",
                        swing.not_finite_at,
                        swing.fastest,
                        swing.fall_speed,
                        swing.rose
                    ));
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "the whipping chain ended a step faster than its fall: {failed:?}"
    );
}


/// The inverse of a multibody's mass matrix holds every degree of freedom with a positive
/// inertia for a chain of twelve 4 kg capsule links, 0.07 m wide and 0.55 m apart, each turned a
/// twentieth of a radian off the one above, carrying a 20 t load from a fixed anchor. Its mass
/// matrix spans nine orders of magnitude, from a link spinning about its own length to the load
/// swinging 7 m below the anchor; factored in `f32`, the inverse read 18 of its 39 diagonal
/// entries negative.
#[test]
#[cfg(feature = "dim3")]
fn a_chain_carrying_a_load_five_thousand_times_a_link_keeps_a_positive_inverse_mass() {
    let mut world = PhysicsWorld::new();
    let (pitch, links) = (0.55 as Real, 12);
    let (anchor, _) = world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::ball(0.1).sensor(true),
    );
    let mut bodies = Vec::new();
    let mut at = Vector::ZERO;
    for i in 0..=links {
        let side = if i % 2 == 0 { -1.0 } else { 1.0 };
        let turn = Rotation::from_scaled_axis(Vector::new(
            0.05 * side,
            0.0,
            0.035 * ((i % 3) as Real - 1.0),
        ));
        let down = turn * Vector::new(0.0, -1.0, 0.0);
        let half = if i == links { 0.5 } else { pitch / 2.0 };
        let collider = if i == links {
            ColliderBuilder::ball(0.42).mass(20_000.0)
        } else {
            ColliderBuilder::capsule_y(pitch * 0.3, 0.07).mass(4.0)
        };
        at += down * half;
        let (body, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(at)
                .rotation(turn.to_scaled_axis())
                .can_sleep(false),
            collider,
        );
        at += down * half;
        bodies.push(body);
    }
    for (i, body) in bodies.iter().enumerate() {
        let joint = SphericalJointBuilder::new()
            .local_anchor1(Vector::new(0.0, if i == 0 { 0.0 } else { -pitch / 2.0 }, 0.0))
            .local_anchor2(Vector::new(
                0.0,
                if i == links { 0.5 } else { pitch / 2.0 },
                0.0,
            ))
            .contacts_enabled(false)
            .build();
        let parent = if i == 0 { anchor } else { bodies[i - 1] };
        world.insert_multibody_joint(parent, *body, joint).unwrap();
    }
    world.step();
    let link = *world.multibody_joints.rigid_body_link(bodies[0]).unwrap();
    let multibody = &world.multibody_joints[link.multibody];
    let ndofs = multibody.ndofs();
    let mut negative = Vec::new();
    for dof in 0..ndofs {
        let mut unit = DVector::zeros(ndofs);
        unit[dof] = 1.0;
        multibody.inv_augmented_mass().solve_mut(&mut unit);
        if !(unit[dof] > 0.0) {
            negative.push((dof, unit[dof]));
        }
    }
    assert!(
        negative.is_empty(),
        "the inverse mass matrix read degrees of freedom without a positive inertia: {negative:?}"
    );
}

/// A heavy box with a forearm hinged to its end, dropped forearm first onto fixed ground fast
/// enough that the continuous-collision pass stops the box's body at its moment of impact, ends
/// every step with the forearm's body where the multibody holds it on the box's body: the turn
/// between the two bodies the hinge's own angle. The pass clamped the root's body alone, about
/// 0.15 m short of where the multibody stood, and the hinge read from the two bodies was up to
/// 25° from the angle the multibody held, which the next step went on from.
#[test]
#[cfg(feature = "dim3")]
fn a_multibody_whose_root_the_continuous_pass_stops_ends_the_step_where_its_links_stand() {
    for (rate, root_mass) in [(30.0 as Real, 100.0 as Real), (30.0, 200.0), (60.0, 200.0)] {
        let mut world = PhysicsWorld::new();
        world.integration_parameters.dt = 1.0 / rate;
        world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
            ColliderBuilder::cuboid(2.5, 0.5, 2.5).friction(0.7),
        );
        let turn = Rotation::from_axis_angle(Vector::Z, -40.0_f32.to_radians() as Real);
        let at = Vector::new(0.0, 2.0, 0.0);
        let (root, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(at)
                .rotation(turn.to_scaled_axis())
                .can_sleep(false),
            ColliderBuilder::cuboid(0.15, 0.1, 0.1)
                .mass(root_mass)
                .friction(0.7),
        );
        let (forearm, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(at + turn * Vector::new(0.3, 0.0, 0.0))
                .rotation(turn.to_scaled_axis())
                .can_sleep(false),
            ColliderBuilder::cuboid(0.15, 0.04, 0.04)
                .mass(1.5)
                .friction(0.7),
        );
        let elbow = RevoluteJointBuilder::new(Vector::Z)
            .local_anchor1(Vector::new(0.15, 0.0, 0.0))
            .local_anchor2(Vector::new(-0.15, 0.0, 0.0))
            .limits([-140.0_f32.to_radians() as Real, 0.0])
            .contacts_enabled(false);
        world.insert_multibody_joint(root, forearm, elbow).unwrap();
        let mut widest_gap: Real = 0.0;
        let mut widest_turn: Real = 0.0;
        for _ in 0..(2.0 * rate) as usize {
            world.step();
            let link = *world.multibody_joints.rigid_body_link(forearm).unwrap();
            let multibody = &world.multibody_joints[link.multibody];
            let held = multibody.link(0).unwrap().local_to_world.inverse()
                * multibody.link(link.id).unwrap().local_to_world;
            let angle = multibody.link(link.id).unwrap().joint.coords[3];
            let bodies = world.bodies[root].position().inverse() * *world.bodies[forearm].position();
            widest_gap = widest_gap.max((bodies.translation - held.translation).length());
            let between =
                world.bodies[root].rotation().inverse() * *world.bodies[forearm].rotation();
            let read = 2.0 * between.z.atan2(between.w);
            widest_turn = widest_turn.max((read - angle).abs().to_degrees());
        }
        assert!(
            widest_gap < 1.0e-3 && widest_turn < 0.05,
            "{rate} Hz, {root_mass} kg: the forearm's body stood {widest_gap} m from where the \
             multibody holds it on the box's body, and the bodies read the hinge {widest_turn}° \
             from the multibody's angle"
        );
    }
}

/// A 10 kg box with two legs hanging below it (a thigh and a shin each, on revolute multibody
/// joints), falling at 20, 40 and 80 m/s onto a fixed plate 0.05 m thick at 30 Hz, ends with
/// every link above the plate's underside, as the same bodies falling free or on impulse joints
/// do. The legs strike the plate first: the continuous pass stops the multibody as a whole at the
/// fraction its first link strikes. With the pass sweeping only the multibody's root, 4 of its 5
/// links ended below the plate at every speed, the legs having passed through it before the box
/// reached it; with each link's body clamped on its own, 2 of 5 at 20 and 40 m/s and 4 at 80.
#[test]
#[cfg(feature = "dim3")]
fn a_multibody_whose_legs_strike_a_thin_plate_first_stops_as_a_whole() {
    let thick: Real = 0.05;
    for speed in [20.0 as Real, 40.0, 80.0] {
        let mut world = PhysicsWorld::new();
        world.integration_parameters.dt = 1.0 / 30.0;
        world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -thick / 2.0, 0.0)),
            ColliderBuilder::cuboid(2.5, thick / 2.0, 2.5).friction(0.7),
        );
        let mut part = |at: Vector, half: Vector, mass: Real| {
            world
                .insert(
                    RigidBodyBuilder::dynamic().translation(at).can_sleep(false),
                    ColliderBuilder::cuboid(half.x, half.y, half.z)
                        .mass(mass)
                        .friction(0.7),
                )
                .0
        };
        let top = 2.0;
        let root = part(
            Vector::new(0.0, top, 0.0),
            Vector::new(0.15, 0.1, 0.1),
            10.0,
        );
        let mut links = vec![root];
        let mut joints = Vec::new();
        for side in [-1.0 as Real, 1.0] {
            let thigh = part(
                Vector::new(0.1 * side, top - 0.3, 0.0),
                Vector::new(0.06, 0.2, 0.06),
                6.0,
            );
            let shin = part(
                Vector::new(0.1 * side, top - 0.7, 0.0),
                Vector::new(0.05, 0.2, 0.05),
                4.0,
            );
            joints.push((
                root,
                thigh,
                RevoluteJointBuilder::new(Vector::X)
                    .local_anchor1(Vector::new(0.1 * side, -0.1, 0.0))
                    .local_anchor2(Vector::new(0.0, 0.2, 0.0))
                    .limits([-1.0, 1.0])
                    .contacts_enabled(false),
            ));
            joints.push((
                thigh,
                shin,
                RevoluteJointBuilder::new(Vector::X)
                    .local_anchor1(Vector::new(0.0, -0.2, 0.0))
                    .local_anchor2(Vector::new(0.0, 0.2, 0.0))
                    .limits([0.0, 2.4])
                    .contacts_enabled(false),
            ));
            links.push(thigh);
            links.push(shin);
        }
        for (parent, child, joint) in joints {
            world.insert_multibody_joint(parent, child, joint).unwrap();
        }
        let link = *world.multibody_joints.rigid_body_link(root).unwrap();
        let multibody = world
            .multibody_joints
            .get_multibody_mut(link.multibody)
            .unwrap();
        let mut velocity = multibody.generalized_velocity_mut();
        velocity.fill(0.0);
        velocity[1] = -speed;
        for _ in 0..30 {
            world.step();
        }
        let below: Vec<_> = links
            .iter()
            .map(|h| world.bodies[*h].translation().y)
            .filter(|y| *y < -thick)
            .collect();
        assert!(
            below.is_empty(),
            "{speed} m/s: {} of the multibody's 5 links ended below the plate, at {below:?}",
            below.len()
        );
    }
}

/// The ragdoll of [`ragdoll_by_stairs`] (its knees and elbows bending the other way, 0.7
/// friction), its pelvis' centre `height` + 0.2 m above a flight of eight stairs whose foot is at
/// `at`, pushed by `push` N·s on its chest and stepped for five seconds at `rate` steps a second
/// in `substeps` substeps, with `ccd_substeps` continuous-collision passes. Answers how far, in
/// degrees, any joint went past its limit, read from the bodies' poses after every step, and how
/// deep, in metres, any of its boxes stood in a step at the end of any step.
#[cfg(feature = "dim3")]
fn ragdoll_landing_on_stairs(
    at: Vector,
    rate: Real,
    substeps: usize,
    ccd_substeps: usize,
    height: Real,
    push: Real,
) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    world.integration_parameters.max_ccd_substeps = ccd_substeps;
    world.integration_parameters.normalized_max_linear_velocity = Real::MAX;
    let mut stairs = Vec::new();
    for step in 0..8 {
        let top = 1.6 - 0.2 * step as Real;
        let (_, stair) = world.insert(
            RigidBodyBuilder::fixed()
                .translation(at + Vector::new(0.0, top * 0.5, 0.3 * step as Real + 0.15)),
            ColliderBuilder::cuboid(1.5, top * 0.5, 0.15).friction(0.7),
        );
        stairs.push(stair);
    }
    let base = at + Vector::new(0.0, height, 0.15);
    let mut boxes = Vec::new();
    let mut part = |world: &mut PhysicsWorld, offset: Vector, half: Vector, mass: Real| {
        let (body, collider) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(base + offset)
                .allow_fast_rotation(true)
                .can_sleep(false),
            ColliderBuilder::cuboid(half.x, half.y, half.z)
                .mass(mass)
                .friction(0.7),
        );
        boxes.push(collider);
        body
    };
    let frame = |axis: Vector| Rotation::from_rotation_arc(Vector::X, axis);
    // Each joint with its swing cone (none for a hinge) and its twist or hinge range.
    type Held = (GenericJoint, Option<Real>, [Real; 2]);
    let ball =
        |axis: Vector, on_parent: Vector, on_child: Vector, swing: Real, twist: Real| -> Held {
            let mut joint: GenericJoint = SphericalJointBuilder::new()
                .local_anchor1(on_parent)
                .local_anchor2(on_child)
                .contacts_enabled(false)
                .build()
                .into();
            joint.local_frame1.rotation = frame(axis);
            joint.local_frame2.rotation = frame(axis);
            let (swing, twist) = (swing.to_radians(), twist.to_radians());
            joint.coupled_axes |= JointAxesMask::ANG_Y | JointAxesMask::ANG_Z;
            joint.set_limits(JointAxis::AngY, [-swing, swing]);
            joint.set_limits(JointAxis::AngZ, [-swing, swing]);
            joint.set_limits(JointAxis::AngX, [-twist, twist]);
            (joint, Some(swing), [-twist, twist])
        };
    let hinge = |axis: Vector, on_parent: Vector, on_child: Vector| -> Held {
        let mut joint: GenericJoint = RevoluteJointBuilder::new(Vector::X)
            .local_anchor1(on_parent)
            .local_anchor2(on_child)
            .contacts_enabled(false)
            .build()
            .into();
        joint.local_frame1.rotation = frame(axis);
        joint.local_frame2.rotation = frame(axis);
        let range = [-140.0_f32.to_radians() as Real, 0.0];
        joint.set_limits(JointAxis::AngX, range);
        (joint, None, range)
    };
    let v = Vector::new;
    let pelvis = part(&mut world, v(0.0, 0.2, 0.0), v(0.15, 0.1, 0.1), 10.0);
    let chest = part(&mut world, v(0.0, 0.5, 0.0), v(0.15, 0.2, 0.1), 15.0);
    let head = part(&mut world, v(0.0, 0.8, 0.0), v(0.1, 0.1, 0.1), 5.0);
    let mut joints = vec![
        (
            pelvis,
            chest,
            ball(Vector::Y, v(0.0, 0.1, 0.0), v(0.0, -0.2, 0.0), 30.0, 20.0),
        ),
        (
            chest,
            head,
            ball(Vector::Y, v(0.0, 0.2, 0.0), v(0.0, -0.1, 0.0), 40.0, 30.0),
        ),
    ];
    for side in [-1.0 as Real, 1.0] {
        let upper = part(
            &mut world,
            v(0.3 * side, 0.65, 0.0),
            v(0.15, 0.05, 0.05),
            2.0,
        );
        let fore = part(
            &mut world,
            v(0.6 * side, 0.65, 0.0),
            v(0.15, 0.04, 0.04),
            1.5,
        );
        let thigh = part(
            &mut world,
            v(0.1 * side, -0.1, 0.0),
            v(0.06, 0.2, 0.06),
            6.0,
        );
        let shin = part(
            &mut world,
            v(0.1 * side, -0.5, 0.0),
            v(0.05, 0.2, 0.05),
            4.0,
        );
        joints.push((
            chest,
            upper,
            ball(
                v(side, 0.0, 0.0),
                v(0.15 * side, 0.15, 0.0),
                v(-0.15 * side, 0.0, 0.0),
                70.0,
                40.0,
            ),
        ));
        joints.push((
            upper,
            fore,
            hinge(
                Vector::Z,
                v(0.15 * side, 0.0, 0.0),
                v(-0.15 * side, 0.0, 0.0),
            ),
        ));
        joints.push((
            pelvis,
            thigh,
            ball(
                -Vector::Y,
                v(0.1 * side, -0.1, 0.0),
                v(0.0, 0.2, 0.0),
                60.0,
                30.0,
            ),
        ));
        joints.push((
            thigh,
            shin,
            hinge(Vector::X, v(0.0, -0.2, 0.0), v(0.0, 0.2, 0.0)),
        ));
    }
    for (parent, child, (joint, _, _)) in &joints {
        world
            .insert_multibody_joint(*parent, *child, *joint)
            .unwrap();
    }
    let link = *world.multibody_joints.rigid_body_link(chest).unwrap();
    world
        .multibody_joints
        .get_multibody_mut(link.multibody)
        .unwrap()
        .apply_link_impulse(
            &world.bodies,
            link.id,
            Vector::new(0.0, 0.0, push),
            AngVector::ZERO,
        );
    let (mut past_limits, mut deepest): (Real, Real) = (0.0, 0.0);
    for _ in 0..(5.0 * rate) as usize {
        world.step();
        for (parent, child, (joint, swing, range)) in &joints {
            let f1 = *world.bodies[*parent].position() * joint.local_frame1;
            let f2 = *world.bodies[*child].position() * joint.local_frame2;
            let mut relative = f1.rotation.inverse() * f2.rotation;
            if relative.w < 0.0 {
                relative = -relative;
            }
            let angle = 2.0 * relative.x.atan2(relative.w);
            let mut past = (range[0] - angle).max(angle - range[1]);
            if let Some(cone) = swing {
                let axis1 = f1.rotation * Vector::X;
                let axis2 = f2.rotation * Vector::X;
                past = past.max(axis1.dot(axis2).clamp(-1.0, 1.0).acos() - cone);
            }
            past_limits = past_limits.max(past.to_degrees());
        }
        for part in &boxes {
            for stair in &stairs {
                let (a, b) = (&world.colliders[*part], &world.colliders[*stair]);
                if let Ok(Some(contact)) =
                    parry::query::contact(a.position(), a.shape(), b.position(), b.shape(), 0.0)
                {
                    deepest = deepest.max(-contact.dist);
                }
            }
        }
    }
    (past_limits, deepest)
}

/// The ragdoll of [`ragdoll_landing_on_stairs`] landing on the stairs from three heights under
/// three pushes, at 30 steps a second in 2 substeps, with the continuous-collision pass and
/// without it. The pass stops the ragdoll as a whole where its first box strikes a step, its
/// joints at the same share of their motion as its root: every box ends the step where the
/// ragdoll stood then, and none of them deeper in a step than the ragdoll's boxes go with the
/// pass off, nor any joint further past its limit than a degree or than with the pass off. A
/// tenth of a degree either way turns on where the ragdoll lands: hung 63 m along X, the
/// ragdoll of 6320f80d went 0.42 degrees past a limit with the pass and 0.15 without it, and
/// 0.14 and 0.15 where it hangs here. With the pass clamping the root alone and
/// carrying the other boxes rigidly with it, the joints' motion over the whole step stood on the
/// stopped root: a box went 0.22 m into a step, where it went 0.155 m with the pass off, and a
/// joint 18.4 degrees past its limit, where it went 13.1.
#[test]
#[cfg(feature = "dim3")]
fn a_ragdoll_landing_on_stairs_with_the_continuous_pass_goes_no_deeper_into_them_nor_further_past_its_limits()
 {
    let at = Vector::new(1837.0, 20.0, 6.0);
    let mut with = (0.0 as Real, 0.0 as Real);
    let mut without = (0.0 as Real, 0.0 as Real);
    for height in [2.45 as Real, 2.65, 2.85] {
        for push in [40.0 as Real, 60.0, 80.0] {
            let (past, deep) = ragdoll_landing_on_stairs(at, 30.0, 2, 1, height, push);
            with = (with.0.max(past), with.1.max(deep));
            let (past, deep) = ragdoll_landing_on_stairs(at, 30.0, 2, 0, height, push);
            without = (without.0.max(past), without.1.max(deep));
        }
    }
    assert!(
        with.1 < without.1,
        "with the pass a box stood {} m in a step, without it {} m",
        with.1,
        without.1
    );
    assert!(
        with.0 < without.0.max(1.0),
        "with the pass a joint went {} degrees past its limit, without it {}",
        with.0,
        without.0
    );
}

/// A ball of radius 0.2 at x = -3 sliding right at `speed` on a prismatic joint from a fixed
/// root, toward a fixed wall 0.05 m thick at x = 0, stepped for one second at 60 Hz, on a
/// multibody joint or an impulse joint. Answers the ball's x at the end.
#[cfg(feature = "dim3")]
fn ball_sliding_into_a_thin_wall(multibody: bool, speed: Real) -> Real {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = 1.0 / 60.0;
    world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(0.025, 2.0, 2.0),
    );
    let root =
        world.insert_body(RigidBodyBuilder::fixed().translation(Vector::new(-3.0, 0.0, 0.0)));
    let (ball, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(-3.0, 0.0, 0.0))
            .linvel(Vector::new(speed, 0.0, 0.0))
            .can_sleep(false),
        ColliderBuilder::ball(0.2),
    );
    let joint = PrismaticJointBuilder::new(Vector::X).contacts_enabled(false);
    if multibody {
        world.insert_multibody_joint(root, ball, joint).unwrap();
        let link = *world.multibody_joints.rigid_body_link(ball).unwrap();
        let multibody = world
            .multibody_joints
            .get_multibody_mut(link.multibody)
            .unwrap();
        let id = multibody.link(link.id).unwrap().assembly_id();
        multibody.generalized_velocity_mut()[id] = speed;
    } else {
        world.insert_impulse_joint(root, ball, joint);
    }
    for _ in 0..60 {
        world.step();
    }
    world.bodies[ball].translation().x
}

/// A ball on a prismatic multibody joint from a fixed root, sliding at 30, 120 and 400 m/s into
/// a fixed wall 0.05 m thick, ends the second in front of the wall, as the same ball on an
/// impulse joint does: the continuous pass stops the multibody at the fraction the ball strikes,
/// its joint at that share of its motion. With the root alone clamped, a root that does not move
/// stopped nothing, and the ball went through the wall at every speed.
#[test]
#[cfg(feature = "dim3")]
fn a_multibody_link_sliding_into_a_thin_wall_stops_in_front_of_it() {
    for speed in [30.0 as Real, 120.0, 400.0] {
        let impulse = ball_sliding_into_a_thin_wall(false, speed);
        let multibody = ball_sliding_into_a_thin_wall(true, speed);
        assert!(
            impulse < -0.2,
            "{speed} m/s on an impulse joint: x = {impulse}"
        );
        assert!(
            multibody < -0.2,
            "{speed} m/s on a multibody joint: x = {multibody}"
        );
    }
}

/// The chain of [`chain_by_its_anchor`] clear of its anchor's body, released at 55, 70 and 85
/// degrees, swings its ball no faster than its fall gives it, to within a twentieth, and never
/// lifts it a tenth of a metre above where it was let go, at 30 steps a second in 2 and 4
/// substeps and at 60 in 2. A chain of light links under a heavy ball whips its links against
/// one another at tens of radians a second as it swings, which a 30 Hz substep moves through
/// a few tenths of a radian: the ball reached 16.4 m/s from 70 degrees at 30 steps in 2
/// substeps, where its fall gives it 9.6, and at 60 in 2 from 85 degrees it rose 1.4 m above its
/// release.
#[test]
#[cfg(feature = "dim3")]
fn a_free_chain_under_a_heavy_ball_swings_no_faster_nor_higher_than_its_fall() {
    let mut failed = Vec::new();
    for angle in [55.0, 70.0, 85.0] {
        for (rate, substeps) in [(30.0, 2), (30.0, 4), (60.0, 2)] {
            let swing = chain_by_its_anchor(angle, rate, substeps, false, 0.0);
            if swing.not_finite_at.is_some()
                || swing.fastest > swing.fall_speed * 1.05
                || swing.rose > 0.1
            {
                failed.push(alloc::format!(
                    "{angle}°, {rate} Hz x {substeps}: not finite at {:?}, fastest {} m/s (its fall {} m/s), rose {} m",
                    swing.not_finite_at,
                    swing.fastest,
                    swing.fall_speed,
                    swing.rose
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "the free chain swung faster or higher than its fall: {failed:?}"
    );
}

/// An articulation of a 5 kg box and a 1 kg arm on an undamped hinge, dropped from rest, falls as
/// far in one second as a lone 5 kg box does, to within a two-hundredth, at `rate` steps a second
/// in `substeps` substeps. Answers how far each fell.
#[cfg(feature = "dim3")]
fn articulation_and_box_falling(rate: Real, substeps: usize) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let (root, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 100.0, 0.0))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.2, 0.2, 0.2).mass(5.0),
    );
    let (arm, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.5, 100.0, 0.0))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.2, 0.05, 0.05).mass(1.0),
    );
    let joint = RevoluteJointBuilder::new(Vector::Z)
        .local_anchor1(Vector::new(0.25, 0.0, 0.0))
        .local_anchor2(Vector::new(-0.25, 0.0, 0.0))
        .contacts_enabled(false)
        .build();
    world.insert_multibody_joint(root, arm, joint).unwrap();
    let link = *world.multibody_joints.rigid_body_link(root).unwrap();
    world
        .multibody_joints
        .get_multibody_mut(link.multibody)
        .unwrap()
        .damping_mut()
        .fill(0.0);
    let (lone, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(10.0, 100.0, 0.0))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.2, 0.2, 0.2).mass(5.0),
    );
    for _ in 0..rate as usize {
        world.step();
    }
    (
        100.0 - world.bodies[root].translation().y,
        100.0 - world.bodies[lone].translation().y,
    )
}

/// An articulation dropped from rest falls as a lone body does, to within a two-hundredth of
/// its fall over one second, at 30 steps a second in 2 substeps and at 60 in 2 and in 4. A
/// multibody's kinetic energy is held each substep to what it moved with plus the work of the
/// forces on it, and that work was read as `dt · v · f` alone: a body at rest, or at the turn of
/// its swing, gains `dt² · f · M⁻¹f / 2` from its forces over a substep and nothing else, so the
/// hold took most of what gravity gave every slowly moving multibody. The articulation fell
/// 4.846 m where the lone box fell 4.946 m at 60 steps in 2 substeps, and 4.799 m where it fell
/// 4.987 m at 30.
#[test]
#[cfg(feature = "dim3")]
fn an_articulation_dropped_from_rest_falls_as_a_lone_body_does() {
    let mut failed = Vec::new();
    for (rate, substeps) in [(30.0, 2), (60.0, 2), (60.0, 4)] {
        let (articulation, lone) = articulation_and_box_falling(rate, substeps);
        if (articulation - lone).abs() > lone * 0.005 {
            failed.push(alloc::format!(
                "{rate} Hz x {substeps}: the articulation fell {articulation} m, the lone box {lone} m"
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "an articulation fell unlike a lone body: {failed:?}"
    );
}

/// The least and the most the kinetic and potential energy of a 1 kg capsule `length` m long,
/// hung from a fixed body on an undamped hinge and let go `angle` degrees from hanging, stood
/// above or below where it was let go over ten seconds at `rate` steps a second in `substeps`
/// substeps, as shares of the energy its swing has.
#[cfg(feature = "dim3")]
fn undamped_hinge_swing(length: Real, angle: Real, rate: Real, substeps: usize) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let anchor_at = Vector::new(0.0, 20.0, 0.0);
    let angle = angle.to_radians();
    let raised = Rotation::from_axis_angle(Vector::X, angle);
    let down = Vector::new(0.0, -angle.cos(), -angle.sin());
    let (anchor, _) = world.insert(
        RigidBodyBuilder::fixed()
            .translation(anchor_at)
            .rotation(raised.to_scaled_axis()),
        ColliderBuilder::ball(0.05).sensor(true),
    );
    let (link, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(anchor_at + down * (length * 0.5))
            .rotation(raised.to_scaled_axis())
            .can_sleep(false),
        ColliderBuilder::capsule_y(length * 0.3, 0.07).mass(1.0),
    );
    let joint = RevoluteJointBuilder::new(Vector::X)
        .local_anchor1(Vector::ZERO)
        .local_anchor2(Vector::new(0.0, length * 0.5, 0.0))
        .contacts_enabled(false)
        .build();
    world.insert_multibody_joint(anchor, link, joint).unwrap();
    let id = *world.multibody_joints.rigid_body_link(link).unwrap();
    world
        .multibody_joints
        .get_multibody_mut(id.multibody)
        .unwrap()
        .damping_mut()
        .fill(0.0);
    let swing = 9.81 * length * 0.5 * (1.0 - angle.cos());
    let start = links_energy(&world.bodies, &[link], world.gravity);
    let (mut least, mut most): (Real, Real) = (0.0, 0.0);
    for _ in 0..(10.0 * rate) as usize {
        world.step();
        let energy = links_energy(&world.bodies, &[link], world.gravity) - start;
        least = least.min(energy / swing);
        most = most.max(energy / swing);
    }
    (least, most)
}

/// An undamped hinge let go from 30 degrees on a 1 m link, and from 10 degrees on a 0.3 m one,
/// swings for ten seconds at 30 and 60 steps a second in 2 substeps with its energy within 7%
/// of the energy its swing has, above or below: the bounded swing of a semi-implicit step's
/// energy through each period, which a 0.3 m link at 30 steps a second spans at about 6%. With
/// the work of gravity read as `dt · v · f` alone, the energy hold took a share of that work at
/// every turn of the swing, and the 0.3 m hinge kept 24% of its energy over the ten seconds at
/// 30 steps a second, the 1 m hinge 62%.
#[test]
#[cfg(feature = "dim3")]
fn an_undamped_hinge_swings_as_high_as_it_was_let_go() {
    let mut failed = Vec::new();
    for (length, angle) in [(1.0 as Real, 30.0 as Real), (0.3, 10.0)] {
        for rate in [30.0, 60.0] {
            let (least, most) = undamped_hinge_swing(length, angle, rate, 2);
            if least < -0.07 || most > 0.07 {
                failed.push(alloc::format!(
                    "{length} m from {angle}° at {rate} Hz x 2: energy from {least} to {most} of its swing's"
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "an undamped hinge lost or gained energy: {failed:?}"
    );
}

/// The ragdoll of [`ragdoll_landing_on_stairs`] landing on the stairs from seven heights (2.35
/// to 2.95 m) under five pushes (30 to 90 N·s), at 30 steps a second in 2 substeps, with the
/// continuous-collision pass and without it, goes no more than a degree past any limit. Its
/// head, struck round its swing cone as the chest hit a step, turned 50 degrees in a substep:
/// the cone's row, read where the substep began along the way the head then stood off the
/// cone's axis, held neither the head swinging through the cone's axis nor sliding round its
/// bound, and a joint went 14.1 degrees past its limit with the pass and 13.1 without it; with
/// the substeps of the commit before, 1.9 degrees.
#[test]
#[cfg(feature = "dim3")]
fn a_ragdoll_landing_on_stairs_keeps_every_joint_within_a_degree_of_its_limits() {
    let at = Vector::new(1837.0, 20.0, 6.0);
    let mut failed = Vec::new();
    for height in [2.35 as Real, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95] {
        for push in [30.0 as Real, 45.0, 60.0, 75.0, 90.0] {
            for ccd_substeps in [1, 0] {
                let (past, _) =
                    ragdoll_landing_on_stairs(at, 30.0, 2, ccd_substeps, height, push);
                if !(past <= 1.0) {
                    failed.push(alloc::format!(
                        "height {height}, push {push}, continuous passes {ccd_substeps}: {past} degrees past a limit"
                    ));
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "a joint of the ragdoll went past its limit: {failed:?}"
    );
}

/// A link on a spherical joint whose swing is held to a 30° cone, standing 29° off the cone's
/// axis and spun round the cone at 300 rad/s, with no gravity, stays within one and a half
/// degrees of its cone over ten steps at 30 steps a second in 1 substep. Its substeps each turn
/// it more than a radian: its cone row read where a whole substep carried the joint, 2.5 rad on,
/// left it 48.7 degrees off the cone's axis.
#[test]
#[cfg(feature = "dim3")]
fn a_link_spun_round_its_cone_faster_than_its_substeps_hold_stays_on_its_cone() {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = 1.0 / 30.0;
    world.integration_parameters.num_solver_iterations = 1;
    let (root, _) = world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::ball(0.05).sensor(true),
    );
    let off = Rotation::from_axis_angle(Vector::Z, (29.0 as Real).to_radians());
    let (link, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(off * Vector::new(0.5, 0.0, 0.0))
            .rotation(off.to_scaled_axis())
            .can_sleep(false),
        ColliderBuilder::cuboid(0.5, 0.05, 0.05).mass(1.0),
    );
    let mut joint: GenericJoint = SphericalJointBuilder::new()
        .local_anchor1(Vector::ZERO)
        .local_anchor2(Vector::new(-0.5, 0.0, 0.0))
        .contacts_enabled(false)
        .build()
        .into();
    let cone = (30.0 as Real).to_radians();
    joint.coupled_axes |= JointAxesMask::ANG_Y | JointAxesMask::ANG_Z;
    joint.set_limits(JointAxis::AngY, [-cone, cone]);
    joint.set_limits(JointAxis::AngZ, [-cone, cone]);
    world.insert_multibody_joint(root, link, joint).unwrap();
    let id = *world.multibody_joints.rigid_body_link(link).unwrap();
    let multibody = world
        .multibody_joints
        .get_multibody_mut(id.multibody)
        .unwrap();
    multibody.damping_mut().fill(0.0);
    let first = multibody.link(id.id).unwrap().assembly_id;
    // Square to the link and to the cone's axis: the link's own way round the cone.
    let round = (off * Vector::X).cross(Vector::Z).normalize() * 300.0;
    multibody.velocities[first] = round.x;
    multibody.velocities[first + 1] = round.y;
    multibody.velocities[first + 2] = round.z;
    let mut widest: Real = 0.0;
    for _ in 0..10 {
        world.step();
        let axis = *world.bodies[link].rotation() * Vector::X;
        widest = widest.max(axis.dot(Vector::X).clamp(-1.0, 1.0).acos().to_degrees());
    }
    assert!(
        widest <= 31.5,
        "the link stood {widest} degrees off its cone's axis, on a 30 degree cone"
    );
}

/// A hinge left by its substep turning far faster than the energy it was held to allows is held
/// to that energy plus the work gravity does at the velocity the hold leaves it with, not at
/// the velocity the substep blew up to. A 1 kg arm 0.5 m from its hinge, held to the energy of
/// 1 rad/s and left turning down at 1000 rad/s, was allowed the work gravity does at 1000 rad/s
/// over a 60 Hz step, 82 J, and kept turning at 25.5 rad/s.
#[test]
#[cfg(feature = "dim3")]
fn a_blown_up_velocity_is_held_with_the_work_of_the_velocity_it_is_held_to() {
    let dt: Real = 1.0 / 60.0;
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = dt;
    let (root, _) = world.insert(RigidBodyBuilder::fixed(), ColliderBuilder::ball(0.05));
    let (arm, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.5, 0.0, 0.0))
            .can_sleep(false),
        ColliderBuilder::ball(0.05).mass(1.0),
    );
    let joint = RevoluteJointBuilder::new(Vector::Z)
        .local_anchor2(Vector::new(-0.5, 0.0, 0.0))
        .contacts_enabled(false)
        .build();
    world.insert_multibody_joint(root, arm, joint).unwrap();
    world.step();
    let link = *world.multibody_joints.rigid_body_link(arm).unwrap();
    let multibody = world
        .multibody_joints
        .get_multibody_mut(link.multibody)
        .unwrap();
    multibody.damping_mut().fill(0.0);
    let turn_at =
        |multibody: &mut crate::dynamics::Multibody, bodies: &mut RigidBodySet, omega: Real| {
            multibody.velocities.fill(omega);
            multibody.update_velocities(bodies);
            multibody.update_mass_matrix(dt, bodies);
            multibody.update_acceleration(dt, bodies, super::multibody::MultibodyForces::Applied);
            multibody
                .kinetic_energy_to_hold(multibody.velocities.as_view(), bodies)
                .unwrap()
        };
    let held = turn_at(multibody, &mut world.bodies, -1.0);
    assert_eq!(
        multibody.step_end_energy_scale(held, true, dt, &world.bodies),
        Some(1.0),
        "a hinge turning at the energy it was held to is left as it is"
    );
    turn_at(multibody, &mut world.bodies, -1000.0);
    let scale = multibody
        .step_end_energy_scale(held, true, dt, &world.bodies)
        .expect("a hinge on a fixed root has no drift");
    let turning = 1000.0 * scale;
    assert!(
        turning < 2.0,
        "the hinge held to the energy of 1 rad/s kept turning at {turning} rad/s"
    );
}

/// The hooks of [`carried_by_a_conveyor`]: every contact's surface slides at 2 m/s along x.
#[cfg(feature = "dim3")]
struct Conveyor;

#[cfg(feature = "dim3")]
impl PhysicsHooks for Conveyor {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        if let Some(rigid) = context.rigid_mut() {
            for contact in rigid.solver_contacts.iter_mut() {
                contact.tangent_velocity = Vector::new(2.0, 0.0, 0.0);
            }
        }
    }
}

/// A lone 5 kg box, and an articulation of two 5 kg boxes on a revolute joint, resting on a fixed
/// ground whose surface slides at 2 m/s along x (a conveyor's, [`Conveyor`]), at `rate` steps a
/// second in `substeps` substeps. Answers how fast the lone box and the articulation's root move
/// along the ground after one second.
#[cfg(feature = "dim3")]
fn carried_by_a_conveyor(rate: Real, substeps: usize) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let hooked = |collider: ColliderBuilder| {
        collider
            .friction(1.0)
            .active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS)
    };
    world.insert(
        RigidBodyBuilder::fixed(),
        hooked(ColliderBuilder::cuboid(100.0, 0.5, 100.0)),
    );
    let mut boxed = |x: Real, z: Real| {
        world
            .insert(
                RigidBodyBuilder::dynamic()
                    .translation(Vector::new(x, 0.7, z))
                    .can_sleep(false),
                hooked(ColliderBuilder::cuboid(0.2, 0.2, 0.2).mass(5.0)),
            )
            .0
    };
    let lone = boxed(0.0, -5.0);
    let root = boxed(0.0, 5.0);
    let arm = boxed(0.5, 5.0);
    let joint = RevoluteJointBuilder::new(Vector::Z)
        .local_anchor1(Vector::new(0.25, 0.0, 0.0))
        .local_anchor2(Vector::new(-0.25, 0.0, 0.0))
        .contacts_enabled(false)
        .build();
    world.insert_multibody_joint(root, arm, joint).unwrap();
    for _ in 0..rate as usize {
        world.step_with_events(&Conveyor, &());
    }
    let along = |body: RigidBodyHandle| {
        let v = world.bodies[body].linvel();
        (v.x * v.x + v.z * v.z).sqrt()
    };
    (along(lone), along(root))
}

/// An articulation resting on a conveyor is carried at the belt's speed, as a lone box is, after
/// a second at 30 steps a second in 2 substeps and at 60 in 2 and in 4. The contact of a fixed
/// body whose surface slides along itself does work on what rests on it through its friction
/// rows' target velocity: a multibody whose rows tied it to nothing that moves was held to the
/// energy it started each substep with, and after a second on a 2 m/s belt the articulation
/// moved at 0.69 m/s at 60 steps a second in 4 substeps, 0.98 m/s in 2 and 1.42 m/s at 30.
#[test]
#[cfg(feature = "dim3")]
fn an_articulation_on_a_conveyor_is_carried_as_a_lone_box_is() {
    let mut failed = Vec::new();
    for (rate, substeps) in [(30.0, 2), (60.0, 2), (60.0, 4)] {
        let (lone, articulation) = carried_by_a_conveyor(rate, substeps);
        if (lone - 2.0).abs() > 0.05 || (articulation - lone).abs() > 0.05 {
            failed.push(alloc::format!(
                "{rate} Hz x {substeps}: the lone box moved at {lone} m/s, the articulation at {articulation} m/s"
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "an articulation on a 2 m/s conveyor was not carried as a lone box is: {failed:?}"
    );
}

/// A lone 5 kg box, and the 5 kg root box of an articulation carrying a 5 kg arm on a revolute
/// joint, each held to a fixed body by an impulse revolute joint about y whose motor turns it at
/// 2 rad/s, with no gravity, at `rate` steps a second in `substeps` substeps. Answers how fast the
/// lone box and the articulation's root turn about y after one second.
#[cfg(feature = "dim3")]
fn turned_by_a_motor_from_a_fixed_body(rate: Real, substeps: usize) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let motor = || {
        RevoluteJointBuilder::new(Vector::Y)
            .motor_velocity(2.0, 1.0e4)
            .contacts_enabled(false)
            .build()
    };
    let mut held_at = |z: Real| {
        let (anchor, _) = world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, 0.0, z)),
            ColliderBuilder::ball(0.05).sensor(true),
        );
        let (body, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 0.0, z))
                .can_sleep(false),
            ColliderBuilder::cuboid(0.2, 0.2, 0.2).mass(5.0),
        );
        (anchor, body)
    };
    let (lone_anchor, lone) = held_at(-5.0);
    let (root_anchor, root) = held_at(5.0);
    let (arm, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.5, 0.0, 5.0))
            .can_sleep(false),
        ColliderBuilder::cuboid(0.2, 0.2, 0.2).mass(5.0),
    );
    let hinge = RevoluteJointBuilder::new(Vector::Z)
        .local_anchor1(Vector::new(0.25, 0.0, 0.0))
        .local_anchor2(Vector::new(-0.25, 0.0, 0.0))
        .contacts_enabled(false)
        .build();
    world.insert_multibody_joint(root, arm, hinge).unwrap();
    world.insert_impulse_joint(lone_anchor, lone, motor());
    world.insert_impulse_joint(root_anchor, root, motor());
    for _ in 0..rate as usize {
        world.step();
    }
    (world.bodies[lone].angvel().y, world.bodies[root].angvel().y)
}

/// An articulation whose root an impulse joint's motor turns from a fixed body turns at the
/// motor's speed, as a lone box does, after a second at 30 steps a second in 2 substeps and at 60
/// in 2 and in 4. The motor's row does work on the multibody from a body that does not move: a
/// multibody whose rows tied it to nothing that moves was held to the energy it started each
/// substep with, and an articulation at rest never turned.
#[test]
#[cfg(feature = "dim3")]
fn an_articulation_turned_by_a_motor_from_a_fixed_body_turns_as_a_lone_box_does() {
    let mut failed = Vec::new();
    for (rate, substeps) in [(30.0, 2), (60.0, 2), (60.0, 4)] {
        let (lone, articulation) = turned_by_a_motor_from_a_fixed_body(rate, substeps);
        if (lone - 2.0).abs() > 0.05 || (articulation - lone).abs() > 0.05 {
            failed.push(alloc::format!(
                "{rate} Hz x {substeps}: the lone box turned at {lone} rad/s, the articulation at {articulation} rad/s"
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "an articulation turned by a 2 rad/s motor did not turn as a lone box does: {failed:?}"
    );
}

/// How a free chain under a heavy ball swung ([`heavy_ball_on_light_links`]).
#[cfg(feature = "dim3")]
struct HeavyBallSwing {
    /// The least the chain's kinetic and potential energy stood from where it started, as a
    /// share of the energy its swing has (negative below).
    least: Real,
    /// The most it stood above where it started, as a share of that energy.
    most: Real,
    /// Where it stood at the end, as a share of that energy.
    end: Real,
    /// How far the ball rose above where it was let go, after its first two seconds.
    rose: Real,
    /// The step its state stopped being a number at, if it did.
    not_finite_at: Option<usize>,
    /// The most the chain's energy fell over one step, as a share of the energy its swing has.
    steepest_fall: Real,
    /// The number of steps in which two of the chain's links, the ball among them, touched.
    knocks: usize,
}

/// Twelve 4 kg capsule links 0.55 m apart on spherical multibody joints, hung from a fixed body
/// whose box is a sensor, a ball of `ball_mass` kg 0.42 m wide on the end, raised 70 degrees
/// and let go with nothing touching it, stepped for eight seconds at `rate` steps a second in
/// `substeps` substeps. The joints are undamped with `undamped`, and keep the damping a
/// multibody joint is built with otherwise.
#[cfg(feature = "dim3")]
fn heavy_ball_on_light_links(
    ball_mass: Real,
    rate: Real,
    substeps: usize,
    undamped: bool,
) -> HeavyBallSwing {
    heavy_ball_on_light_links_framed(
        ball_mass,
        rate,
        substeps,
        undamped,
        Rotation::IDENTITY,
        60.0,
    )
}

/// The chain of [`heavy_ball_on_light_links`] hung at `x` along X, with each joint's frame turned
/// by `frame` in both of its bodies. The frames turned from the X axis to the Y axis, as an engine builds a joint
/// about an axis along each link, round the chain's state off the plane it was raised in: the
/// chain leaves that plane as it swings, and its links knock on one another.
#[cfg(feature = "dim3")]
fn heavy_ball_on_light_links_framed(
    ball_mass: Real,
    rate: Real,
    substeps: usize,
    undamped: bool,
    frame: Rotation,
    x: Real,
) -> HeavyBallSwing {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let anchor_at = Vector::new(x, 16.0, 60.0);
    let (pitch, count, ball_radius, eye) = (0.55 as Real, 12, 0.42 as Real, 0.12 as Real);
    let angle = (70.0 as Real).to_radians();
    let raised = Rotation::from_axis_angle(Vector::X, angle);
    let down = Vector::new(0.0, -angle.cos(), -angle.sin());
    let (anchor, _) = world.insert(
        RigidBodyBuilder::fixed()
            .translation(anchor_at)
            .rotation(raised.to_scaled_axis()),
        ColliderBuilder::cuboid(0.5, 0.5, 0.5).sensor(true),
    );
    let mut bodies = Vec::new();
    for i in 0..count {
        let (link, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(anchor_at + down * (pitch * (i as Real + 0.5)))
                .rotation(raised.to_scaled_axis())
                .can_sleep(false),
            ColliderBuilder::capsule_y(pitch * 0.3, 0.07).mass(4.0),
        );
        bodies.push(link);
    }
    let (ball, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(anchor_at + down * (pitch * count as Real + eye + ball_radius))
            .rotation(raised.to_scaled_axis())
            .can_sleep(false),
        ColliderBuilder::ball(ball_radius).mass(ball_mass),
    );
    bodies.push(ball);
    for (i, body) in bodies.iter().enumerate() {
        let parent = if i == 0 { anchor } else { bodies[i - 1] };
        let anchor1 = if i == 0 {
            Vector::ZERO
        } else {
            Vector::new(0.0, -pitch / 2.0, 0.0)
        };
        let anchor2 = if i == count {
            Vector::new(0.0, ball_radius + eye, 0.0)
        } else {
            Vector::new(0.0, pitch / 2.0, 0.0)
        };
        let joint = SphericalJointBuilder::new()
            .local_frame1(Pose::from_parts(anchor1, frame))
            .local_frame2(Pose::from_parts(anchor2, frame))
            .contacts_enabled(false)
            .build();
        world.insert_multibody_joint(parent, *body, joint).unwrap();
    }
    if undamped {
        let id = *world.multibody_joints.rigid_body_link(ball).unwrap();
        world
            .multibody_joints
            .get_multibody_mut(id.multibody)
            .unwrap()
            .damping_mut()
            .fill(0.0);
    }
    let swing = (ball_mass + 4.0 * count as Real / 2.0)
        * 9.81
        * (pitch * count as Real + eye + ball_radius)
        * (1.0 - angle.cos());
    let released = world.bodies[ball].translation().y;
    let start = links_energy(&world.bodies, &bodies, world.gravity);
    let mut result = HeavyBallSwing {
        least: 0.0,
        most: 0.0,
        end: 0.0,
        rose: Real::MIN,
        not_finite_at: None,
        steepest_fall: 0.0,
        knocks: 0,
    };
    for step in 0..(8.0 * rate) as usize {
        world.step();
        if world
            .narrow_phase
            .contact_pairs()
            .any(|pair| pair.has_any_active_contact())
        {
            result.knocks += 1;
        }
        let body = &world.bodies[ball];
        if !body.translation().is_finite() || !body.linvel().is_finite() {
            result.not_finite_at = Some(step);
            break;
        }
        let energy = (links_energy(&world.bodies, &bodies, world.gravity) - start) / swing;
        result.steepest_fall = result.steepest_fall.max(result.end - energy);
        result.least = result.least.min(energy);
        result.most = result.most.max(energy);
        result.end = energy;
        if step as Real >= 2.0 * rate {
            result.rose = result.rose.max(body.translation().y - released);
        }
    }
    result
}

/// A free chain of twelve 4 kg links on undamped spherical joints with a 5000 kg and a 20000
/// kg ball on the end, let go 70 degrees up with nothing touching it, keeps its energy within
/// 2% of the energy its swing has, above and below, over eight seconds at 30 steps a second in
/// 2 substeps, 60 in 2 and in 4, 120 in 1 and 240 in 1, and its ball never rises a tenth of a
/// metre above where it was let go. Each substep's energy was held only from above: the semi-implicit Coriolis
/// update of a move takes energy from it, the rounding of the move gives some back as often,
/// and every gain was taken away while every loss was kept. The 5000 kg chain ended its eight
/// seconds 74% of its swing down at 30 steps in 2 substeps, after standing 16% above it, and
/// 48% down at 120 in 1, and at 60 in 2 its ball rose a metre above where it was let go; the
/// 20000 kg chain ended 88% down at 60 in 2.
#[test]
#[cfg(feature = "dim3")]
fn a_free_chain_under_a_heavy_ball_keeps_the_energy_of_its_swing() {
    let mut failed = Vec::new();
    for ball_mass in [5000.0 as Real, 20000.0] {
        for (rate, substeps) in [(30.0, 2), (60.0, 2), (60.0, 4), (120.0, 1), (240.0, 1)] {
            let swing = heavy_ball_on_light_links(ball_mass, rate, substeps, true);
            if swing.not_finite_at.is_some()
                || swing.least < -0.02
                || swing.most > 0.02
                || swing.end.abs() > 0.02
                || swing.rose > 0.1
            {
                failed.push(alloc::format!(
                    "{ball_mass} kg, {rate} Hz x {substeps}: not finite at {:?}, energy from {} \
                     to {} of its swing's, {} at the end, rose {} m",
                    swing.not_finite_at,
                    swing.least,
                    swing.most,
                    swing.end,
                    swing.rose
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "a free chain under a heavy ball lost or gained energy: {failed:?}"
    );
}

/// The chain of [`a_free_chain_under_a_heavy_ball_keeps_the_energy_of_its_swing`] with a 2000,
/// a 5000 and a 20000 kg ball, its joints keeping the damping a multibody joint is built with,
/// swings at 30 steps a second in 2 substeps, 60 in 2 and in 4, 120 in 1 and 240 in 1 with its
/// energy never 3% of its swing's below where it started nor 2% above, no step taking 2% of
/// it, and its ball never a tenth of a metre above where it was let go. What the damping took
/// from a kick was solved through the mass matrix the accelerations are solved with, whose
/// semi-implicit Coriolis terms are near singular while a light link whips under the heavy
/// ball: the share came out many times the energy the chain had, the hold stood the chain at
/// less than nothing and stopped it dead within a step, and the 20000 kg chain lost 84% of its
/// swing at 30 steps in 2 substeps.
#[test]
#[cfg(feature = "dim3")]
fn a_free_chain_under_a_heavy_ball_on_damped_joints_keeps_its_swing() {
    let mut failed = Vec::new();
    for ball_mass in [2000.0 as Real, 5000.0, 20000.0] {
        for (rate, substeps) in [(30.0, 2), (60.0, 2), (60.0, 4), (120.0, 1), (240.0, 1)] {
            let swing = heavy_ball_on_light_links(ball_mass, rate, substeps, false);
            if swing.not_finite_at.is_some()
                || swing.least < -0.03
                || swing.most > 0.02
                || swing.steepest_fall > 0.02
                || swing.rose > 0.1
            {
                failed.push(alloc::format!(
                    "{ball_mass} kg, {rate} Hz x {substeps}: not finite at {:?}, energy from {} \
                     to {} of its swing's, {} at the end, {} in one step at most, rose {} m",
                    swing.not_finite_at,
                    swing.least,
                    swing.most,
                    swing.end,
                    swing.steepest_fall,
                    swing.rose
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "a free chain under a heavy ball on damped joints did not keep its swing: {failed:?}"
    );
}

/// The chain of [`a_free_chain_under_a_heavy_ball_on_damped_joints_keeps_its_swing`] hung at x
/// 760 on joints whose frames are turned from the X axis to the Y axis
/// ([`heavy_ball_on_light_links_framed`]), with a 2000, a 5000 and a 20000 kg ball, at 30 steps
/// a second in 2 substeps, 60 in 2 and in 4, 120 in 1 and 240 in 1: it leaves the plane it was
/// raised in, and in some of these swings its links knock on one another (the 5000 kg chain at
/// 30 and 60 steps a second in 2 substeps; the 20000 kg chain at 60 in 2 and 120 in 1 with
/// `enhanced-determinism`). Its energy never stands 3% of its swing's below where it started
/// nor 2% above, no step takes 2% of it, and its state stays a number. A substep of a knock
/// held the chain's energy only from above, keeping every loss the update of its whipping
/// links made and none of the gains: the 5000 kg chain lost 19% of its swing in one step and
/// 72% over the eight seconds at 30 steps a second in 2 substeps, and 8% at 60 in 2; with
/// `enhanced-determinism`, the 20000 kg chain lost 23% in one step and 58% at 60 in 2, and 9%
/// at 120 in 1. A knock in the first substep of a step read the energy the chain stood at from
/// the velocity it opened the step with, 53 kJ below the energy its last step ended at: with
/// `enhanced-determinism`, the 20000 kg chain at 120 steps a second lost 6% of its swing in the
/// step its ball struck its eleventh link, of which the knock took 0.3%.
#[test]
#[cfg(feature = "dim3")]
fn a_free_chain_whose_links_knock_on_one_another_keeps_its_swing() {
    let frame = Rotation::from_rotation_arc(Vector::X, Vector::Y);
    let mut failed = Vec::new();
    let mut knocked = Vec::new();
    for ball_mass in [2000.0 as Real, 5000.0, 20000.0] {
        for (rate, substeps) in [(30.0, 2), (60.0, 2), (60.0, 4), (120.0, 1), (240.0, 1)] {
            let swing =
                heavy_ball_on_light_links_framed(ball_mass, rate, substeps, false, frame, 760.0);
            if swing.knocks > 0 {
                knocked.push((ball_mass, rate, substeps, swing.knocks));
            }
            if swing.not_finite_at.is_some()
                || swing.least < -0.03
                || swing.most > 0.02
                || swing.steepest_fall > 0.02
            {
                failed.push(alloc::format!(
                    "{ball_mass} kg, {rate} Hz x {substeps}: not finite at {:?}, energy from {} \
                     to {} of its swing's, {} at the end, {} in one step at most, {} steps \
                     knocking",
                    swing.not_finite_at,
                    swing.least,
                    swing.most,
                    swing.end,
                    swing.steepest_fall,
                    swing.knocks
                ));
            }
        }
    }
    assert!(
        knocked.len() >= 2,
        "the chain's links knocked on one another in fewer than two swings: {knocked:?}"
    );
    assert!(
        failed.is_empty(),
        "a free chain whose links knock on one another did not keep its swing: {failed:?}"
    );
}

/// A link of 1 kg spinning at 5 rad/s on a revolute joint about its centre, hung from a fixed
/// body with no gravity, nothing touching it and its joint's `damping` the one force on it,
/// stepped for `seconds` at `rate` steps a second in `substeps` substeps: answers its angular
/// velocity at the end and the one the joint's implicit damping gives it,
/// `5 · (I / (I + h · damping))ⁿ` over the `n` substeps of length `h`.
#[cfg(feature = "dim3")]
fn spin_on_a_damped_joint(
    damping: Real,
    rate: Real,
    substeps: usize,
    seconds: Real,
) -> (Real, Real) {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    let at = Vector::new(0.0, 5.0, 0.0);
    let root = world.insert_body(RigidBodyBuilder::fixed().translation(at));
    let (link, _) = world.insert(
        RigidBodyBuilder::dynamic().translation(at).can_sleep(false),
        ColliderBuilder::cuboid(1.0, 0.1, 0.1).mass(1.0),
    );
    let joint = RevoluteJointBuilder::new(Vector::Z)
        .contacts_enabled(false)
        .build();
    world.insert_multibody_joint(root, link, joint).unwrap();
    let id = *world.multibody_joints.rigid_body_link(link).unwrap();
    {
        let multibody = world
            .multibody_joints
            .get_multibody_mut(id.multibody)
            .unwrap();
        multibody.damping_mut().fill(damping);
        multibody.velocities.fill(5.0);
    }
    let inertia = world.bodies[link]
        .mass_properties()
        .local_mprops
        .principal_inertia()
        .z;
    let steps = (seconds * rate) as usize;
    for _ in 0..steps {
        world.step();
    }
    let h = 1.0 / (rate * substeps as Real);
    let implicit = 5.0 * (inertia / (inertia + h * damping)).powi((steps * substeps) as i32);
    (world.bodies[link].angvel().z, implicit)
}

/// A link spinning on a revolute joint whose damping is the one force on it slows as the
/// joint's implicit damping slows it, to within a thousandth, at 30 steps a second in 2
/// substeps, 60 in 2 and in 4, 120 in 1 and 240 in 1, with a damping of 0.3, 1, 10 and 100
/// times its moment of inertia per second. A substep no row bears on is held to the energy
/// the multibody stood at before its move, less what the damping takes. Counted over the half
/// kick that leads to the energy it stands at next and not over the half that follows the
/// energy it stood at, the hold gave back half of what the damping took, and a step's end
/// held its velocity to that energy without the damping's share; under the light dampings
/// the link slowed half as fast as they set, and under the heaviest it kept thousands of
/// times the velocity the damping leaves it.
#[test]
#[cfg(feature = "dim3")]
fn a_damped_joint_slows_its_link_as_its_damping_does() {
    let mut failed = Vec::new();
    for (rate, substeps) in [(30.0, 2), (60.0, 2), (60.0, 4), (120.0, 1), (240.0, 1)] {
        for (per_inertia, seconds) in [
            (0.3 as Real, 2.0 as Real),
            (1.0, 2.0),
            (10.0, 0.5),
            (100.0, 0.1),
        ] {
            // The link's moment of inertia about Z: 1 kg over a 2 m by 0.2 m box.
            let damping = per_inertia * (2.0 * 2.0 + 0.2 * 0.2) / 12.0;
            let (spin, implicit) = spin_on_a_damped_joint(damping, rate, substeps, seconds);
            if !((spin / implicit - 1.0).abs() <= 1.0e-3) {
                failed.push(alloc::format!(
                    "{per_inertia} I/s, {rate} Hz x {substeps}: {spin} rad/s after {seconds} s, \
                     the joint's damping leaves {implicit}"
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "a link on a damped joint did not slow as its damping does: {failed:?}"
    );
}


/// How far a free chain's linear momentum moved over its flight
/// ([`free_chain_flight`]).
#[cfg(feature = "dim3")]
struct ChainFlight {
    /// The largest change of the chain's linear momentum, `Σ mᵢ · vᵢ` over its links, from the
    /// momentum it was given, over the steps of the flight, as a share of that momentum.
    drift: Real,
    /// The first step at which the chain's momentum stopped being finite.
    not_finite_at: Option<usize>,
}

/// Nine 0.4 m rods, 1 kg each and the last 10 kg, joined end to end by spherical multibody
/// joints with the first the free root, flying through empty space for two seconds at `rate`
/// steps a second in `substeps` substeps: no gravity, every collider a sensor, no joint limit or
/// motor. After one step at rest the root is given 5 m/s along X and every degree of freedom
/// from the root's turn on a velocity of `omega` times a sine of its index and `seed`. Each
/// link's velocity is read, after each step, from the multibody's velocity at the pose the step
/// ended at.
#[cfg(feature = "dim3")]
fn free_chain_flight(omega: Real, rate: Real, substeps: usize, seed: usize) -> ChainFlight {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    world.gravity = Vector::ZERO;
    let mut above = None;
    let mut links = Vec::new();
    let mut first = None;
    for i in 0..9 {
        let mass = if i == 8 { 10.0 } else { 1.0 };
        let (link, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.45 * i as Real, 0.0, 0.0))
                .allow_fast_rotation(true)
                .can_sleep(false),
            ColliderBuilder::cuboid(0.2, 0.05, 0.05)
                .mass(mass)
                .sensor(true),
        );
        if let Some(parent) = above {
            let joint: GenericJoint = SphericalJointBuilder::new()
                .local_anchor1(Vector::new(0.225, 0.0, 0.0))
                .local_anchor2(Vector::new(-0.225, 0.0, 0.0))
                .contacts_enabled(false)
                .build()
                .into();
            let handle = world.insert_multibody_joint(parent, link, joint).unwrap();
            first.get_or_insert(handle);
        }
        links.push(link);
        above = Some(link);
    }
    let first = first.unwrap();
    world.step();
    {
        let (multibody, _) = world.multibody_joints.get_mut(first).unwrap();
        let ndofs = multibody.ndofs();
        let mut velocity = multibody.generalized_velocity_mut();
        velocity[0] = 5.0;
        for k in 3..ndofs {
            velocity[k] = omega * ((k as Real) * 1.37 + (seed as Real) * 0.71).sin();
        }
        multibody.update_velocities(&mut world.bodies);
    }
    let momentum = |bodies: &RigidBodySet| -> Vector {
        links
            .iter()
            .map(|link| bodies[*link].linvel() * bodies[*link].mass())
            .fold(Vector::ZERO, |sum, p| sum + p)
    };
    let given = momentum(&world.bodies);
    let mut flight = ChainFlight {
        drift: 0.0,
        not_finite_at: None,
    };
    for step in 0..(2.0 * rate) as usize {
        world.step();
        let (multibody, _) = world.multibody_joints.get_mut(first).unwrap();
        multibody.update_velocities(&mut world.bodies);
        let p = momentum(&world.bodies);
        if !p.is_finite() {
            flight.not_finite_at = Some(step);
            break;
        }
        flight.drift = flight.drift.max((p - given).length() / given.length());
    }
    flight
}

/// A free chain of nine rods with a heavy end, flying through empty space with its root and its
/// joints turning at up to 10, 20 and 40 rad/s, keeps the linear momentum it was given to within
/// a thousandth over two seconds, at 30 steps a second in 2 substeps, 60 in 2 and in 4, 120 in 1
/// and 240 in 1. Nothing acts on it, and its momentum changes only through the forces on it. A
/// move carried each joint's velocity unchanged in its parent's frame, the semi-implicit
/// Coriolis forces read where the move ended gave back only part of what that did to the links'
/// velocities in space, and the energy hold then scaled the whole velocity, the drift of the
/// chain's centre of mass with it: the chain lost 5% of its momentum over a substep, its
/// momentum moved by 4% to 11% of what it was given at 10 rad/s and by 90% to 190% at 40.
#[test]
#[cfg(feature = "dim3")]
fn a_free_chain_flying_through_empty_space_keeps_its_linear_momentum() {
    let mut failed = Vec::new();
    for omega in [10.0 as Real, 20.0, 40.0] {
        for (rate, substeps) in [(30.0, 2), (60.0, 2), (60.0, 4), (120.0, 1), (240.0, 1)] {
            for seed in 0..2 {
                let flight = free_chain_flight(omega, rate, substeps, seed);
                if flight.not_finite_at.is_some() || !(flight.drift <= 1.0e-3) {
                    failed.push(alloc::format!(
                        "{omega} rad/s, {rate} Hz x {substeps}, seed {seed}: momentum moved by \
                         {} of what it was given, not finite at {:?}",
                        flight.drift,
                        flight.not_finite_at
                    ));
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "a free chain flying through empty space did not keep its linear momentum: {failed:?}"
    );
}

/// A free chain of `links` 0.4 m boxes, 2 kg each, joined end to end by spherical multibody
/// joints with the first the free root, each box turned `tilt` radians about Z more than the one
/// before it, dropped from `height` m onto fixed ground with no restitution and stepped for four
/// seconds at `rate` steps a second in `substeps` substeps. Answers the fastest its centre of
/// mass rose, in m/s, once its lowest box reached the ground.
#[cfg(feature = "dim3")]
fn free_chain_landing(rate: Real, substeps: usize, height: Real, links: usize, tilt: Real) -> Real {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    world.insert(
        RigidBodyBuilder::fixed(),
        ColliderBuilder::cuboid(20.0, 0.5, 20.0).translation(Vector::new(0.0, -0.5, 0.0)),
    );
    let mut above = None;
    let mut boxes = Vec::new();
    for i in 0..links {
        let (link, _) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.45 * i as Real, height, 0.0))
                .rotation(AngVector::new(0.0, 0.0, tilt * i as Real))
                .can_sleep(false),
            ColliderBuilder::cuboid(0.2, 0.05, 0.05)
                .mass(2.0)
                .restitution(0.0)
                .friction(0.6),
        );
        if let Some(parent) = above {
            let joint: GenericJoint = SphericalJointBuilder::new()
                .local_anchor1(Vector::new(0.225, 0.0, 0.0))
                .local_anchor2(Vector::new(-0.225, 0.0, 0.0))
                .contacts_enabled(false)
                .build()
                .into();
            world.insert_multibody_joint(parent, link, joint).unwrap();
        }
        boxes.push(link);
        above = Some(link);
    }
    let mut landed = false;
    let mut fastest: Real = 0.0;
    for _ in 0..(4.0 * rate) as usize {
        world.step();
        let mut rising = 0.0;
        let mut lowest = Real::MAX;
        for body in &boxes {
            let body = &world.bodies[*body];
            rising += body.linvel().y / links as Real;
            lowest = lowest.min(body.center_of_mass().y);
        }
        landed |= lowest < 0.12;
        if landed {
            fastest = fastest.max(rising);
        }
    }
    fastest
}

/// A free chain dropped from 3 m onto fixed ground with no restitution, landing at 7.5 m/s,
/// leaves it no faster than 1 m/s at 30 steps a second in 2 and in 4 substeps and at 60 in 2,
/// lying along it or bent. The energy hold of a substep its contacts bore on kept the drift of
/// its centre of mass, which the contacts' biased pass had pushed up, as it was, whatever energy
/// the drift carried: the chain left the ground at 2 m/s at 30 steps a second in 2 substeps,
/// where on impulse joints it leaves it at 0.6.
#[test]
#[cfg(feature = "dim3")]
fn a_free_chain_dropped_onto_the_ground_leaves_it_no_faster_than_a_metre_a_second() {
    let mut failed = Vec::new();
    for (rate, substeps) in [(30.0 as Real, 2), (30.0, 4), (60.0, 2)] {
        for (links, tilt) in [(5, 0.05 as Real), (3, 0.2)] {
            let fastest = free_chain_landing(rate, substeps, 3.0, links, tilt);
            if !(fastest <= 1.0) {
                failed.push(alloc::format!(
                    "{rate} Hz x {substeps}, {links} links: rose at {fastest} m/s"
                ));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "a free chain dropped onto the ground left it too fast: {failed:?}"
    );
}
