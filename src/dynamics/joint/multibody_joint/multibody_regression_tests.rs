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

/// A level chain of eight thin rods (0.4 m by 0.1 m along X, 1 kg each, 10 kg the last)
/// joined by spherical joints whose swing is held to a 40° cone about the joint frames' X
/// axis, which `frame` turns from the rods' length, hung from a fixed root at `at` and let fall
/// for 5 s at `rate` steps a second. Returns the step a link's pose first read not finite at,
/// if one did, and the widest any joint swung before it.
#[cfg(feature = "dim3")]
fn level_coned_chain(at: Vector, rate: Real, frame: Rotation) -> (Option<usize>, Real) {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / rate;
    // A breath of sideways pull, so the chain does not fall in one exact plane.
    world.gravity.z = 0.01;
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
    let mut widest: Real = 0.0;
    for step in 0..(5.0 * rate) as usize {
        world.step();
        for (k, link) in links.iter().enumerate() {
            let pose = *world.bodies[*link].position();
            if !pose.translation.is_finite() || !pose.rotation.is_finite() {
                return (Some(step), widest);
            }
            let parent = if k == 0 { root } else { links[k - 1] };
            let rel = world.bodies[parent].rotation().inverse() * pose.rotation;
            let axis = frame * Vector::X;
            widest = widest.max(axis.dot(rel * axis).clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    (None, widest)
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
                let (nan, swing) = level_coned_chain(at, rate, frame);
                widest = widest.max(swing);
                if let Some(step) = nan {
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
