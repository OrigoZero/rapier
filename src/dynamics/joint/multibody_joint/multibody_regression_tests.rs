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
}

/// Twelve 4 kg capsule links, 0.55 m apart, hung on spherical multibody joints from a fixed
/// anchor whose collider is a 1 m box centred on the first joint, with a 500 kg ball on the
/// end, raised `angle` degrees and let go for 8 s at `rate` steps a second in `substeps`
/// substeps. The joints switch contacts off between the bodies they join, so the second link
/// is the first that meets the anchor's box, near the root. With `anchor_collides` false the
/// box is a sensor and the chain swings free of it.
#[cfg(feature = "dim3")]
fn chain_by_its_anchor(
    angle: Real,
    rate: Real,
    substeps: usize,
    anchor_collides: bool,
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
        ColliderBuilder::cuboid(0.5, 0.5, 0.5).sensor(!anchor_collides),
    );
    let mut bodies = Vec::new();
    for i in 0..links {
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
            .translation(anchor_at + down * (pitch * links as Real + eye + ball_radius))
            .rotation(raised.to_scaled_axis())
            .can_sleep(false),
        ColliderBuilder::ball(ball_radius).mass(500.0),
    );
    bodies.push(ball);
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
    let mut swing = AnchoredChainSwing {
        not_finite_at: None,
        fastest: 0.0,
        rose: Real::MIN,
    };
    for step in 0..(8.0 * rate) as usize {
        world.step();
        let ball = &world.bodies[ball];
        if !ball.translation().is_finite() || !ball.linvel().is_finite() {
            swing.not_finite_at = Some(step);
            return swing;
        }
        swing.fastest = swing.fastest.max(ball.linvel().length());
        if step as Real >= 2.0 * rate {
            swing.rose = swing.rose.max(ball.translation().y - released);
        }
    }
    swing
}

/// A chain of light links with a heavy ball on the end, whose second link slides on its fixed
/// anchor's body as it swings, swings as the same chain clear of that body does: no more than a
/// tenth faster, and never a quarter metre above where it was released. Friction is solved in
/// the stabilization pass that ends each substep, and the Coriolis forces of the next substep
/// were read from the velocity the substep moved with, before that pass: they did work on the
/// velocity the chain carried on, each substep, and at 60 steps a second in 4 substeps the ball
/// was flung at 1000 m/s and rose 8 m above its release.
#[test]
#[cfg(feature = "dim3")]
fn a_chain_sliding_on_its_anchors_body_swings_as_the_free_chain_does() {
    let mut failed = Vec::new();
    for angle in [40.0, 55.0, 70.0, 85.0] {
        for substeps in [4, 8] {
            let free = chain_by_its_anchor(angle, 60.0, substeps, false);
            let swing = chain_by_its_anchor(angle, 60.0, substeps, true);
            let finite = swing.not_finite_at.is_none() && free.not_finite_at.is_none();
            if !finite || swing.fastest > free.fastest * 1.1 || swing.rose > 0.25 {
                failed.push(alloc::format!(
                    "{angle}°, {substeps} substeps: not finite at {:?}, fastest {} m/s (free {} m/s), rose {} m",
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
