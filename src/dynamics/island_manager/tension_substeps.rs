//! The extra substeps the tension of its joints asks of a body.
//!
//! A joint pulling a body at an arm from its centre of mass with a force `f` holds the body
//! upright against a tilt as a spring of stiffness `|f| |arm|` would: a light link of a chain
//! under a heavy load sits between two such springs, the tension of everything hanging below it.
//! A substep solves each joint's rows along the directions they stand in at its start, so that
//! stiffness is stepped forward explicitly, and an explicit step of a spring of angular frequency
//! `w` grows without bound once `w dt` passes 2. A chain of 30 links of 1 kg and 0.5 m holding a
//! 50 kg block tilts its top links at about 133 rad/s: in steps of a 30th of a second (`w dt`
//! 4.4) its end stood 7000 times the chain's length from its anchor after 3 s, in steps of a 60th
//! (2.2) it held to 0.07%, and with each link's inertia nine times larger (`w dt` 1.5 at a 30th)
//! it held exactly.
//!
//! Each body asks for as many substeps as keep that tilt stable ([`tension_extra_substeps`]),
//! and the substep solve groups take it the way they take
//! [`crate::dynamics::RigidBody::additional_solver_iterations`]: a connected component runs the
//! most any of its bodies asks for, and the rest of the world runs the substeps it is set to.

use alloc::vec::Vec;

use crate::dynamics::{ImpulseJointSet, JointAxesMask, RigidBodyHandle, RigidBodySet};
use crate::math::{AngularInertia, Real, Rotation, SpatialVector, Vector};

/// The most phase, in radians, the fastest tilt the tension of a body's joints gives it advances
/// in one substep. Three quarters of the bound of 2 leaves room for a load that pulls harder than
/// it did over the step before, and stays above the 1.1 a chain of five 1 kg rods whipping round
/// as it swings reaches at a 120th of a second, which keeps its energy in the substeps it was set
/// to.
const MAX_TENSION_TILT_PHASE: Real = 1.5;

/// The most substeps a body's tension asks for. At steps of a 30th of a second it keeps a tilt
/// of up to 2880 rad/s within the bound; the top link of a chain of 0.5 m, 1 kg links holding ten
/// tonnes tilts at about 1500.
const MAX_TENSION_SUBSTEPS: usize = 64;

/// The greatest inverse inertia `ii` gives a turn about an axis square to the unit vector
/// `dir`: the largest eigenvalue of `ii` on the plane square to it, the axes a pull along `dir`
/// tilts a body about.
#[cfg(feature = "dim3")]
fn inverse_inertia_square_to(ii: &AngularInertia, dir: Vector) -> Real {
    let (u, v) = dir.any_orthonormal_pair();
    let (iu, iv) = (*ii * u, *ii * v);
    let (p, q, s) = (u.dot(iu), v.dot(iv), u.dot(iv));
    let half = (p - q) * 0.5;
    (p + q) * 0.5 + (half * half + s * s).sqrt()
}

/// The greatest inverse inertia `ii` gives a tilt that a pull along the unit vector `dir` holds
/// against and that a joint whose frame stands at `frame` with `locked` axes leaves free: about
/// the plane square to `dir` when the joint locks none of its angles
/// ([`inverse_inertia_square_to`]), and otherwise the greatest, over each angle it leaves free,
/// of the inverse inertia about that angle's axis times the share of the pull's stiffness about
/// it (`1 - (axis . dir)^2`). A turn about a locked angle is held by the joint's own rows.
#[cfg(feature = "dim3")]
fn free_tilt_inverse_inertia(
    ii: &AngularInertia,
    dir: Vector,
    frame: Rotation,
    locked: JointAxesMask,
) -> Real {
    if !locked.intersects(JointAxesMask::ANG_AXES) {
        return inverse_inertia_square_to(ii, dir);
    }
    let angles = [
        (JointAxesMask::ANG_X, Vector::X),
        (JointAxesMask::ANG_Y, Vector::Y),
        (JointAxesMask::ANG_Z, Vector::Z),
    ];
    let mut most: Real = 0.0;
    for (mask, local) in angles {
        if locked.contains(mask) {
            continue;
        }
        let axis = frame * local;
        let along = axis.dot(dir);
        most = most.max((1.0 - along * along) * axis.dot(*ii * axis));
    }
    most
}

/// The 2D form of the 3D `free_tilt_inverse_inertia`: a body turns about one axis, square to
/// every direction of the plane, which the joint leaves free unless it locks its angle.
#[cfg(feature = "dim2")]
fn free_tilt_inverse_inertia(
    ii: &AngularInertia,
    _dir: Vector,
    _frame: Rotation,
    locked: JointAxesMask,
) -> Real {
    if locked.contains(JointAxesMask::ANG_X) {
        0.0
    } else {
        *ii
    }
}

/// The size of the linear impulse a joint's locked translations applied in the last substep that
/// solved it ([`crate::dynamics::ImpulseJoint::impulses`]).
#[cfg(feature = "dim3")]
fn last_substep_linear_impulse(impulses: &SpatialVector) -> Real {
    Vector::new(impulses[0], impulses[1], impulses[2]).length()
}

/// The 2D form of the 3D `last_substep_linear_impulse`.
#[cfg(feature = "dim2")]
fn last_substep_linear_impulse(impulses: &SpatialVector) -> Real {
    Vector::new(impulses.x, impulses.y).length()
}

/// Writes into `rates` (one entry per awake body, by the slot `slot` names it with) the square of
/// the angular frequency at which the tension of its joints tilts each body: the sum, over its
/// enabled impulse joints, of the force each pulls with, times its arm from the body's centre of
/// mass, times the body's inverse inertia about the axes square to that arm that the joint leaves
/// free ([`free_tilt_inverse_inertia`]). That sum bounds the stiffest tilt the joints hold the
/// body against, a spring of stiffness `|f| |arm|` per joint. `slot` answers `None` for a body
/// that is no awake dynamic body; such a side takes nothing.
///
/// A joint with a side on a body that locks an axis is left out, as it is from the joint trees
/// the solver solves exactly: the locks and the joint's rows can ask for the same motion, and the
/// impulses they then share say nothing of the tension the joint holds. A chain of hinges on
/// bodies held to a plane, falling onto a floor, read tilts of up to 3000 radians a substep off
/// them while it fell to rest in the substeps it was set to.
///
/// A joint's force is the greater of the mean force it applied over its last step (`dt` long)
/// and the force of the last substep of that step. The impulses of the substeps of a step add up
/// to the change of momentum they made, and a load the joints throw back up against them cancels
/// its fall's in that sum: a 1000 kg block on a chain of 1 kg links, rebounding off the
/// correction of the stretch it took while it settled, read a mean force that dropped its chain
/// from 32 substeps to 16 for the next step, and the chain folded at 1900 rad/s within it.
pub(super) fn tension_tilt_rates(
    rates: &mut Vec<Real>,
    num_bodies: usize,
    dt: Real,
    bodies: &RigidBodySet,
    impulse_joints: &ImpulseJointSet,
    slot: impl Fn(RigidBodyHandle) -> Option<u32>,
) {
    rates.clear();
    rates.resize(num_bodies, 0.0);
    if dt <= 0.0 {
        return;
    }
    let locks = |handle| {
        bodies
            .get(handle)
            .is_some_and(|b| !b.locked_axes().is_empty())
    };
    for (_, joint) in impulse_joints.iter() {
        if !joint.data.is_enabled() || locks(joint.body1) || locks(joint.body2) {
            continue;
        }
        let mean = joint.reaction.linear.length() / dt;
        let last = last_substep_linear_impulse(&joint.impulses);
        let frames = [&joint.data.local_frame1, &joint.data.local_frame2];
        for (side, handle) in [joint.body1, joint.body2].into_iter().enumerate() {
            let Some(at) = slot(handle).map(|s| s as usize).filter(|s| *s < num_bodies) else {
                continue;
            };
            let body = &bodies[handle];
            // The last substep of the step that solved the joint: this body's substeps of it.
            let substeps = body.solved_substeps.max(1) as Real;
            let force = mean.max(last * substeps / dt);
            if force.is_nan() || force <= 0.0 {
                continue;
            }
            let local_arm = frames[side].translation - body.mprops.local_mprops.local_com;
            let arm = body.pos.position.rotation * local_arm;
            let reach = arm.length();
            if reach > 0.0 {
                let frame = body.pos.position.rotation * frames[side].rotation;
                let ii = free_tilt_inverse_inertia(
                    &body.mprops.effective_world_inv_inertia,
                    arm / reach,
                    frame,
                    joint.data.locked_axes,
                );
                rates[at] += force * reach * ii;
            }
        }
    }
}

/// The substeps beyond the `substeps` a world steps in that a body whose joints' tension tilts
/// it at the square frequency `rate` (from [`tension_tilt_rates`]) asks for, in steps `dt` long:
/// the `substeps` doubled as many times as keeps its tilt within [`MAX_TENSION_TILT_PHASE`] a
/// substep, up to [`MAX_TENSION_SUBSTEPS`], less the `substeps`. Doubling keeps the substep
/// length a step is cut to a function of the substep length it was set to: a world of 30 steps a
/// second in 4 substeps and one of 120 in 1 whose chain needs shorter substeps both run them at a
/// 240th of a second.
pub(super) fn tension_extra_substeps(substeps: usize, dt: Real, rate: Real) -> u32 {
    let turn = rate.sqrt() * dt;
    let base = substeps.max(1);
    if !turn.is_finite() {
        return 0;
    }
    let most = MAX_TENSION_SUBSTEPS.max(base);
    let mut n = base;
    while n < most && turn > MAX_TENSION_TILT_PHASE * n as Real {
        n = (n * 2).min(most);
    }
    (n - base) as u32
}
