#![allow(missing_docs)] // For downcast.

use crate::dynamics::integration_parameters::SpringCoefficients;
use crate::dynamics::joint::MultibodyLink;
use crate::dynamics::solver::{GenericJointConstraint, WritebackId};
use crate::dynamics::{IntegrationParameters, JointMotor, Multibody};
use crate::math::{DVector, Real};

/// The bound a limit row holds a coordinate to, and what it asks of the coordinate's velocity.
struct LimitRow {
    impulse_bounds: [Real; 2],
    rhs_wo_bias: Real,
    rhs_bias: Real,
}

/// The row holding `curr_pos` inside `limits`, on the side of the range it stands nearer to.
///
/// Inside the range the row is speculative: it lets the coordinate close the gap to the
/// bound over the substep and no further, so a link driven fast into its stop stops at it
/// rather than passing it and being pushed back. Past the bound it pushes the coordinate back
/// at the joint's error-reduction rate.
fn limit_row(
    params: &IntegrationParameters,
    softness: SpringCoefficients<Real>,
    limits: [Real; 2],
    curr_pos: Real,
) -> LimitRow {
    let inv_dt = params.inv_dt();
    let erp_inv_dt = softness.erp_inv_dt(params.dt);
    if curr_pos >= (limits[0] + limits[1]) * 0.5 {
        let past = curr_pos - limits[1];
        LimitRow {
            impulse_bounds: [0.0, Real::MAX],
            rhs_wo_bias: past.min(0.0) * inv_dt,
            rhs_bias: past.max(0.0) * erp_inv_dt,
        }
    } else {
        let past = limits[0] - curr_pos;
        LimitRow {
            impulse_bounds: [-Real::MAX, 0.0],
            rhs_wo_bias: -past.min(0.0) * inv_dt,
            rhs_bias: -past.max(0.0) * erp_inv_dt,
        }
    }
}

/// Initializes and generate the velocity constraints applicable to the multibody links attached
/// to this multibody_joint.
pub fn unit_joint_limit_constraint(
    params: &IntegrationParameters,
    multibody: &Multibody,
    link: &MultibodyLink,
    limits: [Real; 2],
    curr_pos: Real,
    dof_id: usize,
    j_id: &mut usize,
    jacobians: &mut DVector,
    constraints: &mut [GenericJointConstraint],
    insert_at: &mut usize,
    softness: SpringCoefficients<Real>,
) {
    let ndofs = multibody.ndofs();
    let cfm_coeff = softness.cfm_coeff(params.dt);
    let LimitRow {
        impulse_bounds,
        rhs_wo_bias,
        rhs_bias,
    } = limit_row(params, softness, limits, curr_pos);

    let dof_j_id = *j_id + dof_id + link.assembly_id;
    jacobians.rows_mut(*j_id, ndofs * 2).fill(0.0);
    jacobians[dof_j_id] = 1.0;
    jacobians[dof_j_id + ndofs] = 1.0;
    multibody
        .inv_augmented_mass()
        .solve_mut(&mut jacobians.rows_mut(*j_id + ndofs, ndofs));

    let lhs = jacobians[dof_j_id + ndofs]; // = J^t * M^-1 J
    let cfm_gain = lhs * cfm_coeff;

    let constraint = GenericJointConstraint {
        is_rigid_body1: false,
        solver_vel1: u32::MAX,
        ndofs1: 0,
        j_id1: 0,

        is_rigid_body2: false,
        solver_vel2: multibody.solver_id,
        ndofs2: ndofs,
        j_id2: *j_id,
        joint_id: usize::MAX, // TODO: we don’t support impulse writeback for internal constraints yet.
        impulse: 0.0,
        impulse_bounds,
        inv_lhs: crate::utils::inv(lhs + cfm_gain),
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        cfm_coeff,
        cfm_gain,
        writeback_id: WritebackId::Limit(dof_id),
    };

    constraints[*insert_at] = constraint;
    *insert_at += 1;

    *j_id += 2 * ndofs;
}

/// Generates the dry-friction (MuJoCo `frictionloss`) velocity constraint for
/// one generalized DoF.
///
/// `softness` supplies the CFM compliance (MuJoCo's `solreffriction`); only CFM
/// applies, never ERP, since there is no position error for a bias to chase.
#[allow(clippy::too_many_arguments)]
pub fn unit_joint_friction_constraint(
    params: &IntegrationParameters,
    multibody: &Multibody,
    link: &MultibodyLink,
    friction: Real,
    dof_id: usize,
    j_id: &mut usize,
    jacobians: &mut DVector,
    constraints: &mut [GenericJointConstraint],
    insert_at: &mut usize,
    softness: SpringCoefficients<Real>,
) {
    let ndofs = multibody.ndofs();
    let cfm_coeff = softness.cfm_coeff(params.dt);

    let dof_j_id = *j_id + dof_id + link.assembly_id;
    jacobians.rows_mut(*j_id, ndofs * 2).fill(0.0);
    jacobians[dof_j_id] = 1.0;
    jacobians[dof_j_id + ndofs] = 1.0;
    multibody
        .inv_augmented_mass()
        .solve_mut(&mut jacobians.rows_mut(*j_id + ndofs, ndofs));

    let lhs = jacobians[dof_j_id + ndofs]; // = J^t * M^-1 J
    let max_impulse = friction * params.dt;
    let cfm_gain = lhs * cfm_coeff;

    let constraint = GenericJointConstraint {
        is_rigid_body1: false,
        solver_vel1: u32::MAX,
        ndofs1: 0,
        j_id1: 0,
        is_rigid_body2: false,
        solver_vel2: multibody.solver_id,
        ndofs2: ndofs,
        j_id2: *j_id,
        joint_id: usize::MAX, // TODO: we don’t support impulse writeback for internal constraints yet.
        impulse: 0.0,
        impulse_bounds: [-max_impulse, max_impulse],
        inv_lhs: crate::utils::inv(lhs + cfm_gain),
        rhs: 0.0,
        rhs_wo_bias: 0.0,
        cfm_coeff,
        cfm_gain,
        writeback_id: WritebackId::Friction(dof_id),
    };

    constraints[*insert_at] = constraint;
    *insert_at += 1;

    *j_id += 2 * ndofs;
}

/// Initializes and generate the velocity constraints applicable to the multibody links attached
/// to this multibody_joint.
pub fn unit_joint_motor_constraint(
    params: &IntegrationParameters,
    multibody: &Multibody,
    link: &MultibodyLink,
    motor: &JointMotor,
    curr_pos: Real,
    limits: Option<[Real; 2]>,
    dof_id: usize,
    j_id: &mut usize,
    jacobians: &mut DVector,
    constraints: &mut [GenericJointConstraint],
    insert_at: &mut usize,
) {
    let inv_dt = params.inv_dt();
    let ndofs = multibody.ndofs();
    let motor_params = motor.motor_params(params.dt);

    let dof_j_id = *j_id + dof_id + link.assembly_id;
    jacobians.rows_mut(*j_id, ndofs * 2).fill(0.0);
    jacobians[dof_j_id] = 1.0;
    jacobians[dof_j_id + ndofs] = 1.0;
    multibody
        .inv_augmented_mass()
        .solve_mut(&mut jacobians.rows_mut(*j_id + ndofs, ndofs));

    let lhs = jacobians[dof_j_id + ndofs]; // = J^t * M^-1 J
    let impulse_bounds = [-motor_params.max_impulse, motor_params.max_impulse];
    let cfm_gain = lhs * motor_params.cfm_coeff + motor_params.cfm_gain;

    let mut rhs_wo_bias = 0.0;
    if motor_params.erp_inv_dt != 0.0 {
        rhs_wo_bias += (curr_pos - motor_params.target_pos) * motor_params.erp_inv_dt;
    }

    let mut target_vel = motor_params.target_vel;
    if let Some(limits) = limits {
        target_vel = target_vel.clamp(
            (limits[0] - curr_pos) * inv_dt,
            (limits[1] - curr_pos) * inv_dt,
        );
    };

    rhs_wo_bias += -target_vel;

    let constraint = GenericJointConstraint {
        is_rigid_body1: false,
        solver_vel1: u32::MAX,
        ndofs1: 0,
        j_id1: 0,

        is_rigid_body2: false,
        solver_vel2: multibody.solver_id,
        ndofs2: ndofs,
        j_id2: *j_id,
        joint_id: usize::MAX, // TODO: we don’t support impulse writeback for internal constraints yet.
        impulse: 0.0,
        impulse_bounds,
        cfm_coeff: motor_params.cfm_coeff,
        cfm_gain,
        inv_lhs: crate::utils::inv(lhs + cfm_gain),
        rhs: rhs_wo_bias,
        rhs_wo_bias,
        writeback_id: WritebackId::Limit(dof_id),
    };

    constraints[*insert_at] = constraint;
    *insert_at += 1;

    *j_id += 2 * ndofs;
}

/// Initializes the velocity constraint holding a measure of a link's position, `curr_pos`,
/// inside `limits` along a direction that spans several of its joint's degrees of freedom:
/// `dir[k]` is the measure's rate per unit of the joint's dof `first_dof + k`, so the row is
/// the measure's gradient over the joint's generalized velocities.
///
/// A spherical joint's swing cone and twist are measured on its rotation as a whole: each is
/// moved by all three of its angular velocities, which a row on a single dof cannot hold.
#[allow(clippy::too_many_arguments)]
pub fn unit_joint_limit_constraint_along(
    params: &IntegrationParameters,
    multibody: &Multibody,
    link: &MultibodyLink,
    limits: [Real; 2],
    curr_pos: Real,
    first_dof: usize,
    dir: &[Real],
    j_id: &mut usize,
    jacobians: &mut DVector,
    constraints: &mut [GenericJointConstraint],
    insert_at: &mut usize,
    softness: SpringCoefficients<Real>,
    writeback_id: WritebackId,
) {
    let ndofs = multibody.ndofs();
    let cfm_coeff = softness.cfm_coeff(params.dt);
    let LimitRow {
        impulse_bounds,
        rhs_wo_bias,
        rhs_bias,
    } = limit_row(params, softness, limits, curr_pos);

    let base = *j_id + first_dof + link.assembly_id;
    jacobians.rows_mut(*j_id, ndofs * 2).fill(0.0);
    for (k, weight) in dir.iter().enumerate() {
        jacobians[base + k] = *weight;
        jacobians[base + k + ndofs] = *weight;
    }
    multibody
        .inv_augmented_mass()
        .solve_mut(&mut jacobians.rows_mut(*j_id + ndofs, ndofs));

    // J^t * M^-1 * J, over the entries of J that are not zero.
    let lhs: Real = dir
        .iter()
        .enumerate()
        .map(|(k, weight)| *weight * jacobians[base + k + ndofs])
        .sum();
    let cfm_gain = lhs * cfm_coeff;

    constraints[*insert_at] = GenericJointConstraint {
        is_rigid_body1: false,
        solver_vel1: u32::MAX,
        ndofs1: 0,
        j_id1: 0,

        is_rigid_body2: false,
        solver_vel2: multibody.solver_id,
        ndofs2: ndofs,
        j_id2: *j_id,
        joint_id: usize::MAX,
        impulse: 0.0,
        impulse_bounds,
        inv_lhs: crate::utils::inv(lhs + cfm_gain),
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        cfm_coeff,
        cfm_gain,
        writeback_id,
    };
    *insert_at += 1;

    *j_id += 2 * ndofs;
}
