use crate::dynamics::solver::SolverVel;
use crate::dynamics::solver::joint_constraint::{AngularLimitParams, JointConstraintHelper};
use crate::dynamics::{
    GenericJoint, IntegrationParameters, JointAxesMask, JointGraphEdge, JointIndex, JointMotor,
};
use crate::math::{ANG_DIM, DIM, Real, SPATIAL_DIM};
use crate::utils::{ComponentMul, DotProduct, ScalarType, SimdRealCopy};

use crate::dynamics::solver::solver_body::SolverBodies;
use crate::math::{SIMD_WIDTH, SimdReal};
use na::SimdValue;
use parry::math::Pose;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct MotorParameters<N: SimdRealCopy> {
    pub erp_inv_dt: N,
    pub cfm_coeff: N,
    pub cfm_gain: N,
    pub target_pos: N,
    pub target_vel: N,
    pub max_impulse: N,
}

impl<N: SimdRealCopy> Default for MotorParameters<N> {
    fn default() -> Self {
        Self {
            erp_inv_dt: N::zero(),
            cfm_coeff: N::zero(),
            cfm_gain: N::zero(),
            target_pos: N::zero(),
            target_vel: N::zero(),
            max_impulse: N::zero(),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum WritebackId {
    Dof(usize),
    Limit(usize),
    Motor(usize),
    Friction(usize),
}

// TODO: right now we only use this for impulse_joints.
// However, it may actually be a good idea to use this everywhere in
// the solver, to avoid fetching data from the rigid-body set
// every time.
#[derive(Copy, Clone)]
pub struct JointSolverBody<N: ScalarType, const LANES: usize> {
    pub im: N::Vector,
    pub ii: N::AngInertia,
    pub world_com: N::Vector, // TODO: is this still needed now that the solver body poses are expressed at the center of mass?
    pub solver_vel: [u32; LANES],
}

impl<N: ScalarType, const LANES: usize> JointSolverBody<N, LANES> {
    pub fn invalid() -> Self {
        Self {
            im: Default::default(),
            ii: N::AngInertia::default(),
            world_com: Default::default(),
            solver_vel: [u32::MAX; LANES],
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct JointConstraint<N: ScalarType, const LANES: usize> {
    pub solver_vel1: [u32; LANES],
    pub solver_vel2: [u32; LANES],

    pub joint_id: [JointIndex; LANES],

    pub impulse: N,
    pub impulse_bounds: [N; 2],
    pub lin_jac: N::Vector,
    pub ang_jac1: N::AngVector,
    pub ang_jac2: N::AngVector,

    pub ii_ang_jac1: N::AngVector,
    pub ii_ang_jac2: N::AngVector,

    pub inv_lhs: N,
    /// The row's inverse-mass scale, from [`JointConstraint::with_lhs_scale`].
    pub lhs_scale: N,
    pub rhs: N,
    pub rhs_wo_bias: N,
    pub cfm_gain: N,
    pub cfm_coeff: N,

    pub im1: N::Vector,
    pub im2: N::Vector,

    /// On a row holding two points together along `lin_jac`: each point's offset from its
    /// body's centre of mass, which [`Self::carry_turn`] turns at the bodies' angular
    /// velocities. Zero on every other row.
    pub turn_arms: [N::Vector; 2],
    /// Half the substep length on a row that carries its points' turning through the
    /// stabilization pass ([`Self::carry_turn`]), zero on every other row.
    pub turn_half_dt: N,
    /// On a row holding the distance between two points within limits (a rope): the velocity
    /// by which that distance grows over a substep beyond its rate of change, the chord of the
    /// points' paths, which the stabilization pass carries while the row bears on the bodies
    /// (see [`JointConstraintHelper::limit_linear_coupled`]). Zero on every other row.
    pub bearing_turn: N,

    pub writeback_id: WritebackId,
}

impl<N: ScalarType, const LANES: usize> JointConstraint<N, LANES> {
    #[profiling::function]
    pub fn solve_generic(
        &mut self,
        solver_vel1: &mut SolverVel<N>,
        solver_vel2: &mut SolverVel<N>,
    ) {
        let dlinvel = self.lin_jac.gdot(solver_vel2.linear - solver_vel1.linear);
        let dangvel =
            self.ang_jac2.gdot(solver_vel2.angular) - self.ang_jac1.gdot(solver_vel1.angular);

        let rhs = dlinvel + dangvel + self.rhs;
        let total_impulse = (self.impulse + self.inv_lhs * (rhs - self.cfm_gain * self.impulse))
            .simd_clamp(self.impulse_bounds[0], self.impulse_bounds[1]);
        let delta_impulse = total_impulse - self.impulse;
        self.impulse = total_impulse;

        let lin_impulse = self.lin_jac * delta_impulse;
        let ii_ang_impulse1 = self.ii_ang_jac1 * delta_impulse;
        let ii_ang_impulse2 = self.ii_ang_jac2 * delta_impulse;

        solver_vel1.linear += lin_impulse.component_mul(&self.im1);
        solver_vel1.angular += ii_ang_impulse1;
        solver_vel2.linear -= lin_impulse.component_mul(&self.im2);
        solver_vel2.angular -= ii_ang_impulse2;
    }

    pub fn remove_bias_from_rhs(&mut self) {
        self.rhs = self.rhs_wo_bias;
    }

    /// The row's direction along the axes on which the bodies' centres of mass can move: its
    /// components along an axis both bodies' translations are locked on are zero.
    fn movable_dir(&self) -> N::Vector {
        self.lin_jac
            .component_mul(&(self.im1 + self.im2).component_nonzero())
    }

    /// The velocity along the row by which the bodies' turning at `angvel1` and `angvel2`
    /// parts its two points over half a substep: `dt / 2` of the difference, along the row's
    /// movable direction ([`Self::movable_dir`]), of the arms' centripetal accelerations
    /// `ω × (ω × r)`. It reads angular velocities and arms alone, so it holds to the same
    /// precision wherever in the world the joint stands.
    #[cfg(feature = "dim3")]
    fn turn_vel(&self, angvel1: N::AngVector, angvel2: N::AngVector) -> N {
        let n: N::AngVector = self.movable_dir().into();
        // `n · (ω × (ω × r)) = (n · ω) (ω · r) - (ω · ω) (n · r)`.
        let along = |angvel: N::AngVector, arm: N::Vector| {
            let arm: N::AngVector = arm.into();
            n.gdot(angvel) * angvel.gdot(arm) - angvel.gdot(angvel) * n.gdot(arm)
        };
        let turn1 = along(angvel1, self.turn_arms[0]);
        let turn2 = along(angvel2, self.turn_arms[1]);
        (turn2 - turn1) * self.turn_half_dt
    }

    /// The 2D form of the 3D `turn_vel`: an arm turning at `ω` accelerates its point at
    /// `-ω² r`, of which `-ω² (r · n)` lies along the row's movable direction `n`.
    #[cfg(feature = "dim2")]
    fn turn_vel(&self, angvel1: N::AngVector, angvel2: N::AngVector) -> N {
        let n = self.movable_dir();
        let turn1 = -(angvel1.gdot(angvel1) * n.gdot(self.turn_arms[0]));
        let turn2 = -(angvel2.gdot(angvel2) * n.gdot(self.turn_arms[1]));
        (turn2 - turn1) * self.turn_half_dt
    }

    /// Sets the row's stabilization target to what it is without bias, plus the velocity by
    /// which its points' turning parts them over a substep (`turn_vel`), read off the angular
    /// velocities `solver_vel1` and `solver_vel2` the substep moved the bodies with. Called on
    /// the first iteration of the stabilization pass, after the bodies moved, for every row of
    /// a joint before any of them is solved.
    ///
    /// A substep moves each body's centre of mass along a straight line and turns its arm about
    /// it, so a row that held two points together leaves them apart at the substep's end by
    /// half a substep squared of the arms' centripetal accelerations: the chord a turning arm
    /// cuts across the arc of its point. The next substep's biased pass closes that gap at
    /// `dt / 2` of those accelerations. A stabilization pass that set the row's velocity back to
    /// zero took that velocity out of the bodies' motion at every substep, a projection of the
    /// velocity they move on with: a 1 kg rod swinging from a ball joint lost 56% of its energy
    /// in 10 s at 30 steps a second in 4 substeps and 39% at 240 steps in 1. Carried, it is the
    /// velocity the next substep turns back onto the arc, and the swing keeps its energy.
    ///
    /// The carried velocity is read from velocities and arms alone, so a joint keeps its swing
    /// to the same precision at any distance from the origin. It is the chord the substep cut:
    /// the turning is read after the biased pass, which takes out of a body the spin its joints
    /// and locks leave it no room for, and the chord is taken along the axes on which the
    /// bodies' centres of mass can move, where carrying it moves them onto the arc. A hinge
    /// chain held to a plane, whose rows along the plane's normal only turning its links can
    /// meet, spun up and climbed when its rows carried the turning along that normal too.
    ///
    /// A row holding a distance within limits carries the chord of its points' paths
    /// (`bearing_turn`) when its biased pass bore on the bodies, that is when the limit held
    /// them: a slack rope carries nothing.
    pub fn carry_turn_generic(&mut self, solver_vel1: &SolverVel<N>, solver_vel2: &SolverVel<N>) {
        let zero = N::zero();
        let bearing = self.impulse.simd_lt(zero) | zero.simd_lt(self.impulse);
        let turn = self.turn_vel(solver_vel1.angular, solver_vel2.angular)
            + self.bearing_turn.select(bearing, zero);
        self.rhs_wo_bias += turn;
        self.rhs = self.rhs_wo_bias;
    }

    /// Applies the currently-accumulated impulse to the body velocities. Only used when
    /// `IntegrationParameters::warmstart_joints` is enabled: the constraint's `impulse` was
    /// kept from the previous substep (or seeded from last step's writeback) by the update.
    pub fn warmstart_generic(
        &mut self,
        solver_vel1: &mut SolverVel<N>,
        solver_vel2: &mut SolverVel<N>,
    ) {
        let lin_impulse = self.lin_jac * self.impulse;
        let ii_ang_impulse1 = self.ii_ang_jac1 * self.impulse;
        let ii_ang_impulse2 = self.ii_ang_jac2 * self.impulse;

        solver_vel1.linear += lin_impulse.component_mul(&self.im1);
        solver_vel1.angular += ii_ang_impulse1;
        solver_vel2.linear -= lin_impulse.component_mul(&self.im2);
        solver_vel2.angular -= ii_ang_impulse2;
    }
}

impl JointConstraint<Real, 1> {
    pub fn update(
        params: &IntegrationParameters,
        joint_id: JointIndex,
        body1: &JointSolverBody<Real, 1>,
        body2: &JointSolverBody<Real, 1>,
        frame1: &Pose,
        frame2: &Pose,
        joint: &GenericJoint,
        // The angular limits, in the form the rows consume (built once per assembly by
        // `JointConstraintBuilder`, not once per substep).
        ang_limits: &[AngularLimitParams<Real>; ANG_DIM],
        // The velocities the two bodies start the substep with.
        vels: [SolverVel<Real>; 2],
        out: &mut [Self],
    ) -> usize {
        let mut len = 0;
        let locked_axes = joint.locked_axes.bits();
        let motor_axes = joint.motor_axes.bits() & !locked_axes;
        let limit_axes = joint.limit_axes.bits() & !locked_axes;
        let coupled_axes = joint.coupled_axes.bits();

        // Compute per-joint ERP and CFM coefficients
        let erp_inv_dt = joint.softness.erp_inv_dt(params.dt);
        let cfm_coeff = joint.softness.cfm_coeff(params.dt);

        // The has_lin/ang_coupling test is needed to avoid shl overflow later.
        let has_lin_coupling = (coupled_axes & JointAxesMask::LIN_AXES.bits()) != 0;
        let first_coupled_lin_axis_id =
            (coupled_axes & JointAxesMask::LIN_AXES.bits()).trailing_zeros() as usize;

        #[cfg(feature = "dim3")]
        let has_ang_coupling = (coupled_axes & JointAxesMask::ANG_AXES.bits()) != 0;
        #[cfg(feature = "dim3")]
        let first_coupled_ang_axis_id =
            (coupled_axes & JointAxesMask::ANG_AXES.bits()).trailing_zeros() as usize;

        let builder = JointConstraintHelper::<Real>::new(
            frame1,
            frame2,
            &body1.world_com,
            &body2.world_com,
            locked_axes,
        )
        .with_motion(
            vels[0].linear,
            vels[0].angular,
            vels[1].linear,
            vels[1].angular,
        );

        let start = len;
        // An angular motor drives its own axis whether or not that axis is
        // coupled: coupling two angles shares their limit (a swing cone), and
        // each angle's motor still measures and drives that angle alone.
        for i in DIM..SPATIAL_DIM {
            if motor_axes & (1 << i) != 0 {
                out[len] = builder.motor_angular(
                    [joint_id],
                    body1,
                    body2,
                    i - DIM,
                    &joint.motors[i].motor_params(params.dt),
                    WritebackId::Motor(i),
                );
                len += 1;
            }
        }
        for i in 0..DIM {
            if (motor_axes & !coupled_axes) & (1 << i) != 0 {
                let limits = if limit_axes & (1 << i) != 0 {
                    Some([joint.limits[i].min, joint.limits[i].max])
                } else {
                    None
                };

                out[len] = builder.motor_linear(
                    params,
                    [joint_id],
                    body1,
                    body2,
                    i,
                    &joint.motors[i].motor_params(params.dt),
                    limits,
                    WritebackId::Motor(i),
                );
                len += 1;
            }
        }

        if (motor_axes & coupled_axes) & JointAxesMask::LIN_AXES.bits() != 0 {
            let limits = if (limit_axes & (1 << first_coupled_lin_axis_id)) != 0 {
                Some([
                    joint.limits[first_coupled_lin_axis_id].min,
                    joint.limits[first_coupled_lin_axis_id].max,
                ])
            } else {
                None
            };

            out[len] = builder.motor_linear_coupled(
                params,
                [joint_id],
                body1,
                body2,
                coupled_axes,
                &joint.motors[first_coupled_lin_axis_id].motor_params(params.dt),
                limits,
                WritebackId::Motor(first_coupled_lin_axis_id),
            );
            len += 1;
        }

        JointConstraintHelper::finalize_constraints(&mut out[start..len]);

        let start = len;
        for i in DIM..SPATIAL_DIM {
            if locked_axes & (1 << i) != 0 {
                out[len] = builder.lock_angular(
                    params,
                    [joint_id],
                    body1,
                    body2,
                    i - DIM,
                    WritebackId::Dof(i),
                    erp_inv_dt,
                    cfm_coeff,
                );
                len += 1;
            }
        }
        for i in 0..DIM {
            if locked_axes & (1 << i) != 0 {
                out[len] = builder.lock_linear(
                    params,
                    [joint_id],
                    body1,
                    body2,
                    i,
                    WritebackId::Dof(i),
                    erp_inv_dt,
                    cfm_coeff,
                );
                len += 1;
            }
        }

        for i in DIM..SPATIAL_DIM {
            if (limit_axes & !coupled_axes) & (1 << i) != 0 {
                out[len] = builder.limit_angular(
                    params,
                    [joint_id],
                    body1,
                    body2,
                    i - DIM,
                    ang_limits[i - DIM],
                    WritebackId::Limit(i),
                    erp_inv_dt,
                    cfm_coeff,
                );
                out[len].bound_limit_force(joint.limits[i].max_force, params.dt);
                len += 1;
            }
        }
        for i in 0..DIM {
            if (limit_axes & !coupled_axes) & (1 << i) != 0 {
                out[len] = builder.limit_linear(
                    params,
                    [joint_id],
                    body1,
                    body2,
                    i,
                    [joint.limits[i].min, joint.limits[i].max],
                    WritebackId::Limit(i),
                    erp_inv_dt,
                    cfm_coeff,
                );
                out[len].bound_limit_force(joint.limits[i].max_force, params.dt);
                len += 1;
            }
        }

        #[cfg(feature = "dim3")]
        if has_ang_coupling && (limit_axes & (1 << first_coupled_ang_axis_id)) != 0 {
            out[len] = builder.limit_angular_coupled(
                params,
                [joint_id],
                body1,
                body2,
                coupled_axes,
                [
                    joint.limits[first_coupled_ang_axis_id].min,
                    joint.limits[first_coupled_ang_axis_id].max,
                ],
                WritebackId::Limit(first_coupled_ang_axis_id),
                erp_inv_dt,
                cfm_coeff,
            );
            out[len]
                .bound_limit_force(joint.limits[first_coupled_ang_axis_id].max_force, params.dt);
            len += 1;
        }

        if has_lin_coupling && (limit_axes & (1 << first_coupled_lin_axis_id)) != 0 {
            out[len] = builder.limit_linear_coupled(
                params,
                [joint_id],
                body1,
                body2,
                coupled_axes,
                [
                    joint.limits[first_coupled_lin_axis_id].min,
                    joint.limits[first_coupled_lin_axis_id].max,
                ],
                WritebackId::Limit(first_coupled_lin_axis_id),
                erp_inv_dt,
                cfm_coeff,
            );
            out[len]
                .bound_limit_force(joint.limits[first_coupled_lin_axis_id].max_force, params.dt);
            len += 1;
        }
        JointConstraintHelper::finalize_constraints(&mut out[start..len]);

        len
    }

    /// Bound a limit row's impulse to what `max_force` applies over a substep of `dt`
    /// ([`crate::dynamics::JointLimits::max_force`]); an unbounded limit is left as built.
    pub fn bound_limit_force(&mut self, max_force: Real, dt: Real) {
        if max_force == Real::MAX {
            return;
        }
        let bound = JointMotor::max_impulse(max_force.max(0.0), dt);
        self.impulse_bounds[0] = self.impulse_bounds[0].max(-bound);
        self.impulse_bounds[1] = self.impulse_bounds[1].min(bound);
    }

    pub fn solve(&mut self, solver_vels: &mut SolverBodies) {
        let mut solver_vel1 = solver_vels.get_vel(self.solver_vel1[0]);
        let mut solver_vel2 = solver_vels.get_vel(self.solver_vel2[0]);

        self.solve_generic(&mut solver_vel1, &mut solver_vel2);

        solver_vels.set_vel(self.solver_vel1[0], solver_vel1);
        solver_vels.set_vel(self.solver_vel2[0], solver_vel2);
    }

    /// [`Self::carry_turn_generic`] on the velocities the row's bodies hold.
    pub fn carry_turn(&mut self, solver_vels: &SolverBodies) {
        let solver_vel1 = solver_vels.get_vel(self.solver_vel1[0]);
        let solver_vel2 = solver_vels.get_vel(self.solver_vel2[0]);
        self.carry_turn_generic(&solver_vel1, &solver_vel2);
    }

    pub fn warmstart(&mut self, solver_vels: &mut SolverBodies) {
        let mut solver_vel1 = solver_vels.get_vel(self.solver_vel1[0]);
        let mut solver_vel2 = solver_vels.get_vel(self.solver_vel2[0]);

        self.warmstart_generic(&mut solver_vel1, &mut solver_vel2);

        solver_vels.set_vel(self.solver_vel1[0], solver_vel1);
        solver_vels.set_vel(self.solver_vel2[0], solver_vel2);
    }

    pub fn writeback_impulses(&self, joints_all: &mut [JointGraphEdge]) {
        let joint = &mut joints_all[self.joint_id[0]].weight;
        match self.writeback_id {
            WritebackId::Dof(i) => joint.impulses[i] = self.impulse,
            WritebackId::Limit(i) => joint.data.limits[i].impulse = self.impulse,
            WritebackId::Motor(i) => joint.data.motors[i].impulse = self.impulse,
            WritebackId::Friction(_) => {}
        }
    }
}

impl JointConstraint<SimdReal, SIMD_WIDTH> {
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        params: &IntegrationParameters,
        joint_id: [JointIndex; SIMD_WIDTH],
        body1: &JointSolverBody<SimdReal, SIMD_WIDTH>,
        body2: &JointSolverBody<SimdReal, SIMD_WIDTH>,
        frame1: &<SimdReal as ScalarType>::Pose,
        frame2: &<SimdReal as ScalarType>::Pose,
        locked_axes: u8,
        limit_axes: u8,
        limits: &[[SimdReal; 2]; DIM],
        ang_limits: &[AngularLimitParams<SimdReal>; ANG_DIM],
        softness: crate::dynamics::SpringCoefficients<SimdReal>,
        // `Some` = emit the (2D) angular motor row. Kept out of 3D until the
        // wide builder gathers per-axis motors.
        ang_motor: Option<&MotorParameters<SimdReal>>,
        // The velocities the two bodies start the substep with.
        vels: [SolverVel<SimdReal>; 2],
        out: &mut [Self],
    ) -> usize {
        let dt = SimdReal::splat(params.dt);
        let erp_inv_dt = softness.erp_inv_dt(dt);
        let cfm_coeff = softness.cfm_coeff(dt);

        let builder = JointConstraintHelper::new(
            frame1,
            frame2,
            &body1.world_com,
            &body2.world_com,
            locked_axes,
        )
        .with_motion(
            vels[0].linear,
            vels[0].angular,
            vels[1].linear,
            vels[1].angular,
        );

        let mut len = 0;

        // Motor rows come first and are orthogonalized in their own group,
        // exactly like the scalar row emission.
        if let Some(motor_params) = ang_motor {
            out[len] = builder.motor_angular(
                joint_id,
                body1,
                body2,
                0,
                motor_params,
                WritebackId::Motor(DIM),
            );
            len += 1;
            JointConstraintHelper::finalize_constraints(&mut out[..len]);
        }
        let group_start = len;

        for i in 0..DIM {
            if locked_axes & (1 << i) != 0 {
                out[len] = builder.lock_linear(
                    params,
                    joint_id,
                    body1,
                    body2,
                    i,
                    WritebackId::Dof(i),
                    erp_inv_dt,
                    cfm_coeff,
                );
                len += 1;
            }
        }

        for i in DIM..SPATIAL_DIM {
            if locked_axes & (1 << i) != 0 {
                out[len] = builder.lock_angular(
                    params,
                    joint_id,
                    body1,
                    body2,
                    i - DIM,
                    WritebackId::Dof(i),
                    erp_inv_dt,
                    cfm_coeff,
                );
                len += 1;
            }
        }

        for i in DIM..SPATIAL_DIM {
            if limit_axes & (1 << i) != 0 {
                out[len] = builder.limit_angular(
                    params,
                    joint_id,
                    body1,
                    body2,
                    i - DIM,
                    ang_limits[i - DIM],
                    WritebackId::Limit(i),
                    erp_inv_dt,
                    cfm_coeff,
                );
                len += 1;
            }
        }
        for i in 0..DIM {
            if limit_axes & (1 << i) != 0 {
                out[len] = builder.limit_linear(
                    params,
                    joint_id,
                    body1,
                    body2,
                    i,
                    limits[i],
                    WritebackId::Limit(i),
                    erp_inv_dt,
                    cfm_coeff,
                );
                len += 1;
            }
        }

        JointConstraintHelper::finalize_constraints(&mut out[group_start..len]);
        len
    }

    pub fn solve(&mut self, solver_vels: &mut SolverBodies) {
        let mut solver_vel1 = solver_vels.gather_vels(self.solver_vel1);
        let mut solver_vel2 = solver_vels.gather_vels(self.solver_vel2);

        self.solve_generic(&mut solver_vel1, &mut solver_vel2);

        solver_vels.scatter_vels(self.solver_vel1, solver_vel1);
        solver_vels.scatter_vels(self.solver_vel2, solver_vel2);
    }

    /// [`Self::carry_turn_generic`] on the velocities the row's bodies hold.
    pub fn carry_turn(&mut self, solver_vels: &SolverBodies) {
        let solver_vel1 = solver_vels.gather_vels(self.solver_vel1);
        let solver_vel2 = solver_vels.gather_vels(self.solver_vel2);
        self.carry_turn_generic(&solver_vel1, &solver_vel2);
    }

    pub fn warmstart(&mut self, solver_vels: &mut SolverBodies) {
        let mut solver_vel1 = solver_vels.gather_vels(self.solver_vel1);
        let mut solver_vel2 = solver_vels.gather_vels(self.solver_vel2);

        self.warmstart_generic(&mut solver_vel1, &mut solver_vel2);

        solver_vels.scatter_vels(self.solver_vel1, solver_vel1);
        solver_vels.scatter_vels(self.solver_vel2, solver_vel2);
    }

    pub fn writeback_impulses(&self, joints_all: &mut [JointGraphEdge]) {
        let impulses: [_; SIMD_WIDTH] = self.impulse.into();

        // TODO: should we move the iteration on ii deeper in the nested match?
        for ii in 0..SIMD_WIDTH {
            let joint = &mut joints_all[self.joint_id[ii]].weight;
            match self.writeback_id {
                WritebackId::Dof(i) => joint.impulses[i] = impulses[ii],
                WritebackId::Limit(i) => joint.data.limits[i].impulse = impulses[ii],
                WritebackId::Motor(i) => joint.data.motors[i].impulse = impulses[ii],
                WritebackId::Friction(_) => {}
            }
        }
    }
}
