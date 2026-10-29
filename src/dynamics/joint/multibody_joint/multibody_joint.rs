use crate::dynamics::solver::GenericJointConstraint;
use crate::dynamics::{
    FixedJointBuilder, GenericJoint, IntegrationParameters, Multibody, MultibodyLink,
    RigidBodyVelocity, joint,
};
use crate::math::{
    ANG_DIM, DIM, DVector, JacobianViewMut, Pose, Real, Rotation, SPATIAL_DIM, SpatialVector,
    Vector,
};
use parry::math::VectorExt;

#[cfg(feature = "dim2")]
use crate::math::rotation_from_angle;
#[cfg(feature = "dim3")]
use crate::utils::RotationOps;
use crate::utils::vect_to_na;
use na::DVectorViewMut;

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
/// An joint attached to two bodies based on the reduced coordinates formalism.
pub struct MultibodyJoint {
    /// The joint’s description.
    pub data: GenericJoint,
    /// Is the joint a kinematic joint?
    ///
    /// Kinematic joint velocities are never changed by the physics engine. This gives the user
    /// total control over the values of their degrees of freedoms.
    pub kinematic: bool,
    pub(crate) coords: SpatialVector,
    pub(crate) joint_rot: Rotation,
    /// Per-DoF spring stiffness (a passive joint spring integrated implicitly
    /// in the generalized dynamics). Zero on axes with no spring.
    pub(crate) spring_stiffness: SpatialVector,
    /// Per-DoF spring rest position, in the joint's generalized coordinate
    /// (same convention as [`Self::coords`]). Only meaningful where
    /// `spring_stiffness` is non-zero.
    pub(crate) spring_ref: SpatialVector,
}

impl MultibodyJoint {
    /// Creates a new multibody joint from its description.
    pub fn new(data: GenericJoint, kinematic: bool) -> Self {
        Self {
            data,
            kinematic,
            coords: Default::default(),
            joint_rot: Rotation::IDENTITY,
            spring_stiffness: Default::default(),
            spring_ref: Default::default(),
        }
    }

    /// Sets a passive joint spring on `axis` (an index into the 6-DoF spatial
    /// layout: `0..DIM` linear, `DIM..SPATIAL_DIM` angular). The spring applies
    /// a generalized force `-stiffness · (q − rest)` integrated implicitly.
    pub fn set_spring(&mut self, axis: usize, stiffness: Real, rest: Real) {
        self.spring_stiffness[axis] = stiffness;
        self.spring_ref[axis] = rest;
    }

    /// The passive joint spring on `axis` as `(stiffness, rest)`, as set by
    /// [`Self::set_spring`]. `(0, 0)` if no spring is set on that axis.
    pub fn spring(&self, axis: usize) -> (Real, Real) {
        (self.spring_stiffness[axis], self.spring_ref[axis])
    }

    pub(crate) fn free(pos: Pose) -> Self {
        let mut result = Self::new(GenericJoint::default(), false);
        result.set_free_pos(pos);
        result
    }

    pub(crate) fn fixed(pos: Pose) -> Self {
        Self::new(
            FixedJointBuilder::new().local_frame1(pos).build().into(),
            false,
        )
    }

    pub(crate) fn set_free_pos(&mut self, pos: Pose) {
        #[cfg(feature = "dim2")]
        {
            self.coords.x = pos.translation.x;
            self.coords.y = pos.translation.y;
        }
        #[cfg(feature = "dim3")]
        {
            self.coords[0] = pos.translation.x;
            self.coords[1] = pos.translation.y;
            self.coords[2] = pos.translation.z;
        }
        self.joint_rot = pos.rotation;
    }

    /// The joint’s angular coordinates converted to a rotation.
    pub fn joint_rot(&self) -> Rotation {
        self.joint_rot
    }

    fn num_free_lin_dofs(&self) -> usize {
        let locked_bits = self.data.locked_axes.bits();
        DIM - (locked_bits & ((1 << DIM) - 1)).count_ones() as usize
    }

    /// Generalized coordinates for this joint.
    pub fn coords(&self) -> SpatialVector {
        self.coords
    }

    /// The number of degrees of freedom allowed by the multibody_joint.
    pub fn ndofs(&self) -> usize {
        SPATIAL_DIM - self.data.locked_axes.bits().count_ones() as usize
    }

    /// The position of the multibody link containing this multibody_joint relative to its parent.
    pub fn body_to_parent(&self) -> Pose {
        let locked_bits = self.data.locked_axes.bits();
        let mut transform = Pose::from_rotation(self.joint_rot) * self.data.local_frame2.inverse();

        for i in 0..DIM {
            if (locked_bits & (1 << i)) == 0 {
                // Create a translation along axis i with the coordinate value
                let translation = Vector::ith(i, self.coords[i]);
                transform = Pose::from_translation(translation) * transform;
            }
        }

        self.data.local_frame1 * transform
    }

    /// Integrate the position of this multibody_joint.
    #[profiling::function]
    pub fn integrate(&mut self, dt: Real, vels: &[Real]) {
        let locked_bits = self.data.locked_axes.bits();
        let mut curr_free_dof = 0;

        for i in 0..DIM {
            if (locked_bits & (1 << i)) == 0 {
                self.coords[i] += vels[curr_free_dof] * dt;
                curr_free_dof += 1;
            }
        }

        let locked_ang_bits = locked_bits >> DIM;
        let num_free_ang_dofs = ANG_DIM - locked_ang_bits.count_ones() as usize;
        match num_free_ang_dofs {
            0 => { /* No free dofs. */ }
            1 => {
                let dof_id = (!locked_ang_bits).trailing_zeros() as usize;
                self.coords[DIM + dof_id] += vels[curr_free_dof] * dt;
                #[cfg(feature = "dim2")]
                {
                    self.joint_rot = rotation_from_angle(self.coords[DIM + dof_id]);
                }
                #[cfg(feature = "dim3")]
                {
                    self.joint_rot = Rotation::from_axis_angle(
                        Vector::ith(dof_id, 1.0),
                        self.coords[DIM + dof_id],
                    );
                }
            }
            2 => {
                todo!()
            }
            #[cfg(feature = "dim3")]
            3 => {
                let angvel = Vector::from_slice(&vels[curr_free_dof..curr_free_dof + 3]);
                let disp = Rotation::from_scaled_axis(angvel * dt);
                self.joint_rot = disp * self.joint_rot;
                self.coords[3] += angvel[0] * dt;
                self.coords[4] += angvel[1] * dt;
                self.coords[5] += angvel[2] * dt;
            }
            _ => unreachable!(),
        }
    }

    /// Apply a displacement to the multibody_joint.
    pub fn apply_displacement(&mut self, disp: &[Real]) {
        self.integrate(1.0, disp);
    }

    /// Sets in `out` the non-zero entries of the multibody_joint jacobian transformed by `transform`.
    pub fn jacobian(&self, transform: &Rotation, out: &mut JacobianViewMut<Real>) {
        let locked_bits = self.data.locked_axes.bits();
        let mut curr_free_dof = 0;

        for i in 0..DIM {
            if (locked_bits & (1 << i)) == 0 {
                let transformed_axis = (*transform) * Vector::ith(i, 1.0);
                out.fixed_view_mut::<DIM, 1>(0, curr_free_dof)
                    .copy_from(&vect_to_na(transformed_axis));
                curr_free_dof += 1;
            }
        }

        let locked_ang_bits = locked_bits >> DIM;
        let num_free_ang_dofs = ANG_DIM - locked_ang_bits.count_ones() as usize;
        match num_free_ang_dofs {
            0 => { /* No free dofs. */ }
            1 => {
                #[cfg(feature = "dim2")]
                {
                    out[(DIM, curr_free_dof)] = 1.0;
                }

                #[cfg(feature = "dim3")]
                {
                    let dof_id = (!locked_ang_bits).trailing_zeros() as usize;
                    let rotmat = transform.to_mat();
                    out.fixed_view_mut::<ANG_DIM, 1>(DIM, curr_free_dof)
                        .copy_from_slice(rotmat.col(dof_id).as_ref());
                }
            }
            2 => {
                todo!()
            }
            #[cfg(feature = "dim3")]
            3 => {
                let rotmat = transform.to_mat();
                out.fixed_view_mut::<3, 3>(3, curr_free_dof)
                    .copy_from_slice(rotmat.as_ref());
            }
            _ => unreachable!(),
        }
    }

    /// Multiply the multibody_joint jacobian by generalized velocities to obtain the
    /// relative velocity of the multibody link containing this multibody_joint.
    pub fn jacobian_mul_coordinates(&self, acc: &[Real]) -> RigidBodyVelocity<Real> {
        let locked_bits = self.data.locked_axes.bits();
        let mut result = RigidBodyVelocity::zero();
        let mut curr_free_dof = 0;

        for i in 0..DIM {
            if (locked_bits & (1 << i)) == 0 {
                result.linvel += Vector::ith(i, acc[curr_free_dof]);
                curr_free_dof += 1;
            }
        }

        let locked_ang_bits = locked_bits >> DIM;
        let num_free_ang_dofs = ANG_DIM - locked_ang_bits.count_ones() as usize;
        match num_free_ang_dofs {
            0 => { /* No free dofs. */ }
            1 => {
                #[cfg(feature = "dim2")]
                {
                    result.angvel += acc[curr_free_dof];
                }
                #[cfg(feature = "dim3")]
                {
                    let dof_id = (!locked_ang_bits).trailing_zeros() as usize;
                    result.angvel[dof_id] += acc[curr_free_dof];
                }
            }
            2 => {
                todo!()
            }
            #[cfg(feature = "dim3")]
            3 => {
                let angvel = Vector::from_slice(&acc[curr_free_dof..curr_free_dof + 3]);
                result.angvel += angvel;
            }
            _ => unreachable!(),
        }
        result
    }

    /// Fill `out` with the non-zero entries of a damping that can be applied by default to ensure a good stability of the multibody_joint.
    pub fn default_damping(&self, out: &mut DVectorViewMut<Real>) {
        let locked_bits = self.data.locked_axes.bits();
        let mut curr_free_dof = self.num_free_lin_dofs();

        // A default damping only for the angular dofs
        for i in DIM..SPATIAL_DIM {
            if locked_bits & (1 << i) == 0 {
                // This is a free angular DOF.
                out[curr_free_dof] = 0.1;
                curr_free_dof += 1;
            }
        }
    }

    /// Maximum number of velocity constrains that can be generated by this multibody_joint.
    pub fn num_velocity_constraints(&self) -> usize {
        let locked_bits = self.data.locked_axes.bits();
        let limit_bits = self.data.limit_axes.bits();
        let motor_bits = self.data.motor_axes.bits();
        let mut num_constraints = 0;

        #[cfg(feature = "dim3")]
        let spherical = self.is_spherical();
        #[cfg(feature = "dim2")]
        let spherical = false;
        let cone = if spherical { self.swing_cone_axes() } else { 0 };
        for i in 0..SPATIAL_DIM {
            if (locked_bits & (1 << i)) == 0 {
                let in_cone = i >= DIM && (cone & (1 << (i - DIM))) != 0;
                if (limit_bits & (1 << i)) != 0 && !in_cone {
                    num_constraints += 1;
                }
                if (motor_bits & (1 << i)) != 0 {
                    num_constraints += 1;
                }
            }
        }
        if cone != 0 {
            num_constraints += 1;
        }

        num_constraints
    }

    /// Whether all three of the joint's angles are free: its rotation is held as a whole
    /// (`joint_rot`), and its angles, swing cone and twist are measured on it.
    #[cfg(feature = "dim3")]
    fn is_spherical(&self) -> bool {
        (self.data.locked_axes.bits() >> DIM) & 0b111 == 0
    }

    /// The two angular axes (bit `i` for angle `i`) whose shared limit is a swing cone about
    /// the third, on a spherical joint whose coupled angular axes are both limited; zero when
    /// the joint has no cone.
    fn swing_cone_axes(&self) -> u8 {
        let coupled = (self.data.coupled_axes.bits() >> DIM) & 0b111;
        let limited = (self.data.limit_axes.bits() >> DIM) & 0b111;
        if coupled.count_ones() == 2 && limited & coupled == coupled {
            coupled
        } else {
            0
        }
    }

    /// Initialize and generate velocity constraints to enforce, e.g., multibody_joint limits and motors.
    pub fn velocity_constraints(
        &self,
        params: &IntegrationParameters,
        multibody: &Multibody,
        link: &MultibodyLink,
        mut j_id: usize,
        jacobians: &mut DVector,
        constraints: &mut [GenericJointConstraint],
    ) -> usize {
        let j_id = &mut j_id;
        let locked_bits = self.data.locked_axes.bits();
        let limit_bits = self.data.limit_axes.bits();
        let motor_bits = self.data.motor_axes.bits();
        let mut num_constraints = 0;
        let mut curr_free_dof = 0;

        for i in 0..DIM {
            if (locked_bits & (1 << i)) == 0 {
                let limits = if (limit_bits & (1 << i)) != 0 {
                    Some([self.data.limits[i].min, self.data.limits[i].max])
                } else {
                    None
                };

                if (motor_bits & (1 << i)) != 0 {
                    joint::unit_joint_motor_constraint(
                        params,
                        multibody,
                        link,
                        &self.data.motors[i],
                        self.coords[i],
                        limits,
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                    );
                }

                if (limit_bits & (1 << i)) != 0 {
                    joint::unit_joint_limit_constraint(
                        params,
                        multibody,
                        link,
                        [self.data.limits[i].min, self.data.limits[i].max],
                        self.coords[i],
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                        self.data.softness,
                    );
                }
                // Dry joint friction (MuJoCo `frictionloss`), keyed per-DoF on
                // the multibody rather than per-axis on the joint. Zero (the
                // default) emits nothing.
                let friction = multibody.frictions()[link.assembly_id + curr_free_dof];
                if friction > 0.0 {
                    joint::unit_joint_friction_constraint(
                        params,
                        multibody,
                        link,
                        friction,
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                        self.data.softness,
                    );
                }
                curr_free_dof += 1;
            }
        }

        /*
        let locked_ang_bits = locked_bits >> DIM;
        let num_free_ang_dofs = ANG_DIM - locked_ang_bits.count_ones() as usize;
        match num_free_ang_dofs {
            0 => { /* No free dofs. */ }
            1 => {}
            2 => {
                todo!()
            }
            3 => {}
            _ => unreachable!(),
        }
         */
        #[cfg(feature = "dim3")]
        if self.is_spherical() {
            self.spherical_velocity_constraints(
                params,
                multibody,
                link,
                curr_free_dof,
                j_id,
                jacobians,
                constraints,
                &mut num_constraints,
            );
            return num_constraints;
        }

        for i in DIM..SPATIAL_DIM {
            if (locked_bits & (1 << i)) == 0 {
                let limits = if (limit_bits & (1 << i)) != 0 {
                    let limits = [self.data.limits[i].min, self.data.limits[i].max];
                    joint::unit_joint_limit_constraint(
                        params,
                        multibody,
                        link,
                        limits,
                        self.coords[i],
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                        self.data.softness,
                    );
                    Some(limits)
                } else {
                    None
                };

                if (motor_bits & (1 << i)) != 0 {
                    joint::unit_joint_motor_constraint(
                        params,
                        multibody,
                        link,
                        &self.data.motors[i],
                        self.coords[i],
                        limits,
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                    );
                }

                // Dry joint friction (MuJoCo `frictionloss`), keyed per-DoF on
                // the multibody rather than per-axis on the joint. Zero (the
                // default) emits nothing.
                let friction = multibody.frictions()[link.assembly_id + curr_free_dof];
                if friction > 0.0 {
                    joint::unit_joint_friction_constraint(
                        params,
                        multibody,
                        link,
                        friction,
                        curr_free_dof,
                        j_id,
                        jacobians,
                        constraints,
                        &mut num_constraints,
                        self.data.softness,
                    );
                }
                curr_free_dof += 1;
            }
        }

        num_constraints
    }

    /// The limit, motor and friction constraints of a joint whose three angles are free.
    ///
    /// The angles are read off the joint's rotation `q` (the second frame in the first) as
    /// an impulse joint reads them: the angle about axis `i` is `2 atan2(q_i, q_w)`, its
    /// limit row pushes along that angle's gradient over the joint's angular velocities
    /// (which are expressed in the first frame), and its motor drives the angular velocity
    /// about the axis. A swing cone (two coupled, limited angular axes) holds the angle
    /// between the two frames' third axes with one row along the axis it turns about.
    #[cfg(feature = "dim3")]
    #[allow(clippy::too_many_arguments)]
    fn spherical_velocity_constraints(
        &self,
        params: &IntegrationParameters,
        multibody: &Multibody,
        link: &MultibodyLink,
        first_ang_dof: usize,
        j_id: &mut usize,
        jacobians: &mut DVector,
        constraints: &mut [GenericJointConstraint],
        num_constraints: &mut usize,
    ) {
        use crate::dynamics::solver::WritebackId;

        let limit_bits = self.data.limit_axes.bits();
        let motor_bits = self.data.motor_axes.bits();
        let cone = self.swing_cone_axes();
        let mut q = self.joint_rot;
        if q.w < 0.0 {
            q = -q;
        }
        let imag = [q.x, q.y, q.z];
        let w = q.w;

        for a in 0..ANG_DIM {
            let i = DIM + a;
            let in_cone = (cone & (1 << a)) != 0;
            let angle = 2.0 * imag[a].atan2(w);
            let limits = [self.data.limits[i].min, self.data.limits[i].max];

            if (limit_bits & (1 << i)) != 0 && !in_cone {
                let center = (limits[0] + limits[1]) * 0.5;
                let half_range = (limits[1] - limits[0]) * 0.5;
                let recentered = wrap_angle(angle - center);
                let gradient = angle_gradient(imag, w, a);
                joint::unit_joint_limit_constraint_along(
                    params,
                    multibody,
                    link,
                    [-half_range, half_range],
                    recentered,
                    first_ang_dof,
                    &gradient,
                    j_id,
                    jacobians,
                    constraints,
                    num_constraints,
                    self.data.softness,
                    WritebackId::Limit(i),
                );
            }

            if (motor_bits & (1 << i)) != 0 {
                joint::unit_joint_motor_constraint(
                    params,
                    multibody,
                    link,
                    &self.data.motors[i],
                    angle,
                    None,
                    first_ang_dof + a,
                    j_id,
                    jacobians,
                    constraints,
                    num_constraints,
                );
            }

            let friction = multibody.frictions()[link.assembly_id + first_ang_dof + a];
            if friction > 0.0 {
                joint::unit_joint_friction_constraint(
                    params,
                    multibody,
                    link,
                    friction,
                    first_ang_dof + a,
                    j_id,
                    jacobians,
                    constraints,
                    num_constraints,
                    self.data.softness,
                );
            }
        }

        if cone != 0 {
            let axis = (!cone & 0b111).trailing_zeros() as usize;
            let axis1 = Vector::ith(axis, 1.0);
            let axis2 = q * axis1;
            let swing = axis1.dot(axis2).clamp(-1.0, 1.0).acos();
            let turn = axis1.cross(axis2);
            let dir = if turn.length() > 1.0e-6 {
                turn.normalize()
            } else {
                Vector::ith((axis + 1) % 3, 1.0)
            };
            let first_coupled = (cone.trailing_zeros() as usize) + DIM;
            let limits = [
                self.data.limits[first_coupled].min,
                self.data.limits[first_coupled].max,
            ];
            joint::unit_joint_limit_constraint_along(
                params,
                multibody,
                link,
                limits,
                swing,
                first_ang_dof,
                &[dir.x, dir.y, dir.z],
                j_id,
                jacobians,
                constraints,
                num_constraints,
                self.data.softness,
                WritebackId::Limit(first_coupled),
            );
        }
    }
}

/// `angle` wrapped into `[-π, π]`.
#[cfg(feature = "dim3")]
fn wrap_angle(angle: Real) -> Real {
    let two_pi = core::f32::consts::TAU as Real;
    let pi = core::f32::consts::PI as Real;
    angle - two_pi * ((angle + pi) / two_pi).floor()
}

/// The gradient of the angle `2 atan2(q_axis, q_w)` of the rotation `q` (its imaginary part
/// `imag`, real part `w`) with respect to an angular velocity expressed in the rotation's
/// first frame: `axis` itself while the two other angles stand at zero, tilted toward them
/// as the rotation swings away from `axis`.
#[cfg(feature = "dim3")]
fn angle_gradient(imag: [Real; 3], w: Real, axis: usize) -> [Real; 3] {
    // `w² + q_axis²` is `cos²(θ / 2)` of the swing `θ` away from `axis`; it vanishes only with
    // the two frames' axes opposite, where the angle is undefined. The floor holds the
    // gradient finite there.
    const MIN_PLANE: Real = 7.6e-5;
    let j = (axis + 1) % 3;
    let k = (axis + 2) % 3;
    let plane = (w * w + imag[axis] * imag[axis]).max(MIN_PLANE);
    let mut gradient = [0.0; 3];
    gradient[axis] = 1.0;
    gradient[j] = (w * imag[k] + imag[axis] * imag[j]) / plane;
    gradient[k] = (imag[axis] * imag[k] - w * imag[j]) / plane;
    gradient
}
