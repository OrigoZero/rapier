//! Point-to-point constraints attaching a soft-body particle to a rigid body
//! (`SoftBody::attach_particle`): `DIM` coupled bilateral rows keeping the particle on a point of
//! the body, solved like a locked joint (both passes, no bias in relax, softness, warm start).
//!
//! The attachments holding particles to one simulated rigid body are solved together as a block
//! ([`solve_attachment_block`]): the body takes the impulses of all of them at once, so it moves
//! as a body carrying the mass of every particle held to it.

use core::ops::Range;

use crate::dynamics::solver::solver_body::SolverBodies;
use crate::math::{AngVector, AngularInertia, DIM, Matrix, Real, SPATIAL_DIM, Vector};
use crate::utils::{AngularInertiaOps, ComponentMul, CrossProduct};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use simba::scalar::{ComplexField as _, RealField as _};

/// One particle attachment, as solved by the staged solver.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SoftAttachmentConstraint {
    /// Index of the soft body in the solver's awake list, and of the attachment in it.
    pub soft_body: u32,
    pub attachment: u32,
    /// The particle's solver slot and inverse mass.
    pub particle: u32,
    pub im_particle: Real,
    /// The particle's body is on the FEM path: the constraint acts on it through the body's
    /// augmented mass (see [`FemAttachment`]).
    pub fem: Option<FemAttachment>,
    /// The rigid body's solver slot (`u32::MAX`: fixed or not simulated, the anchor stays at
    /// `anchor0`) and its attachment point in its CoM-local frame.
    pub body: u32,
    pub body_local_point: Vector,
    pub anchor0: Vector,
    /// Joint softness.
    pub erp_inv_dt: Real,
    pub cfm_coeff: Real,
    // Solve state.
    pub body_im: Vector,
    pub body_ii: AngularInertia,
    /// World lever arm of the anchor about the body's center of mass.
    pub arm: Vector,
    /// The constraints' effective-mass matrix and its (softened) inverse.
    pub lhs: Matrix,
    pub inv_lhs: Matrix,
    pub rhs: Vector,
    pub impulse: Vector,
}

/// A run of attachments solved together: `members` (a range of
/// `SoftConstraintsSet::attachment_block_members`, indices into its `attachments`) hold distinct
/// particles that move (free ones, or any on the FEM path) to the rigid body in solver slot
/// `body`, solved by [`solve_attachment_block`]. `body == u32::MAX` marks the attachments solved
/// one at a time: to a fixed body, of a pinned particle off the FEM path, or of a particle held
/// more than once.
#[derive(Clone, Debug)]
pub(crate) struct SoftAttachmentBlock {
    pub body: u32,
    pub members: Range<usize>,
}

/// The FEM side of an attachment: `DIM` responses `A⁻¹Jᵀ` (one per particle axis) in
/// `SoftConstraintsSet::fem_responses` from `start`, and their values at the particle, the block
/// `(A⁻¹)_pp` standing in for `im · I` in the constraint's effective mass.
#[derive(Copy, Clone, Debug)]
pub(crate) struct FemAttachment {
    pub first_slot: u32,
    pub num_particles: u32,
    pub start: u32,
    pub inv_mass: Matrix,
}

/// The matrix `-[r]× ii [r]×` (3D) or `ii perp(r) perp(r)ᵀ` (2D): the velocity change of the
/// point at lever arm `r` per unit impulse applied there, through the body's rotation.
fn rotational_effective_mass(arm: Vector, ii: AngularInertia) -> Matrix {
    #[cfg(feature = "dim2")]
    {
        let p = Vector::new(-arm.y, arm.x);
        Matrix::from_cols(p * (p.x * ii), p * (p.y * ii))
    }
    #[cfg(feature = "dim3")]
    {
        let rx = Matrix::from_cols(
            Vector::new(0.0, arm.z, -arm.y),
            Vector::new(-arm.z, 0.0, arm.x),
            Vector::new(arm.y, -arm.x, 0.0),
        );
        -(rx * ii.into_matrix() * rx)
    }
}

/// The `DIM × SPATIAL_DIM` map from a body's twist (linear velocity first) to the velocity of
/// its point at lever arm `arm`: `v + ω × arm`.
fn twist_jacobian(arm: Vector) -> na::SMatrix<Real, DIM, SPATIAL_DIM> {
    let mut j = na::SMatrix::<Real, DIM, SPATIAL_DIM>::zeros();
    for k in 0..DIM {
        j[(k, k)] = 1.0;
    }
    #[cfg(feature = "dim2")]
    {
        j[(0, 2)] = -arm.y;
        j[(1, 2)] = arm.x;
    }
    #[cfg(feature = "dim3")]
    {
        // ω × r = -[r]× ω.
        j[(0, 4)] = arm.z;
        j[(0, 5)] = -arm.y;
        j[(1, 3)] = -arm.z;
        j[(1, 5)] = arm.x;
        j[(2, 3)] = arm.y;
        j[(2, 4)] = -arm.x;
    }
    j
}

fn to_na_vector(v: Vector) -> na::SVector<Real, DIM> {
    na::SVector::<Real, DIM>::from_fn(|i, _| v[i])
}

fn to_na_matrix(m: Matrix) -> na::SMatrix<Real, DIM, DIM> {
    na::SMatrix::<Real, DIM, DIM>::from_fn(|i, j| m.col(j)[i])
}

/// The body's inverse mass and inverse world inertia as one `SPATIAL_DIM` square block (linear
/// axes first): the twist change per unit spatial impulse.
fn inverse_spatial_mass(
    im: Vector,
    ii: AngularInertia,
) -> na::SMatrix<Real, SPATIAL_DIM, SPATIAL_DIM> {
    use crate::utils::AngularInertiaOps;
    let mut w = na::SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
    for k in 0..DIM {
        w[(k, k)] = im[k];
    }
    #[cfg(feature = "dim2")]
    {
        w[(2, 2)] = ii;
    }
    #[cfg(feature = "dim3")]
    {
        let m = ii.into_matrix();
        for r in 0..3 {
            for c in 0..3 {
                w[(DIM + r, DIM + c)] = m.col(c)[r];
            }
        }
    }
    w
}

/// Solves the attachments `members` of `constraints`, which hold distinct particles to the rigid
/// body in solver slot `body`, as one block: one Gauss-Seidel step on the body's twist and every
/// particle of the block. Each attachment `i` asks `(Pᵢ + γᵢ) δᵢ = cᵢ - Jᵢ u'` of its impulse
/// `δᵢ`, with `Pᵢ` the particle's velocity change per unit impulse (its inverse mass, or on the
/// FEM path its body's response at the particle), `γᵢ` its softness, `cᵢ` the particle's velocity
/// with the bias and the softness of the impulse it holds, `Jᵢ` the map from the body's twist to
/// its anchor's velocity and `u'` the body's twist after the impulses, `u' = u + W Σ Jᵢᵀ δᵢ` with
/// `W` the body's inverse mass. Eliminating the impulses leaves `(I + W S) u' = u + W b`,
/// `S = Σ Jᵢᵀ Dᵢ Jᵢ`, `b = Σ Jᵢᵀ Dᵢ cᵢ`, `Dᵢ = (Pᵢ + γᵢ)⁻¹`: one `SPATIAL_DIM` square solve,
/// whatever the number of particles, exact for particles that move independently. Each particle
/// then takes its impulse against `u'` in turn, reading its velocity after the members before it
/// (on the FEM path they move it through its body), and the body takes the impulses the particles
/// took. The constraints are [`SoftAttachmentConstraint::update`]d and warm-started beforehand; a
/// block the solve cannot invert is solved one constraint at a time.
pub(crate) fn solve_attachment_block(
    constraints: &mut [SoftAttachmentConstraint],
    members: &[u32],
    body: u32,
    bodies: &mut SolverBodies,
    pool: &[Vector],
) {
    let Some(&first) = members.first() else {
        return;
    };
    if body as usize >= bodies.len() {
        return;
    }
    // Every member stores the body's mass properties from its update.
    let head = &constraints[first as usize];
    let w = inverse_spatial_mass(head.body_im, head.body_ii);
    let response = |c: &SoftAttachmentConstraint, bodies: &SolverBodies| {
        let softness = c.lhs * c.cfm_coeff;
        let d = (c.particle_inv_mass() + softness).inverse();
        let target = bodies.get_vel(c.particle).linear + c.rhs - softness * c.impulse;
        (to_na_matrix(d), to_na_vector(target))
    };
    let mut s = na::SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
    let mut b = na::SVector::<Real, SPATIAL_DIM>::zeros();
    for &m in members {
        let c = &constraints[m as usize];
        let (d, target) = response(c, bodies);
        let j = twist_jacobian(c.arm);
        let jt_d = j.transpose() * d;
        s += jt_d * j;
        b += jt_d * target;
    }
    let vel = bodies.get_vel(body);
    let mut u = na::SVector::<Real, SPATIAL_DIM>::zeros();
    for k in 0..DIM {
        u[k] = vel.linear[k];
    }
    #[cfg(feature = "dim2")]
    {
        u[2] = vel.angular;
    }
    #[cfg(feature = "dim3")]
    for k in 0..3 {
        u[DIM + k] = vel.angular[k];
    }
    let system = na::SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::identity() + w * s;
    let solved = system
        .lu()
        .solve(&(u + w * b))
        .filter(|twist| twist.iter().all(|x| x.is_finite()));
    let Some(twist) = solved else {
        for &m in members {
            constraints[m as usize].solve(bodies, pool);
        }
        return;
    };
    // The body takes the impulses the particles took: on the FEM path a particle's velocity reads
    // the impulses of the members before it through its body's response.
    let mut taken = na::SVector::<Real, SPATIAL_DIM>::zeros();
    for &m in members {
        let c = &mut constraints[m as usize];
        let (d, target) = response(c, bodies);
        let j = twist_jacobian(c.arm);
        let delta = d * (target - j * twist);
        taken += j.transpose() * delta;
        let delta = Vector::from_slice(delta.as_slice());
        c.impulse += delta;
        c.apply_to_particle(bodies, pool, delta);
    }
    let twist = u + w * taken;
    let v = &mut bodies.vels[body as usize];
    for k in 0..DIM {
        v.linear[k] = twist[k];
    }
    #[cfg(feature = "dim2")]
    {
        v.angular = twist[2];
    }
    #[cfg(feature = "dim3")]
    for k in 0..3 {
        v.angular[k] = twist[DIM + k];
    }
}

impl SoftAttachmentConstraint {
    /// Current world position of the body's attachment point.
    #[inline]
    fn anchor(&self, bodies: &SolverBodies) -> Vector {
        if self.body == u32::MAX {
            self.anchor0
        } else {
            bodies.get_pose(self.body).pose() * self.body_local_point
        }
    }

    /// Updates the position error, effective mass and bias from the current solver state.
    /// `wo_bias`: relax pass (no position correction).
    pub fn update(&mut self, bodies: &SolverBodies, wo_bias: bool) {
        let particle = bodies.get_pose(self.particle).translation;
        let anchor = self.anchor(bodies);
        let mut lhs = self.particle_inv_mass();
        if self.body != u32::MAX {
            let pose = bodies.get_pose(self.body);
            self.body_im = pose.im;
            self.body_ii = pose.ii;
            self.arm = anchor - pose.translation;
            lhs += Matrix::from_diagonal(pose.im) + rotational_effective_mass(self.arm, pose.ii);
        }
        self.lhs = lhs;
        let softened = lhs * (1.0 + self.cfm_coeff);
        // No degree of freedom on either side (a pinned particle on a fixed body): inert constraint.
        self.inv_lhs = if softened.determinant().abs() > Real::EPSILON * Real::EPSILON {
            softened.inverse()
        } else {
            Matrix::ZERO
        };
        // Uncapped, like a joint's locked axes: the attachment must win over the contact constraints
        // solved after it (a cloth corner attached to a bar its surface also touches).
        self.rhs = if wo_bias {
            Vector::ZERO
        } else {
            (particle - anchor) * self.erp_inv_dt
        };
    }

    /// The particle's velocity change per unit impulse: its inverse mass, or on the FEM path the
    /// block of its body's response at the particle.
    #[inline]
    fn particle_inv_mass(&self) -> Matrix {
        match &self.fem {
            Some(fem) => fem.inv_mass,
            None => Matrix::IDENTITY * self.im_particle,
        }
    }

    /// Applies `impulse` to the particle side alone: the particle, or on the FEM path its whole
    /// body through the response.
    #[inline]
    fn apply_to_particle(&self, bodies: &mut SolverBodies, pool: &[Vector], impulse: Vector) {
        if let Some(fem) = &self.fem {
            let n = fem.num_particles as usize;
            let first = fem.first_slot as usize;
            for k in 0..DIM {
                let u = &pool[fem.start as usize + k * n..fem.start as usize + (k + 1) * n];
                let lambda = impulse[k];
                if lambda != 0.0 {
                    for (i, ui) in u.iter().enumerate() {
                        bodies.vels[first + i].linear -= *ui * lambda;
                    }
                }
            }
        } else if (self.particle as usize) < bodies.len() {
            bodies.vels[self.particle as usize].linear -= impulse * self.im_particle;
        }
    }

    /// Applies `impulse` (positive along the constraint: the particle is pulled back, the body pulled
    /// toward it).
    #[inline]
    fn apply(&self, bodies: &mut SolverBodies, pool: &[Vector], impulse: Vector) {
        self.apply_to_particle(bodies, pool, impulse);
        if self.body != u32::MAX && (self.body as usize) < bodies.len() {
            let v = &mut bodies.vels[self.body as usize];
            v.linear += self.body_im.component_mul(&impulse);
            let torque: AngVector = self.arm.gcross(impulse);
            v.angular += self.body_ii.transform_vector(torque);
        }
    }

    #[inline]
    pub fn warmstart(&mut self, bodies: &mut SolverBodies, pool: &[Vector], coeff: Real) {
        self.impulse *= coeff;
        self.apply(bodies, pool, self.impulse);
    }

    #[inline]
    pub fn solve(&mut self, bodies: &mut SolverBodies, pool: &[Vector]) {
        let vp = bodies.get_vel(self.particle).linear;
        let va = if self.body == u32::MAX {
            Vector::ZERO
        } else {
            let v = bodies.get_vel(self.body);
            v.linear + v.angular.gcross(self.arm)
        };
        let dv = vp - va;
        let total = self.impulse
            + self.inv_lhs * (dv + self.rhs - (self.lhs * self.impulse) * self.cfm_coeff);
        let delta = total - self.impulse;
        self.impulse = total;
        self.apply(bodies, pool, delta);
    }
}
