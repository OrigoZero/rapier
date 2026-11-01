//! The scalar N-particle constraint: gradients of each element kind, its update, warm start and solve.

use crate::dynamics::solver::solver_body::SolverBodies;
use crate::dynamics::{SoftBody, SoftBodyMaterial};
use crate::math::{DIM, Real, Vector};
use crate::utils::DotProduct;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use simba::scalar::{ComplexField as _, RealField as _};
#[cfg(feature = "dim3")]
use simba::simd::SimdComplexField as _;

/// Number of particles a soft constraint can touch.
pub(crate) const MAX_CONSTRAINT_PARTICLES: usize = DIM + 1;

/// What a soft constraint constrains.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SoftScalarConstraintKind {
    /// `|x1 - x0| - rest`, two particles.
    Distance,
    /// The angle between two triangles sharing an edge, four particles (3D only).
    #[cfg(feature = "dim3")]
    Dihedral,
    /// The signed area/volume of a simplex cell minus its rest value.
    CellVolume,
}

/// Where a soft constraint writes its accumulated impulse back (warm-start state on the soft body).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SoftScalarConstraintWriteback {
    Edge,
    #[cfg(feature = "dim3")]
    Dihedral,
    /// `impulses[0]` of a cell (the volume model).
    CellVolume,
}

/// One scalar constraint over up to `MAX_CONSTRAINT_PARTICLES` particles.
///
/// Solved like a joint constraint: `Δλ = inv_lhs · (Ċ + rhs - cfm_gain·λ)`, clamped to the impulse
/// bounds, then `v_i -= im_i ∘ ∇_i C · Δλ`; gradients follow the positions before every pass.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SoftScalarConstraint {
    pub kind: SoftScalarConstraintKind,
    pub num_particles: u8,
    pub writeback: SoftScalarConstraintWriteback,
    /// Index of the owning soft body in the solver's awake list.
    pub soft_body: u32,
    /// Index of the element in the soft body's element array.
    pub element: u32,
    /// Solver-body slots of the particles (`u32::MAX`: no slot, position frozen).
    pub solver_ids: [u32; MAX_CONSTRAINT_PARTICLES],
    /// Current particle positions (updated by `update`).
    pub pos: [Vector; MAX_CONSTRAINT_PARTICLES],
    /// Particle inverse masses (step-constant).
    pub im: [Real; MAX_CONSTRAINT_PARTICLES],
    /// Current constraint gradients.
    pub grad: [Vector; MAX_CONSTRAINT_PARTICLES],
    /// Rest length / rest volume / rest angle, depending on the kind.
    pub rest: Real,
    pub erp_inv_dt: Real,
    pub cfm_coeff: Real,
    pub inv_lhs: Real,
    pub cfm_gain: Real,
    pub rhs: Real,
    pub impulse: Real,
    /// The largest impulse the constraint had accumulated at the end of a substep of this step (the
    /// force tear criterion reads the step's peak, not its last substep).
    pub peak_impulse: Real,
    /// The most negative impulse the constraint had accumulated at the end of a substep of this
    /// step (a push: the crush criterion reads it).
    pub trough_impulse: Real,
    pub impulse_bounds: [Real; 2],
    /// The impulses per substep a distance constraint's kind and force bound hold it to, as
    /// `[push, pull]` (`Real::MAX`: no bound).
    pub load_bounds: [Real; 2],
    /// The impulses per substep past which a distance constraint gives way, as `[push, pull]`
    /// (`Real::MAX`: it does not), see [`Self::return_map`].
    pub yield_bounds: [Real; 2],
    /// The shortest and the longest rest length a yielding distance constraint's plastic bounds
    /// allow: reaching one, it hardens.
    pub rest_range: [Real; 2],
    /// The squared natural frequency of the element's spring (its softness's), `ω² = k w` with
    /// `w` its effective mass: its stiffness is `ω² / w`, which its elastic energy reads.
    pub omega_sq: Real,
    /// Whether the element's rest length (distance) or rest angle (dihedral) creeps past its
    /// material's edge yield (see [`Self::creep`]).
    pub creeps: bool,
    /// Whether a substep of this step has been solved: the next update creeps the element over
    /// it before reading its error.
    pub creep_pending: bool,
    /// The elastic energy this step's plastic flows took out of the element: its yield force
    /// times the way a yield moved its rest length, and at each creep its energy at the
    /// substep's end against the rest it flowed from, less against the one it flowed to.
    pub flow_work: Real,
}

/// What a creeping element's flow reads besides the constraint: its material (the edge yield,
/// creep rate, flow directions and plastic bounds) and the rest length or angle it was created
/// with.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ElementCreep<'a> {
    pub material: &'a SoftBodyMaterial,
    pub initial: Real,
}

/// How close to its yield bound a constraint's impulse must come for the element to count as
/// giving way: the solve holds a yielding element at its bound up to rounding.
const YIELD_TOLERANCE: Real = 1.0e-3;

/// Gradients of the dihedral angle between the triangles `(p0, p1, p2)` and `(p0, p1, p3)`
/// (Müller et al., 2007); returns the angle minus `rest`.
#[cfg(feature = "dim3")]
pub(crate) fn dihedral_gradients(pos: &[Vector; 4], rest: Real, grad: &mut [Vector; 4]) -> Real {
    let p1 = pos[1] - pos[0];
    let p2 = pos[2] - pos[0];
    let p3 = pos[3] - pos[0];
    let n1_raw = p1.cross(p2);
    let n2_raw = p1.cross(p3);
    let (l1, l2) = (n1_raw.length(), n2_raw.length());
    if l1 < 1.0e-9 || l2 < 1.0e-9 {
        *grad = [Vector::ZERO; 4];
        return 0.0;
    }
    let n1 = n1_raw / l1;
    let n2 = n2_raw / l2;
    let d = n1.dot(n2).clamp(-1.0, 1.0);
    let angle = d.simd_acos();
    let sin = (1.0 - d * d).sqrt();
    if sin < 1.0e-6 {
        // Flat configuration: the angle is stationary, no useful gradient.
        *grad = [Vector::ZERO; 4];
        return angle - rest;
    }
    // The q_i are minus the gradients of d = n1·n2, so ∇C = q / sin (C = acos(d) - rest).
    let q3 = (p1.cross(n2) + n1.cross(p1) * d) / l1;
    let q4 = (p1.cross(n1) + n2.cross(p1) * d) / l2;
    let q2 = -(p2.cross(n2) + n1.cross(p2) * d) / l1 - (p3.cross(n1) + n2.cross(p3) * d) / l2;
    let q1 = -q2 - q3 - q4;
    let scale = 1.0 / sin;
    grad[0] = q1 * scale;
    grad[1] = q2 * scale;
    grad[2] = q3 * scale;
    grad[3] = q4 * scale;
    angle - rest
}

impl SoftScalarConstraint {
    /// Whether this constraint is strained beyond `min_strain` (relative error of a distance constraint; the
    /// other kinds never are): the criterion of the re-sweep after the contacts.
    #[inline]
    pub fn is_strained(&self, min_strain: Real) -> bool {
        self.kind == SoftScalarConstraintKind::Distance
            && self.erp_inv_dt > 0.0
            && self.rest > 0.0
            && (self.rhs / self.erp_inv_dt).abs() > min_strain * self.rest
    }

    /// Updates the positions, gradients, rhs and effective mass from the current solver-body
    /// poses. Soft constraints are springs whose bias is the spring force, not an error-correction
    /// velocity, so they are solved once per substep (biased pass only) with the bias always on.
    /// A yielding distance constraint gives way first (see [`Self::return_map`], `dt` the substep
    /// length), and with `creep` the element creeps over the substep just solved (see
    /// [`Self::creep`]).
    #[inline]
    pub fn update(&mut self, bodies: &SolverBodies, dt: Real, creep: Option<ElementCreep>) {
        self.read_positions(bodies);
        let mut c = self.compute_gradients();
        if self.kind == SoftScalarConstraintKind::Distance {
            c = self.return_map(c, dt);
        }

        // Effective mass and softness.
        let w = self.effective_mass();
        if let Some(creep) = creep {
            c = self.creep(c, w, creep, dt);
        }
        let cfm_gain = w * self.cfm_coeff;
        self.cfm_gain = cfm_gain;
        self.inv_lhs = crate::utils::inv(w + cfm_gain);

        // No corrective-velocity cap: the bias is the spring force itself, and capping it would
        // make the material's response depend on the substep count.
        self.rhs = c * self.erp_inv_dt;
    }

    /// Reads the particles' positions from the solver bodies (the assembly's for those that are
    /// not solver bodies).
    #[inline]
    fn read_positions(&mut self, bodies: &SolverBodies) {
        let n = self.num_particles as usize;
        for k in 0..n {
            let id = self.solver_ids[k];
            if id != u32::MAX {
                self.pos[k] = bodies.get_pose(id).translation;
            }
        }
    }

    /// `Σ_k ∇_k C · (w_k ∇_k C)` over the current gradients.
    #[inline]
    fn effective_mass(&self) -> Real {
        let n = self.num_particles as usize;
        let mut w = 0.0;
        for k in 0..n {
            w += self.grad[k].gdot(self.grad[k] * self.im[k]);
        }
        w
    }

    /// Creeps the element over the substep `dt` that just ended with its particles where the
    /// solver bodies hold them (see [`Self::creep`]): the step's write-back takes its last
    /// substep's flow this way.
    pub fn creep_at(&mut self, bodies: &SolverBodies, creep: ElementCreep, dt: Real) {
        self.read_positions(bodies);
        let c = self.compute_gradients();
        let w = self.effective_mass();
        self.creep(c, w, creep, dt);
    }

    /// The elastic energy `f² / 2k` the element holds at error `c`, its stiffness `k = ω² / w`
    /// (`w` its effective mass) and its force within its load bounds (`dt` the substep length the
    /// bounds are impulses over).
    #[inline]
    fn energy(&self, c: Real, w: Real, dt: Real) -> Real {
        if w <= 0.0 || self.omega_sq <= 0.0 || dt <= 0.0 {
            return 0.0;
        }
        let k = self.omega_sq / w;
        let bound = |impulse: Real| {
            if impulse == Real::MAX {
                Real::MAX
            } else {
                impulse / dt
            }
        };
        let force = (k * c).clamp(-bound(self.load_bounds[0]), bound(self.load_bounds[1]));
        0.5 * force * force / k
    }

    /// Creeps the element's rest length (distance) or rest angle (dihedral), at error `c` and
    /// effective mass `w`, over the substep `dt`: past its material's edge yield the excess flows
    /// at the edge creep rate (`edge_flowed_rest_length`, `dihedral_flowed_rest_angle`). The
    /// warm-start impulse keeps the share of the error the flow left, and the flow's energy is
    /// added to `flow_work`. Returns the error against the rest it leaves.
    fn creep(&mut self, c: Real, w: Real, creep: ElementCreep, dt: Real) -> Real {
        let rest = match self.kind {
            SoftScalarConstraintKind::Distance => {
                crate::dynamics::soft_body::edge_flowed_rest_length(
                    self.rest,
                    creep.initial,
                    self.rest + c,
                    creep.material,
                    dt,
                )
            }
            #[cfg(feature = "dim3")]
            SoftScalarConstraintKind::Dihedral => {
                crate::dynamics::soft_body::dihedral_flowed_rest_angle(
                    self.rest,
                    creep.initial,
                    self.rest + c,
                    creep.material,
                    dt,
                )
            }
            SoftScalarConstraintKind::CellVolume => None,
        };
        let Some(rest) = rest else {
            return c;
        };
        let left = c + self.rest - rest;
        self.flow_work += (self.energy(c, w, dt) - self.energy(left, w, dt)).max(0.0);
        if c != 0.0 {
            self.impulse *= (left / c).clamp(0.0, 1.0);
        }
        self.rest = rest;
        left
    }

    /// Flows a yielding distance constraint's rest length at the start of a substep, given its
    /// length error `c`: an element that carried its yield force through the substep just ended
    /// gave way, so its rest length takes the length the load put it at, and one whose length
    /// reached the end of its plastic range there hardens, its yield in that direction lifted.
    /// It hardens where it stands, so the solve stops it there and never springs it back to the
    /// end of the range (a hit carrying it past that end within one substep sets it that much
    /// further). Giving way adds its yield force (`dt` the substep length its yield impulse is
    /// over) times the way its rest length moved to `flow_work`. Returns the length error against
    /// the rest length it leaves.
    #[inline]
    fn return_map(&mut self, c: Real, dt: Real) -> Real {
        let [push, pull] = self.yield_bounds;
        let length = c + self.rest;
        let gave_way = |bound: Real| bound * (1.0 - YIELD_TOLERANCE);
        let inv_dt = crate::utils::inv(dt);
        if push != Real::MAX && c < 0.0 && self.impulse <= -gave_way(push) {
            self.flow_work += push * inv_dt * (self.rest - length).max(0.0);
            self.rest = length;
            if length <= self.rest_range[0] {
                self.yield_bounds[0] = Real::MAX;
            }
        } else if pull != Real::MAX && c > 0.0 && self.impulse >= gave_way(pull) {
            self.flow_work += pull * inv_dt * (length - self.rest).max(0.0);
            self.rest = length;
            if length >= self.rest_range[1] {
                self.yield_bounds[1] = Real::MAX;
            }
        } else {
            return c;
        }
        self.impulse_bounds = [
            -self.load_bounds[0].min(self.yield_bounds[0]),
            self.load_bounds[1].min(self.yield_bounds[1]),
        ];
        0.0
    }

    /// Recomputes the gradients from `self.pos`; returns the constraint value.
    #[inline]
    pub(super) fn compute_gradients(&mut self) -> Real {
        match self.kind {
            SoftScalarConstraintKind::Distance => {
                let d = self.pos[1] - self.pos[0];
                let len = d.length();
                let n = if len > 1.0e-9 { d / len } else { Vector::X };
                self.grad[0] = -n;
                self.grad[1] = n;
                len - self.rest
            }
            #[cfg(feature = "dim3")]
            SoftScalarConstraintKind::Dihedral => self.dihedral_gradients(),
            SoftScalarConstraintKind::CellVolume => {
                let x: [Vector; DIM + 1] = core::array::from_fn(|k| self.pos[k]);
                let vol = SoftBody::cell_volume(x);
                let grad = SoftBody::cell_volume_gradients(x);
                self.grad[..DIM + 1].copy_from_slice(&grad);
                vol - self.rest
            }
        }
    }

    /// Gradients of the dihedral angle of this constraint's four particles.
    #[cfg(feature = "dim3")]
    fn dihedral_gradients(&mut self) -> Real {
        let pos = [self.pos[0], self.pos[1], self.pos[2], self.pos[3]];
        let mut grad = [Vector::ZERO; 4];
        let c = dihedral_gradients(&pos, self.rest, &mut grad);
        self.grad[..4].copy_from_slice(&grad);
        c
    }

    /// Applies the accumulated impulse to the particle velocities.
    #[inline]
    pub fn warmstart(&self, bodies: &mut SolverBodies) {
        self.apply_impulse(bodies, self.impulse);
    }

    /// One Gauss-Seidel iteration on this constraint.
    #[inline]
    pub fn solve(&mut self, bodies: &mut SolverBodies) {
        let n = self.num_particles as usize;
        let mut dc = 0.0;
        for k in 0..n {
            let id = self.solver_ids[k];
            if id != u32::MAX {
                dc += self.grad[k].gdot(bodies.get_vel(id).linear);
            }
        }
        let rhs = dc + self.rhs;
        let total = (self.impulse + self.inv_lhs * (rhs - self.cfm_gain * self.impulse))
            .clamp(self.impulse_bounds[0], self.impulse_bounds[1]);
        let delta = total - self.impulse;
        self.impulse = total;
        self.apply_impulse(bodies, delta);
    }

    #[inline]
    fn apply_impulse(&self, bodies: &mut SolverBodies, impulse: Real) {
        let n = self.num_particles as usize;
        for k in 0..n {
            let id = self.solver_ids[k];
            if id != u32::MAX && (id as usize) < bodies.len() {
                let dv = self.grad[k] * self.im[k] * impulse;
                bodies.vels[id as usize].linear -= dv;
            }
        }
    }
}
