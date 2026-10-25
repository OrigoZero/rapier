use crate::alloc_prelude::*;
use crate::dynamics::solver::solver_body::SolverBodies;
use crate::dynamics::{MultibodyLinkId, solver::SolverVel};
#[cfg(feature = "dim3")]
use crate::math::AngVector;
use crate::math::{DVector, Real, Rotation, Vector};

/// Per-solver-body data for the staged solver's per-substep gyroscopic pass (3D
/// only). `enabled` is false for bodies whose gyroscopic forces are off (the
/// default), for frontier/sleeping bodies, and for multibody links.
#[cfg(feature = "dim3")]
#[derive(Copy, Clone)]
pub(crate) struct GyroParams {
    /// Principal inertia, i.e. the diagonal of the inertia tensor in the
    /// principal-axes frame.
    pub principal_inertia: AngVector,
    pub inv_principal_inertia: AngVector,
    /// Orientation of the principal-axes frame relative to the body's local frame
    /// (`MassProperties::principal_inertia_local_frame`); the gyroscopic pass composes it with
    /// the world rotation, so bodies with tilted principal axes are handled correctly.
    pub principal_frame: Rotation,
    pub enabled: bool,
}

#[cfg(feature = "dim3")]
impl Default for GyroParams {
    fn default() -> Self {
        Self {
            principal_inertia: AngVector::ZERO,
            inv_principal_inertia: AngVector::ZERO,
            principal_frame: Rotation::IDENTITY,
            enabled: false,
        }
    }
}

/// The solver-body buffers shared by the staged island solver. The actual solve (constraint
/// init, PGS sweeps, integration, writeback) is driven by the staged solver's worker stages;
/// this type only owns the reusable buffers they operate on.
pub(crate) struct VelocitySolver {
    pub solver_bodies: SolverBodies,
    pub solver_vels_increment: Vec<SolverVel<Real>>,
    pub generic_solver_vels: DVector,
    pub generic_solver_vels_increment: DVector,
    pub multibody_roots: Vec<MultibodyLinkId>,
    /// Per-solver-body gyroscopic parameters (3D), indexed like `solver_bodies`; consumed each
    /// substep by the velocity-increment stage, gated by a global "any gyroscopic body" flag so
    /// gyro-free scenes (the common case) skip the pass entirely.
    #[cfg(feature = "dim3")]
    pub solver_gyro: Vec<GyroParams>,
    /// The velocity and centre-of-mass pose each recording solver body
    /// (`SOLVER_BODY_RECORD_SUBSTEPS`: a rigid body that records its substeps, or a recorded
    /// soft-body particle) ended each substep but the last with, indexed like `solver_bodies`.
    /// Only the slots of recording bodies are written, each by the worker that claims the slot;
    /// the list only grows, so a step that records nothing allocates nothing.
    pub substep_vels: Vec<Vec<SubstepRecord>>,
}

/// The velocity, position and orientation one solver body ended one substep with, as
/// `VelocitySolver::substep_vels` keeps it for a recording body.
pub(crate) type SubstepRecord = (SolverVel<Real>, Vector, Rotation);

impl VelocitySolver {
    pub fn new() -> Self {
        Self {
            solver_bodies: SolverBodies::default(),
            solver_vels_increment: Vec::new(),
            generic_solver_vels: DVector::zeros(0),
            generic_solver_vels_increment: DVector::zeros(0),
            multibody_roots: Vec::new(),
            #[cfg(feature = "dim3")]
            solver_gyro: Vec::new(),
            substep_vels: Vec::new(),
        }
    }
}
