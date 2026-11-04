use crate::alloc_prelude::*;
use crate::dynamics::solver::coupled_multibody_rows::CoupledRows;
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

/// What a multibody's energy is held to over a substep (see `update_multibody_dynamics`).
pub(crate) struct MultibodyLedger {
    /// The kinetic energy the multibody stood at before its move
    /// (`Multibody::kicked_kinetic_energy`), or, for the first substep of a step, the energy the
    /// last step left it standing at (`Multibody::open_step`).
    pub standing: Real,
    /// The kinetic energy the joints' damping took from the second half of the kick the
    /// multibody moved with, between the energy it stood at and its move
    /// (`Multibody::moving_damping_loss`); 0 where the substep's rows changed that velocity.
    pub damped: Real,
    /// The kinetic energy a substep its rows bore on is held to: the energy of the velocity it
    /// moved with or, as for `standing`, of the velocity it started with if that is less.
    pub held: Real,
    /// The kinetic energy of the velocity it moved with.
    pub moved: Real,
    /// Its run of `VelocitySolver::multibody_ledger_poses`: where its links stood before the move.
    pub poses: core::ops::Range<usize>,
    /// Whether a spring pulls any of its joints.
    pub sprung: bool,
    /// The kinetic energy it stood at once its inertial forces turned its velocity for where
    /// the move took it, before the stabilization pass.
    pub turned: Real,
    /// Whether the substep's rows changed the velocity it moves with.
    pub solved: bool,
    /// Whether rows bore on it over the substep and every one of them is a contact between two
    /// of its own links (`VelocitySolver::multibody_self_contacts`,
    /// `VelocitySolver::multibody_bound_out`): its links knocking on one another.
    pub self_contact_only: bool,
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
    /// What the energy of each multibody of `multibody_roots` is held to over the current
    /// substep, when its energy is held, indexed like `multibody_roots`: read as the substep
    /// moves the multibody, applied once its stabilization pass has settled the velocity it ends
    /// at (`update_multibody_dynamics`).
    pub multibody_ledgers: Vec<Option<MultibodyLedger>>,
    /// The poses the links of the multibodies of `multibody_ledgers` moved from over the
    /// current substep, each ledger's run of them in its `poses`.
    pub multibody_ledger_poses: Vec<(Vector, Rotation)>,
    /// The kinetic energy each multibody of `multibody_roots` stands at as the step begins, where
    /// its last substep left it (`MultibodyLedger::standing`), indexed like `multibody_roots`:
    /// `None` where rows bore on that substep or it has none.
    pub multibody_opening_standings: Vec<Option<Real>>,
    /// The generalized velocity every multibody starts the current substep with, its velocity
    /// increment applied and before any row of the substep is solved, indexed like
    /// `generic_solver_vels`.
    pub generic_free_vels: DVector,
    /// Whether something moves or drives each multibody of `multibody_roots` through the
    /// current substep's rows, indexed like `multibody_roots`.
    pub multibody_driven: Vec<bool>,
    /// Whether a contact between two of its own links bears on each multibody of
    /// `multibody_roots` over the current substep, indexed like `multibody_roots`.
    pub multibody_self_contacts: Vec<bool>,
    /// Whether a row of the current substep other than a contact between two of its own links
    /// bears on each multibody of `multibody_roots`: a contact with anything outside it, a row of
    /// one of its joints (a limit or a motor), or a row of an impulse joint, indexed like
    /// `multibody_roots`.
    pub multibody_bound_out: Vec<bool>,
    /// The multibody joint rows each pass solves its multibody contacts together with, and the
    /// buffers of those blocks (see [`crate::dynamics::solver::coupled_multibody_rows`]).
    pub coupled_rows: CoupledRows,
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
    /// What each solver body flagged `SOLVER_BODY_GRAVITY_FIELD` asks the hooks about, and the
    /// scale the answer is applied with, indexed like `solver_bodies`. Only flagged slots are
    /// written and read, each by the stage that owns the slot.
    pub solver_gravity: Vec<SolverGravity>,
    /// The linear velocity each solver body flagged `SOLVER_BODY_KEEP_ENDED` ended the last
    /// substep with (for the first substep: the one it started the step with), before this
    /// substep's increments, indexed like `solver_bodies`. Only flagged slots are written and
    /// read: written by the increment stage, read by the FEM step after it.
    pub ended_vels: Vec<Vector>,
}

/// What a solver body whose gravity comes from the hooks asks them about, and how the answer is
/// applied: the subject handed to `PhysicsHooks::gravity`, and the per-axis factor the acceleration
/// it answers is multiplied by (the gravity scale on each axis the body is free to move along,
/// zero on a locked one).
#[derive(Copy, Clone, Debug)]
pub(crate) struct SolverGravity {
    pub subject: crate::pipeline::GravitySubject,
    pub scale: Vector,
}

impl Default for SolverGravity {
    fn default() -> Self {
        Self {
            subject: crate::pipeline::GravitySubject::RigidBody(
                crate::dynamics::RigidBodyHandle::invalid(),
            ),
            scale: Vector::ZERO,
        }
    }
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
            multibody_ledgers: Vec::new(),
            multibody_ledger_poses: Vec::new(),
            multibody_opening_standings: Vec::new(),
            generic_free_vels: DVector::zeros(0),
            multibody_driven: Vec::new(),
            multibody_self_contacts: Vec::new(),
            multibody_bound_out: Vec::new(),
            coupled_rows: CoupledRows::default(),
            #[cfg(feature = "dim3")]
            solver_gyro: Vec::new(),
            substep_vels: Vec::new(),
            solver_gravity: Vec::new(),
            ended_vels: Vec::new(),
        }
    }
}
