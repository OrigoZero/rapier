//! The `SoftBody` struct and the shared constants of the soft-body constraints.
#[cfg(feature = "dim3")]
use super::SoftBodyDihedral;
#[cfg(feature = "fem")]
use super::SoftBodySolver;
#[cfg(feature = "serde-serialize")]
use super::soft_body_material::default_true;
use super::{
    SoftBodyCell, SoftBodyCellModel, SoftBodyEdge, SoftBodyMaterial, SoftBodyParticle,
    SoftParticleAttachment,
};
use crate::alloc_prelude::*;
use crate::dynamics::RigidBodyHandle;
use crate::math::{DIM, Real, Vector};

/// Number of particles a soft-body constraint can touch (a simplex cell).
pub const SOFT_BODY_MAX_CONSTRAINT_PARTICLES: usize = DIM + 1;

/// Color id marking a soft-body element that could not be given a parallel color
/// (solved serially by the solver).
pub(crate) const SOFT_BODY_OVERFLOW_COLOR: u8 = 128;

/// A deformable body: [`SoftBodyParticle`]s linked by elastic elements ([`SoftBodyEdge`]s, bending,
/// [`SoftBodyCell`]s, volume preservation, shape matching). Its hidden root rigid body (tagged by
/// [`crate::dynamics::RigidBody::soft_body`]) must never be moved, removed or attached to joints.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SoftBody {
    pub(crate) particles: Vec<SoftBodyParticle>,
    /// The hidden rigid body standing for this soft body in the islands and holding its
    /// colliders (invalid until the soft body is inserted in a set): the whole-body cluster's
    /// proxy, re-pointed to another live cluster's proxy if that cluster is removed.
    pub(crate) root_body: RigidBodyHandle,
    /// The clusters of this soft body (index 0: the whole-body cluster, created at insertion).
    /// Removed clusters leave a dead slot so indices stay stable.
    pub(crate) clusters: Vec<super::SoftBodyCluster>,
    /// Number of live clusters referencing each particle: a particle is removed when it drops
    /// to zero.
    pub(crate) cluster_refs: Vec<u32>,
    /// The particles attached to rigid bodies.
    pub(crate) attachments: Vec<SoftParticleAttachment>,
    /// Whether the soft body is currently asleep (mirrors the root body's state at the end of
    /// each step).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) sleeping: bool,
    /// Whether the soft body takes part in the simulation (applied to the root body at the start
    /// of the next step; cleared by the NaN quarantine).
    #[cfg_attr(feature = "serde-serialize", serde(default = "default_true"))]
    pub(crate) enabled: bool,
    /// Set by the setters that move particles between steps (positions, pinning): the colliders
    /// are updated at the start of the next step so the narrow phase sees the new geometry.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) positions_modified: bool,
    /// Set when the attachments changed since the last step: the island links are updated at
    /// the start of the next step.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) attachments_modified: bool,
    pub(crate) edges: Vec<SoftBodyEdge>,
    #[cfg(feature = "dim3")]
    pub(crate) dihedrals: Vec<SoftBodyDihedral>,
    pub(crate) cells: Vec<SoftBodyCell>,
    /// Each cell's block as the solver last built it (see `SoftCellBlock`), indexed like
    /// `cells`: a cell whose rest shape, masses, material and substep length stayed the same
    /// takes it again.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) cell_blocks: Vec<crate::dynamics::solver::SoftCellBlock>,
    /// The computational mesh's boundary: segments (2D) or triangles (3D), oriented outward.
    /// Used by the volume constraints, and as the geometry of the default collision mesh.
    pub(crate) boundary: Vec<[u32; DIM]>,
    /// Whether the boundary is closed (every segment vertex / triangle edge shared by exactly
    /// two elements).
    pub(crate) boundary_closed: bool,
    /// The cell owning each boundary element (`u32::MAX`: none): the tearing pass follows it,
    /// and the winding of an element whose cell is inverted no longer tells where the outside
    /// is.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) boundary_element_cells: Vec<u32>,
    pub(crate) material: SoftBodyMaterial,
    /// The materials of the body's regions: an element whose `material` is `k > 0` takes
    /// `region_materials[k - 1]`, one whose `material` is `0` the body's own `material`.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) region_materials: Vec<SoftBodyMaterial>,
    /// The plane the particles are held to, as its unit normal and its offset along it (a
    /// particle at `x` stands on it when `normal · x == offset`); `None`: the particles move in
    /// all three dimensions.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) plane: Option<(Vector, Real)>,
    pub(crate) cell_model: SoftBodyCellModel,
    /// Which solver simulates this body's elasticity.
    #[cfg(feature = "fem")]
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) solver: SoftBodySolver,
    /// Whether the volume pieces are constrained (only set while there is at least one piece).
    pub(crate) volume_preservation: bool,
    /// The pieces of material enclosed by a closed boundary, each with its own volume constraint
    /// (rebuilt with the boundary tables).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) volume_pieces: Vec<super::SoftVolumePiece>,
    /// Multiplier of the target volume (`> 1` inflates the body).
    pub(crate) volume_factor: Real,
    /// The rest shape's center of mass at insertion (world space).
    pub(crate) rest_com: Vector,
    /// The particle spacing: the impact-adaptive substeps bound the travel per substep by it,
    /// and the collision meshes' colliders take their default contact skin from it.
    pub(crate) particle_radius: Real,
    /// Rigid-body settings applied to the particles at insertion.
    pub(crate) particle_settings: super::SoftBodyParticleSettings,
    /// Number of parallel colors used by the elements (`0..num_colors`, plus possibly the
    /// overflow color).
    pub(crate) num_colors: u8,
    /// Whether some element has the overflow color.
    pub(crate) has_overflow_color: bool,
    /// Set by the setters that change the dynamics (material, volume factor, shape matching...):
    /// the pipeline wakes the soft body up at the start of the next step so the change takes
    /// effect even if it was sleeping.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) modified: bool,
    /// Set by the solver when a cell flowed plastically by a significant amount during the last
    /// step: the body is kept awake so its creep goes on.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) plastic_flowing: bool,
    /// Set when the elements flowed plastically: the particles' rest positions are fitted to the
    /// flowed rest shapes at the end of the step, over the following steps until they settle
    /// (see `SoftBody::fit_rest_positions`).
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) rest_fit_pending: bool,
    /// The speed the sleep rule reads for this body's cluster proxies: the farthest any particle
    /// moved over the last step, divided by the step's length, or `Real::MAX` while the body
    /// flows plastically (kept awake). A step with no [`Self::sleep_reference`] to measure from
    /// reads the fastest particle's velocity instead.
    /// `RigidBodyActivation::update_energy` compares it against the proxies' activation threshold.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) sleep_speed: Real,
    /// Each particle's position at the end of the last step this body was simulated in, which
    /// [`Self::sleep_speed`] measures the next step's travel from. Empty until the body's first
    /// step, and after its particles were placed by `SoftBodySet::rest_at`; a reference whose
    /// length differs from the particles' (a tear or a cut changed them) is not read.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) sleep_reference: Vec<Vector>,
    /// Set when some edge or cell has a `torn` mark: the tearing pass at the end of the step
    /// removes them (see `SoftBodySet::tear`).
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) tearing_pending: bool,
    /// Bumped whenever the topology changes (a tear removed elements or duplicated particles):
    /// renderers keying their meshes on the particles and elements rebuild them on change.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) topology_version: u32,
    /// Largest normal approach speed of the rigid bodies met by the surface, for the last step
    /// and the one before (`None`: no contact constraint that step); with the particle speeds, it drives
    /// the impact-adaptive substeps (`IntegrationParameters::soft_bodies.max_extra_substeps`).
    /// A step split into several CCD passes counts once, with the largest speed of its passes.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) contact_approach_speeds: [Option<Real>; 2],
    /// Set by the first solve of a step, which shifted `contact_approach_speeds`: the step's later
    /// CCD passes merge into its entry. Cleared by the end-of-step sync.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) contact_approach_step_open: bool,
    /// Total normal impulse of the surface's contact constraints over the last step, and the extra
    /// substeps currently requested from that load (see `sync_particle_positions`).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) contact_load: Real,
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) load_extra_substeps: u8,
    /// The particles whose state the solver keeps at the end of every substep
    /// ([`Self::set_recorded_particles`]), sorted and unique.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) recorded_particles: Vec<u32>,
    /// The state each recorded particle ended each substep of the last step the solver moved
    /// this body in, substep-major: `substep_particles[k * recorded + r]` is the velocity and
    /// position of `recorded_particles[r]` after substep `k`.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) substep_particles: Vec<(Vector, Vector)>,
    /// The length of the substeps `substep_particles` holds.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) substep_particles_dt: Real,
    /// The velocity the impulses applied since [`Self::clear_impulse_particles`] gave each
    /// recorded particle, parallel to `recorded_particles` ([`Self::impulse_particles`]).
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) impulse_particles: Vec<Vector>,
    /// The soft body this one was split off from by a tear or a cut (see
    /// [`Self::origin`]).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) origin: Option<crate::dynamics::SoftBodyHandle>,
    /// The soft bodies split off from this one, in creation order (see [`Self::pieces`]).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) pieces: Vec<crate::dynamics::SoftBodyHandle>,
    /// User-defined data associated to this soft body.
    pub user_data: u128,
}
