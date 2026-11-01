//! The element types of a soft body: particles, edges, dihedrals, cells, and the per-body solver and cell-model choices.
use super::SoftBodyMaterial;
#[cfg(feature = "serde-serialize")]
use super::soft_body_material::one;
use crate::dynamics::SpringCoefficients;
use crate::math::{DIM, Matrix, Real, Rotation, Vector};
use na::SimdRealField as _;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use simba::scalar::{ComplexField as _, RealField as _};

/// One particle (point mass) of a soft body: plain state owned by its
/// [`crate::dynamics::SoftBody`], given its own solver body for the step when the body is awake.
/// A pinned particle (`inv_mass == 0`) is kinematic: it moves at its velocity or toward its target.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SoftBodyParticle {
    /// World-space position.
    pub(crate) position: Vector,
    /// World-space velocity.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) velocity: Vector,
    /// Position in the body's rest shape, relative to the rest center of mass (the
    /// shape-matching goal shape). Follows the plastic flow of the elements (see
    /// [`Self::initial_rest_position`]).
    pub(crate) rest_position: Vector,
    /// The rest position before any plastic flow: the world-space position at insertion,
    /// relative to the rest center of mass. The cells' initial rest shapes are read from it.
    pub(crate) initial_rest_position: Vector,
    /// Nominal mass (used by shape matching even for pinned particles).
    pub(crate) mass: Real,
    /// `0.0` for pinned particles.
    pub(crate) inv_mass: Real,
    /// User force applied to the particle every step until reset (`SoftBody::reset_forces`).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) force: Vector,
    /// Where a pinned particle must be at the end of the next step (`None`: it keeps moving at
    /// its velocity).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) next_position: Option<Vector>,
    /// Set once an element touching the particle tore (see [`Self::is_damaged`]).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) damaged: bool,
    /// Whether some surface element has this particle as a vertex (updated with the boundary).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) on_surface: bool,
    /// The particle this one was split from, shared by every copy of a split particle (its own
    /// index when it was never split; `u32::MAX` also reads as its own index, see
    /// [`Self::split_root`]).
    #[cfg_attr(feature = "serde-serialize", serde(default = "no_split_root"))]
    pub(crate) split_root: u32,
    /// A number the body's author gave this particle (see [`Self::tag`]); `u32::MAX` when none.
    #[cfg_attr(feature = "serde-serialize", serde(default = "no_tag"))]
    pub(crate) tag: u32,
}

#[cfg(feature = "serde-serialize")]
fn no_tag() -> u32 {
    u32::MAX
}

#[cfg(feature = "serde-serialize")]
fn infinite() -> Real {
    Real::INFINITY
}

#[cfg(feature = "serde-serialize")]
fn no_split_root() -> u32 {
    u32::MAX
}

impl SoftBodyParticle {
    /// The root of this particle's split family, `index` being this particle's own index: two
    /// particles with the same root are copies of one particle a crack went through.
    pub(crate) fn split_root(&self, index: u32) -> u32 {
        if self.split_root == u32::MAX {
            index
        } else {
            self.split_root
        }
    }

    /// The number the body's author gave this particle ([`crate::dynamics::SoftBodyBuilder::particle_tags`]),
    /// `u32::MAX` when none. A particle a tear splits keeps it in every copy, and a piece split
    /// off keeps the tags of its particles, so it names the authored particle wherever it went.
    pub fn tag(&self) -> u32 {
        self.tag
    }

    /// The world-space position of this particle.
    pub fn position(&self) -> Vector {
        self.position
    }

    /// The world-space velocity of this particle.
    pub fn velocity(&self) -> Vector {
        self.velocity
    }

    /// The user force currently applied to this particle (see [`crate::dynamics::SoftBody::add_particle_force`]).
    pub fn force(&self) -> Vector {
        self.force
    }

    /// The position a pinned particle is moving to over the next step, if one was set with
    /// [`crate::dynamics::SoftBody::set_particle_kinematic_target`].
    pub fn kinematic_target(&self) -> Option<Vector> {
        self.next_position
    }

    /// The position of this particle in the rest shape (relative to the rest center of mass).
    /// With a plastic material the rest shape follows the elements' flow (see
    /// [`SoftBodyMaterial::plastic_yield`]); [`Self::initial_rest_position`] predates any flow.
    pub fn rest_position(&self) -> Vector {
        self.rest_position
    }

    /// The position of this particle in the soft body's rest shape before any plastic flow
    /// (its position at insertion, relative to the rest center of mass).
    pub fn initial_rest_position(&self) -> Vector {
        self.initial_rest_position
    }

    /// The nominal mass of this particle.
    pub fn mass(&self) -> Real {
        self.mass
    }

    /// The inverse mass of this particle (`0.0` for pinned particles).
    pub fn inv_mass(&self) -> Real {
        self.inv_mass
    }

    /// Whether this particle is pinned (infinite mass, moved kinematically).
    pub fn is_pinned(&self) -> bool {
        self.inv_mass == 0.0
    }

    /// Whether an element touching this particle has torn (or [`crate::dynamics::SoftBody::set_particle_damaged`]
    /// marked it). The elements around a damaged particle lose the interior strength of
    /// [`SoftBodyMaterial::interior_strength`]: a tear runs on from where it started.
    pub fn is_damaged(&self) -> bool {
        self.damaged
    }

    /// Whether this particle is a vertex of the body's surface (a boundary segment in 2D, a
    /// boundary triangle in 3D). An element whose particles are all interior gets the material's
    /// [`SoftBodyMaterial::interior_strength`].
    pub fn is_on_surface(&self) -> bool {
        self.on_surface
    }
}

/// The role of a soft-body edge (distance constraint).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum SoftBodyEdgeKind {
    /// A structural edge, keeping two neighboring particles at their rest distance
    /// (softness: [`SoftBodyMaterial::edge_softness`]).
    Structural,
    /// A bending edge, keeping two second-neighbor particles at their rest distance
    /// (softness: [`SoftBodyMaterial::bend_softness`]).
    Bend,
}

/// A spring an edge is held by in physical units, in place of the material's softness: its
/// stiffness in newtons per metre of stretch and its damping in newton-seconds per metre of
/// stretching speed. The constraint takes the natural frequency and damping ratio these give over
/// the edge's effective mass at each step, so the spring holds whatever its particles weigh.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SoftEdgeSpring {
    /// Stiffness, in newtons per metre (force per metre of stretch or squeeze).
    pub stiffness: Real,
    /// Damping, in newton-seconds per metre (force per metre per second of stretching speed).
    pub damping: Real,
}

impl SoftEdgeSpring {
    /// The softness of this spring between particles of inverse masses summing to `inv_mass`
    /// (`None` when they are both pinned or the spring has no stiffness).
    pub fn softness(&self, inv_mass: Real) -> Option<SpringCoefficients<Real>> {
        if inv_mass <= 0.0 || self.stiffness <= 0.0 || !self.stiffness.is_finite() {
            return None;
        }
        // ω² = k / m with m = 1 / w the effective mass; ζ = c / (2 √(k m)).
        let omega = (self.stiffness * inv_mass).sqrt();
        let zeta = self.damping.max(0.0) * inv_mass / (2.0 * omega);
        Some(SpringCoefficients::new(
            omega / Real::simd_two_pi(),
            zeta,
        ))
    }
}

/// An edge whose rest length is driven: each step it moves toward `target` at `speed`, the way a
/// hydraulic ram or a muscle extends and retracts. With [`SoftBodyEdge::max_force`] the edge
/// pushes and pulls no harder than that, so a blocked actuator stalls instead of tearing the body.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SoftEdgeActuator {
    /// The length the edge is driven to, in metres.
    pub target: Real,
    /// How fast the rest length moves toward the target, in metres per second; `Real::INFINITY`
    /// sets it at once.
    pub speed: Real,
}

/// A distance constraint between two particles of a soft body.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SoftBodyEdge {
    /// The two particles.
    pub vertices: [u32; 2],
    /// The distance the edge tries to maintain.
    pub rest_length: Real,
    /// Structural or bending edge.
    pub kind: SoftBodyEdgeKind,
    /// If `true`, the edge only resists stretching (a rope segment), never compression.
    pub tension_only: bool,
    /// If `true`, the edge only resists squeezing (a strut resting on a stop), never stretching.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub compression_only: bool,
    /// A spring in physical units holding this edge, overriding [`Self::softness`] and the
    /// material's softness when set.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub spring: Option<SoftEdgeSpring>,
    /// The most force the edge pushes or pulls with, in newtons (default: `Real::INFINITY`, no
    /// bound): past it the edge gives way at that force and springs back once the load drops.
    #[cfg_attr(feature = "serde-serialize", serde(default = "infinite"))]
    pub max_force: Real,
    /// The drive of the edge's rest length, when it is an actuator.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub actuator: Option<SoftEdgeActuator>,
    /// Which material of the body the edge takes: `0` the body's own, `k` its `k`-th region
    /// material (see [`crate::dynamics::SoftBody::region_materials`]).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub material: u16,
    /// A number the body's author gave this edge (`u32::MAX` when none). An edge with a tag is an
    /// element of its own: tearing it removes it rather than opening a crack through its
    /// particles, and the tear event lists it by its tag.
    #[cfg_attr(feature = "serde-serialize", serde(default = "no_tag"))]
    pub tag: u32,
    /// The break group of the edge (`0`: none): when one edge of a group tears, every edge of the
    /// group tears with it in the same step.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub break_group: u32,
    /// Softness of this edge, overriding the material's (`edge_softness` / `bend_softness`)
    /// when set: anisotropic cloth (warp, weft and shear stiffness), stiffer seams...
    pub softness: Option<SpringCoefficients<Real>>,
    /// Multiplier of the material's tear thresholds for this edge (default `1.0`): a seam that
    /// holds, a perforation that gives. Usually set through [`crate::dynamics::SoftBody::set_edge_tear_resistance`]
    /// or [`crate::dynamics::SoftBody::set_cluster_tear_resistance`].
    #[cfg_attr(feature = "serde-serialize", serde(default = "one"))]
    pub tear_resistance: Real,
    /// Accumulated impulse of the last step (warm-start state).
    pub(crate) impulse: Real,
    /// Accumulated plastic strain: `rest_length = initial rest length * (1 + plastic_strain)`
    /// (see [`Self::plastic_strain`]).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) plastic_strain: Real,
    /// Parallel solve color.
    pub(crate) color: u8,
    /// Set when the edge was loaded past the material's tear threshold during the last step
    /// (or by [`crate::dynamics::SoftBody::tear_edge`]): removed by the tearing pass at the end of the step.
    pub(crate) torn: bool,
    /// The load of the edge as a fraction of its tear threshold, smoothed over the material's
    /// `tear_smoothing` (see [`Self::stress`]).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) stress: Real,
    /// The force the edge carried over the last step, along its direction: positive pulling its
    /// particles together (resisting a stretch), negative pushing them apart.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) force: Real,
    /// The work the plastic flow of the last step's FEM substeps took out of the edge (see
    /// [`Self::flow_work`]).
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) flow_work: Real,
}

impl SoftBodyEdge {
    /// The work plastic flow took out of this edge within the last step, in joules, on the FEM
    /// solver ([`crate::dynamics::SoftBodySolver::Fem`]), which flows it at every substep: at each
    /// flow, the elastic energy it held at that substep's configuration against the rest length it
    /// flowed from, less against the one it flowed to; for a yield force, that force times the way
    /// its rest length moved. A body the constraint solver stepped reads `0.0`.
    pub fn flow_work(&self) -> Real {
        self.flow_work
    }

    /// An edge between `vertices` at rest length `rest_length`, of `kind`, with every per-edge
    /// setting at its default (the material's softness, no bound, no tag).
    pub(crate) fn new(vertices: [u32; 2], rest_length: Real, kind: SoftBodyEdgeKind) -> Self {
        Self {
            vertices,
            rest_length,
            kind,
            tension_only: false,
            compression_only: false,
            spring: None,
            max_force: Real::INFINITY,
            actuator: None,
            material: 0,
            tag: u32::MAX,
            break_group: 0,
            softness: None,
            tear_resistance: 1.0,
            impulse: 0.0,
            plastic_strain: 0.0,
            color: 0,
            torn: false,
            stress: 0.0,
            force: 0.0,
            flow_work: 0.0,
        }
    }

    /// The impulse this edge applied during the last substep, along the edge direction
    /// (positive when the edge was resisting stretching).
    pub fn impulse(&self) -> Real {
        self.impulse
    }

    /// The force this edge carried over the last step, in newtons along its direction: positive
    /// when it pulled its particles together (it was stretched), negative when it pushed them
    /// apart (it was squeezed).
    pub fn force(&self) -> Real {
        self.force
    }

    /// The bounds the edge's kind and [`Self::max_force`] put on its force, as `(push, pull)`,
    /// without its yield: a tension-only edge pushes with nothing, a compression-only one pulls
    /// with nothing.
    pub(crate) fn load_bounds(&self) -> (Real, Real) {
        let cap = self.max_force.max(0.0);
        let push = if self.tension_only { 0.0 } else { cap };
        let pull = if self.compression_only { 0.0 } else { cap };
        (push, pull)
    }

    /// The forces past which the edge yields, as `(push, pull)`: the material's
    /// [`SoftBodyMaterial::edge_yield_force`] in each direction the edge may still take a
    /// permanent set in, `Real::INFINITY` in the others.
    pub(crate) fn yield_limits(&self, material: &SoftBodyMaterial) -> (Real, Real) {
        let (mut push, mut pull) = (Real::INFINITY, Real::INFINITY);
        if let Some(yield_force) = material.edge_yield_force {
            let yield_force = yield_force.max(0.0);
            let flows_in = |strain: Real| {
                use super::SoftEdgePlasticFlow as Flow;
                let allowed = match material.edge_plastic_flow {
                    Flow::Both => true,
                    Flow::Compression => strain < 0.0,
                    Flow::Tension => strain > 0.0,
                };
                // Up to its bound, short of the rounding the flow's clamp leaves.
                let bound = material.edge_plastic_bound(strain) * (1.0 - PLASTIC_BOUND_TOLERANCE);
                allowed && self.plastic_strain * strain.signum() < bound
            };
            if flows_in(-1.0) {
                push = yield_force;
            }
            if flows_in(1.0) {
                pull = yield_force;
            }
        }
        (push, pull)
    }

    /// Load of this edge as a fraction of its tear threshold, smoothed over
    /// [`SoftBodyMaterial::tear_smoothing`]: `0.0` slack, `1.0` tearing. The larger of the stretch
    /// over `tear_strain` and the force over `tear_force`; `0.0` while the material has neither.
    pub fn stress(&self) -> Real {
        self.stress
    }
}

/// The share of its plastic bound an edge's set may fall short of and still count as at the
/// bound: the flow clamps the rest length to the bound, and the set read back from it rounds.
const PLASTIC_BOUND_TOLERANCE: Real = 1.0e-4;

/// A dihedral bending constraint between two triangles sharing an edge (3D only).
///
/// `vertices[0..2]` is the shared edge, `vertices[2]` and `vertices[3]` the two opposite
/// vertices.
#[cfg(feature = "dim3")]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SoftBodyDihedral {
    /// The four particles.
    pub vertices: [u32; 4],
    /// The angle between the two triangle normals at rest.
    pub rest_angle: Real,
    /// Accumulated impulse of the last step (warm-start state).
    pub(crate) impulse: Real,
    /// Accumulated plastic set of the rest angle: `rest_angle = initial rest angle +
    /// plastic_set` (see [`Self::plastic_set`]).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) plastic_set: Real,
    /// Parallel solve color.
    pub(crate) color: u8,
    /// Which material of the body the dihedral takes (see [`SoftBodyEdge::material`]).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub material: u16,
    /// The work the plastic flow of the last step's FEM substeps took out of the dihedral (see
    /// [`Self::flow_work`]).
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) flow_work: Real,
}

#[cfg(feature = "dim3")]
impl SoftBodyDihedral {
    /// The work plastic flow took out of this dihedral within the last step, in joules, on the FEM
    /// solver ([`crate::dynamics::SoftBodySolver::Fem`]), which flows it at every substep: at each
    /// flow, the elastic energy it held at that substep's configuration against the rest angle it
    /// flowed from, less against the one it flowed to. A body the constraint solver stepped reads
    /// `0.0`.
    pub fn flow_work(&self) -> Real {
        self.flow_work
    }

    /// The permanent set of this dihedral: how far its rest angle has flowed from the angle it
    /// was created with, in radians (see [`SoftBodyMaterial::edge_plastic_yield`]).
    pub fn plastic_set(&self) -> Real {
        self.plastic_set
    }

    /// The rest angle this dihedral was created with (its `rest_angle` before any plastic
    /// flow).
    pub fn initial_rest_angle(&self) -> Real {
        self.rest_angle - self.plastic_set
    }
}

/// Which solver simulates a soft body's elasticity (requires the `fem` cargo feature).
#[cfg(feature = "fem")]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum SoftBodySolver {
    /// Every elastic element becomes constraints, warm-started and swept Gauss-Seidel by color once
    /// per substep with the contacts and joints. Cheap and robust, but converged only as far as the
    /// sweep count: a stiff body keeps a residual compliance, a load crosses a long body slowly.
    #[default]
    Constraints,
    /// Implicit Euler elasticity: forces and tangent stiffness from the strain-energy density,
    /// `(M + h D + h² K) Δv = b` solved over the whole body once per substep; constraints see the
    /// body through its augmented mass. Costs a factorization per step and a solve per constraint.
    Fem,
}

/// The constitutive model of a soft body's cells (triangles in 2D, tetrahedra in 3D).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum SoftBodyCellModel {
    /// One constraint per cell keeping its signed area/volume at its rest value
    /// (softness: [`SoftBodyMaterial::volume_softness`]). Cheap; combined with the edges this
    /// gives a convincing jelly.
    #[default]
    Volume,
    /// Corotational elasticity (linear elasticity in the cell's rotation-free frame) plus an exact
    /// volume constraint, parameterized by [`SoftBodyMaterial::young_modulus`], `poisson_ratio` and
    /// `elastic_damping_ratio`. Stable at any stiffness, recovers from inverted cells.
    Corotational,
    /// Stable Neo-Hookean hyperelasticity (Smith et al. 2018): the parameters (plasticity too) and
    /// small-strain behavior of [`SoftBodyCellModel::Corotational`], but Neo-Hookean at large
    /// strains (stiffer than linear elasticity under compression, softer under tension, inverted
    /// cells pushed back through the collapse); 2x cost.
    NeoHookean,
}

/// Number of warm-start impulses stored per cell (the corotational model's strain rows plus its
/// volumetric row).
pub(crate) const CELL_IMPULSES: usize = DIM * (DIM + 1) / 2 + 1;

/// A simplex cell of a soft body: a triangle in 2D, a tetrahedron in 3D.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SoftBodyCell {
    /// The cell's particles, positively oriented (positive signed area/volume at rest).
    pub vertices: [u32; DIM + 1],
    /// The signed area (2D) or volume (3D) of the cell at rest.
    pub rest_volume: Real,
    /// Inverse of the rest edge matrix `[x1 - x0, x2 - x0, (x3 - x0)]`.
    pub(crate) inv_rest_matrix: Matrix,
    /// Accumulated plastic stretch of the rest shape (material frame, unit determinant): the rest
    /// edge matrix is `plastic_stretch * Dm₀`, `Dm₀` the initial rest edge matrix. Bounded by the
    /// material's `plastic_max`, which keeps flowing cells from degenerating into slivers.
    #[cfg_attr(feature = "serde-serialize", serde(default = "identity_matrix"))]
    pub(crate) plastic_stretch: Matrix,
    /// Accumulated impulses of the last step (warm-start state): the volume constraint's in `[0]`, or
    /// the corotational constraints' (strain rows in the cell frame, then volumetric).
    pub(crate) impulses: [Real; CELL_IMPULSES],
    /// Rotation of the polar decomposition of the cell's deformation gradient at the last step
    /// (warm start of the corotational constraints' rotation extraction).
    pub(crate) rotation: Rotation,
    /// Multiplier of the material's Young modulus for this cell (default `1.0`): per-region
    /// stiffness, usually set through [`crate::dynamics::SoftBody::set_cluster_stiffness_scale`].
    #[cfg_attr(feature = "serde-serialize", serde(default = "one"))]
    pub stiffness_scale: Real,
    /// Multiplier of the material's tear strain for this cell (default `1.0`), see
    /// [`crate::dynamics::SoftBody::set_cell_tear_resistance`].
    #[cfg_attr(feature = "serde-serialize", serde(default = "one"))]
    pub tear_resistance: Real,
    /// Which material of the body the cell takes (see [`SoftBodyEdge::material`]): its Young
    /// modulus, Poisson's ratio, damping, plastic and tear values.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub material: u16,
    /// Parallel solve color.
    pub(crate) color: u8,
    /// Set when the cell was strained past the material's `tear_strain` during the last step
    /// (or by [`crate::dynamics::SoftBody::tear_cell`]): removed by the tearing pass at the end of the step.
    pub(crate) torn: bool,
    /// The tensile strain of the cell as a fraction of its tear threshold, smoothed over the
    /// material's `tear_smoothing` (see [`Self::stress`]).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub(crate) stress: Real,
    /// The work the plastic flow of the last step's FEM substeps took out of the cell (see
    /// [`Self::flow_work`]).
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    pub(crate) flow_work: Real,
}

#[cfg(feature = "serde-serialize")]
fn identity_matrix() -> Matrix {
    Matrix::IDENTITY
}

impl SoftBodyCell {
    /// The work plastic flow took out of this cell within the last step, in joules, on the FEM
    /// solver ([`crate::dynamics::SoftBodySolver::Fem`]), which flows it at every substep: at each
    /// flow, the elastic energy it held at that substep's configuration against the rest shape it
    /// flowed from, less against the one it flowed to. A body the constraint solver stepped reads
    /// `0.0`.
    pub fn flow_work(&self) -> Real {
        self.flow_work
    }

    /// The accumulated plastic stretch of this cell's rest shape (material frame, unit
    /// determinant): its current rest shape is this stretch applied to the shape it was created
    /// with (see [`SoftBodyMaterial::plastic_yield`]); the identity before any flow.
    pub fn plastic_stretch(&self) -> Matrix {
        self.plastic_stretch
    }

    /// The largest tensile strain of this cell as a fraction of its tear threshold, smoothed
    /// over the material's [`SoftBodyMaterial::tear_smoothing`]: `0.0` slack, `1.0` tearing.
    /// Stays `0.0` while the material has no `tear_strain` (cells only tear on their strain).
    pub fn stress(&self) -> Real {
        self.stress
    }
}
