//! The particle settings, the builder struct and its `Default` and `From` impls.

use crate::alloc_prelude::*;
use crate::dynamics::SpringCoefficients;
use crate::geometry::ColliderBuilder;
use crate::math::{DIM, Real, Vector};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use simba::scalar::{ComplexField as _, RealField as _};

use super::super::{SoftBody, SoftBodyCellModel, SoftBodyMaterial};

/// The dynamics settings shared by every particle of a soft body.
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SoftBodyParticleSettings {
    /// Linear damping of the particles (air friction).
    pub linear_damping: Real,
    /// Gravity scale of the particles.
    pub gravity_scale: Real,
    /// Extra solver substeps requested for the particles (and everything they touch).
    pub additional_solver_iterations: usize,
    /// Extra internal PGS iterations per substep for the particles and everything they touch, on
    /// top of `IntegrationParameters::num_internal_pgs_iterations`; stamped on the root proxy each
    /// step (see [`crate::dynamics::RigidBody::set_additional_pgs_iterations`]) (default: `3`).
    pub additional_pgs_iterations: usize,
    /// Whether the soft body may fall asleep.
    pub can_sleep: bool,
    /// Dominance group of the soft body (see [`crate::dynamics::RigidBody::dominance_group`]).
    pub dominance_group: i8,
}

impl Default for SoftBodyParticleSettings {
    fn default() -> Self {
        Self {
            linear_damping: 0.0,
            gravity_scale: 1.0,
            additional_solver_iterations: 0,
            additional_pgs_iterations: 3,
            can_sleep: true,
            dominance_group: 0,
        }
    }
}

/// A builder for [`crate::dynamics::SoftBody`]s: particles are given in world space and elements
/// reference particles by index. The generators ([`Self::rope`], plus `cloth`/`cuboid`/`sphere`
/// in 3D and `polygon`/`disk`/`grid` in 2D) make complete bodies; the setters build any body.
#[derive(Clone, Debug)]
pub struct SoftBodyBuilder {
    /// World-space particle positions.
    pub positions: Vec<Vector>,
    /// Per-particle masses (empty = uniform `particle_mass`).
    pub masses: Vec<Real>,
    /// Uniform particle mass, used when `masses` is empty.
    pub particle_mass: Real,
    /// Indices of the pinned particles.
    pub pinned: Vec<u32>,
    /// Structural edges (particle index pairs).
    pub edges: Vec<[u32; 2]>,
    /// Bending edges (particle index pairs).
    pub bend_edges: Vec<[u32; 2]>,
    /// Edges that only resist stretching, never compression (indices into the concatenation of
    /// `edges` then `bend_edges`).
    pub tension_only_edges: Vec<u32>,
    /// Per-edge softness overrides (indices into the concatenation of `edges` then
    /// `bend_edges`); the material's softness applies to the others.
    pub edge_softness: Vec<(u32, SpringCoefficients<Real>)>,
    /// Per-edge tear-threshold multipliers (indices into the concatenation of `edges` then
    /// `bend_edges`, see [`crate::dynamics::SoftBodyEdge::tear_resistance`]); `1.0` for the others.
    pub edge_tear_resistance: Vec<(u32, Real)>,
    /// Per-edge laws, parallel to the concatenation of `edges` then `bend_edges` (an edge past
    /// the end takes the default law); the tension-only, softness and tear-resistance lists
    /// apply on top of them.
    pub edge_laws: Vec<SoftEdgeLaw>,
    /// The numbers the author gave the particles, parallel to `positions` (a particle past the
    /// end has none; see [`crate::dynamics::SoftBodyParticle::tag`]).
    pub particle_tags: Vec<u32>,
    /// The materials of the body's regions (see [`crate::dynamics::SoftBody::region_materials`]).
    pub region_materials: Vec<SoftBodyMaterial>,
    /// The material index of each cell, parallel to `cells` (a cell past the end takes `0`, the
    /// body's own material).
    pub cell_materials: Vec<u16>,
    /// The plane the particles are held to, as a normal and an offset (see
    /// [`crate::dynamics::SoftBody::set_plane`]).
    pub plane: Option<(Vector, Real)>,
    /// Dihedral bending constraints: shared edge then the two opposite vertices.
    #[cfg(feature = "dim3")]
    pub dihedrals: Vec<[u32; 4]>,
    /// Simplex cells (triangles in 2D, tetrahedra in 3D).
    pub cells: Vec<[u32; DIM + 1]>,
    /// Boundary elements (segments in 2D, triangles in 3D), oriented outward. Derived from the
    /// cells when empty.
    pub surface: Vec<[u32; DIM]>,
    /// The material.
    pub material: SoftBodyMaterial,
    /// The constitutive model of the cells.
    pub cell_model: SoftBodyCellModel,
    /// Which solver simulates the body's elasticity.
    #[cfg(feature = "fem")]
    pub solver: super::SoftBodySolver,
    /// Whether area/volume preservation is enabled (one constraint per piece of material enclosed
    /// by a closed surface).
    pub volume_preservation: bool,
    /// Target volume multiplier (`> 1` inflates the body).
    pub volume_factor: Real,
    /// Whether shape matching is enabled.
    pub shape_matching: bool,
    /// Whether the body's surface collides with itself.
    pub self_contacts: bool,
    /// Whether the default collision surface's shape is built with parry's `ORIENTED` flag
    /// (see [`Self::oriented`]); `None`: when the surface is closed.
    pub oriented: Option<bool>,
    /// Thickness of the particles: the contact skin of the body's surface collider.
    pub particle_radius: Real,
    /// Template of the colliders of the body's collision meshes (their shape is the mesh's own);
    /// `None` disables collisions.
    pub collider_template: Option<ColliderBuilder>,
    /// Mesh held by the cells, if any: `(vertices, elements)` in world space, bound to the
    /// cells at build time.
    pub skin: Option<(Vec<Vector>, Vec<[u32; DIM]>)>,
    /// The segments the body collides through when it has no surface: a rope or a wire (3D
    /// only; in 2D the surface is already made of segments).
    #[cfg(feature = "dim3")]
    pub wire: Vec<[u32; 2]>,
    /// Whether the body meets the world through its skin rather than through its cells' boundary
    /// (off by default, and no effect without a skin).
    pub skin_collision: bool,
    /// Dynamics settings of the particles.
    pub particle_settings: SoftBodyParticleSettings,
    /// User data of the soft body.
    pub user_data: u128,
}

/// The settings an authored edge is built with: what kind of load it carries, the spring
/// holding it, its bound, its drive, its material, the number that names it and its break group
/// (see the fields of [`crate::dynamics::SoftBodyEdge`] of the same names).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SoftEdgeLaw {
    /// Only resists stretching (a cable).
    pub tension_only: bool,
    /// Only resists squeezing (a strut against a stop).
    pub compression_only: bool,
    /// The spring in physical units; `None`: the material's softness.
    pub spring: Option<super::super::SoftEdgeSpring>,
    /// The most force the edge carries, in newtons.
    pub max_force: Real,
    /// The drive of the edge's rest length.
    pub actuator: Option<super::super::SoftEdgeActuator>,
    /// The material index (see [`crate::dynamics::SoftBody::region_materials`]).
    pub material: u16,
    /// The number that names the edge (`u32::MAX`: none).
    pub tag: u32,
    /// The break group (`0`: none).
    pub break_group: u32,
}

impl Default for SoftEdgeLaw {
    fn default() -> Self {
        Self {
            tension_only: false,
            compression_only: false,
            spring: None,
            max_force: Real::INFINITY,
            actuator: None,
            material: 0,
            tag: u32::MAX,
            break_group: 0,
        }
    }
}

impl SoftEdgeLaw {
    /// Writes this law into the edge `edge`.
    pub(crate) fn apply(&self, edge: &mut crate::dynamics::SoftBodyEdge) {
        edge.tension_only = self.tension_only;
        edge.compression_only = self.compression_only;
        edge.spring = self.spring;
        edge.max_force = self.max_force.max(0.0);
        edge.actuator = self.actuator;
        edge.material = self.material;
        edge.tag = self.tag;
        edge.break_group = self.break_group;
    }
}

impl Default for SoftBodyBuilder {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl From<SoftBodyBuilder> for SoftBody {
    fn from(builder: SoftBodyBuilder) -> Self {
        builder.build()
    }
}
