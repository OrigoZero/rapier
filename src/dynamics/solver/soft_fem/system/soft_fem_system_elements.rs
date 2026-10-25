//! The element state of a FEM soft body and the `SoftFemSystem` that owns it.

use super::super::soft_fem_sparse::{BlockMatrix, ConjugateGradient};
use crate::alloc_prelude::*;
use crate::dynamics::solver::soft_constraint::soft_element_constraint::{
    MAX_CONSTRAINT_PARTICLES, StrainMatrix, StrainVector,
};
use crate::math::{Matrix, Real, Rotation, Vector};

/// Number of `DIM × DIM` blocks a cell writes into the system matrix.
pub(super) const CELL_BLOCKS: usize = MAX_CONSTRAINT_PARTICLES * MAX_CONSTRAINT_PARTICLES;

/// One finite element (a triangle in 2D, a tetrahedron in 3D) of a FEM soft body:
/// [`SoftFemSystem::prepare`] sets its per-step constants (material, shape functions, Rayleigh
/// coefficient), [`SoftFemSystem::assemble`] its rotation, strain, force and tangent per substep.
#[derive(Copy, Clone, Debug)]
pub(super) struct FemCell {
    pub(super) vertices: [u32; MAX_CONSTRAINT_PARTICLES],
    /// `∂F_cj/∂(x_a)_c`: the constraints of `Dm⁻¹`, particle 0 minus their sum.
    pub(super) coeffs: [Vector; MAX_CONSTRAINT_PARTICLES],
    pub(super) inv_rest_matrix: Matrix,
    /// `μ V₀` and `λ V₀` (Lamé parameters scaled by the rest volume).
    pub(super) mu: Real,
    pub(super) lambda: Real,
    /// Rayleigh damping coefficient `β` (the damping matrix of the cell is `β K`).
    pub(super) beta: Real,
    /// Polar rotation of the deformation gradient, frozen over the substep and warm-started
    /// from the previous one.
    pub(super) rotation: Rotation,
    /// Corotated strain `sym(RᵀF) - I` in the strain-constraint coordinates.
    pub(super) strain: StrainVector,
    /// Whether the cell was inverted (`det F < 0`) at the last assembly.
    pub(super) inverted: bool,
    /// Whether the cell uses the stable Neo-Hookean energy instead of the linear-elastic
    /// (corotational) one.
    pub(super) neo_hookean: bool,
    /// The tangent `V₀ ∂²Ψ/∂ε²` last computed, and the strain it was computed at
    /// (`Real::MAX`: never). Constant for the corotational model; the Neo-Hookean one
    /// re-linearizes when the strain moved by more than `STIFFNESS_UPDATE_STRAIN`.
    pub(super) tangent: StrainMatrix,
    pub(super) tangent_strain: StrainVector,
}

/// One volume element of a FEM soft body with the [`SoftBodyCellModel::Volume`] cell model: energy
/// `½ k (V − V₀)²` over the signed area/volume, Gauss-Newton tangent `k ∇V ∇Vᵀ` (indefinite term
/// dropped); `k = ω² / w₀` from `volume_softness`, `w₀` the constraint's rest effective mass.
#[derive(Copy, Clone, Debug)]
pub(super) struct FemVolumeCell {
    /// Index of the cell in the body (its vertices and cached block slots).
    pub(super) cell: u32,
    pub(super) rest_volume: Real,
    pub(super) stiffness: Real,
    pub(super) beta: Real,
}

/// One distance element (a structural or bending edge) of a FEM soft body.
///
/// Energy `½ k (|d| − L)²`, with the Gauss-Newton tangent `k ∇C ∇Cᵀ` and, while the element
/// pulls, the tensile part of `C ∂²C` (`f / |d|` across it); a pushed element's `C ∂²C` is
/// indefinite and dropped, since an implicit step needs a positive semi-definite stiffness.
#[derive(Copy, Clone, Debug)]
pub(super) struct FemSpring {
    pub(super) vertices: [u32; 2],
    /// Slots of the element's four blocks, row-major.
    pub(super) blocks: [u32; 4],
    pub(super) rest_length: Real,
    pub(super) stiffness: Real,
    pub(super) beta: Real,
    /// The largest compressive and tensile force the element carries, `(push, pull)`: a rope
    /// segment pushes with nothing, a strut pulls with nothing, a bounded element carries no
    /// more than its bound (see `SoftBodyEdge::load_bounds`). Past a bound the element's force
    /// is the bound (see [`Self::force`]).
    pub(super) bounds: (Real, Real),
    /// The forces past which the element yields, `(push, pull)` (`Real::INFINITY`: it does not):
    /// past one, its rest length flows within the substep so it carries exactly that force
    /// (see [`Self::return_map`]), down to `rest_range.0` and up to `rest_range.1`.
    pub(super) yield_limits: (Real, Real),
    /// The shortest and the longest rest length the element's plastic bounds allow.
    pub(super) rest_range: (Real, Real),
}

impl FemSpring {
    /// The element's force at length error `c` (positive pulling its particles together),
    /// clamped to its bounds, and whether it carries stiffness: an element slack in the direction
    /// it does not load (a cable squeezed, a strut stretched) carries none, while one held at a
    /// nonzero bound keeps its elastic stiffness in the implicit solve, which holds the substep's
    /// motion against it the way the bound's own constraint would.
    pub(super) fn force(&self, c: Real) -> (Real, bool) {
        let force = self.stiffness * c;
        let (push, pull) = self.bounds;
        if force > pull {
            (pull, pull > 0.0)
        } else if force < -push {
            (-push, push > 0.0)
        } else {
            (force, true)
        }
    }

    /// Flows the element's rest length so that, at `length`, it carries no more than its yield
    /// force (an elastoplastic return mapping), within its rest range.
    pub(super) fn return_map(&mut self, length: Real) {
        if self.stiffness <= 0.0 {
            return;
        }
        let (push, pull) = self.yield_limits;
        let (shortest, longest) = self.rest_range;
        let trial = self.stiffness * (length - self.rest_length);
        if trial > pull && self.rest_length < longest {
            let rest = (length - pull / self.stiffness).min(longest);
            self.rest_length = self.rest_length.max(rest);
        } else if trial < -push && self.rest_length > shortest {
            let rest = (length + push / self.stiffness).max(shortest);
            self.rest_length = self.rest_length.min(rest);
        }
    }
}

/// One dihedral bending element of a FEM soft body (3D): energy `½ k (θ − θ₀)²`, Gauss-Newton
/// tangent, gradients shared with the constraint path.
#[cfg(feature = "dim3")]
#[derive(Copy, Clone, Debug)]
pub(super) struct FemDihedral {
    pub(super) vertices: [u32; 4],
    /// Slots of the element's sixteen blocks, row-major.
    pub(super) blocks: [u32; 16],
    pub(super) rest_angle: Real,
    pub(super) stiffness: Real,
    pub(super) beta: Real,
}

/// The FEM system of one soft body: the block-sparse `A`, its conjugate-gradient workspace, and
/// the per-particle vectors the two solves work on. The sparsity pattern is built once from the
/// element graph and kept across steps; the values are refilled every substep.
#[derive(Clone, Debug, Default)]
pub(crate) struct SoftFemSystem {
    /// Topology version the pattern was built for (bumped by tearing).
    pub(super) topology_version: u32,
    pub(super) matrix: BlockMatrix,
    pub(super) cg: ConjugateGradient,
    pub(super) cells: Vec<FemCell>,
    /// Slots of every cell's `(DIM+1)²` blocks, row-major over its particles.
    pub(super) cell_blocks: Vec<[u32; CELL_BLOCKS]>,
    pub(super) volume_cells: Vec<FemVolumeCell>,
    pub(super) springs: Vec<FemSpring>,
    #[cfg(feature = "dim3")]
    pub(super) dihedrals: Vec<FemDihedral>,
    /// Particle masses (`0` for a pinned particle: its constraint is a Dirichlet constraint).
    pub(super) mass: Vec<Real>,
    /// Solver-body slot of every particle.
    pub(super) slots: Vec<u32>,
    /// Whether the particle is pinned (`inv_mass == 0`).
    pub(super) pinned: Vec<bool>,
    pub(super) position: Vec<Vector>,
    pub(super) velocity: Vec<Vector>,
    /// Elastic force of the current configuration.
    pub(super) force: Vec<Vector>,
    /// `Σ_{q pinned} A_pq v_q` per free row: the pinned particles' right-hand-side term, kept when
    /// their rows become Dirichlet rows.
    pub(super) pinned_coupling: Vec<Vector>,
    /// The operator (`matrix`, `force`, `pinned_coupling`) was just assembled at the current
    /// positions by the step-start factorization: the first substep's predict reuses it.
    pub(super) operator_fresh: bool,
    pub(super) rhs: Vec<Vector>,
    /// Solution of the last predictor solve (the warm start of the next one).
    pub(super) delta: Vec<Vector>,
    /// `A` at the step's start (the substep length of the body's group): the operator every
    /// constraint of the step sees the body through, kept for the PCG fallback.
    pub(super) step_matrix: BlockMatrix,
    /// The PCG of `step_matrix` (its preconditioner), used when the direct factorization is
    /// not available.
    pub(super) step_cg: ConjugateGradient,
    /// The direct (skyline Cholesky) factorization of `step_matrix`, for bodies up to
    /// `SoftFemParameters::max_dense_dofs`; ordering and envelope persist with the pattern (`None`
    /// after a rebuild), and `direct_valid` says whether this step's factorization succeeded.
    pub(super) direct: Option<super::super::soft_fem_skyline::SkylineCholesky>,
    pub(super) direct_valid: bool,
    pub(super) response_rhs: Vec<Vector>,
    pub(super) response: Vec<Vector>,
    /// The column responses of the step (see [`Self::load_particle`]): per loaded particle, `DIM`
    /// axis-major responses `A_step⁻¹ (e_p ⊗ axis)` of `num_particles` vectors each, plus every
    /// particle's column index (`u32::MAX`: not loaded).
    pub(super) columns: Vec<Vector>,
    pub(super) column_of: Vec<u32>,
    pub(super) num_columns: usize,
}
