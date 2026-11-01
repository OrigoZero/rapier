//! Plastic flow of the elastic cells from the strain of a solve, shared by the constraint solver's
//! substeps and the FEM assembly, the energy a flow takes out of a cell, and the total (elastic
//! plus plastic) stretch a cell tears on.

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use simba::scalar::{ComplexField as _, RealField as _};
use simba::simd::SimdComplexField as _;

use crate::dynamics::soft_body::creep_share;
use crate::dynamics::soft_body::soft_body_shape_matching::extract_rotation_within;
use crate::dynamics::{SoftBody, SoftBodyCellModel};
use crate::math::{DIM, Matrix, Real, Rotation, Vector};
use crate::utils::RotationOps;

use super::super::soft_element_constraint::{
    CELL_ROTATION_TOLERANCE, ElementCreep, MAX_CONSTRAINT_PARTICLES, SoftElasticConstraint,
    SoftScalarConstraint, SoftScalarConstraintWriteback, StrainVector, max_symmetric_eigenvalue,
    strain_matrix, strain_rows_of,
};
use super::super::{ElasticCellTerms, elastic_cell_rows, elastic_cell_weight};

/// A cell's rest shape after plastic flow (see [`flowed_rest_shape`]).
#[derive(Copy, Clone, Debug)]
pub(crate) struct FlowedRestShape {
    /// The new plastic stretch `P'` (unit determinant).
    pub plastic_stretch: Matrix,
    /// The inverse of the new rest edge matrix `P' Dm₀`.
    pub inv_rest_matrix: Matrix,
    /// The share of the cell's strain the flow took into its rest shape.
    pub share: Real,
}

/// Plastic flow of an elastic cell of plastic stretch `plastic_stretch` held at `strain` for
/// `dt` (see `SoftBodyMaterial::plastic_yield`): past the yield the rest shape flows toward the
/// deviatoric stretch at the creep rate (`Dm' = S_p Dm`, unit determinant, `Dm = P Dm₀` bounded by
/// `plastic_max`); an inverted cell does not flow. `None` when it does not flow.
pub(crate) fn flowed_rest_shape(
    plastic_stretch: &Matrix,
    strain: &StrainVector,
    inverted: bool,
    rest0: &[Vector; DIM + 1],
    material: &crate::dynamics::SoftBodyMaterial,
    dt: Real,
) -> Option<FlowedRestShape> {
    if inverted || material.plastic_yield <= 0.0 {
        return None;
    }
    let eps = strain_matrix(strain);
    // The diagonal strain rows come first.
    let trace = strain.fixed_rows::<DIM>(0).sum();
    let dev = eps - Matrix::IDENTITY * (trace / DIM as Real);
    let norm = frobenius_norm(&dev);
    if norm <= material.plastic_yield {
        return None;
    }
    let share = (norm - material.plastic_yield) / norm * creep_share(material.plastic_creep, dt);
    if share <= 0.0 {
        return None;
    }
    // The plastic stretch: a fraction of the full stretch `S = I + ε`, normalized to a unit
    // determinant so the flow keeps the rest volume.
    let s_p = Matrix::IDENTITY + eps * share;
    let det = s_p.determinant();
    if det <= 1.0e-6 {
        return None;
    }
    let s_p = s_p * (1.0 / det.simd_powf(1.0 / DIM as Real));
    let total = s_p * *plastic_stretch;
    let deviation = total - Matrix::IDENTITY;
    let deviation_norm = frobenius_norm(&deviation);
    let total = if deviation_norm > material.plastic_max {
        Matrix::IDENTITY + deviation * (material.plastic_max / deviation_norm)
    } else {
        total
    };
    let total_det = total.determinant();
    if total_det <= 1.0e-6 {
        return None;
    }
    let total = total * (1.0 / total_det.simd_powf(1.0 / DIM as Real));
    // The new rest edge matrix `P' Dm₀` and its signed volume (`det(Dm) / DIM!`).
    let dm = total * SoftBody::cell_edge_matrix(*rest0);
    let factorial = if DIM == 2 { 2.0 } else { 6.0 };
    let volume = dm.determinant() / factorial;
    if !volume.is_finite() || volume.abs() <= Real::EPSILON {
        return None;
    }
    Some(FlowedRestShape {
        plastic_stretch: total,
        inv_rest_matrix: dm.inverse(),
        share,
    })
}

/// Whether a cell's plastic stretch moved from `before` to `after` by a significant amount (such
/// a body is kept awake).
pub(crate) fn stretch_moved(before: &Matrix, after: &Matrix) -> bool {
    frobenius_norm(&(*after - *before)) > 1.0e-4
}

/// The elastic energy `V₀ Ψ` a cell of the constraint solver holds at deformation gradient `f`,
/// its rotation extracted from `rotation` (returned): `μ V₀ ‖ε‖² + ½ λ V₀ (det F − 1)²` with the
/// corotated strain `ε = sym(RᵀF) − I` (the corotational model's strain rows and volumetric row),
/// or the stable Neo-Hookean energy `½ μ (‖S‖² − d) − μ (J − 1) + ½ (λ + μ) (J − 1)²` of
/// `S = sym(RᵀF)`, `J = det S` (`mu`, `lambda`: the Lamé parameters times the rest volume). Never
/// below zero.
pub(crate) fn corotated_cell_energy(
    f: &Matrix,
    rotation: Rotation,
    mu: Real,
    lambda: Real,
    neo_hookean: bool,
) -> (Real, Rotation) {
    let rotation = extract_rotation_within(*f, rotation, CELL_ROTATION_TOLERANCE);
    let strain = strain_rows_of(&(rotation.to_mat().transpose() * *f - Matrix::IDENTITY));
    let eps = strain_matrix(&strain);
    let energy = if neo_hookean {
        let s = Matrix::IDENTITY + eps;
        let norm2 = frobenius_norm(&s).powi(2);
        let j = s.determinant() - 1.0;
        0.5 * mu * (norm2 - DIM as Real) - mu * j + 0.5 * (lambda + mu) * j * j
    } else {
        let j = f.determinant() - 1.0;
        mu * frobenius_norm(&eps).powi(2) + 0.5 * lambda * j * j
    };
    (energy.max(0.0), rotation)
}

/// One plastic flow of an elastic cell of the constraint solver over the substep `dt` that ended
/// with its particles at `x` (see [`SoftCellFlow`](super::super::soft_element_constraint::SoftCellFlow)).
#[derive(Copy, Clone, Debug)]
pub(crate) struct CellSubstepFlow {
    /// The share of the cell's strain the flow took into its rest shape: its strain rows' warm
    /// start keeps `1 - share` of their impulses.
    pub share: Real,
}

/// Flows the plastic elastic cell `constraint` of `sb` over the substep `dt` it was just solved
/// through, at the stress its strain rows carried over it (`strain_impulse`, the impulses the
/// substep accumulated, read through `strain_per_impulse`: the sweep's unconverged residual is
/// geometric but transmits no load), the particles where the substep left them (`x`). Past the
/// yield the rest shape flows at the creep rate over the substep (see [`flowed_rest_shape`]); an
/// inverted cell does not flow. The constraint's plastic state takes the new rest shape and the
/// flow's energy, and with `refit` its strain coefficients and rows follow the new rest shape for
/// the substeps still to come. `max_corrective_velocity` bounds the rows' snap-back. `None` when
/// the cell does not flow.
pub(crate) fn flow_elastic_cell(
    constraint: &mut SoftElasticConstraint,
    sb: &SoftBody,
    x: &[Vector; MAX_CONSTRAINT_PARTICLES],
    strain_impulse: &StrainVector,
    dt: Real,
    max_corrective_velocity: Real,
    refit: bool,
) -> Option<CellSubstepFlow> {
    let state = constraint.flow?;
    let cell = sb.cells.get(constraint.element as usize)?;
    let material = sb.element_material(cell.material);
    let rest0: [Vector; DIM + 1] =
        core::array::from_fn(|k| sb.particles[cell.vertices[k] as usize].initial_rest_position);
    let stress_strain = strain_impulse.component_mul(&constraint.strain_per_impulse);
    let ds = SoftBody::cell_edge_matrix(*x);
    let f = ds * constraint.inv_rest_matrix;
    let flowed = flowed_rest_shape(
        &state.plastic_stretch,
        &stress_strain,
        f.determinant() < 0.0,
        &rest0,
        material,
        dt,
    )?;
    let (mu, lambda) = material.lame_parameters();
    let volume = cell.rest_volume.abs();
    let (mu, lambda) = (
        mu * cell.stiffness_scale * volume,
        lambda * cell.stiffness_scale * volume,
    );
    let neo_hookean = sb.cell_model == SoftBodyCellModel::NeoHookean;
    let (held, rotation) = corotated_cell_energy(&f, constraint.rotation, mu, lambda, neo_hookean);
    let (left, _) = corotated_cell_energy(
        &(ds * flowed.inv_rest_matrix),
        rotation,
        mu,
        lambda,
        neo_hookean,
    );
    let work = (held - left).max(0.0);
    constraint.flow = Some(super::super::soft_element_constraint::SoftCellFlow {
        plastic_stretch: flowed.plastic_stretch,
        work: state.work + work,
        ..state
    });
    constraint.inv_rest_matrix = flowed.inv_rest_matrix;
    if refit {
        constraint.coeffs = SoftElasticConstraint::coefficients(&constraint.inv_rest_matrix);
        let (mu, lambda) = material.lame_parameters();
        let terms = ElasticCellTerms {
            mu: mu * cell.stiffness_scale,
            lambda: lambda * cell.stiffness_scale,
            volume,
            damping_ratio: material.elastic_damping_ratio,
            dt,
            neo_hookean,
            max_corrective_velocity,
        };
        let w_vol = elastic_cell_weight(&constraint.coeffs, &constraint.im);
        let (rows, _) = elastic_cell_rows(&terms, &constraint.coeffs, &constraint.im, w_vol, None);
        constraint.set_rows(&rows);
    }
    Some(CellSubstepFlow {
        share: flowed.share,
    })
}

/// What scalar element `constraint` of `sb` creeps by, when it creeps (see
/// `SoftScalarConstraint::creep`): its material and the rest length (an edge) or rest angle (a
/// dihedral) it was created with.
pub(crate) fn element_creep<'a>(
    sb: &'a SoftBody,
    constraint: &SoftScalarConstraint,
) -> Option<ElementCreep<'a>> {
    match constraint.writeback {
        SoftScalarConstraintWriteback::Edge => {
            let e = sb.edges.get(constraint.element as usize)?;
            Some(ElementCreep {
                material: sb.element_material(e.material),
                initial: e.initial_rest_length(),
            })
        }
        #[cfg(feature = "dim3")]
        SoftScalarConstraintWriteback::Dihedral => {
            let d = sb.dihedrals.get(constraint.element as usize)?;
            Some(ElementCreep {
                material: sb.element_material(d.material),
                initial: d.initial_rest_angle(),
            })
        }
        SoftScalarConstraintWriteback::CellVolume => None,
    }
}

/// The largest tensile strain of a cell, plastic flow included: the largest principal value of the
/// symmetric part of the total stretch `(I + ε) P` minus one (the tearing criterion, so a creeping
/// cell tears at the strain an elastic one would); without plastic flow, that of `ε`.
pub(crate) fn total_tensile_strain(strain: &StrainVector, plastic_stretch: &Matrix) -> Real {
    let stretch = (Matrix::IDENTITY + strain_matrix(strain)) * *plastic_stretch;
    let sym = (stretch + stretch.transpose()) * 0.5 - Matrix::IDENTITY;
    max_symmetric_eigenvalue(&sym)
}

/// Frobenius norm of a matrix.
#[inline]
fn frobenius_norm(m: &Matrix) -> Real {
    let mut norm_sq = 0.0;
    for j in 0..DIM {
        norm_sq += m.col(j).length_squared();
    }
    norm_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::super::super::soft_element_constraint::STRAIN_ROWS;
    use super::*;
    use crate::dynamics::{SoftBodyCell, SoftBodyMaterial};

    /// Plastic flow of an elastic cell over `dt` (see [`flowed_rest_shape`]), its strain rows' warm
    /// start following what is left of the strain. Returns whether it flowed significantly.
    fn plastic_flow(
        cell: &mut crate::dynamics::SoftBodyCell,
        strain: &StrainVector,
        inverted: bool,
        rest0: &[Vector; DIM + 1],
        material: &crate::dynamics::SoftBodyMaterial,
        dt: Real,
    ) -> bool {
        let Some(flowed) =
            flowed_rest_shape(&cell.plastic_stretch, strain, inverted, rest0, material, dt)
        else {
            return false;
        };
        cell.inv_rest_matrix = flowed.inv_rest_matrix;
        let factorial = if DIM == 2 { 2.0 } else { 6.0 };
        cell.rest_volume =
            (flowed.plastic_stretch * SoftBody::cell_edge_matrix(*rest0)).determinant() / factorial;
        let moved = stretch_moved(&cell.plastic_stretch, &flowed.plastic_stretch);
        cell.plastic_stretch = flowed.plastic_stretch;
        for impulse in &mut cell.impulses[..STRAIN_ROWS] {
            *impulse *= 1.0 - flowed.share;
        }
        moved
    }

    /// A unit right simplex cell at rest.
    fn unit_cell() -> (SoftBodyCell, [Vector; DIM + 1]) {
        let mut rest0 = [Vector::ZERO; DIM + 1];
        for k in 0..DIM {
            let mut axis = Vector::ZERO;
            axis[k] = 1.0;
            rest0[k + 1] = axis;
        }
        let volume = SoftBody::cell_volume(rest0);
        let cell = SoftBodyCell {
            vertices: core::array::from_fn(|k| k as u32),
            rest_volume: volume,
            inv_rest_matrix: SoftBody::cell_edge_matrix(rest0).inverse(),
            plastic_stretch: Matrix::IDENTITY,
            impulses: [1.0; STRAIN_ROWS + 1],
            rotation: Rotation::IDENTITY,
            stiffness_scale: 1.0,
            tear_resistance: 1.0,
            material: 0,
            color: 0,
            torn: false,
            stress: 0.0,
            flow_work: 0.0,
        };
        (cell, rest0)
    }

    fn clay() -> SoftBodyMaterial {
        SoftBodyMaterial {
            plastic_yield: 0.05,
            plastic_creep: Real::INFINITY,
            plastic_max: 1.0,
            ..Default::default()
        }
    }

    /// A shear past the yield flows the rest shape, keeps its volume and rebuilds the rest
    /// matrix from the initial rest shape: the elastic strain left is the yield.
    #[test]
    fn shear_flows_and_keeps_the_rest_volume() {
        let (mut cell, rest0) = unit_cell();
        let mut strain = StrainVector::zeros();
        // The first off-diagonal row: an `xy` shear of 0.2.
        strain[DIM] = 0.2;
        let flowed = plastic_flow(&mut cell, &strain, false, &rest0, &clay(), 1.0 / 60.0);
        assert!(flowed);
        assert!((cell.rest_volume - SoftBody::cell_volume(rest0)).abs() < 1.0e-5);
        assert!((cell.plastic_stretch.determinant() - 1.0).abs() < 1.0e-5);
        // `Dm = P Dm₀` exactly.
        let dm = cell.plastic_stretch * SoftBody::cell_edge_matrix(rest0);
        let err = frobenius_norm(&(cell.inv_rest_matrix * dm - Matrix::IDENTITY));
        assert!(err < 1.0e-5, "rest matrix drifted: {err}");
        // The strain rows' warm start followed the flow: the yield is what stays elastic, the
        // share `yield / ‖dev ε‖` of the strain (the shear's deviatoric norm is `0.2 √2`).
        let retained = 0.05 / (0.2 * Real::sqrt(2.0));
        assert!(
            cell.impulses[..STRAIN_ROWS]
                .iter()
                .all(|i| (i - retained).abs() < 1.0e-5),
            "{:?}",
            cell.impulses
        );
        assert_eq!(cell.impulses[STRAIN_ROWS], 1.0);
    }

    /// An inverted cell never flows, whatever its strain rows look like.
    #[test]
    fn inverted_cells_do_not_flow() {
        let (mut cell, rest0) = unit_cell();
        let before = cell;
        // A reflection about an oblique plane: no diagonal strain reveals the inversion.
        let mut strain = StrainVector::zeros();
        for k in 0..DIM {
            strain[k] = -2.0 / DIM as Real;
        }
        for k in DIM..STRAIN_ROWS {
            strain[k] = -2.0 / DIM as Real;
        }
        // A partial creep: the full stretch is the reflection itself, which the determinant
        // guard rejects on its own; a fraction of it is a plain squash.
        let material = SoftBodyMaterial {
            plastic_creep: 1.0,
            ..clay()
        };
        assert!(!plastic_flow(
            &mut cell, &strain, true, &rest0, &material, 0.3
        ));
        assert_eq!(cell.plastic_stretch, before.plastic_stretch);
        assert_eq!(cell.inv_rest_matrix, before.inv_rest_matrix);
        // The same rows on a cell that is not inverted flow, bounded by the plastic maximum.
        let mut flowed = 0;
        for _ in 0..200 {
            flowed += plastic_flow(&mut cell, &strain, false, &rest0, &material, 0.3) as usize;
        }
        assert!(flowed > 0);
        let deviation = frobenius_norm(&(cell.plastic_stretch - Matrix::IDENTITY));
        assert!(
            deviation <= 1.0 + 1.0e-3,
            "flow past the maximum: {deviation}"
        );
    }

    /// The tear strain counts the plastic stretch: an unflowed cell reads its largest
    /// principal strain, a flowed one its total stretch.
    #[test]
    fn total_tensile_strain_counts_the_flow() {
        let mut strain = StrainVector::zeros();
        strain[0] = 0.3;
        strain[1] = -0.1;
        assert!((total_tensile_strain(&strain, &Matrix::IDENTITY) - 0.3).abs() < 1.0e-6);
        let mut p = Matrix::IDENTITY;
        p.x_axis[0] = 1.5;
        p.y_axis[1] = 1.0 / 1.5;
        assert!((total_tensile_strain(&StrainVector::zeros(), &p) - 0.5).abs() < 1.0e-6);
        assert!((total_tensile_strain(&strain, &p) - (1.3 * 1.5 - 1.0)).abs() < 1.0e-6);
    }
}
