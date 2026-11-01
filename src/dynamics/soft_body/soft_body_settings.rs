//! Simulation settings shared by every soft body of a world, grouped on
//! [`IntegrationParameters::soft_bodies`](crate::dynamics::IntegrationParameters::soft_bodies).

use super::SoftRecoverySettings;
use crate::math::Real;

/// Simulation settings shared by every soft body of a world.
///
/// Lives on [`IntegrationParameters::soft_bodies`], so it can be changed between steps.
///
/// [`IntegrationParameters::soft_bodies`]: crate::dynamics::IntegrationParameters::soft_bodies
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde-serialize", serde(default))]
pub struct SoftBodiesSettings {
    /// Runtime toggles and tuning for the soft-body tangle detection and recovery stack
    /// (see [`SoftRecoverySettings`]). Every mechanism can be switched off individually.
    pub recovery: SoftRecoverySettings,
    /// Strain beyond which a soft-body constraint (an elastic cell's strain component, a distance
    /// constraint's relative error) is re-solved after the contacts inside every substep, so a
    /// light body buried under heavier ones is not torn (default: `0.75`; `Real::MAX` to disable).
    pub resweep_strain: Real,
    /// Most extra substeps a second a soft body adds to its island component while it is hit
    /// fast or loaded hard (default: `120`; `0` disables the adaptive substeps). A body hit fast
    /// asks for enough substeps to bound the per-substep travel of its particles and of the rigid
    /// bodies approaching it by one particle radius, and a body loaded hard for the substep rate
    /// its load calls for; whatever the step length, the extra substeps a step runs are at most
    /// this many a second ([`Self::max_extra_substeps`]): 4 a step at 30 steps a second, 2 at 60,
    /// 1 at 120 and none from 240, so a hit costs about the same soft body work per simulated
    /// second at every tick rate.
    pub max_extra_substep_rate: Real,
    /// Factor applied to the [`IntegrationParameters`](crate::dynamics::IntegrationParameters)
    /// contact softness natural frequencies for the soft-body contacts (default: `4.0`); they run
    /// stiffer because a contact's effective mass is a few particles' worth, not the whole body's.
    pub contact_stiffening: Real,
    /// Tuning of the FEM soft-body solver (see
    /// [`SoftBodySolver::Fem`](crate::dynamics::SoftBodySolver::Fem)); only read by the soft
    /// bodies that select it.
    #[cfg(feature = "fem")]
    pub fem: SoftFemParameters,
}

impl Default for SoftBodiesSettings {
    fn default() -> Self {
        Self {
            recovery: SoftRecoverySettings::default(),
            resweep_strain: 0.75,
            max_extra_substep_rate: 120.0,
            contact_stiffening: 4.0,
            #[cfg(feature = "fem")]
            fem: SoftFemParameters::default(),
        }
    }
}

impl SoftBodiesSettings {
    /// The most extra substeps a step of `dt` seconds may add to a soft body's island:
    /// [`Self::max_extra_substep_rate`] over that step, rounded down.
    pub fn max_extra_substeps(&self, dt: Real) -> usize {
        // A rate the step divides evenly (120 a second over 1/30 s) allows exactly that many.
        (self.max_extra_substep_rate * dt + 1.0e-4).floor().max(0.0) as usize
    }
}

/// Tuning of the FEM soft-body solver (see
/// [`SoftBodySolver::Fem`](crate::dynamics::SoftBodySolver::Fem)), which solves a linear system
/// per substep by conjugate gradient and factorizes it once per step (see `max_dense_dofs`).
#[cfg(feature = "fem")]
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct SoftFemParameters {
    /// Relative residual at which the conjugate gradient stops (default: `1.0e-5`).
    pub linear_tolerance: Real,
    /// Hard cap on the conjugate-gradient iterations, whatever the residual (default: `20`).
    ///
    /// A very stiff, finely meshed body can need several hundred iterations to converge; the
    /// truncated step it gets instead is under-relaxed (safe), only slower to settle.
    pub max_linear_iterations: usize,
    /// Largest number of degrees of freedom (`DIM` per particle) for which the step-start
    /// matrix `A` of a body is factorized directly (default: `600`). Each constraint on a FEM body
    /// needs one solve against `A` per step; larger bodies fall back to a conjugate gradient.
    pub max_dense_dofs: usize,
}

#[cfg(feature = "fem")]
impl Default for SoftFemParameters {
    fn default() -> Self {
        Self {
            linear_tolerance: 1.0e-5,
            max_linear_iterations: 20,
            max_dense_dofs: 600,
        }
    }
}
