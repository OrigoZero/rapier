use crate::alloc_prelude::*;
use crate::dynamics::RigidBodyVelocity;
use crate::math::{DVector, Pose, Real};

/// A temporary workspace for various updates of the multibody.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct MultibodyWorkspace {
    pub accs: Vec<RigidBodyVelocity<Real>>,
    /// The generalized forces that derive from positions or external inputs (gravity, applied
    /// forces, spring position terms), as the last acceleration update that measured them read
    /// them.
    pub position_forces: DVector,
    /// How the multibody's last step left it, when no row bore on its last substep and the
    /// continuous pass did not take it back: the step that follows reads the energy it stands at
    /// from it (see `VelocitySolver::integrate_multibody_positions`).
    pub step_end: Option<MultibodyStepEnd>,
}

/// How a step left a multibody no row bore on as it ended (`MultibodyWorkspace::step_end`).
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct MultibodyStepEnd {
    /// The kinetic energy it stood at, half a substep on from the velocity it ended with
    /// (`Multibody::kicked_kinetic_energy`).
    pub standing: Real,
    /// The generalized velocity it ended with: a multibody that stands at another velocity when
    /// the next step begins was pushed between the two, and that step reads the energy it stands
    /// at afresh.
    pub velocities: DVector,
    /// The pose each link ended at, in link order: a multibody whose links stand elsewhere when
    /// the next step begins was moved between the two, and that step reads the energy it stands
    /// at afresh.
    pub poses: Vec<Pose>,
}

impl MultibodyWorkspace {
    /// Create an empty workspace.
    pub fn new() -> Self {
        MultibodyWorkspace {
            accs: Vec::new(),
            position_forces: DVector::zeros(0),
            step_end: None,
        }
    }

    /// Resize the workspace so it is enough for `nlinks` links.
    pub fn resize(&mut self, nlinks: usize, ndofs: usize) {
        self.accs.resize(nlinks, RigidBodyVelocity::zero());
        self.position_forces = DVector::zeros(ndofs)
    }
}
