use crate::alloc_prelude::*;
use crate::dynamics::RigidBodyVelocity;
use crate::math::{DVector, Real};

/// A temporary workspace for various updates of the multibody.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct MultibodyWorkspace {
    pub accs: Vec<RigidBodyVelocity<Real>>,
    /// The generalized forces that derive from positions or external inputs (gravity, applied
    /// forces, spring position terms), as the last acceleration update that measured them read
    /// them.
    pub position_forces: DVector,
}

impl MultibodyWorkspace {
    /// Create an empty workspace.
    pub fn new() -> Self {
        MultibodyWorkspace {
            accs: Vec::new(),
            position_forces: DVector::zeros(0),
        }
    }

    /// Resize the workspace so it is enough for `nlinks` links.
    pub fn resize(&mut self, nlinks: usize, ndofs: usize) {
        self.accs.resize(nlinks, RigidBodyVelocity::zero());
        self.position_forces = DVector::zeros(ndofs)
    }
}
