pub use self::impulse_joint::{ImpulseJoint, ImpulseJointReaction};
pub use self::impulse_joint_handle::ImpulseJointHandle;
#[cfg(feature = "alloc")]
pub use self::impulse_joint_set::ImpulseJointSet;
#[cfg(feature = "alloc")]
pub(crate) use self::impulse_joint_set::{JointGraphEdge, JointIndex};

mod impulse_joint;
mod impulse_joint_handle;
#[cfg(all(test, feature = "dim3"))]
mod impulse_joint_regression_tests;
#[cfg(feature = "alloc")]
mod impulse_joint_set;
