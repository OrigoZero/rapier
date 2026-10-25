use crate::dynamics::{GenericJoint, ImpulseJointHandle, RigidBodyHandle, RigidBodySet};
use crate::math::{AngVector, SpatialVector, Vector};
#[cfg(feature = "dim3")]
use crate::utils::CrossProduct;

/// The impulse a joint applied to its two bodies over the last step that solved
/// it, summed over that step's substeps and over every row the solver built for
/// the joint (its locked axes, limits and motors), in world space. Divided by
/// the step's duration it is the average force and moment the joint held.
///
/// Each row's impulse acts along that row's jacobian as the solver held it, after
/// the rows were orthogonalized against each other, so this is the impulse the
/// bodies received rather than a reading of the joint's own axes.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImpulseJointReaction {
    /// The linear impulse applied to the first body; the second body received
    /// its opposite.
    pub linear: Vector,
    /// The angular impulse applied to the first body about the point the solver
    /// held it about: its center of mass while it took part in the solve.
    pub angular1: AngVector,
    /// The angular impulse applied to the second body about its center of mass
    /// while it took part in the solve, with the sign the first body's reaction
    /// carries: the second body received its opposite.
    pub angular2: AngVector,
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
/// An impulse-based joint attached to two bodies.
pub struct ImpulseJoint {
    /// Handle to the first body attached to this joint.
    ///
    /// Kept crate-private so the attached bodies can only be changed via
    /// [`ImpulseJointSet::set_bodies`], which also updates the interaction
    /// graph and island-manager bookkeeping. Read it through
    /// [`ImpulseJoint::body1`].
    pub(crate) body1: RigidBodyHandle,
    /// Handle to the second body attached to this joint. See
    /// [`ImpulseJoint::body1`] for why this is not directly writeable.
    pub(crate) body2: RigidBodyHandle,

    /// The joint’s description.
    pub data: GenericJoint,

    /// The impulses applied by this joint.
    pub impulses: SpatialVector,

    /// The impulse this joint applied to its bodies over the last step that
    /// solved it, in world space. Recorded for a joint solved between rigid
    /// bodies; read it through [`ImpulseJoint::reaction_about`].
    pub reaction: ImpulseJointReaction,

    // A joint needs to know its handle to simplify its removal.
    pub(crate) handle: ImpulseJointHandle,

    /// The solver-body index (`active_set_id`) of each attached body, or
    /// `u32::MAX` for a side that is world-attached from the solver's point of
    /// view (fixed or sleeping).
    #[cfg_attr(
        feature = "serde-serialize",
        serde(skip, default = "default_solver_body_ids")
    )]
    pub(crate) solver_body_ids: [u32; 2],
    /// The solver graph color assigned to this joint.
    #[cfg_attr(feature = "serde-serialize", serde(default = "default_solver_color"))]
    pub(crate) solver_color: u8,
}

#[cfg(feature = "serde-serialize")]
fn default_solver_color() -> u8 {
    crate::geometry::contact_pair::SOLVER_COLOR_UNCOLORED
}

#[cfg(feature = "serde-serialize")]
fn default_solver_body_ids() -> [u32; 2] {
    [u32::MAX; 2]
}

impl ImpulseJoint {
    /// Handle to the first body attached to this joint.
    #[inline]
    pub fn body1(&self) -> RigidBodyHandle {
        self.body1
    }

    /// Handle to the second body attached to this joint.
    #[inline]
    pub fn body2(&self) -> RigidBodyHandle {
        self.body2
    }

    /// The impulse this joint applied to its first body over the last step
    /// that solved it, as a linear impulse and the angular impulse about the
    /// world point `point`.
    ///
    /// The angular part is moved from the center of mass of a body that took
    /// part in that solve: the first body's, or the second body's when the
    /// first was held fixed (a fixed or sleeping body), whose reaction is the
    /// same wrench with its sign turned.
    pub fn reaction_about(&self, bodies: &RigidBodySet, point: Vector) -> (Vector, AngVector) {
        let reaction = &self.reaction;
        let (angular, body) = if self.solver_body_ids[0] != u32::MAX || self.solver_body_ids[1] == u32::MAX {
            (reaction.angular1, self.body1)
        } else {
            (reaction.angular2, self.body2)
        };
        let center = bodies.get(body).map_or(point, |rb| rb.center_of_mass());
        // The moment of the linear impulse about `point`, taken off the one the
        // solver applied about the center: M_p = M_c + (c - p) x L.
        #[cfg(feature = "dim3")]
        let angular = angular + (center - point).gcross(reaction.linear);
        #[cfg(feature = "dim2")]
        let angular = angular + (center - point).perp_dot(reaction.linear);
        (reaction.linear, angular)
    }
}
