use super::multibody_link::{MultibodyLink, MultibodyLinkVec};
use super::multibody_workspace::{MultibodyStepEnd, MultibodyWorkspace};
use crate::alloc_prelude::*;
use crate::dynamics::integration_parameters::SpringCoefficients;
use crate::dynamics::solver::{GenericJointConstraint, WritebackId};
use crate::dynamics::{
    IntegrationParameters, RigidBodyHandle, RigidBodySet, RigidBodyType, RigidBodyVelocity,
};
use crate::math::{
    ANG_DIM, AngDim, AngVector, DIM, DVector, Dim, Jacobian, Pose, Real, Rotation, SPATIAL_DIM,
    SimdAngVector, SpatialVector, Vector,
};
use crate::prelude::MultibodyJoint;
#[cfg(feature = "dim3")]
use crate::utils::mat_to_na;
use crate::utils::{AngularInertiaOps, CrossProduct, CrossProductMatrix, IndexMut2, vect_to_na};
use na::{
    self, DMatrix, DVectorView, DVectorViewMut, Dyn, LU, OMatrix, SMatrix, SVector, StorageMut,
};
use simba::simd::SimdComplexField as _;

#[cfg(doc)]
use crate::prelude::{GenericJoint, RigidBody};

/// The fraction of its kinetic energy a multibody may gain over a substep past what the forces
/// on it did, for the rounding and the drift of a healthy substep. Every substep its rows bear on
/// is held to it (see `VelocitySolver::update_multibody_dynamics`),
/// however far it turns the multibody's coordinates: a chain whose fastest joint turns a few
/// hundredths of a radian a substep can still gain a percent of its energy each substep, when a
/// link pressed against a body spins about its own length and the joints below it turn back
/// against it, through the Coriolis update of those joints, and a chain held only from 0.08 rad
/// a substep was flung off its anchor's body.
const ENERGY_HOLD_MARGIN: Real = 1.0e-3;

/// The share of a multibody's kinetic energy that stands within the rounding of the energies a
/// substep reads: a velocity, less its drift, that carries no more than this share of the energy
/// is the rounding of the drift (see [`Multibody::energy_ledger_scale`]). A multibody falling
/// from rest moves with its drift alone, and the energy ledger turned the rounding of its energy
/// into a velocity hundreds of millions of times the rounding left: a box with an arm on a hinge,
/// dropped at 30 steps a second in 2 substeps, started swinging its arm at its first substep and
/// fell 9% short of a lone box.
const ROUNDING_ENERGY_SHARE: Real = 1.0e-6;

/// Which of the generalized forces on a multibody an acceleration update reads
/// ([`Multibody::update_acceleration`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum MultibodyForces {
    /// The forces applied to its links (gravity and every force and torque added to their
    /// bodies), the forces of its joints' springs and the damping of its joints.
    Applied,
    /// The inertial forces of its motion at the configuration it stands at: the Coriolis and
    /// centrifugal forces of its joints' velocities and the gyroscopic torques of its links.
    Inertial,
}

/// The factorization of a multibody's mass matrix, held in `f64` whatever precision the rest of
/// the solver works in.
///
/// A multibody's mass matrix spans the inertia of everything its joints move: a light link
/// turning about its own long axis and a heavy load swinging at the end of a long chain differ by
/// eight orders of magnitude or more (a 4 kg capsule 0.07 m wide spins against 0.01 kg·m², a
/// 500 kg ball 7 m down a chain swings against 10⁵), which is past the seven digits an `f32`
/// holds. Assembled and factored in `f32`, the smallest of those inertias came out of the
/// rounding with the wrong sign, and a contact solved through the factorization moved the
/// multibody with a negative effective mass: each impulse pushed further into the contact it was
/// meant to stop and a chain resting on a body was flung. The matrix is assembled from the
/// solver's own values (whose rounding moves the large inertias only), and factored and solved
/// in `f64`.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct MultibodyMassFactorization(LU<f64, Dyn, Dyn>);

impl MultibodyMassFactorization {
    fn new(mass: DMatrix<f64>) -> Self {
        Self(LU::new(mass))
    }

    /// Solves `M x = b` in place for each column of `b`, where `M` is the factored matrix.
    /// Returns `false` if the matrix is singular, leaving `b` as the factorization left it.
    pub fn solve_mut<R: na::Dim, C: na::Dim, S: StorageMut<Real, R, C>>(
        &self,
        b: &mut na::Matrix<Real, R, C, S>,
    ) -> bool {
        let mut solved = true;
        let mut wide = na::DVector::<f64>::zeros(b.nrows());
        for mut column in b.column_iter_mut() {
            for (w, x) in wide.iter_mut().zip(column.iter()) {
                *w = *x as f64;
            }
            solved &= self.0.solve_mut(&mut wide);
            for (x, w) in column.iter_mut().zip(wide.iter()) {
                *x = *w as Real;
            }
        }
        solved
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct Force {
    linear: Vector,
    angular: AngVector,
}

impl Force {
    fn new(linear: Vector, angular: AngVector) -> Self {
        Self { linear, angular }
    }

    fn as_vector(&self) -> &SVector<Real, SPATIAL_DIM> {
        unsafe { core::mem::transmute(self) }
    }
}

#[cfg(feature = "dim2")]
fn concat_rb_mass_matrix(mass: Vector, inertia: Real) -> SMatrix<Real, SPATIAL_DIM, SPATIAL_DIM> {
    let mut result = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
    result[(0, 0)] = mass.x;
    result[(1, 1)] = mass.y;
    result[(2, 2)] = inertia;
    result
}

#[cfg(feature = "dim3")]
fn concat_rb_mass_matrix(
    mass: Vector,
    inertia: na::Matrix3<Real>,
) -> SMatrix<Real, SPATIAL_DIM, SPATIAL_DIM> {
    let mut result = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
    result[(0, 0)] = mass.x;
    result[(1, 1)] = mass.y;
    result[(2, 2)] = mass.z;
    result
        .fixed_view_mut::<ANG_DIM, ANG_DIM>(DIM, DIM)
        .copy_from(&inertia);
    result
}

/// A holonomic coupling between two generalized coordinates of a single
/// [`Multibody`], `q2 = coeff · q1 + offset`, enforced as a velocity-level
/// equality constraint. This is how MuJoCo's `<equality><joint>` (a polynomial
/// joint-to-joint coupling) is represented for the linear (first-order) case —
/// e.g. the robotiq gripper's two driver joints moving together.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct MultibodyDofCoupling {
    /// Internal id of the link owning the first joint.
    pub link1: usize,
    /// Local free-DoF index of the coupled DoF within `link1` (its position in
    /// that link's slice of the generalized vectors).
    pub dof1: usize,
    /// Spatial-coordinate axis (`0..6`) of `link1`'s coupled DoF, used to read
    /// its generalized position from the joint coords.
    pub axis1: usize,
    /// Internal id of the link owning the second joint.
    pub link2: usize,
    /// Local free-DoF index of the coupled DoF within `link2`.
    pub dof2: usize,
    /// Spatial-coordinate axis (`0..6`) of `link2`'s coupled DoF.
    pub axis2: usize,
    /// Linear coupling coefficient: the constraint is `q2 − coeff·q1 − offset = 0`.
    pub coeff: Real,
    /// Constant offset of the coupling.
    pub offset: Real,
}

/// An articulated body simulated using the reduced-coordinates approach.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Multibody {
    // TODO: serialization: skip the workspace fields.
    pub(crate) links: MultibodyLinkVec,
    pub(crate) velocities: DVector,
    pub(crate) damping: DVector,
    /// Per-DoF reflected rotor inertia (matches MuJoCo’s concept of `armature`).
    pub(crate) armature: DVector,
    /// Per-DoF dry joint friction (matches MuJoCo’s `frictionloss`).
    pub(crate) frictions: DVector,
    pub(crate) accelerations: DVector,

    body_jacobians: Vec<Jacobian<Real>>,
    // NOTE: the mass matrices are dimensioned based on the non-kinematic degrees of
    //       freedoms only. The `Self::augmented_mass_permutation` sequence can be used to
    //       move dofs from/to a format that matches the augmented mass.
    // TODO: use sparse matrices?
    augmented_mass: DMatrix<f64>,
    inv_augmented_mass: MultibodyMassFactorization,
    // The indexing sequence for moving all kinematics degrees of
    // freedoms to the end of the generalized coordinates vector.
    augmented_mass_indices: IndexSequence,

    acc_augmented_mass: DMatrix<f64>,
    acc_inv_augmented_mass: MultibodyMassFactorization,

    ndofs: usize,
    pub(crate) root_is_dynamic: bool,
    pub(crate) solver_id: u32,
    self_contacts_enabled: bool,
    /// Holonomic couplings between two of this multibody's generalized
    /// coordinates (`q2 = coeff·q1 + offset`), e.g. MuJoCo's
    /// `<equality><joint>`. Resolved as velocity constraints each step.
    couplings: Vec<MultibodyDofCoupling>,

    /*
     * Workspaces.
     */
    workspace: MultibodyWorkspace,
    coriolis_v: Vec<OMatrix<Real, Dim, Dyn>>,
    coriolis_w: Vec<OMatrix<Real, AngDim, Dyn>>,
    i_coriolis_dt: Jacobian<Real>,
}
impl Default for Multibody {
    fn default() -> Self {
        Multibody::new()
    }
}

impl Multibody {
    /// Creates a new multibody with no link.
    pub fn new() -> Self {
        Self::with_self_contacts(true)
    }

    pub(crate) fn with_self_contacts(self_contacts_enabled: bool) -> Self {
        Multibody {
            links: MultibodyLinkVec(Vec::new()),
            velocities: DVector::zeros(0),
            damping: DVector::zeros(0),
            armature: DVector::zeros(0),
            frictions: DVector::zeros(0),
            accelerations: DVector::zeros(0),
            body_jacobians: Vec::new(),
            augmented_mass: DMatrix::zeros(0, 0),
            inv_augmented_mass: MultibodyMassFactorization::new(DMatrix::zeros(0, 0)),
            acc_augmented_mass: DMatrix::zeros(0, 0),
            acc_inv_augmented_mass: MultibodyMassFactorization::new(DMatrix::zeros(0, 0)),
            augmented_mass_indices: IndexSequence::new(),
            ndofs: 0,
            solver_id: 0,
            workspace: MultibodyWorkspace::new(),
            coriolis_v: Vec::new(),
            coriolis_w: Vec::new(),
            i_coriolis_dt: Jacobian::zeros(0),
            root_is_dynamic: false,
            self_contacts_enabled,
            couplings: Vec::new(),
            // solver_workspace: Some(SolverWorkspace::new()),
        }
    }

    pub(crate) fn with_root(handle: RigidBodyHandle, self_contacts_enabled: bool) -> Self {
        let mut mb = Multibody::with_self_contacts(self_contacts_enabled);
        // NOTE: we have no way of knowing if the root in fixed at this point, so
        //       we mark it as dynamic and will fix later with `Self::update_root_type`.
        mb.root_is_dynamic = true;
        let joint = MultibodyJoint::free(Pose::IDENTITY);
        mb.add_link(None, joint, handle);
        mb
    }

    pub(crate) fn remove_link(self, to_remove: usize, joint_only: bool) -> Vec<Multibody> {
        let mut result = vec![];
        let mut link2mb = vec![usize::MAX; self.links.len()];
        let mut link_id2new_id = vec![usize::MAX; self.links.len()];
        // Links whose joint is replaced by a fixed one: their DoFs no longer exist.
        let mut joint_replaced = vec![false; self.links.len()];

        // Split multibody and update the set of links and ndofs.
        for (i, mut link) in self.links.0.into_iter().enumerate() {
            let is_new_root = i == 0
                || !joint_only && link.parent_internal_id == to_remove
                || joint_only && i == to_remove;

            if !joint_only && i == to_remove {
                continue;
            } else if is_new_root {
                link2mb[i] = result.len();
                result.push(Multibody::with_self_contacts(self.self_contacts_enabled));
            } else {
                link2mb[i] = link2mb[link.parent_internal_id]
            }

            let curr_mb = &mut result[link2mb[i]];
            link_id2new_id[i] = curr_mb.links.len();

            if is_new_root {
                let joint = MultibodyJoint::fixed(*link.local_to_world());
                link.joint = joint;
                joint_replaced[i] = true;
            }

            curr_mb.ndofs += link.joint().ndofs();
            curr_mb.links.push(link);
        }

        // Adjust all the internal ids, and copy the data from the
        // previous multibody to the new one.
        for mb in &mut result {
            mb.grow_buffers(mb.ndofs, mb.links.len());
            mb.workspace.resize(mb.links.len(), mb.ndofs);

            let mut assembly_id = 0;
            for (i, link) in mb.links.iter_mut().enumerate() {
                let link_ndofs = link.joint().ndofs();
                mb.velocities
                    .rows_mut(assembly_id, link_ndofs)
                    .copy_from(&self.velocities.rows(link.assembly_id, link_ndofs));
                mb.damping
                    .rows_mut(assembly_id, link_ndofs)
                    .copy_from(&self.damping.rows(link.assembly_id, link_ndofs));
                mb.armature
                    .rows_mut(assembly_id, link_ndofs)
                    .copy_from(&self.armature.rows(link.assembly_id, link_ndofs));
                mb.frictions
                    .rows_mut(assembly_id, link_ndofs)
                    .copy_from(&self.frictions.rows(link.assembly_id, link_ndofs));
                mb.accelerations
                    .rows_mut(assembly_id, link_ndofs)
                    .copy_from(&self.accelerations.rows(link.assembly_id, link_ndofs));

                link.internal_id = i;
                link.assembly_id = assembly_id;

                // NOTE: for the root, the current`link.parent_internal_id` is invalid since that
                //       parent lies in a different multibody now.
                link.parent_internal_id = if i != 0 {
                    link_id2new_id[link.parent_internal_id]
                } else {
                    0
                };
                assembly_id += link_ndofs;
            }
        }

        // Keep the couplings whose DoFs still exist and ended up in the same multibody.
        for coupling in &self.couplings {
            let (l1, l2) = (coupling.link1, coupling.link2);
            let kept = |l: usize| link_id2new_id[l] != usize::MAX && !joint_replaced[l];
            if kept(l1) && kept(l2) && link2mb[l1] == link2mb[l2] {
                result[link2mb[l1]].couplings.push(MultibodyDofCoupling {
                    link1: link_id2new_id[l1],
                    link2: link_id2new_id[l2],
                    ..*coupling
                });
            }
        }

        result
    }

    pub(crate) fn append(&mut self, mut rhs: Multibody, parent: usize, joint: MultibodyJoint) {
        let joint_ndofs = joint.ndofs();
        let rhs_root_ndofs = rhs.links[0].joint.ndofs();
        let ndofs_before_append = self.velocities.len();
        let base_internal_id = self.links.len();
        // Values for rhs will be copied into the buffers of `self` starting at this index.
        let rhs_copy_shift = ndofs_before_append + joint_ndofs;
        // Number of dofs to copy from rhs. The root’s dofs isn’t included because it will be
        // replaced by `joint`.
        let rhs_copy_ndofs = rhs.ndofs - rhs_root_ndofs;

        // Adjust the ids of all the rhs links except the first one.
        for link in &mut rhs.links.0[1..] {
            link.assembly_id =
                (link.assembly_id + ndofs_before_append + joint_ndofs) - rhs_root_ndofs;
            link.internal_id += base_internal_id;
            link.parent_internal_id += base_internal_id;
        }

        // Adjust the first link.
        {
            rhs.links[0].joint = joint;
            rhs.links[0].assembly_id = ndofs_before_append;
            rhs.links[0].internal_id = base_internal_id;
            rhs.links[0].parent_internal_id = parent;
        }

        // Grow buffers then append data from rhs.
        self.grow_buffers(rhs_copy_ndofs + joint_ndofs, rhs.links.len());

        if rhs_copy_ndofs > 0 {
            self.velocities
                .rows_mut(rhs_copy_shift, rhs_copy_ndofs)
                .copy_from(&rhs.velocities.rows(rhs_root_ndofs, rhs_copy_ndofs));
            self.damping
                .rows_mut(rhs_copy_shift, rhs_copy_ndofs)
                .copy_from(&rhs.damping.rows(rhs_root_ndofs, rhs_copy_ndofs));
            self.armature
                .rows_mut(rhs_copy_shift, rhs_copy_ndofs)
                .copy_from(&rhs.armature.rows(rhs_root_ndofs, rhs_copy_ndofs));
            self.frictions
                .rows_mut(rhs_copy_shift, rhs_copy_ndofs)
                .copy_from(&rhs.frictions.rows(rhs_root_ndofs, rhs_copy_ndofs));
            self.accelerations
                .rows_mut(rhs_copy_shift, rhs_copy_ndofs)
                .copy_from(&rhs.accelerations.rows(rhs_root_ndofs, rhs_copy_ndofs));
        }

        // Set the default damping for the new joint.
        rhs.links[0]
            .joint
            .default_damping(&mut self.damping.rows_mut(ndofs_before_append, joint_ndofs));

        self.links.append(&mut rhs.links);
        self.ndofs = self.velocities.len();
        self.workspace.resize(self.links.len(), self.ndofs);

        // The rhs couplings follow its links, except the ones on its root whose joint was replaced.
        self.couplings.extend(
            rhs.couplings
                .iter()
                .filter(|c| c.link1 != 0 && c.link2 != 0)
                .map(|c| MultibodyDofCoupling {
                    link1: c.link1 + base_internal_id,
                    link2: c.link2 + base_internal_id,
                    ..*c
                }),
        );
        // Self-contacts stay disabled for the links of a multibody that had them disabled.
        self.self_contacts_enabled &= rhs.self_contacts_enabled;
    }

    /// Whether self-contacts are enabled on this multibody.
    ///
    /// If set to `false` no two link from this multibody can generate contacts, even
    /// if the contact is enabled on the individual joint with [`GenericJoint::contacts_enabled`].
    pub fn self_contacts_enabled(&self) -> bool {
        self.self_contacts_enabled
    }

    /// Sets whether self-contacts are enabled on this multibody.
    ///
    /// If set to `false` no two link from this multibody can generate contacts, even
    /// if the contact is enabled on the individual joint with [`GenericJoint::contacts_enabled`].
    pub fn set_self_contacts_enabled(&mut self, enabled: bool) {
        self.self_contacts_enabled = enabled;
    }

    /// The inverse augmented mass matrix of this multibody, factored in `f64`
    /// (see [`MultibodyMassFactorization`]).
    pub fn inv_augmented_mass(&self) -> &MultibodyMassFactorization {
        &self.inv_augmented_mass
    }

    /// The first link of this multibody.
    #[inline]
    pub fn root(&self) -> &MultibodyLink {
        &self.links[0]
    }

    /// Mutable reference to the first link of this multibody.
    #[inline]
    pub fn root_mut(&mut self) -> &mut MultibodyLink {
        &mut self.links[0]
    }

    /// Reference `i`-th multibody link of this multibody.
    ///
    /// Return `None` if there is less than `i + 1` multibody links.
    #[inline]
    pub fn link(&self, id: usize) -> Option<&MultibodyLink> {
        self.links.get(id)
    }

    /// Mutable reference to the multibody link with the given id.
    ///
    /// Return `None` if the given id does not identifies a multibody link part of `self`.
    #[inline]
    pub fn link_mut(&mut self, id: usize) -> Option<&mut MultibodyLink> {
        self.links.get_mut(id)
    }

    /// The number of links on this multibody.
    pub fn num_links(&self) -> usize {
        self.links.len()
    }

    /// Iterator through all the links of this multibody.
    ///
    /// All link are guaranteed to be yielded before its descendant.
    pub fn links(&self) -> impl Iterator<Item = &MultibodyLink> {
        self.links.iter()
    }

    /// Mutable iterator through all the links of this multibody.
    ///
    /// All link are guaranteed to be yielded before its descendant.
    pub fn links_mut(&mut self) -> impl Iterator<Item = &mut MultibodyLink> {
        self.links.iter_mut()
    }

    /// The vector of damping applied to this multibody.
    #[inline]
    pub fn damping(&self) -> &DVector {
        &self.damping
    }

    /// Mutable vector of damping applied to this multibody.
    #[inline]
    pub fn damping_mut(&mut self) -> &mut DVector {
        &mut self.damping
    }

    /// The vector of per-DoF armature (reflected rotor inertia) of this multibody: additional
    /// inertia added directly to the mass matrix, simulating the joint's own weight distribution.
    #[inline]
    pub fn armature(&self) -> &DVector {
        &self.armature
    }

    /// Mutable vector of per-DoF armature (reflected rotor inertia) of this
    /// multibody.
    #[inline]
    pub fn armature_mut(&mut self) -> &mut DVector {
        &mut self.armature
    }

    /// The vector of per-DoF dry joint friction (MuJoCo's `frictionloss`, in N
    /// or N·m): the largest force friction may generate on that DoF.
    #[inline]
    pub fn frictions(&self) -> &DVector {
        &self.frictions
    }

    /// Mutable vector of per-DoF dry joint friction of this multibody.
    #[inline]
    pub fn frictions_mut(&mut self) -> &mut DVector {
        &mut self.frictions
    }

    pub(crate) fn add_link(
        &mut self,
        parent: Option<usize>, // TODO: should be a RigidBodyHandle?
        dof: MultibodyJoint,
        body: RigidBodyHandle,
    ) -> &mut MultibodyLink {
        assert!(
            parent.is_none() || !self.links.is_empty(),
            "Multibody::build_body: invalid parent id."
        );

        /*
         * Compute the indices.
         */
        let assembly_id = self.velocities.len();
        let internal_id = self.links.len();

        /*
         * Grow the buffers.
         */
        let ndofs = dof.ndofs();
        self.grow_buffers(ndofs, 1);
        self.ndofs += ndofs;

        /*
         * Setup default damping.
         */
        dof.default_damping(&mut self.damping.rows_mut(assembly_id, ndofs));

        /*
         * Create the multibody.
         */
        let local_to_parent = dof.body_to_parent();
        let local_to_world;

        let parent_internal_id;
        if let Some(parent) = parent {
            parent_internal_id = parent;
            let parent_link = &mut self.links[parent_internal_id];
            local_to_world = parent_link.local_to_world * local_to_parent;
        } else {
            parent_internal_id = 0;
            local_to_world = local_to_parent;
        }

        let rb = MultibodyLink::new(
            body,
            internal_id,
            assembly_id,
            parent_internal_id,
            dof,
            local_to_world,
            local_to_parent,
        );

        self.links.push(rb);
        self.workspace.resize(self.links.len(), self.ndofs);

        &mut self.links[internal_id]
    }

    fn grow_buffers(&mut self, ndofs: usize, num_jacobians: usize) {
        let len = self.velocities.len();
        self.velocities.resize_vertically_mut(len + ndofs, 0.0);
        self.damping.resize_vertically_mut(len + ndofs, 0.0);
        self.armature.resize_vertically_mut(len + ndofs, 0.0);
        self.frictions.resize_vertically_mut(len + ndofs, 0.0);
        self.accelerations.resize_vertically_mut(len + ndofs, 0.0);
        self.body_jacobians
            .extend((0..num_jacobians).map(|_| Jacobian::zeros(0)));
    }

    /// The accelerations the generalized forces `forces` give the multibody over a substep of
    /// `dt`, at the configuration and the velocity its links' bodies stand at, solved through the
    /// mass matrix [`Self::update_mass_matrix`] last built.
    ///
    /// The applied forces are read as a substep begins and the inertial forces as its move ends:
    /// the Coriolis, centrifugal and gyroscopic forces turn the velocity the multibody moved with
    /// for the configuration the move reached, before the substep's stabilization pass reads that
    /// velocity (see `VelocitySolver::integrate_multibody_positions`). The applied forces'
    /// position part (gravity, the forces added to the links' bodies and the springs' pull) is
    /// kept as the work [`Self::kinetic_energy_scale`] reads.
    pub(crate) fn update_acceleration(
        &mut self,
        dt: Real,
        bodies: &RigidBodySet,
        forces: MultibodyForces,
    ) {
        if self.ndofs == 0 {
            return; // Nothing to do.
        }
        let applied = forces == MultibodyForces::Applied;
        let inertial = forces == MultibodyForces::Inertial;

        self.accelerations.fill(0.0);

        // If there is significant movement, the semi-implicit coriolis term
        // occasionally cause instabilities due to the matrix being near singular.
        // To prevent that, we enable a check that verifies the result and fallbacks
        // to the explicit term if needed
        let check_implicit_coriolis_divergence = dt * self.velocities.amax() >= 1.0e-3;
        // Generalized forces that derive from positions or external inputs
        // (gravity, user forces, spring position terms), which the energy hold
        // reads as the work of a substep (`kinetic_energy_scale`).
        let mut pos_forces = DVector::zeros(self.ndofs);

        // Eqn 42 to 45
        for i in 0..self.links.len() {
            let link = &self.links[i];
            let rb = &bodies[link.rigid_body];

            let mut acc = RigidBodyVelocity::zero();

            if i != 0 {
                let parent_id = link.parent_internal_id;
                let parent_link = &self.links[parent_id];
                let parent_rb = &bodies[parent_link.rigid_body];

                acc += self.workspace.accs[parent_id];
                // The 2.0 originates from the two identical terms of Jdot (the terms become
                // identical once they are multiplied by the generalized velocities).
                acc.linvel += 2.0 * parent_rb.vels.angvel.gcross(link.joint_velocity.linvel);
                #[cfg(feature = "dim3")]
                {
                    acc.angvel += parent_rb.vels.angvel.cross(link.joint_velocity.angvel);
                }

                acc.linvel += parent_rb
                    .vels
                    .angvel
                    .gcross(parent_rb.vels.angvel.gcross(link.shift02));
                acc.linvel += self.workspace.accs[parent_id].angvel.gcross(link.shift02);
            }

            acc.linvel += rb.vels.angvel.gcross(rb.vels.angvel.gcross(link.shift23));
            acc.linvel += acc.angvel.gcross(link.shift23);

            self.workspace.accs[i] = acc;

            // TODO: should gyroscopic forces already be computed by the rigid-body itself
            //       (at the same time that we add the gravity force)?
            let gyroscopic;
            let rb_inertia = rb.mprops.effective_angular_inertia();
            let rb_mass = rb.mprops.effective_mass();

            #[cfg(feature = "dim3")]
            {
                let angvel = rb.vels.angvel;
                let inertia_times_angvel = rb_inertia * angvel;
                gyroscopic = angvel.cross(inertia_times_angvel);
            }
            #[cfg(feature = "dim2")]
            {
                gyroscopic = 0.0;
            }

            if inertial {
                let inertial_forces = Force::new(
                    -(rb_mass * acc.linvel),
                    -(gyroscopic + rb_inertia * acc.angvel),
                );
                self.accelerations.gemv_tr(
                    1.0,
                    &self.body_jacobians[i],
                    inertial_forces.as_vector(),
                    1.0,
                );
            }

            if applied {
                let applied_forces = Force::new(rb.forces.force, rb.forces.torque);
                self.accelerations.gemv_tr(
                    1.0,
                    &self.body_jacobians[i],
                    applied_forces.as_vector(),
                    1.0,
                );
                pos_forces.gemv_tr(
                    1.0,
                    &self.body_jacobians[i],
                    applied_forces.as_vector(),
                    1.0,
                );
            }
        }

        if applied {
            self.accelerations
                .cmpy(-1.0, &self.damping, &self.velocities, 1.0);
        }

        // Implicit joint springs: backward-Euler at `q⁺ = q + dt·v⁺` gives `-k·(q − rest) − k·dt·v⁺`,
        // with the `v⁺` part implicit via the `dt²·k` mass-matrix diagonal (see `update_mass_matrix`).
        // Consistency requires BOTH `-k·(q − rest)` and `-k·dt·v` here, else the spring is only semi-implicit.
        for li in 0..self.links.len() {
            let mut idx = self.links[li].assembly_id;
            let locked = self.links[li].joint.data.locked_axes.bits();
            for a in 0..SPATIAL_DIM {
                if (locked >> a) & 1 == 0 {
                    let k = self.links[li].joint.spring_stiffness[a];
                    if k != 0.0 && applied {
                        let q = self.links[li].joint.coords[a];
                        let rest = self.links[li].joint.spring_ref[a];
                        let spring_pos_force = -k * (q - rest);
                        self.accelerations[idx] += spring_pos_force - k * dt * self.velocities[idx];
                        pos_forces[idx] += spring_pos_force;
                    }
                    idx += 1;
                }
            }
        }

        // Snapshot the full generalized forces: the energy guard may need them
        // for a re-solve with the plain mass matrix.
        let gen_forces = if check_implicit_coriolis_divergence {
            self.accelerations.clone()
        } else {
            DVector::zeros(0)
        };
        let root_forces: na::DVector<f64> = if applied {
            self.accelerations
                .rows(0, self.free_root_translations())
                .map(|x| x as f64)
        } else {
            na::DVector::zeros(0)
        };

        self.augmented_mass_indices
            .with_rearranged_rows_mut(&mut self.accelerations, |accs| {
                self.acc_inv_augmented_mass.solve_mut(accs);
            });

        if check_implicit_coriolis_divergence {
            self.free_velocity_energy_guard(dt, &gen_forces, &pos_forces);
        }
        if applied {
            self.hold_drift_to_forces(&root_forces);
            self.workspace.position_forces = pos_forces;
        }
    }

    /// Makes the accelerations change the linear momentum of a multibody along its root's free
    /// translations ([`Self::translation_momentum`]) by `forces`, the generalized forces on those
    /// translations, and by nothing else, through a translation of the whole multibody: the rows
    /// of the mass matrix with its joints' damping, armature and springs, without the
    /// semi-implicit Coriolis and gyroscopic terms, times the accelerations equal `forces` on the
    /// root's free translations.
    ///
    /// The applied forces are solved through the mass matrix with the semi-implicit Coriolis
    /// terms, which read how the forces' kick turns the inertial forces of the joints' motion.
    /// Those turn the links' velocities against one another, and through the root's rows they
    /// also moved the linear momentum of a free multibody each substep a force bent its joints,
    /// where the force along a translation of the whole multibody is the whole of what changes
    /// its linear momentum along it.
    fn hold_drift_to_forces(&mut self, forces: &na::DVector<f64>) {
        let n = forces.len();
        if n == 0 {
            return;
        }
        let effective_dim = self.augmented_mass_indices.dim_after_removal(self.ndofs);
        let mut accelerations = self.accelerations.clone();
        self.augmented_mass_indices
            .rearrange_rows(&mut accelerations, true);
        let accelerations = accelerations.rows(0, effective_dim).map(|x| x as f64);
        let rows = self.augmented_mass.view((0, 0), (n, effective_dim)) * accelerations;
        let block = self.augmented_mass.view((0, 0), (n, n)).into_owned();
        if let Some(correction) = block.lu().solve(&(forces - rows)) {
            if correction.iter().all(|x| x.is_finite()) {
                for k in 0..n {
                    self.accelerations[k] += correction[k] as Real;
                }
            }
        }
    }

    /// The fastest any joint of the multibody turns its link about the link it hangs from, in
    /// radians per second, read from the joints' velocities: the angular velocity of each link
    /// relative to its parent. The root's own turning moves the multibody as a whole and is not
    /// read.
    pub(crate) fn fastest_joint_turn(&self) -> Real {
        let mut fastest: Real = 0.0;
        for link in self.links.iter().skip(1) {
            let velocity = link
                .joint
                .jacobian_mul_coordinates(&self.velocities.as_slice()[link.assembly_id..]);
            #[cfg(feature = "dim2")]
            let turn = velocity.angvel.abs();
            #[cfg(feature = "dim3")]
            let turn = velocity.angvel.length();
            fastest = fastest.max(turn);
        }
        fastest
    }

    /// The kinetic energy of the links at the generalized `velocities`, at the configuration the
    /// body jacobians and the links' mass properties stand at, with the energy of the joints'
    /// armature (the rotor inertia each degree of freedom carries on the mass matrix's diagonal).
    fn kinetic_energy_at(&self, velocities: DVectorView<Real>, bodies: &RigidBodySet) -> Real {
        let mut energy = 0.0;
        if self.armature.len() == velocities.len() {
            for (armature, velocity) in self.armature.iter().zip(velocities.iter()) {
                energy += *armature * *velocity * *velocity;
            }
        }
        for i in 0..self.links.len() {
            let rb = &bodies[self.links[i].rigid_body];
            if !rb.is_dynamic() {
                continue;
            }
            let spatial = &self.body_jacobians[i] * velocities;
            let spatial = spatial.as_slice();
            let linvel = Vector::from_slice(&spatial[..DIM]);
            energy += linvel.dot(rb.mprops.effective_mass() * linvel);
            #[cfg(feature = "dim2")]
            {
                energy += spatial[DIM] * spatial[DIM] * rb.mprops.effective_angular_inertia();
            }
            #[cfg(feature = "dim3")]
            {
                let angvel = Vector::from_slice(&spatial[DIM..]);
                energy += angvel.dot(rb.mprops.effective_angular_inertia() * angvel);
            }
        }
        0.5 * energy
    }

    /// The number of the root joint's free translations, the first degrees of freedom of the
    /// generalized velocity: those of a dynamic root whose joint is not kinematic, none for a
    /// fixed or kinematic root. A free translation of the root moves every link with it.
    fn free_root_translations(&self) -> usize {
        match self.links.first() {
            Some(root) if self.root_is_dynamic && !root.joint.kinematic => {
                root.joint.num_free_lin_dofs()
            }
            _ => 0,
        }
    }

    /// The generalized momentum of each of the root's free translations
    /// ([`Self::free_root_translations`]) at the generalized `velocities`, at the configuration
    /// the body jacobians stand at: the multibody's linear momentum along the translation, with
    /// the translation's armature.
    fn translation_momentum(
        &self,
        velocities: DVectorView<Real>,
        bodies: &RigidBodySet,
    ) -> na::DVector<f64> {
        let n = self.free_root_translations();
        let mut momentum = na::DVector::<f64>::zeros(n);
        for k in 0..n.min(self.armature.len()) {
            momentum[k] += (self.armature[k] * velocities[k]) as f64;
        }
        for i in 0..self.links.len() {
            let rb = &bodies[self.links[i].rigid_body];
            if !rb.is_dynamic() {
                continue;
            }
            let jacobian = &self.body_jacobians[i];
            let linear = jacobian.fixed_rows::<DIM>(0) * velocities;
            let mass = rb.mprops.effective_mass();
            for k in 0..n {
                for d in 0..DIM {
                    momentum[k] += (jacobian[(d, k)] * mass[d] * linear[d]) as f64;
                }
            }
        }
        momentum
    }

    /// The mass the root's free translations move ([`Self::free_root_translations`]), the block
    /// of the mass matrix the kinetic energy reads ([`Self::kinetic_energy_at`]) over them: the
    /// multibody's whole mass along each, with the translations' armature.
    fn translation_inertia(&self, bodies: &RigidBodySet) -> DMatrix<f64> {
        let n = self.free_root_translations();
        let mut inertia = DMatrix::<f64>::zeros(n, n);
        for k in 0..n.min(self.armature.len()) {
            inertia[(k, k)] += self.armature[k] as f64;
        }
        for i in 0..self.links.len() {
            let rb = &bodies[self.links[i].rigid_body];
            if !rb.is_dynamic() {
                continue;
            }
            let jacobian = &self.body_jacobians[i];
            let mass = rb.mprops.effective_mass();
            for k in 0..n {
                for l in 0..n {
                    for d in 0..DIM {
                        inertia[(k, l)] += (jacobian[(d, k)] * mass[d] * jacobian[(d, l)]) as f64;
                    }
                }
            }
        }
        inertia
    }

    /// The generalized velocity that translates the whole multibody, along its root's free
    /// translations, with the translation momentum `momentum` ([`Self::translation_momentum`]):
    /// `Eₜ · Mₜₜ⁻¹ · momentum`, `Mₜₜ` the mass the translations move
    /// ([`Self::translation_inertia`]). `None` for a multibody with no free translation.
    fn translation_velocity(
        &self,
        momentum: &na::DVector<f64>,
        bodies: &RigidBodySet,
    ) -> Option<DVector> {
        let n = momentum.len();
        if n == 0 {
            return None;
        }
        let along = self.translation_inertia(bodies).lu().solve(momentum)?;
        if !along.iter().all(|x| x.is_finite()) {
            return None;
        }
        let mut velocity = DVector::zeros(self.ndofs);
        for k in 0..n {
            velocity[k] = along[k] as Real;
        }
        Some(velocity)
    }

    /// The drift of the generalized velocity `velocities`: the translation of the whole
    /// multibody that carries all of its linear momentum along its root's free translations
    /// ([`Self::translation_velocity`]), the velocity of its centre of mass along them. Zero for
    /// a multibody with no free translation.
    ///
    /// The rest of the velocity carries no linear momentum along those translations and is
    /// orthogonal to the drift through the mass matrix: the kinetic energy of a velocity is the
    /// energy of its drift, `|p|² / 2m` along the translations, plus the energy of the rest,
    /// and the drift's energy is the same at every configuration.
    fn drift(&self, velocities: DVectorView<Real>, bodies: &RigidBodySet) -> DVector {
        let momentum = self.translation_momentum(velocities, bodies);
        self.translation_velocity(&momentum, bodies)
            .unwrap_or_else(|| DVector::zeros(velocities.len()))
    }

    /// Scales the generalized velocity, less its drift ([`Self::drift`]), by `scale`, keeping
    /// the drift, and so the multibody's linear momentum along its root's free translations, as
    /// it is; the accelerations too when `with_accelerations`. The energy hold scales a
    /// multibody's velocity this way: the hold answers for the work the integration's own
    /// rounding does on the multibody's motion about its centre of mass, which moves no
    /// momentum, and a free multibody flying through empty space keeps the linear momentum it
    /// carries whatever its joints do. For a multibody with no free translation, the whole
    /// velocity scales.
    pub(crate) fn scale_keeping_drift(
        &mut self,
        scale: Real,
        with_accelerations: bool,
        bodies: &RigidBodySet,
    ) {
        if scale == 1.0 {
            return;
        }
        let drift = self.drift(self.velocities.as_view(), bodies);
        self.velocities -= &drift;
        self.velocities *= scale;
        self.velocities += &drift;
        if with_accelerations {
            let drift = self.drift(self.accelerations.as_view(), bodies);
            self.accelerations -= &drift;
            self.accelerations *= scale;
            self.accelerations += &drift;
        }
    }

    /// The kinetic energy of the drift of `velocities` ([`Self::drift`]) and of the rest, with
    /// the drift itself: the two energies add to the energy of `velocities`.
    fn split_kinetic_energy(
        &self,
        velocities: DVectorView<Real>,
        bodies: &RigidBodySet,
    ) -> (Real, Real, DVector) {
        if self.free_root_translations() == 0 {
            return (
                0.0,
                self.kinetic_energy_at(velocities, bodies),
                DVector::zeros(velocities.len()),
            );
        }
        let drift = self.drift(velocities, bodies);
        let rest = velocities - &drift;
        (
            self.kinetic_energy_at(drift.as_view(), bodies),
            self.kinetic_energy_at(rest.as_view(), bodies),
            drift,
        )
    }

    /// The linear momentum the multibody carries along its root's free translations
    /// ([`Self::translation_momentum`]) before a move, which [`Self::keep_momentum`] gives back to
    /// it once the move and its inertial forces are done. `None` for a multibody with no free
    /// translation.
    pub(crate) fn momentum_before_move(&self, bodies: &RigidBodySet) -> Option<na::DVector<f64>> {
        (self.free_root_translations() > 0)
            .then(|| self.translation_momentum(self.velocities.as_view(), bodies))
    }

    /// Gives the multibody back, after a move and the inertial forces of the configuration it
    /// reached, the linear momentum it carried `before` the move
    /// ([`Self::momentum_before_move`]), through the translation of the whole multibody that
    /// carries the difference ([`Self::translation_velocity`]). The linear momentum of a
    /// multibody changes only through the forces on it, and a move and its inertial forces are
    /// no force on it.
    ///
    /// A move carries each joint's velocity unchanged in its parent's frame and the root's
    /// velocity unchanged at its centre of mass, and the semi-implicit Coriolis and centrifugal
    /// forces read where it ends give back only part of what that does to the links' velocities
    /// in space: a free chain of nine rods flying at 5 m/s, its joints and its root turning at
    /// up to 20 rad/s, lost 5% of its linear momentum over a 120th of a second, and as much as
    /// a third of it within two seconds.
    pub(crate) fn keep_momentum(&mut self, before: &na::DVector<f64>, bodies: &RigidBodySet) {
        if before.len() != self.free_root_translations() {
            return;
        }
        let momentum = self.translation_momentum(self.velocities.as_view(), bodies);
        if let Some(correction) = self.translation_velocity(&(before - momentum), bodies) {
            self.velocities += correction;
        }
    }

    /// The kinetic energy of the links at the generalized `velocities`, at the configuration the
    /// substep starts from, for [`Self::kinetic_energy_scale`] to hold the substep to: `None`
    /// for a multibody with no degree of freedom, or while a prescribed motion, which does work
    /// no generalized force states, drives it.
    pub(crate) fn kinetic_energy_to_hold(
        &self,
        velocities: DVectorView<Real>,
        bodies: &RigidBodySet,
    ) -> Option<Real> {
        if self.ndofs == 0 {
            return None;
        }
        let prescribed = (0..self.links.len()).any(|i| {
            let link = &self.links[i];
            let rb = &bodies[link.rigid_body];
            let held = link.joint.kinematic || !rb.is_dynamic();
            held && (rb.vels.linvel != Vector::ZERO || rb.vels.angvel != AngVector::default())
        });
        (!prescribed).then(|| self.kinetic_energy_at(velocities, bodies))
    }

    /// Opens a step where the step before left this multibody ([`Self::end_step_standing`]):
    /// moves its velocity back to the one that step ended with before it was held to the energy
    /// it stood at, and answers that energy. `None`, with the velocity left as it is, when a row
    /// bore on its last substep, the continuous pass took it back, or its velocity or
    /// configuration changed between the two steps.
    pub(crate) fn open_step(&mut self) -> Option<Real> {
        let step_end = self.workspace.step_end.as_ref()?;
        let left_as_it_was = self.ndofs > 0
            && step_end.velocities == self.velocities
            && step_end.poses.len() == self.links.len()
            && step_end
                .poses
                .iter()
                .zip(self.links.iter())
                .all(|(pose, link)| *pose == link.local_to_world);
        if !left_as_it_was {
            return None;
        }
        self.velocities.copy_from(&step_end.continued);
        Some(step_end.standing)
    }

    /// Ends a step whose last substep no row bore on: records the kinetic energy the multibody
    /// stands at, half a substep of `dt` on from the velocity it ends with
    /// ([`Self::kicked_kinetic_energy`]), for the next step to open at ([`Self::open_step`]),
    /// and scales the velocity it ends with, less its drift
    /// ([`Self::scale_keeping_drift`]) to no more energy than it stands at, with what the
    /// joints' damping takes from that half kick ([`Self::damping_loss`]), allows.
    ///
    /// A velocity half a kick `h = dt · M⁻¹ f / 2` behind the one it stands at carries
    /// `K(v) = K(v + h) − dt · v · f / 2 − dt² · f · M⁻¹ f / 8`. The kick a substep gives through
    /// the semi-implicit Coriolis terms of the mass matrix turns much of a whipping chain's
    /// velocity back: a chain of twelve 4 kg links with a 20000 kg end, its energy held where it
    /// stood, ended a step 17% of its swing over that energy and was back the step after. The
    /// velocity a step ends with is read between steps and swept by the continuous pass. The
    /// next step that opens here stands the multibody at the energy recorded here and moves on
    /// from the velocity the step ended with before the factor, so the factor shapes the
    /// velocity read between steps and the motion goes on as if no step ended. The factor's
    /// forces `f` leave the joints' damping out, so the damping's share of the half kick is
    /// added back to what the velocity may carry: a heavily damped joint ends a step at the
    /// velocity its damping left, where held to the energy alone it read slower by that share.
    pub(crate) fn end_step_standing(&mut self, dt: Real, bodies: &RigidBodySet) {
        self.workspace.step_end = None;
        let Some(standing) =
            self.kicked_kinetic_energy(self.velocities.as_view(), 0.5 * dt, bodies)
        else {
            return;
        };
        let allowed = standing + self.damping_loss(dt, bodies);
        let (drift_energy, ended, drift) =
            self.split_kinetic_energy(self.velocities.as_view(), bodies);
        let forces = &self.workspace.position_forces;
        let mut response = forces.clone();
        self.augmented_mass_indices
            .with_rearranged_rows_mut(&mut response, |response| {
                self.inv_augmented_mass.solve_mut(response);
            });
        // With d the drift of v and u = v − d the part the factor scales,
        // K(d + c · u) + (d + c · u) · dt · f / 2 + dt² · f · M⁻¹ f / 8 ≤ allowed.
        let drift_moving = 0.5 * dt * drift.dot(forces);
        let moving = 0.5 * dt * self.velocities.dot(forces) - drift_moving;
        let kicked = 0.125 * dt * dt * response.dot(forces) + drift_energy + drift_moving;
        let scale = if ended <= 0.0 {
            1.0
        } else {
            let discriminant = moving * moving - 4.0 * ended * (kicked - allowed);
            let most = if discriminant <= 0.0 {
                -moving / (2.0 * ended)
            } else {
                (-moving + discriminant.simd_sqrt()) / (2.0 * ended)
            };
            most.clamp(0.0, 1.0)
        };
        let continued = self.velocities.clone();
        self.scale_keeping_drift(scale, false, bodies);
        self.workspace.step_end = Some(MultibodyStepEnd {
            standing,
            continued,
            velocities: self.velocities.clone(),
            poses: self.links.iter().map(|link| link.local_to_world).collect(),
        });
    }

    /// Forgets how the last step left this multibody ([`Self::open_step`]): a row bore on
    /// its last substep.
    pub(crate) fn clear_step_end(&mut self) {
        self.workspace.step_end = None;
    }

    /// The kinetic energy of the generalized velocity `velocities + share · accelerations`, at
    /// the configuration the body jacobians and the links' mass properties stand at; `None`
    /// where [`Self::kinetic_energy_to_hold`] holds none.
    ///
    /// A substep moves a multibody at the velocity its kick left: the velocity it stood at plus
    /// the whole of its accelerations over the substep. The velocity it stands at, between the
    /// move that brought it here and the move that takes it on, is that velocity less half the
    /// kick (`share = −dt / 2`), or the velocity the last move left plus half of that substep's
    /// kick. The kinetic energy read there, with the potential of the forces, is what a
    /// multibody's energy is held to from substep to substep, whatever their length: read at the
    /// velocity it moves with, a body falling at a constant pull stood a share of a substep's fall
    /// above its fall, and that share changes with the length of the substep.
    pub(crate) fn kicked_kinetic_energy(
        &self,
        velocities: DVectorView<Real>,
        share: Real,
        bodies: &RigidBodySet,
    ) -> Option<Real> {
        if self.accelerations.len() != velocities.len() {
            return self.kinetic_energy_to_hold(velocities, bodies);
        }
        let mut kicked = velocities.into_owned();
        kicked.axpy(share, &self.accelerations, 1.0);
        self.kinetic_energy_to_hold(kicked.as_view(), bodies)
    }

    /// Whether a spring pulls any of this multibody's joints. A joint spring is integrated
    /// implicitly so a stiff one on a light link stays bounded, which takes energy from it on
    /// purpose: its energy is held from above (see `VelocitySolver::update_multibody_dynamics`).
    pub(crate) fn has_joint_springs(&self) -> bool {
        self.links.iter().any(|link| {
            let locked = link.joint.data.locked_axes.bits();
            (0..SPATIAL_DIM)
                .any(|axis| (locked >> axis) & 1 == 0 && link.joint.spring_stiffness[axis] != 0.0)
        })
    }

    /// Records where each link stands, its centre of mass and its rotation, in `poses` (one
    /// entry per link, in link order, appended): what [`Self::work_since`] reads the work of
    /// the forces from.
    pub(crate) fn record_poses(&self, bodies: &RigidBodySet, poses: &mut Vec<(Vector, Rotation)>) {
        poses.extend(self.links.iter().map(|link| {
            let rb = &bodies[link.rigid_body];
            (rb.mprops.world_com, rb.position().rotation)
        }));
    }

    /// The work the forces on the links did over their move from `poses` (from
    /// [`Self::record_poses`]) to where they stand: each body's force over the path of its
    /// centre of mass and its torque over its turn. The force on a body is held over a step, so
    /// its work over a move is the fall of its potential, whatever path the move took.
    pub(crate) fn work_since(&self, poses: &[(Vector, Rotation)], bodies: &RigidBodySet) -> Real {
        let mut work = 0.0;
        for (link, (com, rotation)) in self.links.iter().zip(poses) {
            let rb = &bodies[link.rigid_body];
            if !rb.is_dynamic() {
                continue;
            }
            work += rb.forces.force.dot(rb.mprops.world_com - *com);
            #[cfg(feature = "dim2")]
            {
                work += rb.forces.torque * (rb.position().rotation * rotation.inverse()).angle();
            }
            #[cfg(feature = "dim3")]
            {
                let turn = (rb.position().rotation * rotation.inverse()).to_scaled_axis();
                work += rb.forces.torque.dot(turn);
            }
        }
        work
    }

    /// The kinetic energy the joints' damping takes from the first half of the kick the
    /// accelerations give the velocity the multibody ends a substep with, over a substep of `dt`:
    /// from `velocities` to the velocity it stands at, `velocities + dt · accelerations / 2`.
    /// 0 for undamped joints.
    pub(crate) fn damping_loss(&self, dt: Real, bodies: &RigidBodySet) -> Real {
        if self.damping.iter().all(|d| *d == 0.0) {
            return 0.0;
        }
        let mut standing = self.velocities.clone();
        standing.axpy(0.5 * dt, &self.accelerations, 1.0);
        self.half_kick_damping_loss(standing, self.velocities.as_view(), dt, bodies)
    }

    /// The kinetic energy the joints' damping took from the second half of the kick the
    /// multibody moves through a substep of `dt` with: from the velocity it stood at,
    /// `velocities − dt · accelerations / 2`, to `velocities`, the kick read at the velocity it
    /// started the kick from, `velocities − dt · accelerations`. 0 for undamped joints.
    ///
    /// Joint damping is implicit, through the `dt · D` the mass matrix carries, and takes most of
    /// a heavily damped joint's velocity within the kick: the energy the multibody stood at
    /// before its move still held all of that half's share.
    pub(crate) fn moving_damping_loss(&self, dt: Real, bodies: &RigidBodySet) -> Real {
        if self.damping.iter().all(|d| *d == 0.0) {
            return 0.0;
        }
        let mut started = self.velocities.clone();
        started.axpy(-dt, &self.accelerations, 1.0);
        self.half_kick_damping_loss(self.velocities.clone(), started.as_view(), dt, bodies)
    }

    /// The kinetic energy of `reached`, the velocity half of a kick over a substep of `dt`
    /// reached, with the damping's share of that half, `dt · (M + dt · D)⁻¹ · (−D · read_at) / 2`,
    /// left out, less the energy of `reached`: what the joints' damping, read at the velocity
    /// `read_at` the kick's accelerations were read at, took from that half; never less than 0.
    ///
    /// The share is solved through the mass matrix with the damping's `dt · D` and without the
    /// semi-implicit Coriolis terms the accelerations are solved with: that matrix is symmetric
    /// and positive, so the share is at most half of `read_at`, where the Coriolis terms, near
    /// singular while a light link whips under a heavy one, gave a whipping chain's damping a
    /// share many times the energy the chain had.
    fn half_kick_damping_loss(
        &self,
        mut reached: DVector,
        read_at: DVectorView<Real>,
        dt: Real,
        bodies: &RigidBodySet,
    ) -> Real {
        let mut damping = self.damping.component_mul(&read_at);
        self.augmented_mass_indices
            .with_rearranged_rows_mut(&mut damping, |damping| {
                self.inv_augmented_mass.solve_mut(damping);
            });
        let damped = self.kinetic_energy_at(reached.as_view(), bodies);
        reached.axpy(0.5 * dt, &damping, 1.0);
        (self.kinetic_energy_at(reached.as_view(), bodies) - damped).max(0.0)
    }

    /// The factor `s` on the velocity a substep ends with, less its drift (`velocities`,
    /// before the next kick; [`Self::scale_keeping_drift`]) that stands the multibody at the
    /// kinetic energy `target` once half of the next kick (`accelerations`) is applied: with `d`
    /// the drift of `v` ([`Self::drift`]), `s ≥ 0` solves `K(d + s · (v − d) + dt · a / 2) =
    /// target`. Where no factor reaches `target`, the one that comes nearest, and never more
    /// than 1. 1 where the velocity less its drift carries no more than
    /// [`ROUNDING_ENERGY_SHARE`] of the energy: that is the rounding of the drift, which carries
    /// the multibody's linear momentum and whose energy no factor changes, and a factor on it
    /// would turn the rounding of the energy `target` into motion.
    ///
    /// A move carries each joint's velocity unchanged in its parent's frame and the Coriolis
    /// and centrifugal forces read where the move ends turn it back, through the mass matrix
    /// they are solved with semi-implicitly: that takes energy from a substep in proportion to
    /// how far its joints turned, and the move's straight step through the coordinates stands
    /// the links where their forces did more or less work than the kick gave them. Both are
    /// the integration's own and are given back here, up as well as down. Held only from above,
    /// the energy of each substep kept every loss and none of the gain the same rounding gives
    /// as often: a chain of twelve undamped 4 kg links with a 5000 kg end, let go 70 degrees up
    /// with no contact, lost half its swing over ten seconds at 120 steps a second, and at 30
    /// steps in 2 substeps lost three quarters of it after standing a sixth of it above its
    /// release.
    pub(crate) fn energy_ledger_scale(
        &self,
        target: Real,
        dt: Real,
        bodies: &RigidBodySet,
    ) -> Real {
        let (_, moving, drift) = self.split_kinetic_energy(self.velocities.as_view(), bodies);
        if moving <= 0.0 {
            return 1.0;
        }
        let mut half_kick = self.accelerations.clone();
        half_kick *= 0.5 * dt;
        let mut kept = half_kick.clone();
        kept += &drift;
        let kicked = self.kinetic_energy_at(kept.as_view(), bodies);
        if moving <= ROUNDING_ENERGY_SHARE * (moving + kicked) {
            return 1.0;
        }
        let mut both = half_kick;
        both += &self.velocities;
        // With d the drift of v and u = v − d the part the factor scales,
        // K(d + s · u + h) = s² · K(u) + s · cross + K(d + h).
        let cross = self.kinetic_energy_at(both.as_view(), bodies) - moving - kicked;
        let discriminant = cross * cross - 4.0 * moving * (kicked - target);
        if discriminant <= 0.0 {
            (-cross / (2.0 * moving)).clamp(0.0, 1.0)
        } else {
            ((-cross + discriminant.simd_sqrt()) / (2.0 * moving)).max(0.0)
        }
    }

    /// The factor that brings the free velocity of this substep (`velocities + dt · accelerations`)
    /// down to the kinetic energy `held` the substep was held to, plus
    /// the work of the position forces over `dt` and a margin; 1 while it stands within that.
    ///
    /// The work of the position forces `f` is the kinetic energy they give the links over a
    /// substep, `dt · v · f + dt² · f · M⁻¹f / 2`, through the mass matrix `M` alone: a body at
    /// rest, or at the turn of its swing, gains the second term alone, so a multibody falls as
    /// fast as a single body and a pendulum swings as high as it was let go.
    ///
    /// A step moves the links at a fixed generalized velocity and then lets the Coriolis
    /// accelerations turn that velocity for the configuration it reached. Both are first order
    /// in the rotation of a substep, so they cancel only so far: at a few tenths of a radian a
    /// substep, or at a few hundredths for the joints below a link spinning about its own length,
    /// every substep leaves the chain with more energy than the last. This reads the balance
    /// across the whole of a substep, the move included: `free_velocity_energy_guard` reads the
    /// velocity update alone, at one configuration.
    pub(crate) fn kinetic_energy_scale(&self, held: Real, dt: Real, bodies: &RigidBodySet) -> Real {
        let mut free = self.velocities.clone();
        free.axpy(dt, &self.accelerations, 1.0);
        let (drift_energy, rest_energy, _) = self.split_kinetic_energy(free.as_view(), bodies);
        self.energy_hold_scale(drift_energy, rest_energy, held, dt, bodies)
    }

    /// The factor that brings the velocity a step ends with (`velocities`, after its last
    /// substep) down to `cap`, the larger of the kinetic energy that substep was held to and the
    /// energy it moved with, plus the work of the position forces over `dt` and a margin; 1
    /// while it stands within that.
    ///
    /// The velocity a step ends with is the one read between steps, swept by the continuous pass
    /// and the one the next step starts from. [`Self::kinetic_energy_scale`] holds the free
    /// velocity of a next substep, `velocities + dt · accelerations`, whose accelerations are
    /// solved through the semi-implicit Coriolis terms of the mass matrix: for a chain whipping
    /// under a heavy end they turn a large part of the velocity back, and the velocity itself
    /// stood well past the energy the substep was held to. A chain of twelve 4 kg links with a
    /// 500 kg end, its second link knocking on its anchor's box at 30 steps a second in 4
    /// substeps, ended a step with 25.5 kJ, held to 16.2 kJ, and its end left the step 4% faster
    /// than its fall gives it.
    pub(crate) fn step_end_energy_scale(&self, cap: Real, dt: Real, bodies: &RigidBodySet) -> Real {
        let (drift_energy, rest_energy, _) =
            self.split_kinetic_energy(self.velocities.as_view(), bodies);
        self.energy_hold_scale(drift_energy, rest_energy, cap, dt, bodies)
    }

    /// The factor `s` on a velocity that scales with `velocities`, less its drift
    /// ([`Self::scale_keeping_drift`]), that brings its kinetic energy, `drift_energy` in its
    /// drift and `rest_energy` in the rest, down to `held` plus the work of the position forces
    /// over `dt` and [`ENERGY_HOLD_MARGIN`]; 1 while it stands within that.
    ///
    /// The work's first term, `dt · v · f`, is read at the velocity the factor leaves: with `d`
    /// the drift of `v`, the scaled velocity `d + s · (v − d)` does `dt · d · f` and `s` times
    /// `dt · (v − d) · f`, so `s` solves `drift_energy + rest_energy · s² = (held + dt · d · f +
    /// s · dt · (v − d) · f + dt² · f · M⁻¹f / 2)` (with the margin). Read at `velocities` before
    /// the hold, the work of a velocity a substep had blown up allowed that velocity much of its
    /// excess: a chain ending a substep at 17.8 MJ, held to 1.4 kJ, was allowed 22.6 kJ and its
    /// 500 kg end left the step at 9.4 m/s where it hung at 2.4.
    fn energy_hold_scale(
        &self,
        drift_energy: Real,
        rest_energy: Real,
        held: Real,
        dt: Real,
        bodies: &RigidBodySet,
    ) -> Real {
        let forces = &self.workspace.position_forces;
        let mut response = forces.clone();
        self.augmented_mass_indices
            .with_rearranged_rows_mut(&mut response, |response| {
                self.inv_augmented_mass.solve_mut(response);
            });
        let drift = self.drift(self.velocities.as_view(), bodies);
        let margin = 1.0 + ENERGY_HOLD_MARGIN;
        let drift_work = dt * drift.dot(forces);
        let rest_work = dt * self.velocities.dot(forces) - drift_work;
        let response_work = 0.5 * dt * dt * response.dot(forces);
        let energy = drift_energy + rest_energy;
        let allowed = (held + (drift_work + rest_work + response_work).max(0.0)) * margin;
        if energy <= allowed || rest_energy <= 0.0 {
            return 1.0;
        }
        if rest_work <= 0.0 {
            return ((allowed - drift_energy).max(0.0) / rest_energy)
                .simd_sqrt()
                .min(1.0);
        }
        // rest_energy · s² − b · s − c = 0, for s ≥ 0.
        let b = margin * rest_work;
        let c = margin * (held + response_work.max(0.0) + drift_work) - drift_energy;
        let discriminant = b * b + 4.0 * rest_energy * c;
        if discriminant <= 0.0 {
            return (b / (2.0 * rest_energy)).clamp(0.0, 1.0);
        }
        ((b + discriminant.simd_sqrt()) / (2.0 * rest_energy)).clamp(0.0, 1.0)
    }

    // If the semi-implicit coriolis solve introduces more energy than the external
    // forces’ work, we recalculate without the coriolis term in the mass matrix.
    //
    // This operates by calculating the effective kinematic energy added by the
    // external forces after the solve with the semi-implicit mass matrix, and
    // compare it with the force’s work. If the energy delta exceeds a threshold
    // (indicating undesired/unrealistic energy injection), we fall back to re-solving
    // the forces effect but using only the mass matrix without the implicit coriolis
    // terms.
    fn free_velocity_energy_guard(&mut self, dt: Real, gen_forces: &DVector, pos_forces: &DVector) {
        let eff_dim = self.augmented_mass_indices.dim_after_removal(self.ndofs);
        if eff_dim == 0 {
            return;
        }

        // Move all quantities to the kinematic-reduced ordering of the
        // factorized matrices (no-op when there is no kinematic dof).
        let mut v = self.velocities.clone();
        let mut f = gen_forces.clone();
        let mut f_pos = pos_forces.clone();
        self.augmented_mass_indices.rearrange_rows(&mut v, true);
        self.augmented_mass_indices.rearrange_rows(&mut f, true);
        self.augmented_mass_indices.rearrange_rows(&mut f_pos, true);

        let m = self.augmented_mass.view((0, 0), (eff_dim, eff_dim));
        let wide = |x: &DVector| x.rows(0, eff_dim).map(|x| x as f64);
        let v = wide(&v);
        let f = wide(&f);
        let f_pos = wide(&f_pos);
        let dt = dt as f64;

        let energy_delta_minus_work = |accelerations: &DVector| -> Option<Real> {
            let mut a = accelerations.clone();
            self.augmented_mass_indices.rearrange_rows(&mut a, true);
            let a = wide(&a);
            if !a.iter().all(|x| x.is_finite()) {
                return None;
            }
            let m_a = m * &a;
            let energy_delta = dt * v.dot(&m_a) + 0.5 * dt * dt * a.dot(&m_a);
            let work = dt * v.dot(&f) + 0.5 * dt * dt * a.dot(&f_pos);
            // Margin: a small fraction of the current kinetic energy to absorb
            // discretization error on healthy steps.
            let kinetic_energy = 0.5 * v.dot(&(m * &v));
            let energy_margin = 1.0e-2 * kinetic_energy + 1.0e-8;
            Some((energy_delta - work - energy_margin) as Real)
        };

        match energy_delta_minus_work(&self.accelerations) {
            Some(excess) if excess <= 0.0 => (), // Implicit update didn’t introduce too much extra energy.
            _ => {
                // The implicit solve injected energy (or produced non-finite
                // values): fall back to the plain mass matrix.
                self.accelerations.copy_from(gen_forces);
                self.augmented_mass_indices.with_rearranged_rows_mut(
                    &mut self.accelerations,
                    |accs| {
                        self.inv_augmented_mass.solve_mut(accs);
                    },
                );

                // Last-chance check: if all else fails, just clear the accelerations vector
                // instead of breaking the simulation. Hopefully it will get back to a
                // sane result in a close timestep.
                if !self.accelerations.iter().all(|x| x.is_finite()) {
                    self.accelerations.fill(0.0);
                }
            }
        }
    }

    /// Computes the constant terms of the dynamics.
    #[profiling::function]
    pub(crate) fn update_velocities(&mut self, bodies: &mut RigidBodySet) {
        /*
         * Compute velocities.
         * NOTE: this is needed for kinematic bodies too.
         */
        let link = &mut self.links[0];
        let joint_velocity = link
            .joint
            .jacobian_mul_coordinates(&self.velocities.as_slice()[link.assembly_id..]);

        link.joint_velocity = joint_velocity;
        bodies.index_mut_internal(link.rigid_body).vels = link.joint_velocity;

        for i in 1..self.links.len() {
            let (link, parent_link) = self.links.get_mut_with_parent(i);
            let rb = &bodies[link.rigid_body];
            let parent_rb = &bodies[parent_link.rigid_body];

            let joint_velocity = link
                .joint
                .jacobian_mul_coordinates(&self.velocities.as_slice()[link.assembly_id..]);
            link.joint_velocity = joint_velocity.transformed(
                &(parent_link.local_to_world.rotation * link.joint.data.local_frame1.rotation),
            );
            let mut new_rb_vels = parent_rb.vels + link.joint_velocity;
            let shift = rb.mprops.world_com - parent_rb.mprops.world_com;
            new_rb_vels.linvel += parent_rb.vels.angvel.gcross(shift);
            new_rb_vels.linvel += link.joint_velocity.angvel.gcross(link.shift23);

            bodies.index_mut_internal(link.rigid_body).vels = new_rb_vels;
        }
    }

    fn update_body_jacobians(&mut self) {
        for i in 0..self.links.len() {
            let link = &self.links[i];

            if self.body_jacobians[i].ncols() != self.ndofs {
                // TODO: use a resize instead.
                self.body_jacobians[i] = Jacobian::zeros(self.ndofs);
            }

            let parent_to_world;

            if i != 0 {
                let parent_id = link.parent_internal_id;
                let parent_link = &self.links[parent_id];
                parent_to_world = parent_link.local_to_world;

                let (link_j, parent_j) = self.body_jacobians.index_mut_const(i, parent_id);
                link_j.copy_from(parent_j);

                {
                    let mut link_j_v = link_j.fixed_rows_mut::<DIM>(0);
                    let parent_j_w = parent_j.fixed_rows::<ANG_DIM>(DIM);

                    let shift_tr = vect_to_na(link.shift02).gcross_matrix_tr();
                    link_j_v.gemm(1.0, &shift_tr, &parent_j_w, 1.0);
                }
            } else {
                self.body_jacobians[i].fill(0.0);
                parent_to_world = Pose::IDENTITY;
            }

            let ndofs = link.joint.ndofs();
            let mut tmp = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
            let mut link_joint_j = tmp.columns_mut(0, ndofs);
            let mut link_j_part = self.body_jacobians[i].columns_mut(link.assembly_id, ndofs);
            link.joint.jacobian(
                &(parent_to_world.rotation * link.joint.data.local_frame1.rotation),
                &mut link_joint_j,
            );
            link_j_part += link_joint_j;

            {
                let link_j = &mut self.body_jacobians[i];
                let (mut link_j_v, link_j_w) =
                    link_j.rows_range_pair_mut(0..DIM, DIM..DIM + ANG_DIM);
                let shift_tr = vect_to_na(link.shift23).gcross_matrix_tr();
                link_j_v.gemm(1.0, &shift_tr, &link_j_w, 1.0);
            }
        }
    }

    pub(crate) fn update_mass_matrix(&mut self, dt: Real, bodies: &RigidBodySet) {
        if self.ndofs == 0 {
            return; // Nothing to do.
        }

        if self.augmented_mass.ncols() != self.ndofs {
            // TODO: do a resize instead of a full reallocation.
            self.augmented_mass = DMatrix::zeros(self.ndofs, self.ndofs);
            self.acc_augmented_mass = DMatrix::zeros(self.ndofs, self.ndofs);
        } else {
            self.augmented_mass.fill(0.0);
            self.acc_augmented_mass.fill(0.0);
        }

        self.augmented_mass_indices.clear();

        // Resize coriolis workspaces if the link count or number of DOFs change.
        let coriolis_ndofs = self.coriolis_v.first().map(|m| m.ncols());
        if self.coriolis_v.len() != self.links.len() || coriolis_ndofs != Some(self.ndofs) {
            self.coriolis_v = vec![OMatrix::<Real, Dim, Dyn>::zeros(self.ndofs); self.links.len()];
            self.coriolis_w =
                vec![OMatrix::<Real, AngDim, Dyn>::zeros(self.ndofs); self.links.len()];
            self.i_coriolis_dt = Jacobian::zeros(self.ndofs);
        }

        let mut curr_assembly_id = 0;

        for i in 0..self.links.len() {
            let link = &self.links[i];
            let rb = &bodies[link.rigid_body];
            let rb_mass = rb.mprops.effective_mass();
            let rb_inertia = rb.mprops.effective_angular_inertia().into_matrix();
            let body_jacobian = &self.body_jacobians[i];

            // NOTE: the mass matrix index reordering operates on the assumption that the assembly
            //       ids are traversed in order. This assert is here to ensure the assumption always
            //       hold.
            assert_eq!(
                curr_assembly_id, link.assembly_id,
                "Internal error: contiguity assumption on assembly_id does not hold."
            );
            curr_assembly_id += link.joint.ndofs();
            // A link's body jacobian and its Coriolis terms read the velocities of its own joint
            // and of the joints it hangs from, which come before it in assembly order: its part
            // of the mass matrix spans the degrees of freedom up to its own.
            let width = curr_assembly_id;

            if link.joint.kinematic {
                for k in link.assembly_id..link.assembly_id + link.joint.ndofs() {
                    self.augmented_mass_indices.remove(k);
                }
            } else {
                for k in link.assembly_id..link.assembly_id + link.joint.ndofs() {
                    self.augmented_mass_indices.keep(k);
                }
            }

            #[allow(unused_mut)] // mut is needed for 3D but not for 2D.
            let mut augmented_inertia = rb_inertia;

            #[cfg(feature = "dim3")]
            {
                // Derivative of gyroscopic forces.
                let gyroscopic_matrix = rb.vels.angvel.gcross_matrix() * rb_inertia
                    - (rb_inertia * rb.vels.angvel).gcross_matrix();

                augmented_inertia += gyroscopic_matrix * dt;
            }

            // TODO: optimize that (knowing the structure of the augmented inertia matrix).
            // TODO: this could be better optimized in 2D.
            #[allow(clippy::useless_conversion)] // Needed in 3D, no-op in 2D
            let rb_mass_matrix_wo_gyro = concat_rb_mass_matrix(rb_mass, rb_inertia.into());
            #[allow(clippy::useless_conversion)] // Needed in 3D, no-op in 2D
            let rb_mass_matrix = concat_rb_mass_matrix(rb_mass, augmented_inertia.into());
            if width > 0 {
                let body_jacobian = body_jacobian.columns(0, width).map(|x| x as f64);
                self.augmented_mass
                    .view_mut((0, 0), (width, width))
                    .quadform(
                        1.0,
                        &rb_mass_matrix_wo_gyro.map(|x| x as f64),
                        &body_jacobian,
                        1.0,
                    );
                self.acc_augmented_mass
                    .view_mut((0, 0), (width, width))
                    .quadform(1.0, &rb_mass_matrix.map(|x| x as f64), &body_jacobian, 1.0);
            }

            /*
             *
             * Coriolis matrix.
             *
             */
            let rb_j = &self.body_jacobians[i];
            let rb_j_w = rb_j.fixed_rows::<ANG_DIM>(DIM);

            let ndofs = link.joint.ndofs();

            if i != 0 {
                let parent_id = link.parent_internal_id;
                let parent_link = &self.links[parent_id];
                let parent_rb = &bodies[parent_link.rigid_body];
                let parent_j = &self.body_jacobians[parent_id];
                let parent_j_w = parent_j.fixed_rows::<ANG_DIM>(DIM);
                let parent_w = SimdAngVector::<Real>::from(parent_rb.vels.angvel).gcross_matrix();

                let (coriolis_v, parent_coriolis_v) = self.coriolis_v.index_mut2(i, parent_id);
                let (coriolis_w, parent_coriolis_w) = self.coriolis_w.index_mut2(i, parent_id);

                coriolis_v.copy_from(parent_coriolis_v);
                coriolis_w.copy_from(parent_coriolis_w);

                // [c1 - c0].gcross() * (JDot + JDot/u * qdot)"
                let shift_cross_tr = vect_to_na(link.shift02).gcross_matrix_tr();
                coriolis_v.gemm(1.0, &shift_cross_tr, parent_coriolis_w, 1.0);

                // JDot (but the 2.0 originates from the sum of two identical terms in JDot and JDot/u * gdot)
                let dvel_cross = vect_to_na(
                    rb.vels.angvel.gcross(link.shift02) + 2.0 * link.joint_velocity.linvel,
                )
                .gcross_matrix_tr();
                coriolis_v.gemm(1.0, &dvel_cross, &parent_j_w, 1.0);

                // JDot/u * qdot
                coriolis_v.gemm(
                    1.0,
                    &vect_to_na(link.joint_velocity.linvel).gcross_matrix_tr(),
                    &parent_j_w,
                    1.0,
                );
                coriolis_v.gemm(1.0, &(parent_w * shift_cross_tr), &parent_j_w, 1.0);

                #[cfg(feature = "dim3")]
                {
                    let vel_wrt_joint_w = vect_to_na(link.joint_velocity.angvel).gcross_matrix();
                    coriolis_w.gemm(-1.0, &vel_wrt_joint_w, &parent_j_w, 1.0);
                }

                // JDot (but the 2.0 originates from the sum of two identical terms in JDot and JDot/u * gdot)
                if !link.joint.kinematic {
                    let mut coriolis_v_part = coriolis_v.columns_mut(link.assembly_id, ndofs);

                    let mut tmp1 = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
                    let mut rb_joint_j = tmp1.columns_mut(0, ndofs);
                    link.joint.jacobian(
                        &(parent_link.local_to_world.rotation
                            * link.joint.data.local_frame1.rotation),
                        &mut rb_joint_j,
                    );

                    let rb_joint_j_v = rb_joint_j.fixed_rows::<DIM>(0);
                    coriolis_v_part.gemm(2.0, &parent_w, &rb_joint_j_v, 1.0);

                    #[cfg(feature = "dim3")]
                    {
                        let rb_joint_j_w = rb_joint_j.fixed_rows::<ANG_DIM>(DIM);
                        let mut coriolis_w_part = coriolis_w.columns_mut(link.assembly_id, ndofs);
                        coriolis_w_part.gemm(1.0, &parent_w, &rb_joint_j_w, 1.0);
                    }
                }
            } else {
                self.coriolis_v[i].fill(0.0);
                self.coriolis_w[i].fill(0.0);
            }

            let coriolis_v = &mut self.coriolis_v[i];
            let coriolis_w = &mut self.coriolis_w[i];

            {
                // [c3 - c2].gcross() * (JDot + JDot/u * qdot)
                let shift_cross_tr = vect_to_na(link.shift23).gcross_matrix_tr();
                coriolis_v.gemm(1.0, &shift_cross_tr, coriolis_w, 1.0);

                // JDot
                let dvel_cross = vect_to_na(rb.vels.angvel.gcross(link.shift23)).gcross_matrix_tr();
                coriolis_v.gemm(1.0, &dvel_cross, &rb_j_w, 1.0);

                // JDot/u * qdot
                coriolis_v.gemm(
                    1.0,
                    &(SimdAngVector::<Real>::from(rb.vels.angvel).gcross_matrix() * shift_cross_tr),
                    &rb_j_w,
                    1.0,
                );
            }

            let coriolis_v = &mut self.coriolis_v[i];
            let coriolis_w = &mut self.coriolis_w[i];

            /*
             * Meld with the mass matrix.
             */
            {
                let mut i_coriolis_dt_v = self.i_coriolis_dt.fixed_rows_mut::<DIM>(0);
                i_coriolis_dt_v.copy_from(coriolis_v);
                let rb_mass_dt = vect_to_na(rb_mass * dt);
                i_coriolis_dt_v
                    .column_iter_mut()
                    .for_each(|mut c| c.component_mul_assign(&rb_mass_dt));
            }

            #[cfg(feature = "dim2")]
            {
                let mut i_coriolis_dt_w = self.i_coriolis_dt.fixed_rows_mut::<ANG_DIM>(DIM);
                // NOTE: this is just an axpy, but on row columns.
                i_coriolis_dt_w.zip_apply(coriolis_w, |o, x| *o = x * dt * rb_inertia);
            }
            #[cfg(feature = "dim3")]
            {
                let mut i_coriolis_dt_w = self.i_coriolis_dt.fixed_rows_mut::<ANG_DIM>(DIM);
                i_coriolis_dt_w.gemm(dt, &mat_to_na(rb_inertia), coriolis_w, 0.0);
            }

            if width > 0 {
                self.acc_augmented_mass
                    .view_mut((0, 0), (width, width))
                    .gemm_tr(
                        1.0,
                        &rb_j.columns(0, width).map(|x| x as f64),
                        &self.i_coriolis_dt.columns(0, width).map(|x| x as f64),
                        1.0,
                    );
            }
        }

        /*
         * The swing cones' stiffness across their bounds, made implicit the way a spring is:
         * `dt²·K` on the mass matrix, with `K = λ cot θ · t tᵀ` over the joint's angular dofs
         * (see `MultibodyJoint::swing_cone_stiffness_dt`). The force behind it is already the
         * cone limit's, so nothing joins the accelerations.
         */
        #[cfg(feature = "dim3")]
        for li in 0..self.links.len() {
            let link = &self.links[li];
            if link.joint.kinematic {
                continue;
            }
            let Some((t, stiffness_dt)) = link.joint.swing_cone_stiffness_dt() else {
                continue;
            };
            let base = link.assembly_id + link.joint.num_free_lin_dofs();
            let gain = stiffness_dt * dt;
            for r in 0..ANG_DIM {
                for c in 0..ANG_DIM {
                    let v = (gain * t[r] * t[c]) as f64;
                    self.acc_augmented_mass[(base + r, base + c)] += v;
                    self.augmented_mass[(base + r, base + c)] += v;
                }
            }
        }

        /*
         * Damping and armature.
         *
         * Damping is a velocity-proportional force made implicit, so it adds
         * `dt · d` to the mass-matrix diagonal. Armature is a "reflected
         * rotor inertia" (additional inertia to account for the joint’s
         * mass and geometry itself): it adds to the diagonal directly.
         */
        for i in 0..self.ndofs {
            let diag = (self.damping[i] * dt + self.armature[i]) as f64;
            self.acc_augmented_mass[(i, i)] += diag;
            self.augmented_mass[(i, i)] += diag;
        }

        // Implicit joint springs: `dt²·k` on the mass-matrix diagonal (force term in `update_acceleration`)
        // keeps a stiff spring on a low-inertia link stable where an explicit position motor injects energy.
        // The spring lives on the link's `MultibodyJoint` so it travels with the link through topology changes.
        let dt2 = dt * dt;
        for li in 0..self.links.len() {
            let mut idx = self.links[li].assembly_id;
            let locked = self.links[li].joint.data.locked_axes.bits();
            for a in 0..SPATIAL_DIM {
                if (locked >> a) & 1 == 0 {
                    let k = self.links[li].joint.spring_stiffness[a];
                    if k != 0.0 {
                        let d = (k * dt2) as f64;
                        self.acc_augmented_mass[(idx, idx)] += d;
                        self.augmented_mass[(idx, idx)] += d;
                    }
                    idx += 1;
                }
            }
        }

        let effective_dim = self
            .augmented_mass_indices
            .dim_after_removal(self.acc_augmented_mass.nrows());

        // PERF: since we clone the matrix anyway for LU, should be directly output
        //       a new matrix instead of applying permutations?
        self.augmented_mass_indices
            .rearrange_columns(&mut self.acc_augmented_mass, true);
        self.augmented_mass_indices
            .rearrange_columns(&mut self.augmented_mass, true);

        self.augmented_mass_indices
            .rearrange_rows(&mut self.acc_augmented_mass, true);
        self.augmented_mass_indices
            .rearrange_rows(&mut self.augmented_mass, true);

        // TODO: avoid allocation inside LU at each timestep.
        self.acc_inv_augmented_mass = MultibodyMassFactorization::new(
            self.acc_augmented_mass
                .view((0, 0), (effective_dim, effective_dim))
                .into_owned(),
        );
        self.inv_augmented_mass = MultibodyMassFactorization::new(
            self.augmented_mass
                .view((0, 0), (effective_dim, effective_dim))
                .into_owned(),
        );
    }

    /// Per-DoF inverse joint-space inertia `diag(M⁻¹)` at the current configuration, where `M`
    /// includes armature but excludes joint damping and springs — MuJoCo's `dof_invweight0`: the
    /// apparent inverse inertia at each DoF with all other DoFs free, including the articulated
    /// coupling. Re-runs forward kinematics and reassembles the mass matrix, so intended for
    /// occasional use (e.g. sizing `<joint springdamper>` springs at load time), not every step.
    pub fn dof_inverse_inertia(&mut self, bodies: &RigidBodySet) -> DVector {
        // Resolve the root joint type (a fixed base may still be a 6-DoF free root pre-collapse)
        // so `ndofs` is final, then assemble `M`. `dt = 0` drops the `dt·damping` and
        // `dt²·stiffness` diagonal terms, leaving exactly `M + armature`.
        self.forward_kinematics(bodies, false);
        self.update_mass_matrix(0.0, bodies);

        let n = self.ndofs;
        let mut out = DVector::zeros(n);
        if n == 0 {
            return out;
        }
        // `(M⁻¹)[i, i]` for each DoF: solve `M x = e_i` and read `x[i]`. The `inv_augmented_mass`
        // factorization lives in the kinematic-reduced ordering, so route the unit vector through
        // the same rearrangement the solver uses.
        let mut e = DVector::zeros(n);
        for i in 0..n {
            e.fill(0.0);
            e[i] = 1.0;
            self.augmented_mass_indices
                .with_rearranged_rows_mut(&mut e, |b| {
                    self.inv_augmented_mass.solve_mut(b);
                });
            out[i] = e[i];
        }
        out
    }

    /// Applies an impulse to the rigid body of link `link_id`: `linear` through its centre of
    /// mass and `angular` about it, the push a body alone would take as a change of velocity.
    ///
    /// A link's velocity is its multibody's to state, from the generalized velocities, so an
    /// impulse written to the link's rigid body is lost at the next step. This one goes into the
    /// generalized velocities instead, by `M⁻¹ Jᵀ f` at the multibody's current configuration:
    /// the link and every link it is joined to move off together, as the push moves them. It
    /// takes effect whether or not the multibody has stepped since it was built or changed. The
    /// degrees of freedom of kinematic links, whose velocities are prescribed, keep theirs.
    pub fn apply_link_impulse(
        &mut self,
        bodies: &RigidBodySet,
        link_id: usize,
        linear: Vector,
        angular: AngVector,
    ) {
        if link_id >= self.links.len() {
            return;
        }
        self.forward_kinematics(bodies, false);
        if self.ndofs == 0 {
            return;
        }
        // `dt = 0`: the plain mass matrix, with no damping or spring folded into it; an impulse
        // takes no time.
        self.update_mass_matrix(0.0, bodies);
        let impulse = Force::new(linear, angular);
        let mut change = self.body_jacobians[link_id].tr_mul(impulse.as_vector());
        self.augmented_mass_indices
            .with_rearranged_rows_mut(&mut change, |rows| {
                self.inv_augmented_mass.solve_mut(rows);
            });
        for link in self.links.iter() {
            if link.joint.kinematic {
                change
                    .rows_mut(link.assembly_id, link.joint.ndofs())
                    .fill(0.0);
            }
        }
        self.velocities += change;
    }

    /// Adds a holonomic coupling between two of this multibody's generalized
    /// coordinates (`q2 = coeff·q1 + offset`), enforced as a velocity-level
    /// equality constraint each step. See [`MultibodyDofCoupling`].
    pub fn add_dof_coupling(&mut self, coupling: MultibodyDofCoupling) {
        self.couplings.push(coupling);
    }

    /// The DoF couplings declared on this multibody.
    ///
    /// Couplings follow the multibody's topology changes (link insertions, merges and splits);
    /// a coupling is dropped once one of its DoFs no longer exists, or when its two DoFs end up
    /// in different multibodies.
    pub fn couplings(&self) -> &[MultibodyDofCoupling] {
        &self.couplings
    }

    /// Removes the `i`-th DoF coupling (in the order of [`Self::couplings`]) and returns it.
    ///
    /// Returns `None` if there is no such coupling. The couplings after it shift down by one.
    pub fn remove_dof_coupling(&mut self, i: usize) -> Option<MultibodyDofCoupling> {
        (i < self.couplings.len()).then(|| self.couplings.remove(i))
    }

    /// Keeps only the DoF couplings for which `f` returns `true`.
    pub fn retain_dof_couplings(&mut self, f: impl FnMut(&MultibodyDofCoupling) -> bool) {
        self.couplings.retain(f);
    }

    /// Removes all the DoF couplings of this multibody.
    pub fn clear_dof_couplings(&mut self) {
        self.couplings.clear();
    }

    /// The number of dry-friction rows `link_id`'s joint will emit: one per
    /// free DoF whose `frictionloss` entry is non-zero.
    pub(crate) fn num_friction_constraints(&self, link_id: usize) -> usize {
        let Some(link) = self.link(link_id) else {
            return 0;
        };
        let ndofs = link.joint().ndofs();
        (0..ndofs)
            .filter(|i| self.frictions[link.assembly_id + i] > 0.0)
            .count()
    }

    /// Number of coupling constraints "owned" by `owner_link` — couplings whose first joint
    /// (`link1`) is that link. Each coupling is generated once, by `link1` (which always has a
    /// free DoF and so is an active link in the solver island, unlike a possibly-fixed root).
    pub(crate) fn num_couplings_owned_by(&self, owner_link: usize) -> usize {
        self.couplings
            .iter()
            .filter(|c| c.link1 == owner_link)
            .count()
    }

    /// Generates the velocity constraints for the DoF couplings owned by `owner_link` into
    /// `out[..]`: each coupling `q2 = coeff·q1 + offset` is one bilateral constraint with jacobian
    /// `J = e_{q2} − coeff·e_{q1}` and a rhs pulling the position drift back to zero.
    pub(crate) fn coupling_velocity_constraints(
        &self,
        owner_link: usize,
        params: &IntegrationParameters,
        mut j_id: usize,
        jacobians: &mut DVector,
        out: &mut [GenericJointConstraint],
    ) -> usize {
        let ndofs = self.ndofs;
        let erp_inv_dt = SpringCoefficients::<Real>::joint_defaults().erp_inv_dt(params.dt);

        let mut i = 0;
        for c in self.couplings.iter().filter(|c| c.link1 == owner_link) {
            let g1 = self.links[c.link1].assembly_id + c.dof1;
            let g2 = self.links[c.link2].assembly_id + c.dof2;
            let q1 = self.links[c.link1].joint().coords()[c.axis1];
            let q2 = self.links[c.link2].joint().coords()[c.axis2];

            // Jacobian J = e_{g2} − coeff·e_{g1}, then WJ = M⁻¹ J.
            jacobians.rows_mut(j_id, ndofs * 2).fill(0.0);
            jacobians[j_id + g2] += 1.0;
            jacobians[j_id + g1] -= c.coeff;
            for k in 0..ndofs {
                jacobians[j_id + ndofs + k] = jacobians[j_id + k];
            }
            self.inv_augmented_mass
                .solve_mut(&mut jacobians.rows_mut(j_id + ndofs, ndofs));

            // lhs = Jᵀ M⁻¹ J = Σ J[k]·WJ[k]; only g1/g2 entries of J are nonzero.
            let lhs = jacobians[j_id + ndofs + g2] - c.coeff * jacobians[j_id + ndofs + g1];

            let drift = q2 - c.coeff * q1 - c.offset;

            out[i] = GenericJointConstraint {
                is_rigid_body1: false,
                solver_vel1: u32::MAX,
                ndofs1: 0,
                j_id1: 0,
                is_rigid_body2: false,
                solver_vel2: self.solver_id,
                ndofs2: ndofs,
                j_id2: j_id,
                joint_id: usize::MAX, // internal: no impulse writeback.
                impulse: 0.0,
                impulse_bounds: [-Real::MAX, Real::MAX],
                inv_lhs: crate::utils::inv(lhs),
                rhs: drift * erp_inv_dt,
                rhs_wo_bias: 0.0,
                cfm_coeff: 0.0,
                cfm_gain: 0.0,
                writeback_id: WritebackId::Dof(0),
            };

            j_id += 2 * ndofs;
            i += 1;
        }

        i
    }

    /// The generalized velocity at the multibody_joint of the given link.
    #[inline]
    pub fn joint_velocity(&self, link: &MultibodyLink) -> DVectorView<'_, Real> {
        let ndofs = link.joint().ndofs();
        DVectorView::from_slice(
            &self.velocities.as_slice()[link.assembly_id..link.assembly_id + ndofs],
            ndofs,
        )
    }

    /// The generalized accelerations of this multibodies.
    #[inline]
    pub fn generalized_acceleration(&self) -> DVectorView<'_, Real> {
        self.accelerations.rows(0, self.ndofs)
    }

    /// The generalized velocities of this multibodies.
    #[inline]
    pub fn generalized_velocity(&self) -> DVectorView<'_, Real> {
        self.velocities.rows(0, self.ndofs)
    }

    /// The body jacobian for link `link_id` calculated by the last call to [`Multibody::forward_kinematics`].
    #[inline]
    pub fn body_jacobian(&self, link_id: usize) -> &Jacobian<Real> {
        &self.body_jacobians[link_id]
    }

    /// The mutable generalized velocities of this multibodies.
    #[inline]
    pub fn generalized_velocity_mut(&mut self) -> DVectorViewMut<'_, Real> {
        self.velocities.rows_mut(0, self.ndofs)
    }

    #[inline]
    pub(crate) fn integrate(&mut self, dt: Real) {
        for rb in self.links.iter_mut() {
            rb.joint
                .integrate(dt, &self.velocities.as_slice()[rb.assembly_id..])
        }
    }

    /// Apply displacements, in generalized coordinates, to this multibody.
    ///
    /// Note this does **not** updates the link poses, only their generalized coordinates.
    /// To update the link poses and associated rigid-bodies, call [`Self::forward_kinematics`].
    pub fn apply_displacements(&mut self, disp: &[Real]) {
        for link in self.links.iter_mut() {
            link.joint.apply_displacement(&disp[link.assembly_id..])
        }
    }

    pub(crate) fn update_root_type(&mut self, bodies: &RigidBodySet, take_body_pose: bool) {
        if let Some(rb) = bodies.get(self.links[0].rigid_body) {
            if rb.is_dynamic() != self.root_is_dynamic {
                let root_pose = if take_body_pose {
                    *rb.position()
                } else {
                    self.links[0].local_to_world
                };

                if rb.is_dynamic() {
                    let free_joint = MultibodyJoint::free(root_pose);
                    let prev_root_ndofs = self.links[0].joint().ndofs();
                    self.links[0].joint = free_joint;
                    self.links[0].assembly_id = 0;
                    self.ndofs += SPATIAL_DIM;

                    self.velocities = self.velocities.clone().insert_rows(0, SPATIAL_DIM, 0.0);
                    self.damping = self.damping.clone().insert_rows(0, SPATIAL_DIM, 0.0);
                    self.armature = self.armature.clone().insert_rows(0, SPATIAL_DIM, 0.0);
                    self.frictions = self.frictions.clone().insert_rows(0, SPATIAL_DIM, 0.0);
                    self.accelerations =
                        self.accelerations.clone().insert_rows(0, SPATIAL_DIM, 0.0);

                    for link in &mut self.links[1..] {
                        link.assembly_id += SPATIAL_DIM - prev_root_ndofs;
                    }
                } else {
                    assert!(self.velocities.len() >= SPATIAL_DIM);
                    assert!(self.damping.len() >= SPATIAL_DIM);
                    assert!(self.armature.len() >= SPATIAL_DIM);
                    assert!(self.frictions.len() >= SPATIAL_DIM);
                    assert!(self.accelerations.len() >= SPATIAL_DIM);

                    let fixed_joint = MultibodyJoint::fixed(root_pose);
                    let prev_root_ndofs = self.links[0].joint().ndofs();
                    self.links[0].joint = fixed_joint;
                    self.links[0].assembly_id = 0;
                    self.ndofs -= prev_root_ndofs;

                    if self.ndofs == 0 {
                        self.velocities = DVector::zeros(0);
                        self.damping = DVector::zeros(0);
                        self.armature = DVector::zeros(0);
                        self.frictions = DVector::zeros(0);
                        self.accelerations = DVector::zeros(0);
                    } else {
                        self.velocities =
                            self.velocities.index((prev_root_ndofs.., 0)).into_owned();
                        self.damping = self.damping.index((prev_root_ndofs.., 0)).into_owned();
                        self.armature = self.armature.index((prev_root_ndofs.., 0)).into_owned();
                        self.frictions = self.frictions.index((prev_root_ndofs.., 0)).into_owned();
                        self.accelerations = self
                            .accelerations
                            .index((prev_root_ndofs.., 0))
                            .into_owned();
                    }

                    for link in &mut self.links[1..] {
                        link.assembly_id -= prev_root_ndofs;
                    }
                }

                self.root_is_dynamic = rb.is_dynamic();
            }

            // Make sure the positions are properly set to match the rigid-body’s.
            if take_body_pose {
                if self.links[0].joint.data.locked_axes.is_empty() {
                    self.links[0].joint.set_free_pos(*rb.position());
                } else {
                    self.links[0].joint.data.local_frame1 = *rb.position();
                }
            }
        }
    }

    /// Update the rigid-body poses based on this multibody joint poses.
    ///
    /// This is typically called after [`Self::forward_kinematics`] to apply the new joint poses
    /// to the rigid-bodies.
    pub fn update_rigid_bodies(&self, bodies: &mut RigidBodySet, update_mass_properties: bool) {
        self.update_rigid_bodies_internal(bodies, update_mass_properties, false, true)
    }

    pub(crate) fn update_rigid_bodies_internal(
        &self,
        bodies: &mut RigidBodySet,
        update_mass_properties: bool,
        update_next_positions_only: bool,
        change_tracking: bool,
    ) {
        // Handle the children. They all have a parent within this multibody.
        for link in self.links.iter() {
            let rb = if change_tracking {
                bodies.get_mut_internal_with_modification_tracking(link.rigid_body)
            } else {
                bodies.get_mut_internal(link.rigid_body)
            };

            if let Some(rb) = rb {
                rb.pos.next_position = link.local_to_world;

                if !update_next_positions_only {
                    rb.pos.position = link.local_to_world;
                }

                if update_mass_properties {
                    rb.mprops
                        .update_world_mass_properties(rb.body_type, &link.local_to_world);
                }
            }
        }
    }

    /// The coordinates and rotation of every link's joint, in link order: what
    /// [`Self::rewind_joints`] takes back to.
    pub(crate) fn joint_states(&self) -> impl Iterator<Item = (SpatialVector, Rotation)> + '_ {
        self.links
            .iter()
            .map(|link| (link.joint.coords, link.joint.joint_rot))
    }

    /// Puts this multibody where it stood `fraction` of the way through the motion that began
    /// with its joints at `start` (from [`Self::joint_states`], one entry per link): every
    /// joint below the root goes back to that share of its motion, a dynamic root takes the
    /// pose its body's `next_position` holds, and every link's body ends the motion where the
    /// multibody then stands (`next_position`), with the velocity its joints give it there.
    ///
    /// The multibody keeps its generalized velocity through the rewind, at no more kinetic
    /// energy than it ended the motion with. The energy a generalized velocity carries depends
    /// on the configuration it is read at: the joints of a chain whipping under a heavy end turn
    /// against one another, their shares of the end's velocity cancelling where the solver held
    /// the chain's energy, and the same velocity read a few hundredths of a radian back cancels
    /// far less. A chain hanging from a fixed body at 30 steps a second, its second link knocking
    /// on that body's box, ended its solve with 6 kJ of kinetic energy and started the next step,
    /// rewound by the continuous pass, with 152 kJ: its 500 kg end flung at 13.6 m/s where its
    /// fall gives it 9.2. A rewind that would raise the energy scales the velocity's relative
    /// motion ([`Self::scale_keeping_drift`]) down to the energy the motion ended with.
    pub(crate) fn rewind_joints(
        &mut self,
        bodies: &mut RigidBodySet,
        start: &[(SpatialVector, Rotation)],
        fraction: Real,
    ) {
        let ended =
            (self.ndofs > 0).then(|| self.kinetic_energy_at(self.velocities.as_view(), bodies));
        // The configuration the last step ended at is gone.
        self.workspace.step_end = None;
        for (link, start) in self.links.iter_mut().zip(start).skip(1) {
            link.joint.interpolate_from(*start, fraction);
        }
        if self.root_is_dynamic {
            let root_pose = bodies[self.links[0].rigid_body].pos.next_position;
            let root = &mut self.links[0].joint;
            if root.data.locked_axes.is_empty() {
                root.set_free_pos(root_pose);
            } else {
                root.data.local_frame1 = root_pose;
            }
        }
        self.forward_kinematics(bodies, false);
        self.update_rigid_bodies_internal(bodies, true, true, false);
        if let Some(ended) = ended {
            let (drift_energy, rest_energy, _) =
                self.split_kinetic_energy(self.velocities.as_view(), bodies);
            if drift_energy + rest_energy > ended && rest_energy > 0.0 {
                let scale = ((ended - drift_energy).max(0.0) / rest_energy).simd_sqrt();
                self.scale_keeping_drift(scale, false, bodies);
            }
            self.update_velocities(bodies);
        }
    }

    // TODO: make a version that doesn’t write back to bodies and doesn’t update the jacobians
    //       (i.e. just something used by the velocity solver’s small steps).
    /// Apply forward-kinematics to this multibody.
    ///
    /// This will update the [`MultibodyLink`] pose information as wall as the body jacobians.
    /// This will also ensure that the multibody has the proper number of degrees of freedom if
    /// its root node changed between dynamic and non-dynamic.
    ///
    /// Note that this does **not** update the poses of the [`RigidBody`] attached to the joints.
    /// Run [`Self::update_rigid_bodies`] to trigger that update.
    ///
    /// This method updates `self` with the result of the forward-kinematics operation.
    /// For a non-mutable version running forward kinematics on a single link, see
    /// [`Self::forward_kinematics_single_link`].
    ///
    /// ## Parameters
    /// - `bodies`: the set of rigid-bodies.
    /// - `read_root_pose_from_rigid_body`: if set to `true`, the root joint (either a fixed joint,
    ///   or a free joint) will have its pose set to its associated-rigid-body pose. Set this to `true`
    ///   when the root rigid-body pose has been modified and needs to affect the multibody.
    pub fn forward_kinematics(
        &mut self,
        bodies: &RigidBodySet,
        read_root_pose_from_rigid_body: bool,
    ) {
        // Be sure the degrees of freedom match and take the root position if needed.
        self.update_root_type(bodies, read_root_pose_from_rigid_body);

        // Special case for the root, which has no parent.
        {
            let link = &mut self.links[0];
            link.local_to_parent = link.joint.body_to_parent();
            link.local_to_world = link.local_to_parent;
        }

        // Handle the children. They all have a parent within this multibody.
        for i in 1..self.links.len() {
            let (link, parent_link) = self.links.get_mut_with_parent(i);

            link.local_to_parent = link.joint.body_to_parent();
            link.local_to_world = parent_link.local_to_world * link.local_to_parent;

            {
                let parent_rb = &bodies[parent_link.rigid_body];
                let link_rb = &bodies[link.rigid_body];
                let c0 = parent_link.local_to_world * parent_rb.mprops.local_mprops.local_com;
                let c2 = link.local_to_world * link.joint.data.local_frame2.translation;
                let c3 = link.local_to_world * link_rb.mprops.local_mprops.local_com;

                link.shift02 = c2 - c0;
                link.shift23 = c3 - c2;
            }

            assert_eq!(
                bodies[link.rigid_body].body_type,
                RigidBodyType::Dynamic,
                "A rigid-body that is not at the root of a multibody must be dynamic."
            );
        }

        /*
         * Compute body jacobians.
         */
        self.update_body_jacobians();
    }

    /// Computes the ids of all the links between the root and the link identified by `link_id`.
    pub fn kinematic_branch(&self, link_id: usize) -> Vec<usize> {
        let mut branch = vec![]; // Perf: avoid allocation.
        let mut curr_id = Some(link_id);

        while let Some(id) = curr_id {
            branch.push(id);
            curr_id = self.links[id].parent_id();
        }

        branch.reverse();
        branch
    }

    /// Apply forward-kinematics to compute the position of a single link of this multibody.
    ///
    /// If `out_jacobian` is `Some`, this will simultaneously compute the new jacobian of this link.
    /// If `displacement` is `Some`, the generalized position considered during transform propagation
    /// is the sum of the current position of `self` and this `displacement`.
    // TODO: this shares a lot of code with `forward_kinematics` and `update_body_jacobians`, except
    //       that we are only traversing a single kinematic chain. Could this be refactored?
    pub fn forward_kinematics_single_link(
        &self,
        bodies: &RigidBodySet,
        link_id: usize,
        displacement: Option<&[Real]>,
        out_jacobian: Option<&mut Jacobian<Real>>,
    ) -> Pose {
        let branch = self.kinematic_branch(link_id);
        self.forward_kinematics_single_branch(bodies, &branch, displacement, out_jacobian)
    }

    /// Apply forward-kinematics to compute the position of a single sorted branch of this multibody.
    ///
    /// The given `branch` must have the following properties:
    /// - It must be sorted, i.e., `branch[i] < branch[i + 1]`.
    /// - All the indices must be part of the same kinematic branch.
    /// - If a link is `branch[i]`, then `branch[i - 1]` must be its parent.
    ///
    /// In general, this method shouldn’t be used directly and [`Self::forward_kinematics_single_link`]
    /// should be preferred since it computes the branch indices automatically.
    ///
    /// If you want to calculate the branch indices manually, see [`Self::kinematic_branch`].
    ///
    /// If `out_jacobian` is `Some`, this will simultaneously compute the new jacobian of this branch.
    /// This represents the body jacobian for the last link in the branch.
    ///
    /// If `displacement` is `Some`, the generalized position considered during transform propagation
    /// is the sum of the current position of `self` and this `displacement`.
    // TODO: this shares a lot of code with `forward_kinematics` and `update_body_jacobians`, except
    //       that we are only traversing a single kinematic chain. Could this be refactored?
    #[profiling::function]
    pub fn forward_kinematics_single_branch(
        &self,
        bodies: &RigidBodySet,
        branch: &[usize],
        displacement: Option<&[Real]>,
        mut out_jacobian: Option<&mut Jacobian<Real>>,
    ) -> Pose {
        if let Some(out_jacobian) = out_jacobian.as_deref_mut() {
            if out_jacobian.ncols() != self.ndofs {
                *out_jacobian = Jacobian::zeros(self.ndofs);
            } else {
                out_jacobian.fill(0.0);
            }
        }

        let mut parent_link: Option<MultibodyLink> = None;

        for i in branch {
            let mut link = self.links[*i];

            if let Some(displacement) = displacement {
                link.joint
                    .apply_displacement(&displacement[link.assembly_id..]);
            }

            let parent_to_world;

            if let Some(parent_link) = parent_link {
                link.local_to_parent = link.joint.body_to_parent();
                link.local_to_world = parent_link.local_to_world * link.local_to_parent;

                {
                    let parent_rb = &bodies[parent_link.rigid_body];
                    let link_rb = &bodies[link.rigid_body];
                    let c0 = parent_link.local_to_world * parent_rb.mprops.local_mprops.local_com;
                    let c2 = link.local_to_world
                        * Vector::from(link.joint.data.local_frame2.translation);
                    let c3 = link.local_to_world * link_rb.mprops.local_mprops.local_com;

                    link.shift02 = c2 - c0;
                    link.shift23 = c3 - c2;
                }

                parent_to_world = parent_link.local_to_world;

                if let Some(out_jacobian) = out_jacobian.as_deref_mut() {
                    let (mut link_j_v, parent_j_w) =
                        out_jacobian.rows_range_pair_mut(0..DIM, DIM..DIM + ANG_DIM);
                    let shift_tr = vect_to_na(link.shift02).gcross_matrix_tr();
                    link_j_v.gemm(1.0, &shift_tr, &parent_j_w, 1.0);
                }
            } else {
                link.local_to_parent = link.joint.body_to_parent();
                link.local_to_world = link.local_to_parent;
                parent_to_world = Pose::IDENTITY;
            }

            if let Some(out_jacobian) = out_jacobian.as_deref_mut() {
                let ndofs = link.joint.ndofs();
                let mut tmp = SMatrix::<Real, SPATIAL_DIM, SPATIAL_DIM>::zeros();
                let mut link_joint_j = tmp.columns_mut(0, ndofs);
                let mut link_j_part = out_jacobian.columns_mut(link.assembly_id, ndofs);
                link.joint.jacobian(
                    &(parent_to_world.rotation * link.joint.data.local_frame1.rotation),
                    &mut link_joint_j,
                );
                link_j_part += link_joint_j;

                {
                    let (mut link_j_v, link_j_w) =
                        out_jacobian.rows_range_pair_mut(0..DIM, DIM..DIM + ANG_DIM);
                    let shift_tr = vect_to_na(link.shift23).gcross_matrix_tr();
                    link_j_v.gemm(1.0, &shift_tr, &link_j_w, 1.0);
                }
            }

            parent_link = Some(link);
        }

        parent_link
            .map(|link| link.local_to_world)
            .unwrap_or(Pose::IDENTITY)
    }

    /// The total number of freedoms of this multibody.
    #[inline]
    pub fn ndofs(&self) -> usize {
        self.ndofs
    }

    pub(crate) fn fill_jacobians(
        &self,
        link_id: usize,
        unit_force: Vector,
        unit_torque: AngVector,
        j_id: &mut usize,
        jacobians: &mut DVector,
    ) -> (Real, Real) {
        if self.ndofs == 0 {
            return (0.0, 0.0);
        }

        let wj_id = *j_id + self.ndofs;
        let force = Force {
            linear: unit_force,
            #[cfg(feature = "dim2")]
            angular: unit_torque,
            #[cfg(feature = "dim3")]
            angular: unit_torque,
        };

        let link = &self.links[link_id];
        let mut out_j = jacobians.rows_mut(*j_id, self.ndofs);
        self.body_jacobians[link.internal_id].tr_mul_to(force.as_vector(), &mut out_j);

        // TODO: Optimize with a copy_nonoverlapping?
        for i in 0..self.ndofs {
            jacobians[wj_id + i] = jacobians[*j_id + i];
        }

        {
            let mut out_invm_j = jacobians.rows_mut(wj_id, self.ndofs);
            self.augmented_mass_indices
                .with_rearranged_rows_mut(&mut out_invm_j, |out_invm_j| {
                    self.inv_augmented_mass.solve_mut(out_invm_j);
                });
        }

        let j = jacobians.rows(*j_id, self.ndofs);
        let invm_j = jacobians.rows(wj_id, self.ndofs);
        *j_id += self.ndofs * 2;

        (j.dot(&invm_j), j.dot(&self.generalized_velocity()))
    }

    /// Fills `jacobians` with the relative jacobian `J = J2ᵀ·f2 − J1ᵀ·f1` of two links of `self`, then `M⁻¹·J`
    /// (e.g. a loop closure). The difference must be explicit: per-link blocks lose the `J1ᵀ·W·J2` effective-mass
    /// coupling since both act on the same generalized velocities. Cancellation-vanished rows are zeroed so the solver skips them.
    pub(crate) fn fill_relative_jacobians(
        &self,
        link_id1: usize,
        unit_force1: Vector,
        unit_torque1: AngVector,
        link_id2: usize,
        unit_force2: Vector,
        unit_torque2: AngVector,
        j_id: &mut usize,
        jacobians: &mut DVector,
    ) {
        if self.ndofs == 0 {
            return;
        }

        let wj_id = *j_id + self.ndofs;
        let force1 = Force {
            linear: unit_force1,
            angular: unit_torque1,
        };
        let force2 = Force {
            linear: unit_force2,
            angular: unit_torque2,
        };

        let link1 = &self.links[link_id1];
        let link2 = &self.links[link_id2];

        {
            let jb1 = &self.body_jacobians[link1.internal_id];
            let jb2 = &self.body_jacobians[link2.internal_id];

            // Use the (overwritten below) W·J slot as workspace for J1ᵀ·f1.
            let (mut out_j, mut workspace) =
                jacobians.rows_range_pair_mut(*j_id..*j_id + self.ndofs, wj_id..wj_id + self.ndofs);
            jb2.tr_mul_to(force2.as_vector(), &mut out_j);
            jb1.tr_mul_to(force1.as_vector(), &mut workspace);
            out_j.axpy(-1.0, &workspace, 1.0);

            // Cancellation guard: the reference scale is the magnitude of the dot-product operands,
            // not their results (which may be pure cancellation noise when the direction isn’t
            // expressible by the dofs). Rows below ~1000·ε times that scale are noise, not a constraint.
            let scale_sq = jb1.norm_squared() * force1.as_vector().norm_squared()
                + jb2.norm_squared() * force2.as_vector().norm_squared();
            let eps = Real::EPSILON * 1.0e3;
            if out_j.norm_squared() <= eps * eps * scale_sq {
                out_j.fill(0.0);
            }
        }

        // TODO: Optimize with a copy_nonoverlapping?
        for i in 0..self.ndofs {
            jacobians[wj_id + i] = jacobians[*j_id + i];
        }

        {
            let mut out_invm_j = jacobians.rows_mut(wj_id, self.ndofs);
            self.augmented_mass_indices
                .with_rearranged_rows_mut(&mut out_invm_j, |out_invm_j| {
                    self.inv_augmented_mass.solve_mut(out_invm_j);
                });
        }

        *j_id += self.ndofs * 2;
    }
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
struct IndexSequence {
    first_to_remove: u32,
    index_map: Vec<usize>,
}

impl IndexSequence {
    const NONE: u32 = u32::MAX;

    fn new() -> Self {
        Self {
            first_to_remove: Self::NONE,
            index_map: vec![],
        }
    }

    /// Index of the first removed dof, assuming there is one.
    fn start(&self) -> usize {
        self.first_to_remove as usize
    }

    fn clear(&mut self) {
        self.first_to_remove = Self::NONE;
        self.index_map.clear();
    }

    fn keep(&mut self, i: usize) {
        if self.first_to_remove == Self::NONE {
            // Nothing got removed yet. No need to register any
            // special indexing.
            return;
        }

        self.index_map.push(i);
    }

    fn remove(&mut self, i: usize) {
        if self.first_to_remove == Self::NONE {
            self.first_to_remove = i as u32;
        }
    }

    fn dim_after_removal(&self, original_dim: usize) -> usize {
        if self.first_to_remove == Self::NONE {
            original_dim
        } else {
            self.start() + self.index_map.len()
        }
    }

    fn rearrange_columns<T: na::Scalar + Copy + num::Zero, R: na::Dim, C: na::Dim, S: StorageMut<T, R, C>>(
        &self,
        mat: &mut na::Matrix<T, R, C, S>,
        clear_removed: bool,
    ) {
        if self.first_to_remove == Self::NONE {
            // Nothing to rearrange.
            return;
        }

        for (target_shift, source) in self.index_map.iter().enumerate() {
            let target = self.start() + target_shift;
            let (mut target_col, source_col) = mat.columns_range_pair_mut(target, *source);
            target_col.copy_from(&source_col);
        }

        if clear_removed {
            mat.columns_range_mut(self.start() + self.index_map.len()..)
                .fill(T::zero());
        }
    }

    fn rearrange_rows<T: na::Scalar + Copy + num::Zero, R: na::Dim, C: na::Dim, S: StorageMut<T, R, C>>(
        &self,
        mat: &mut na::Matrix<T, R, C, S>,
        clear_removed: bool,
    ) {
        if self.first_to_remove == Self::NONE {
            // Nothing to rearrange.
            return;
        }

        for mut col in mat.column_iter_mut() {
            for (target_shift, source) in self.index_map.iter().enumerate() {
                let target = self.start() + target_shift;
                col[target] = col[*source];
            }

            if clear_removed {
                col.rows_range_mut(self.start() + self.index_map.len()..)
                    .fill(T::zero());
            }
        }
    }

    fn inv_rearrange_rows<R: na::Dim, C: na::Dim, S: StorageMut<Real, R, C>>(
        &self,
        mat: &mut na::Matrix<Real, R, C, S>,
    ) {
        if self.first_to_remove == Self::NONE {
            // Nothing to rearrange.
            return;
        }

        for mut col in mat.column_iter_mut() {
            for (target_shift, source) in self.index_map.iter().enumerate().rev() {
                let target = self.start() + target_shift;
                col[*source] = col[target];
                col[target] = 0.0;
            }
        }
    }

    fn with_rearranged_rows_mut<C: na::Dim, S: StorageMut<Real, Dyn, C>>(
        &self,
        mat: &mut na::Matrix<Real, Dyn, C, S>,
        mut f: impl FnMut(&mut na::MatrixViewMut<Real, Dyn, C, S::RStride, S::CStride>),
    ) {
        self.rearrange_rows(mat, true);
        let effective_dim = self.dim_after_removal(mat.nrows());
        if effective_dim > 0 {
            f(&mut mat.rows_mut(0, effective_dim));
        }
        self.inv_rearrange_rows(mat);
    }
}

#[cfg(test)]
mod test {
    use super::IndexSequence;
    use crate::alloc_prelude::*;
    use crate::dynamics::{ImpulseJointSet, IslandManager};
    #[cfg(feature = "dim3")]
    use crate::math::Vector;
    use crate::math::{Real, SPATIAL_DIM};
    use crate::prelude::{
        ColliderSet, MultibodyJointHandle, MultibodyJointSet, RevoluteJoint, RigidBodyBuilder,
        RigidBodySet, SoftBodySet,
    };
    use na::{DVector, RowDVector};

    #[test]
    fn test_multibody_append() {
        let mut bodies = RigidBodySet::new();
        let mut joints = MultibodyJointSet::new();

        let a = bodies.insert(RigidBodyBuilder::dynamic());
        let b = bodies.insert(RigidBodyBuilder::dynamic());
        let c = bodies.insert(RigidBodyBuilder::dynamic());
        let d = bodies.insert(RigidBodyBuilder::dynamic());

        #[cfg(feature = "dim2")]
        let joint = RevoluteJoint::new();
        #[cfg(feature = "dim3")]
        let joint = RevoluteJoint::new(Vector::X);

        let mb_handle = joints.insert(a, b, joint, true).unwrap();
        joints.insert(c, d, joint, true).unwrap();
        joints.insert(b, c, joint, true).unwrap();

        assert_eq!(joints.get(mb_handle).unwrap().0.ndofs, SPATIAL_DIM + 3);
    }

    #[test]
    fn test_multibody_insert() {
        let mut rnd = oorandom::Rand32::new(1234);

        for k in 0..10 {
            let mut bodies = RigidBodySet::new();
            let mut multibody_joints = MultibodyJointSet::new();

            let num_links = 100;
            let mut handles = vec![];

            for _ in 0..num_links {
                handles.push(bodies.insert(RigidBodyBuilder::dynamic()));
            }

            let mut insertion_id: Vec<_> = (0..num_links - 1).collect();

            #[cfg(feature = "dim2")]
            let joint = RevoluteJoint::new();
            #[cfg(feature = "dim3")]
            let joint = RevoluteJoint::new(Vector::X);

            match k {
                0 => {} // Remove in insertion order.
                1 => {
                    // Remove from leaf to root.
                    insertion_id.reverse();
                }
                _ => {
                    // Shuffle the vector a bit.
                    // (This test checks multiple shuffle arrangements due to k > 2).
                    for l in 0..num_links - 1 {
                        insertion_id.swap(l, rnd.rand_range(0..num_links as u32 - 1) as usize);
                    }
                }
            }

            let mut mb_handle = MultibodyJointHandle::invalid();
            for i in insertion_id {
                mb_handle = multibody_joints
                    .insert(handles[i], handles[i + 1], joint, true)
                    .unwrap();
            }

            assert_eq!(
                multibody_joints.get(mb_handle).unwrap().0.ndofs,
                SPATIAL_DIM + num_links - 1
            );
        }
    }

    #[test]
    fn test_multibody_remove() {
        let mut rnd = oorandom::Rand32::new(1234);

        for k in 0..10 {
            let mut bodies = RigidBodySet::new();
            let mut multibody_joints = MultibodyJointSet::new();
            let mut colliders = ColliderSet::new();
            let mut impulse_joints = ImpulseJointSet::new();
            let mut islands = IslandManager::new();
            let mut soft_bodies = SoftBodySet::new();

            let num_links = 100;
            let mut handles = vec![];

            for _ in 0..num_links {
                handles.push(bodies.insert(RigidBodyBuilder::dynamic()));
            }

            #[cfg(feature = "dim2")]
            let joint = RevoluteJoint::new();
            #[cfg(feature = "dim3")]
            let joint = RevoluteJoint::new(Vector::X);

            for i in 0..num_links - 1 {
                multibody_joints
                    .insert(handles[i], handles[i + 1], joint, true)
                    .unwrap();
            }

            match k {
                0 => {} // Remove in insertion order.
                1 => {
                    // Remove from leaf to root.
                    handles.reverse();
                }
                _ => {
                    // Shuffle the vector a bit.
                    // (This test checks multiple shuffle arrangements due to k > 2).
                    for l in 0..num_links {
                        handles.swap(l, rnd.rand_range(0..num_links as u32) as usize);
                    }
                }
            }

            for handle in handles {
                bodies.remove(
                    handle,
                    &mut islands,
                    &mut colliders,
                    &mut impulse_joints,
                    &mut multibody_joints,
                    &mut soft_bodies,
                    true,
                );
            }
        }
    }

    fn test_sequence() -> IndexSequence {
        let mut seq = IndexSequence::new();
        seq.remove(2);
        seq.remove(3);
        seq.remove(4);
        seq.keep(5);
        seq.keep(6);
        seq.remove(7);
        seq.keep(8);
        seq
    }

    #[test]
    fn index_sequence_rearrange_columns() {
        let seq = test_sequence();
        let mut vec = RowDVector::from_fn(10, |_, c| c as Real);
        seq.rearrange_columns(&mut vec, true);
        assert_eq!(
            vec,
            RowDVector::from(vec![0.0, 1.0, 5.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
    }

    #[test]
    fn index_sequence_rearrange_rows() {
        let seq = test_sequence();
        let mut vec = DVector::from_fn(10, |r, _| r as Real);
        seq.rearrange_rows(&mut vec, true);
        assert_eq!(
            vec,
            DVector::from(vec![0.0, 1.0, 5.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        seq.inv_rearrange_rows(&mut vec);
        assert_eq!(
            vec,
            DVector::from(vec![0.0, 1.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 8.0, 0.0])
        );
    }

    #[test]
    fn index_sequence_with_rearranged_rows_mut() {
        let seq = test_sequence();
        let mut vec = DVector::from_fn(10, |r, _| r as Real);
        seq.with_rearranged_rows_mut(&mut vec, |v| {
            assert_eq!(v.len(), 5);
            assert_eq!(*v, DVector::from(vec![0.0, 1.0, 5.0, 6.0, 8.0]));
            *v *= 10.0;
        });
        assert_eq!(
            vec,
            DVector::from(vec![0.0, 10.0, 0.0, 0.0, 0.0, 50.0, 60.0, 0.0, 80.0, 0.0])
        );
    }
}
