//! Accessors and simple setters of a soft body: particles, elements, material, volume preservation, sleeping and enabling.
#[cfg(feature = "dim3")]
use super::SoftBodyDihedral;
#[cfg(feature = "fem")]
use super::SoftBodySolver;
use super::{
    SoftBody, SoftBodyCell, SoftBodyCellModel, SoftBodyEdge, SoftBodyMaterial, SoftBodyParticle,
    SoftVolumePiece,
};
use crate::alloc_prelude::*;
use crate::dynamics::RigidBodyHandle;
use crate::math::{DIM, Real, Vector};

impl SoftBody {
    /// A counter bumped whenever the topology changes (a tear removed elements or duplicated
    /// particles): a renderer keying its mesh on the particles and elements rebuilds it when
    /// this value changes.
    pub fn topology_version(&self) -> u32 {
        self.topology_version
    }

    /// The particles of this soft body.
    pub fn particles(&self) -> &[SoftBodyParticle] {
        &self.particles
    }

    /// The number of particles of this soft body.
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    /// Have the solver keep the velocity and position the given particles end each substep of a
    /// step with, read back through [`Self::substep_particles`] after the step, and the body keep
    /// the velocity the impulses applied to them between two steps give them
    /// ([`Self::impulse_particles`], starting from zero). The indices are kept sorted and unique;
    /// an empty list stops the recording and drops what was kept. A tear or a cut that renumbers
    /// the particles of a piece split off clears that piece's list.
    pub fn set_recorded_particles(&mut self, mut particles: Vec<u32>) {
        particles.sort_unstable();
        particles.dedup();
        if particles.is_empty() {
            self.substep_particles = Vec::new();
            self.substep_particles_dt = 0.0;
        }
        self.impulse_particles = vec![Vector::ZERO; particles.len()];
        self.recorded_particles = particles;
    }

    /// The velocity the impulses applied since the last [`Self::clear_impulse_particles`] gave
    /// each of [`Self::recorded_particles`], in the same order ([`Self::apply_impulse`],
    /// [`Self::apply_particle_impulse`], [`Self::apply_impulse_at_point`],
    /// [`Self::apply_radial_impulse`]): a particle's velocity less its entry is the velocity it
    /// had before those pushes. A velocity set since ([`Self::set_particle_velocity`], pinning
    /// the particle, placing it, the body falling asleep) is where the particle starts again, so
    /// it zeroes the particle's entry.
    pub fn impulse_particles(&self) -> &[Vector] {
        &self.impulse_particles
    }

    /// Zero what [`Self::impulse_particles`] holds, so it counts the impulses applied from here
    /// on.
    pub fn clear_impulse_particles(&mut self) {
        self.impulse_particles.fill(Vector::ZERO);
    }

    /// Add the velocity change `dv` an impulse gave the `i`-th particle to its entry of
    /// [`Self::impulse_particles`], when the particle is recorded.
    pub(crate) fn note_particle_push(recorded: &[u32], pushes: &mut [Vector], i: usize, dv: Vector) {
        if recorded.is_empty() {
            return;
        }
        if let Ok(column) = recorded.binary_search(&(i as u32))
            && let Some(push) = pushes.get_mut(column)
        {
            *push += dv;
        }
    }

    /// Zero the `i`-th particle's entry of [`Self::impulse_particles`]: its velocity was set.
    pub(crate) fn forget_particle_push(&mut self, i: usize) {
        if self.recorded_particles.is_empty() {
            return;
        }
        if let Ok(column) = self.recorded_particles.binary_search(&(i as u32))
            && let Some(push) = self.impulse_particles.get_mut(column)
        {
            *push = Vector::ZERO;
        }
    }

    /// The particles whose substep states the solver keeps ([`Self::set_recorded_particles`]),
    /// sorted.
    pub fn recorded_particles(&self) -> &[u32] {
        &self.recorded_particles
    }

    /// How the recorded particles ended each substep of the last step the solver moved this body
    /// in: their velocities and positions substep-major (`states[k * recorded + r]` is
    /// [`Self::recorded_particles`]`[r]` after substep `k`, the last substep's after linear
    /// damping, which the step applies once and the states spread evenly: substep `k` of `n`
    /// carries `k / n` of it), and the length of one substep. The states hold as many substeps as the step ran
    /// for the body's island (the world's solver iterations plus the extra ones the island asked
    /// for, a hit soft body's impact and load substeps included), and stay as the solver left them
    /// until it moves the body again or [`Self::clear_substep_particles`] empties them: a sleeping
    /// body records nothing.
    pub fn substep_particles(&self) -> (&[(Vector, Vector)], Real) {
        (&self.substep_particles, self.substep_particles_dt)
    }

    /// Empty what [`Self::substep_particles`] holds, so a read after the next step tells a step
    /// that moved this body from one that did not.
    pub fn clear_substep_particles(&mut self) {
        self.substep_particles.clear();
        self.substep_particles_dt = 0.0;
    }

    /// The world-space position of the `i`-th particle.
    pub fn particle_position(&self, i: usize) -> Vector {
        self.particles[i].position
    }

    /// The world-space positions of every particle.
    pub fn particle_positions(&self) -> impl ExactSizeIterator<Item = Vector> + '_ {
        self.particles.iter().map(|p| p.position)
    }

    /// The world-space velocity of the `i`-th particle.
    pub fn particle_velocity(&self, i: usize) -> Vector {
        self.particles[i].velocity
    }

    /// The world-space velocities of every particle.
    pub fn particle_velocities(&self) -> impl ExactSizeIterator<Item = Vector> + '_ {
        self.particles.iter().map(|p| p.velocity)
    }

    /// Teleports the `i`-th particle to `position` (no velocity change). For a pinned particle
    /// moving along a path, prefer [`Self::set_particle_kinematic_target`], which gives it the
    /// matching velocity so friction and contacts see the motion.
    pub fn set_particle_position(&mut self, i: usize, position: Vector) {
        self.particles[i].position = position;
        self.particles[i].next_position = None;
        self.positions_modified = true;
        self.modified = true;
    }

    /// Sets the velocity of the `i`-th particle. A pinned particle keeps moving at that velocity
    /// (a velocity-driven kinematic particle) until it is set again or given a kinematic target.
    pub fn set_particle_velocity(&mut self, i: usize, velocity: Vector) {
        self.particles[i].velocity = velocity;
        self.particles[i].next_position = None;
        self.forget_particle_push(i);
        self.modified = true;
    }

    /// Clears what the solver carries from one step to the next: every element's warm-start
    /// impulse and smoothed load, the attachments' and volume pieces' warm-start impulses, each
    /// cell's and cluster's rotation (back to the identity a built body starts from), the
    /// clusters' shape-matching history, and the contact load and approach speeds the next step
    /// sizes its substeps by. The body then steps from its particles, rest shape and pins alone,
    /// the way a body just built at those particles does, so two bodies set to the same
    /// particles step the same way whatever each went through before.
    pub fn forget_solver_history(&mut self) {
        for e in &mut self.edges {
            e.impulse = 0.0;
            e.stress = 0.0;
            e.force = 0.0;
        }
        #[cfg(feature = "dim3")]
        for d in &mut self.dihedrals {
            d.impulse = 0.0;
        }
        for c in &mut self.cells {
            c.impulses = Default::default();
            c.stress = 0.0;
            c.rotation = crate::math::Rotation::IDENTITY;
        }
        for piece in &mut self.volume_pieces {
            piece.impulse = 0.0;
        }
        for attachment in &mut self.attachments {
            attachment.impulse = Vector::ZERO;
        }
        for cluster in &mut self.clusters {
            cluster.rotation = crate::math::Rotation::IDENTITY;
            cluster.shape_matching_target = None;
            cluster.prev_shape_matching_target = None;
            cluster.last_gather = Default::default();
            cluster.shape_impulses.clear();
        }
        self.contact_approach_speeds = [None; 2];
        self.contact_approach_step_open = false;
        self.contact_load = 0.0;
        self.load_level = 0;
        self.sleep_speed = 0.0;
        self.sleep_reference.clear();
        self.modified = true;
    }

    /// Moves the pinned `i`-th particle to `position` over the next step (with the matching
    /// velocity, so dragged cloth and friction see the motion), then holds it there until the next
    /// target or velocity is set. Ignored for a free particle.
    pub fn set_particle_kinematic_target(&mut self, i: usize, position: Vector) {
        let particle = &mut self.particles[i];
        if particle.inv_mass == 0.0 {
            particle.next_position = Some(position);
            // A target away from the particle is a motion: the body must be awake to follow it.
            if position != particle.position {
                self.modified = true;
            }
        }
    }

    /// The distance constraints of this soft body.
    pub fn edges(&self) -> &[SoftBodyEdge] {
        &self.edges
    }

    /// The dihedral bending constraints of this soft body.
    #[cfg(feature = "dim3")]
    pub fn dihedrals(&self) -> &[SoftBodyDihedral] {
        &self.dihedrals
    }

    /// The simplex cells (triangles in 2D, tetrahedra in 3D) of this soft body.
    pub fn cells(&self) -> &[SoftBodyCell] {
        &self.cells
    }

    /// The boundary of this soft body's computational mesh: segments in 2D, triangles in 3D
    /// (particle indices), oriented outward. This is what the volume constraint integrates over; the
    /// geometry the body collides through is its [`crate::dynamics::SoftCollisionMesh`].
    pub fn boundary(&self) -> &[[u32; DIM]] {
        &self.boundary
    }

    /// The material (per-family stiffness and damping) of this soft body.
    pub fn material(&self) -> &SoftBodyMaterial {
        &self.material
    }

    /// Sets the material of this soft body.
    pub fn set_material(&mut self, material: SoftBodyMaterial) {
        self.material = material;
        self.modified = true;
    }

    /// Mutable access to the material of this soft body.
    pub fn material_mut(&mut self) -> &mut SoftBodyMaterial {
        self.modified = true;
        &mut self.material
    }

    /// The materials of this body's regions: an element whose `material` index is `k > 0`
    /// takes the `k - 1`-th of them, an element with index `0` the body's own [`Self::material`].
    pub fn region_materials(&self) -> &[SoftBodyMaterial] {
        &self.region_materials
    }

    /// Sets the materials of this body's regions (see [`Self::region_materials`]). An element
    /// whose index names no material takes the body's own.
    pub fn set_region_materials(&mut self, materials: Vec<SoftBodyMaterial>) {
        self.region_materials = materials;
        self.modified = true;
    }

    /// The material an element whose `material` index is `index` takes: the body's own for `0`
    /// or an index naming no region material, its region material otherwise.
    #[inline]
    pub fn element_material(&self, index: u16) -> &SoftBodyMaterial {
        if index == 0 {
            &self.material
        } else {
            self.region_materials
                .get(index as usize - 1)
                .unwrap_or(&self.material)
        }
    }

    /// Whether the body's own material or one of its region materials satisfies `test`.
    pub fn any_material(&self, test: impl Fn(&SoftBodyMaterial) -> bool) -> bool {
        test(&self.material) || self.region_materials.iter().any(test)
    }

    /// Gives every element whose particles all satisfy `selected` the material index `index`
    /// (see [`Self::region_materials`]): the edges, the cells and the dihedrals. Returns how many
    /// elements took it.
    pub fn assign_element_material(&mut self, index: u16, selected: impl Fn(u32) -> bool) -> usize {
        let mut count = 0;
        for e in &mut self.edges {
            if e.vertices.iter().all(|&v| selected(v)) {
                e.material = index;
                count += 1;
            }
        }
        for c in &mut self.cells {
            if c.vertices.iter().all(|&v| selected(v)) {
                c.material = index;
                count += 1;
            }
        }
        #[cfg(feature = "dim3")]
        for d in &mut self.dihedrals {
            if d.vertices.iter().all(|&v| selected(v)) {
                d.material = index;
                count += 1;
            }
        }
        self.modified = true;
        count
    }

    /// Sets the material index of the `i`-th edge (see [`Self::region_materials`]).
    pub fn set_edge_material(&mut self, i: usize, index: u16) {
        if let Some(e) = self.edges.get_mut(i) {
            e.material = index;
            self.modified = true;
        }
    }

    /// Sets the material index of the `i`-th dihedral (see [`Self::region_materials`]).
    #[cfg(feature = "dim3")]
    pub fn set_dihedral_material(&mut self, i: usize, index: u16) {
        if let Some(d) = self.dihedrals.get_mut(i) {
            d.material = index;
            self.modified = true;
        }
    }

    /// Sets the law of the `i`-th edge (see [`super::SoftEdgeLaw`]): its load, spring, bound,
    /// drive, material, number and break group.
    pub fn set_edge_law(&mut self, i: usize, law: &super::SoftEdgeLaw) {
        if let Some(e) = self.edges.get_mut(i) {
            law.apply(e);
            self.modified = true;
        }
    }

    /// Sets the material index of the `i`-th cell (see [`Self::region_materials`]).
    pub fn set_cell_material(&mut self, i: usize, index: u16) {
        if let Some(c) = self.cells.get_mut(i) {
            c.material = index;
            self.modified = true;
        }
    }

    /// Sets the drive of the `i`-th edge's rest length: toward `target` metres at `speed` metres
    /// per second, or no drive with `None` (see [`super::SoftEdgeActuator`]).
    pub fn set_edge_actuator(&mut self, i: usize, actuator: Option<super::SoftEdgeActuator>) {
        if let Some(e) = self.edges.get_mut(i) {
            e.actuator = actuator;
            self.modified = true;
        }
    }

    /// Sets the spring of the `i`-th edge in physical units, or the material's softness with
    /// `None` (see [`super::SoftEdgeSpring`]).
    pub fn set_edge_spring(&mut self, i: usize, spring: Option<super::SoftEdgeSpring>) {
        if let Some(e) = self.edges.get_mut(i) {
            e.spring = spring;
            self.modified = true;
        }
    }

    /// Sets the most force the `i`-th edge pushes or pulls with, in newtons (`Real::INFINITY`:
    /// no bound).
    pub fn set_edge_max_force(&mut self, i: usize, max_force: Real) {
        if let Some(e) = self.edges.get_mut(i) {
            e.max_force = max_force.max(0.0);
            self.modified = true;
        }
    }

    /// The plane the particles are held to, as its unit normal and offset (see
    /// [`Self::set_plane`]); `None` when they move freely.
    pub fn plane(&self) -> Option<(Vector, Real)> {
        self.plane
    }

    /// Holds every particle to the plane of unit normal `normal` through the points `x` with
    /// `normal · x == offset`, or frees them with `None`: each step their velocity across the
    /// plane is removed and they are put back on it, so the body moves as a flat structure (a
    /// side-on bridge, a 2D puzzle). A normal of zero length frees them.
    pub fn set_plane(&mut self, plane: Option<(Vector, Real)>) {
        self.plane = plane.and_then(|(normal, offset)| {
            let length = normal.length();
            (length > Real::EPSILON && length.is_finite() && offset.is_finite())
                .then(|| (normal / length, offset / length))
        });
        self.modified = true;
    }

    /// The constitutive model of this soft body's cells.
    pub fn cell_model(&self) -> SoftBodyCellModel {
        self.cell_model
    }

    /// Which solver simulates this soft body's elasticity.
    #[cfg(feature = "fem")]
    pub fn solver(&self) -> SoftBodySolver {
        self.solver
    }

    /// Selects the solver simulating this soft body's elasticity (see [`SoftBodySolver`]).
    #[cfg(feature = "fem")]
    pub fn set_solver(&mut self, solver: SoftBodySolver) {
        self.solver = solver;
        self.modified = true;
    }

    /// Whether this soft body's elasticity is solved by the FEM path.
    #[inline]
    pub(crate) fn uses_fem(&self) -> bool {
        #[cfg(feature = "fem")]
        {
            self.solver == SoftBodySolver::Fem
        }
        #[cfg(not(feature = "fem"))]
        {
            false
        }
    }

    /// Whether the area/volume preservation constraints are enabled (see [`Self::volume_pieces`]).
    pub fn volume_preservation_enabled(&self) -> bool {
        self.volume_preservation
    }

    /// Enables or disables area/volume preservation: one constraint per volume piece (see
    /// [`Self::volume_pieces`]); it stays disabled for a body without any.
    pub fn enable_volume_preservation(&mut self, enabled: bool) {
        self.volume_preservation = enabled && !self.volume_pieces.is_empty();
        self.modified = true;
    }

    /// The pieces of material enclosed by a closed boundary, each with its own volume constraint
    /// so squeezing one does not inflate another. Pieces joined by no element (appended bodies,
    /// torn-apart halves) are separate; the holes or cavities of a piece belong to it.
    pub fn volume_pieces(&self) -> &[SoftVolumePiece] {
        &self.volume_pieces
    }

    /// The signed area (2D) or volume (3D) enclosed by the volume pieces at rest (`0.0` for a
    /// body without a closed boundary).
    pub fn rest_volume(&self) -> Real {
        self.volume_pieces
            .iter()
            .map(|piece| piece.rest_volume)
            .sum()
    }

    /// The signed area (2D) or volume (3D) currently enclosed by the volume pieces.
    pub fn volume(&self) -> Real {
        self.volume_pieces
            .iter()
            .map(|piece| piece.volume(self))
            .sum()
    }

    /// The multiplier applied to each volume piece's rest volume to obtain its
    /// volume-preservation target (`> 1` inflates the body).
    pub fn volume_factor(&self) -> Real {
        self.volume_factor
    }

    /// Sets the multiplier applied to each volume piece's rest volume to obtain its
    /// volume-preservation target (`> 1` inflates the body).
    pub fn set_volume_factor(&mut self, factor: Real) {
        self.volume_factor = factor;
        self.modified = true;
    }

    /// The particle spacing of this soft body: what the impact-adaptive substeps bound the
    /// travel per substep by, and the default contact skin of its collision meshes' colliders.
    pub fn particle_radius(&self) -> Real {
        self.particle_radius
    }

    /// The rigid body standing for this soft body in the islands and holding the colliders of its
    /// surface (invalid until inserted in a set): joints, forces and impulses applied to it act on
    /// the body as a whole. A tear splitting the body moves each of its joints to the piece closest
    /// to its anchor in the rest shape (see [`crate::dynamics::SoftBodyTearEvent::moved_joints`]).
    /// Its pose is recomputed from the particles at every step, so moving it has no effect, and
    /// removing it dissolves its whole-body cluster like any cluster proxy.
    pub fn root_body(&self) -> RigidBodyHandle {
        self.root_body
    }

    /// The soft body this one was split off from by a tear or a cut (`None` for a user-inserted
    /// body): the piece that kept the handle (see [`crate::dynamics::SoftBodyTearEvent::pieces`]).
    /// Informational only; the handle dangles once that body is removed.
    pub fn origin(&self) -> Option<crate::dynamics::SoftBodyHandle> {
        self.origin
    }

    /// The soft bodies split off from this one by tears and cuts, in creation order (a piece
    /// torn off a piece is listed by that piece, not here). Informational: the engine never
    /// reads it, and a handle dangles once that body is removed.
    pub fn pieces(&self) -> &[crate::dynamics::SoftBodyHandle] {
        &self.pieces
    }

    /// The dynamics settings of the particles (damping, gravity scale, sleeping...).
    pub fn particle_settings(&self) -> &super::SoftBodyParticleSettings {
        &self.particle_settings
    }

    /// Whether the particles' gravity comes from [`crate::pipeline::PhysicsHooks::gravity`]
    /// ([`Self::set_gravity_field`]).
    pub fn gravity_field(&self) -> bool {
        self.particle_settings.gravity_field
    }

    /// Take the particles' gravity from [`crate::pipeline::PhysicsHooks::gravity`] in place of
    /// the step's gravity vector: the solver asks the hooks at every substep where each free
    /// particle stands, and applies the answer times the particles' gravity scale. Takes effect at
    /// the next step; a sleeping body stays asleep until something wakes it.
    pub fn set_gravity_field(&mut self, enabled: bool) {
        self.particle_settings.gravity_field = enabled;
    }

    /// Sets the extra internal PGS iterations per substep requested for this body's island
    /// component (see [`super::SoftBodyParticleSettings::additional_pgs_iterations`]); takes
    /// effect at the next step.
    pub fn set_additional_pgs_iterations(&mut self, iterations: usize) {
        self.particle_settings.additional_pgs_iterations = iterations;
    }

    /// The number of parallel solver colors used by this soft body's elements (informational:
    /// elements of a same color share no particle and are solved concurrently).
    pub fn num_solver_colors(&self) -> usize {
        self.num_colors as usize + self.has_overflow_color as usize
    }

    /// The mass-weighted center of the particles (from the cached positions).
    pub fn center_of_mass(&self) -> Vector {
        let mut com = Vector::ZERO;
        let mut mass = 0.0;
        for p in &self.particles {
            com += p.position * p.mass;
            mass += p.mass;
        }
        if mass > 0.0 { com / mass } else { com }
    }

    /// The total nominal mass of the particles.
    pub fn mass(&self) -> Real {
        self.particles.iter().map(|p| p.mass).sum()
    }

    /// Whether this soft body is currently sleeping (a soft body sleeps and wakes as a unit,
    /// with the island of the rigid bodies it touches). Updated at the end of every step, and
    /// by [`Self::wake_up`].
    pub fn is_sleeping(&self) -> bool {
        self.sleeping
    }

    /// The speed this soft body's sleep rule read at the end of its last simulated step: the
    /// farthest any particle moved over that step, divided by the step's length (the fastest
    /// particle's velocity on a step with no earlier positions to measure from), or `Real::MAX`
    /// while the body flows plastically. The body falls asleep once this stays under its
    /// proxies' activation threshold for their time until sleep.
    pub fn sleep_speed(&self) -> Real {
        self.sleep_speed
    }

    /// Wakes this soft body up (effective at the start of the next step, together with the
    /// island of the bodies it touches).
    pub fn wake_up(&mut self) {
        self.sleeping = false;
        self.modified = true;
    }

    /// Whether this soft body takes part in the simulation. A disabled soft body (by
    /// [`Self::set_enabled`], or by the NaN quarantine: `PhysicsPipeline::quarantine`) is
    /// left where it is, without colliders or constraints, until it is enabled again.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enables or disables this soft body (effective at the start of the next step). Re-enabling
    /// a quarantined body: set its non-finite particles right first
    /// ([`Self::set_particle_position`]), they are not repaired.
    pub fn set_enabled(&mut self, enabled: bool) {
        if self.enabled != enabled {
            self.enabled = enabled;
            self.modified = true;
        }
    }

    /// Whether every particle's state is finite.
    pub(crate) fn is_finite(&self) -> bool {
        self.particles.iter().all(|p| {
            p.position.is_finite()
                && p.velocity.is_finite()
                && p.force.is_finite()
                && p.next_position.is_none_or(|t| t.is_finite())
        })
    }

    /// Neutralizes the dynamics of a non-finite body (velocities, forces, targets); the
    /// positions are left as they are.
    pub(crate) fn sanitize_dynamics(&mut self) {
        for p in &mut self.particles {
            p.velocity = Vector::ZERO;
            p.force = Vector::ZERO;
            p.next_position = None;
        }
        self.clear_impulse_particles();
    }
}
