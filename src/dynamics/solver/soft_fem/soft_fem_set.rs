//! The FEM systems of the soft bodies solved by the staged island solver: the per-substep
//! predict stage's bodies, and the responses of the step's constraints.

use super::system::SoftFemSystem;
use crate::alloc_prelude::*;
use crate::dynamics::solver::soft_constraint::SoftConstraintsSet;
use crate::dynamics::solver::soft_constraint::soft_attachment::FemAttachment;
use crate::dynamics::solver::soft_constraint::soft_constraints_set::{
    FemClusterSide, FemOverlapSide, FemVolumeSide,
};
use crate::dynamics::solver::soft_constraint::soft_contact::{CONTACT_ANCHORS, FemContactSide};
use crate::dynamics::solver::solver_body::SolverBodies;
use crate::dynamics::{SoftBody, SoftBodyHandle, SoftBodySet, SoftFemParameters};
use crate::math::{AngularInertia, DIM, Matrix, Real, SPATIAL_DIM, Vector};
use core::ops::Range;
use parry::utils::hashmap::HashMap;

/// The most multiply-adds a body's direct factorization may cost for it to be refactorized at
/// every substep (see [`SoftFemSet::refactors_every_substep`]): a chain of a few thousand
/// particles, a box of a few dozen.
const SUBSTEP_FACTOR_WORK: usize = 250_000;

/// One awake soft body solved by the FEM path, during one island solve.
struct ActiveFem {
    /// The soft body, valid for the solve's scope (same contract as `AwakeSoftBody::ptr`).
    body: *mut SoftBody,
    /// Its system, owned by `SoftFemSet::systems` and boxed so the pointer stays valid.
    system: *mut SoftFemSystem,
    /// The substep length of the body's solve group.
    dt: Real,
    /// Solver slot of the body's first particle (the particles' slots are contiguous).
    first_slot: u32,
    num_particles: u32,
    params: SoftFemParameters,
}

/// A constraint side answered by a FEM body, filled by the response stage (see
/// `SoftFemSet::plan_responses`).
#[derive(Copy, Clone, Debug)]
enum FemConstraintRef {
    /// `contacts[constraint].fem[side]` (the combined same-body response when it `covers_other`).
    Contact {
        constraint: u32,
        side: u8,
    },
    Attachment(u32),
    Shape(u32),
    /// `overlap_fem_sides[side]` of `overlap_constraints[constraint]`.
    Overlap {
        constraint: u32,
        side: u32,
    },
    /// `clusters[index]`: a joint end on a region of the body, or on its whole frame.
    Cluster(u32),
}

// SAFETY: the pointers are only dereferenced during the solve, where the stage discipline gives
// exactly one worker access to a given index (same contract as `AwakeSoftBody`).
unsafe impl Send for ActiveFem {}
unsafe impl Sync for ActiveFem {}

/// A FEM body's system as seen by the soft constraints set (`AwakeSoftBody::fem`), for the
/// constraints updating their responses per substep.
#[derive(Copy, Clone)]
pub(crate) struct AwakeFem {
    /// The body's system, valid for the solve's scope (see `ActiveFem::system`).
    pub system: *mut SoftFemSystem,
    pub params: SoftFemParameters,
}

// SAFETY: same contract as `ActiveFem`.
unsafe impl Send for AwakeFem {}
unsafe impl Sync for AwakeFem {}

/// The FEM systems of every soft body that selected the FEM solver, looked up by handle and kept
/// across steps (the sparsity pattern and element tables are the expensive part; values refill
/// every substep); `active` holds the awake ones of the island being solved, group-major.
#[derive(Default)]
pub(crate) struct SoftFemSet {
    systems: HashMap<SoftBodyHandle, Box<SoftFemSystem>>,
    active: Vec<ActiveFem>,
    /// Slice of `active` belonging to each substep group.
    groups: Vec<Range<usize>>,
    /// Index into `active` of every awake soft body (`u32::MAX`: not on the FEM path).
    fem_of_awake: Vec<u32>,
    /// Per active body, the constraint sides it answers this step.
    refs: Vec<Vec<FemConstraintRef>>,
}

impl SoftFemSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// The `active` range of a substep group.
    pub fn group(&self, group: usize) -> Range<usize> {
        self.groups
            .get(group)
            .cloned()
            .unwrap_or(self.active.len()..self.active.len())
    }

    /// Collects the island's awake FEM soft bodies (after `SoftConstraintsSet::assemble` laid out
    /// the particle slots) and updates their systems' step-constant state; the factorization runs
    /// in the parallel response stage ([`Self::compute_responses`]).
    pub fn assemble(
        &mut self,
        soft_constraints: &mut SoftConstraintsSet,
        soft_bodies: &SoftBodySet,
        num_groups: usize,
        group_dt: impl Fn(usize) -> Real,
        params: &SoftFemParameters,
    ) {
        self.active.clear();
        self.groups.clear();
        self.fem_of_awake.clear();
        self.fem_of_awake
            .resize(soft_constraints.awake.len(), u32::MAX);
        self.groups.resize(num_groups.max(1), 0..0);
        if soft_constraints.is_empty() {
            // Bodies whose soft body was removed would otherwise keep their system forever; the
            // sweep is only worth its cost once the map has grown well past what is in use.
            if self.systems.len() > 64 {
                self.systems.retain(|h, _| soft_bodies.get(*h).is_some());
            }
            return;
        }

        let mut group_starts = vec![0usize; self.groups.len() + 1];
        for awake in &soft_constraints.awake {
            // SAFETY: read-only access during assembly, same contract as the constraint assembly.
            let sb = unsafe { &*awake.ptr };
            if !awake.frozen && sb.uses_fem() {
                group_starts[awake.group as usize + 1] += 1;
            }
        }
        for k in 1..group_starts.len() {
            group_starts[k] += group_starts[k - 1];
        }
        for (g, range) in self.groups.iter_mut().enumerate() {
            *range = group_starts[g]..group_starts[g + 1];
        }

        // `awake` is already sorted by group, so a single pass keeps the group-major order.
        let slots_all = &soft_constraints.slots;
        for (ai, awake) in soft_constraints.awake.iter_mut().enumerate() {
            awake.fem = None;
            let sb = unsafe { &*awake.ptr };
            if awake.frozen || !sb.uses_fem() {
                continue;
            }
            let slots = &slots_all[awake.slot_start..awake.slot_start + awake.num_particles];
            let dt = group_dt(awake.group as usize);
            let system = self.systems.entry(awake.handle).or_default();
            system.prepare(sb, slots);
            awake.fem = Some(AwakeFem {
                system: &mut **system as *mut SoftFemSystem,
                params: *params,
            });
            self.fem_of_awake[ai] = self.active.len() as u32;
            self.active.push(ActiveFem {
                body: awake.ptr,
                system: &mut **system as *mut SoftFemSystem,
                dt,
                params: *params,
                first_slot: slots.first().copied().unwrap_or(u32::MAX),
                num_particles: awake.num_particles as u32,
            });
        }
    }

    /// The index into `active` of an awake soft body, if it is on the FEM path.
    #[inline]
    fn fem_of(&self, awake: u32) -> Option<usize> {
        if awake == u32::MAX {
            return None;
        }
        let idx = self.fem_of_awake[awake as usize];
        (idx != u32::MAX).then_some(idx as usize)
    }

    /// Plans the responses of the step's constraints on the FEM bodies, once all are assembled:
    /// picks which constraint sides a FEM body answers, applies the lever freezes, reserves the
    /// ranges in `soft.fem_responses` and records the sides [`Self::compute_responses`] fills.
    pub fn plan_responses(&mut self, soft: &mut SoftConstraintsSet) {
        soft.fem_responses.clear();
        for refs in &mut self.refs {
            refs.clear();
        }
        self.refs.resize_with(self.active.len(), Vec::new);
        if self.active.is_empty() {
            return;
        }
        let SoftConstraintsSet {
            awake,
            contacts,
            attachments,
            shape_constraints,
            volume_constraints,
            overlap_constraints,
            overlap_grads,
            overlap_warm_impulses,
            overlap_fem_sides,
            fem_responses,
            clusters,
            ..
        } = soft;
        let mut pool_len = 0u32;
        let mut reserve = |count: usize| -> u32 {
            let start = pool_len;
            pool_len += count as u32;
            start
        };

        for (constraint, c) in contacts.iter_mut().enumerate() {
            c.fem = [None, None];
            let mut support = self.fem_of(c.support_body);
            let mut other = self.fem_of(c.other_body);
            if support.is_none() && other.is_none() {
                continue;
            }
            // A side acting through pinned particles but for a sliver of weight on a free one is a
            // lever (the constraint would fling the body to move its point): frozen for the constraint, as
            // the lumped path does (`SoftContact::update`).
            if support.is_some() && is_lever(&c.weights, &c.im_particles) {
                c.im_particles = [0.0; CONTACT_ANCHORS];
                support = None;
            }
            if let (Some(e), true) = (&mut c.element, other.is_some()) {
                if is_lever(&e.weights, &e.im_particles) {
                    e.im_particles = [0.0; CONTACT_ANCHORS];
                    other = None;
                }
            }
            // A self contact (two points of one body): one combined response
            // `A⁻¹(J_support - J_other)ᵀ`, applied once; separate sides would lose the
            // `J_s A⁻¹ J_oᵀ` coupling, as `Multibody::fill_relative_jacobians` explains.
            let same_body = support.is_some() && support == other;
            for (side, fi) in [(0usize, support), (1, other)] {
                let Some(fi) = fi else {
                    continue;
                };
                if side == 1 && same_body {
                    continue;
                }
                let entry = &self.active[fi];
                let n = entry.num_particles as usize;
                c.fem[side] = Some(FemContactSide {
                    first_slot: entry.first_slot,
                    num_particles: entry.num_particles,
                    start: reserve(DIM * n),
                    gains: [0.0; DIM],
                    covers_other: side == 0 && same_body,
                });
                self.refs[fi].push(FemConstraintRef::Contact {
                    constraint: constraint as u32,
                    side: side as u8,
                });
            }
        }

        // Attachments and shape-matching constraints: one response per axis of the particle.
        for (constraint, r) in attachments.iter_mut().enumerate() {
            r.fem = None;
            let Some(fi) = self.fem_of(r.soft_body) else {
                continue;
            };
            let entry = &self.active[fi];
            if r.particle < entry.first_slot || r.particle >= entry.first_slot + entry.num_particles
            {
                continue;
            }
            r.fem = Some(FemAttachment {
                first_slot: entry.first_slot,
                num_particles: entry.num_particles,
                start: reserve(DIM * entry.num_particles as usize),
                inv_mass: Matrix::ZERO,
            });
            self.refs[fi].push(FemConstraintRef::Attachment(constraint as u32));
        }
        for (ai, body) in awake.iter().enumerate() {
            let fem = self.fem_of(ai as u32);
            for constraint in body.shape_constraints.clone() {
                let r = &mut shape_constraints[constraint];
                r.fem = None;
                let Some(fi) = fem else {
                    continue;
                };
                let entry = &self.active[fi];
                if r.solver_id < entry.first_slot
                    || r.solver_id >= entry.first_slot + entry.num_particles
                {
                    continue;
                }
                r.fem = Some(FemAttachment {
                    first_slot: entry.first_slot,
                    num_particles: entry.num_particles,
                    start: reserve(DIM * entry.num_particles as usize),
                    inv_mass: Matrix::ZERO,
                });
                self.refs[fi].push(FemConstraintRef::Shape(constraint as u32));
            }
        }

        // Active clusters (the proxies a joint, an impulse or a force acts on): one response per
        // spatial axis of the cluster's rigid fit.
        for (index, record) in clusters.iter_mut().enumerate() {
            record.fem = None;
            let Some(fi) = self.fem_of(record.awake) else {
                continue;
            };
            let entry = &self.active[fi];
            record.fem = Some(FemClusterSide {
                first_slot: entry.first_slot,
                num_particles: entry.num_particles,
                start: reserve(SPATIAL_DIM * entry.num_particles as usize),
                inv_im: Vector::ZERO,
                inv_ii: AngularInertia::default(),
            });
            self.refs[fi].push(FemConstraintRef::Cluster(index as u32));
        }

        // Volume-piece constraints: the range is reserved here, the response updated with the
        // gradients at each substep's prepare (`SoftConstraintsSet::prepare_awake_body`).
        for constraint in volume_constraints.iter_mut() {
            constraint.fem = None;
            let Some(fi) = self.fem_of(constraint.soft_body) else {
                continue;
            };
            let entry = &self.active[fi];
            constraint.fem = Some(FemVolumeSide {
                start: reserve(entry.num_particles as usize),
                #[cfg(feature = "fem")]
                gain: 0.0,
                u_max: 0.0,
            });
        }

        // Intersection-volume constraints: one response per FEM body of the patch (its gradients)
        // plus the response to the warm impulses; the body's entries then leave the lumped loops
        // (zero inverse mass), after taking the lumped gain (ill-conditioned response fallback).
        overlap_fem_sides.clear();
        for (constraint_id, constraint) in overlap_constraints.iter_mut().enumerate() {
            let sides_start = overlap_fem_sides.len() as u32;
            let grads = &mut overlap_grads[constraint.grads.clone()];
            let warm_impulses = &overlap_warm_impulses[constraint.grads.clone()];
            for (fi, entry) in self.active.iter().enumerate() {
                let in_body = |slot: u32| {
                    slot != u32::MAX
                        && slot >= entry.first_slot
                        && slot < entry.first_slot + entry.num_particles
                };
                if !grads.iter().any(|g| in_body(g.0) && g.1 > 0.0) {
                    continue;
                }
                let n = entry.num_particles as usize;
                let lumped: Real = grads
                    .iter()
                    .filter(|g| in_body(g.0))
                    .map(|g| g.1 * g.2.length_squared())
                    .sum();
                let warm_pending = constraint.warm_pending
                    && constraint.warm.is_some()
                    && grads
                        .iter()
                        .zip(warm_impulses)
                        .any(|(g, p)| in_body(g.0) && g.1 > 0.0 && *p != Vector::ZERO);
                let start = reserve(n);
                let warm = if warm_pending { reserve(n) } else { u32::MAX };
                for g in grads.iter_mut() {
                    if in_body(g.0) {
                        g.1 = 0.0;
                    }
                }
                self.refs[fi].push(FemConstraintRef::Overlap {
                    constraint: constraint_id as u32,
                    side: overlap_fem_sides.len() as u32,
                });
                overlap_fem_sides.push(FemOverlapSide {
                    first_slot: entry.first_slot,
                    num_particles: entry.num_particles,
                    start,
                    warm,
                    gain: lumped,
                    u_max: 0.0,
                });
            }
            constraint.fem_sides = sides_start..overlap_fem_sides.len() as u32;
        }
        fem_responses.resize(pool_len as usize, Vector::ZERO);
    }

    /// The response stage's work for `active[index]`: factorizes the body's step matrix and fills
    /// the responses and gains of every constraint side planned by [`Self::plan_responses`], and
    /// gives the proxy slot of each of the body's clusters that only joints act on the mass the
    /// cluster has through the body (see [`FemClusterSide`]).
    /// # Safety
    /// One worker per index per stage; the body writes only its own system, ranges, side gains
    /// and its own clusters' proxy slots in `bodies`.
    pub unsafe fn compute_responses(
        &self,
        index: usize,
        soft: *mut SoftConstraintsSet,
        bodies: *mut SolverBodies,
    ) {
        let entry = &self.active[index];
        let system = unsafe { &mut *entry.system };
        let params = &entry.params;
        system.factorize_step_matrix(entry.dt, params);
        let soft = unsafe { &mut *soft };
        let n = entry.num_particles as usize;
        let first = entry.first_slot;
        let pool = soft.fem_responses.as_mut_ptr();
        // SAFETY: the ranges were reserved for this body alone (see `plan_responses`).
        let range = |start: u32, len: usize| unsafe {
            core::slice::from_raw_parts_mut(pool.add(start as usize), len)
        };
        let in_body = |slot: u32| slot != u32::MAX && slot >= first && slot < first + n as u32;

        // The particles with contact, attachment or shape constraints: their axis responses are
        // solved once (the columns) and every constraint on them combines those (the overlap constraints'
        // patches and the volume constraints keep their own solves).
        system.clear_columns();
        for r in &self.refs[index] {
            match *r {
                FemConstraintRef::Contact { constraint, side } => {
                    let c = &soft.contacts[constraint as usize];
                    let Some(fem) = c.fem[side as usize] else {
                        continue;
                    };
                    if side == 0 {
                        for k in 0..CONTACT_ANCHORS {
                            if c.particles[k] != u32::MAX && c.weights[k] != 0.0 {
                                system.load_particle(c.support_particle[k]);
                            }
                        }
                    }
                    if side == 1 || fem.covers_other {
                        match &c.element {
                            Some(e) => {
                                for k in 0..CONTACT_ANCHORS {
                                    if in_body(e.particles[k]) && e.weights[k] != 0.0 {
                                        system.load_particle(e.particles[k] - first);
                                    }
                                }
                            }
                            None => {
                                if in_body(c.body) {
                                    system.load_particle(c.body - first);
                                }
                            }
                        }
                    }
                }
                FemConstraintRef::Attachment(constraint) => {
                    let r = &soft.attachments[constraint as usize];
                    if r.fem.is_some() {
                        system.load_particle(r.particle - first);
                    }
                }
                FemConstraintRef::Shape(constraint) => {
                    let r = &soft.shape_constraints[constraint as usize];
                    if r.fem.is_some() {
                        system.load_particle(r.solver_id - first);
                    }
                }
                FemConstraintRef::Overlap { .. } | FemConstraintRef::Cluster(_) => {}
            }
        }
        system.compute_columns(params);

        for r in &self.refs[index] {
            match *r {
                FemConstraintRef::Contact { constraint, side } => {
                    let c = &mut soft.contacts[constraint as usize];
                    let side = side as usize;
                    let Some(fem) = c.fem[side] else {
                        continue;
                    };
                    // The anchors of each side with their barycentric weights, the other
                    // side's negated (the constraint pushes the support along `+d` and the other
                    // side along `-d`).
                    let mut anchors: [(u32, Real); 2 * CONTACT_ANCHORS] =
                        [(0, 0.0); 2 * CONTACT_ANCHORS];
                    let mut num = 0;
                    let mut lumped = [0.0; DIM];
                    let support_side = side == 0;
                    if support_side {
                        for k in 0..CONTACT_ANCHORS {
                            if c.particles[k] != u32::MAX && c.weights[k] != 0.0 {
                                anchors[num] = (c.support_particle[k], c.weights[k]);
                                num += 1;
                                let w = c.weights[k] * c.weights[k] * c.im_particles[k];
                                lumped.iter_mut().for_each(|l| *l += w);
                            }
                        }
                    }
                    if !support_side || fem.covers_other {
                        match &c.element {
                            Some(e) => {
                                for k in 0..CONTACT_ANCHORS {
                                    if in_body(e.particles[k]) && e.weights[k] != 0.0 {
                                        anchors[num] = (e.particles[k] - first, -e.weights[k]);
                                        num += 1;
                                        let w = e.weights[k] * e.weights[k] * e.im_particles[k];
                                        lumped.iter_mut().for_each(|l| *l += w);
                                    }
                                }
                            }
                            None => {
                                if in_body(c.body) {
                                    let p = c.body - first;
                                    anchors[num] = (p, -1.0);
                                    num += 1;
                                    // SAFETY: read-only access to the body's particles.
                                    let im = unsafe { &*entry.body }.particles[p as usize].inv_mass;
                                    lumped.iter_mut().for_each(|l| *l += im);
                                }
                            }
                        }
                    }
                    let anchors = &anchors[..num];
                    let directions: [Vector; DIM] =
                        core::array::from_fn(|k| if k == 0 { c.dir } else { c.tangents[k - 1] });
                    let mut gains = [0.0; DIM];
                    for (k, d) in directions.iter().enumerate() {
                        let d = *d;
                        let entries = anchors.iter().map(move |&(p, w)| (p, d * w));
                        let out = range(fem.start + (k * n) as u32, n);
                        gains[k] = match system.response_from_columns(entries.clone(), out) {
                            Some(gain) => gain,
                            None => system.response_into(entries, out, params),
                        };
                    }
                    if !well_conditioned(range(fem.start, DIM * n), &gains, n) {
                        // Ill-conditioned (`MAX_RESPONSE_AMPLIFICATION`): the lumped gain (never
                        // below the augmented one, `A ≥ M`) under-relaxes it; its impulses still
                        // spread through the true response, keeping its constraints consistent.
                        gains = lumped;
                    }
                    c.fem[side].as_mut().unwrap().gains = gains;
                }
                FemConstraintRef::Attachment(constraint) => {
                    let r = &mut soft.attachments[constraint as usize];
                    let Some(fem) = r.fem.as_mut() else {
                        continue;
                    };
                    let p = r.particle - first;
                    fem.inv_mass = particle_response(system, p, range(fem.start, DIM * n), params);
                }
                FemConstraintRef::Shape(constraint) => {
                    let r = &mut soft.shape_constraints[constraint as usize];
                    let Some(fem) = r.fem.as_mut() else {
                        continue;
                    };
                    let p = r.solver_id - first;
                    fem.inv_mass = particle_response(system, p, range(fem.start, DIM * n), params);
                }
                FemConstraintRef::Overlap { constraint, side } => {
                    let oc = &soft.overlap_constraints[constraint as usize];
                    let grads = &soft.overlap_grads[oc.grads.clone()];
                    let warm_impulses = &soft.overlap_warm_impulses[oc.grads.clone()];
                    let sd = &mut soft.overlap_fem_sides[side as usize];
                    let lumped = sd.gain;
                    let entries = grads
                        .iter()
                        .filter(|g| in_body(g.0))
                        .map(|g| (g.0 - first, g.2));
                    let out = range(sd.start, n);
                    let gain = system.response_into(entries, out, params);
                    let u_max = out.iter().map(|v| v.length()).fold(0.0, Real::max);
                    sd.u_max = u_max;
                    sd.gain = if u_max > MAX_RESPONSE_AMPLIFICATION * gain {
                        lumped
                    } else {
                        gain
                    };
                    if sd.warm != u32::MAX {
                        let entries = grads
                            .iter()
                            .zip(warm_impulses)
                            .filter(|(g, p)| in_body(g.0) && **p != Vector::ZERO)
                            .map(|(g, p)| (g.0 - first, *p));
                        system.response_into(entries, range(sd.warm, n), params);
                    }
                }
                FemConstraintRef::Cluster(cluster_index) => {
                    let record = &mut soft.clusters[cluster_index as usize];
                    let Some(fem) = record.fem else {
                        continue;
                    };
                    // SAFETY: read-only access to the body's particles and clusters.
                    let sb = unsafe { &*entry.body };
                    let Some(cluster) = sb.clusters.get(record.cluster as usize) else {
                        continue;
                    };
                    // The cluster's free particles: their masses and lever arms about their
                    // centroid, as the gather fits the cluster's rigid motion.
                    let free: Vec<(u32, Real, Vector)> = cluster
                        .particles
                        .iter()
                        .filter(|&&p| sb.particles[p as usize].inv_mass > 0.0)
                        .map(|&p| {
                            let particle = &sb.particles[p as usize];
                            (p, particle.mass, particle.position)
                        })
                        .collect();
                    let mass: Real = free.iter().map(|f| f.1).sum();
                    if mass <= 0.0 {
                        record.fem = None;
                        continue;
                    }
                    let com = free.iter().map(|f| f.2 * f.1).sum::<Vector>() / mass;
                    let lumped_ii = record.lumped_ii;
                    // `Jᵀ eₖ` at particle `p`: the impulse on it of a unit impulse along spatial axis `k`.
                    let jacobian = |k: usize, m: Real, x: Vector| -> Vector {
                        if k < DIM {
                            let mut e = Vector::ZERO;
                            e[k] = m / mass;
                            e
                        } else {
                            cluster_angular_jacobian(&lumped_ii, k - DIM, m, x - com)
                        }
                    };
                    for k in 0..SPATIAL_DIM {
                        let entries = free.iter().map(move |&(p, m, x)| (p, jacobian(k, m, x)));
                        system.response_into(entries, range(fem.start + (k * n) as u32, n), params);
                    }
                    // `W = J A⁻¹ Jᵀ`: the cluster's velocity change per unit impulse on it.
                    let mut w = [[0.0; SPATIAL_DIM]; SPATIAL_DIM];
                    for l in 0..SPATIAL_DIM {
                        let response = range(fem.start + (l * n) as u32, n);
                        for k in 0..SPATIAL_DIM {
                            w[k][l] = free
                                .iter()
                                .map(|&(p, m, x)| jacobian(k, m, x).dot(response[p as usize]))
                                .sum();
                        }
                    }
                    let side = record.fem.as_mut().unwrap();
                    // The slot is given a mass model `D` the joints can hold (an inverse mass per
                    // axis and an inverse inertia) that bounds `W` from above, `W ≼ D`: an impulse
                    // the joints solve through `D` then changes the cluster's velocity by at most
                    // what they asked for, so the iteration between the joint solve and the
                    // gather of the true response converges, where the diagonal of `W` alone
                    // overshoots by the coupling it leaves out and diverges on a compliant body.
                    // `W ≼ 2 diag(W_lin, W_ang)` for any symmetric positive semi-definite `W`, and
                    // the linear block is bounded by its largest row sum.
                    let linear_bound = (0..DIM)
                        .map(|k| (0..DIM).map(|l| w[k][l].abs()).sum::<Real>())
                        .fold(0.0, Real::max);
                    let im = Vector::splat(2.0 * linear_bound);
                    let usable = linear_bound.is_finite() && linear_bound > 0.0;
                    if record.joints_only && usable {
                        let ii = cluster_angular_block(&w) * 2.0;
                        side.inv_im = Vector::ONE / im;
                        side.inv_ii = crate::dynamics::soft_body::pseudo_inverse_inertia(ii, 0.0);
                        // SAFETY: the proxy slot is this cluster's alone (see the method's contract).
                        let bodies = unsafe { &mut *bodies };
                        let pose = &mut bodies.poses[record.slot as usize];
                        pose.im = im;
                        pose.ii = ii;
                    } else {
                        // The slot keeps the cluster's lumped mass, which the contacts on its own
                        // colliders and its external impulses were solved with.
                        let lumped_im = record.lumped_im;
                        side.inv_im = Vector::from_array(core::array::from_fn(|k| {
                            if lumped_im[k] > 0.0 { 1.0 / lumped_im[k] } else { 0.0 }
                        }));
                        side.inv_ii =
                            crate::dynamics::soft_body::pseudo_inverse_inertia(lumped_ii, 0.0);
                    }
                }
            }
        }
    }

    /// Whether `active[index]` refactorizes its operator, and answers its constraints through
    /// it, at every substep rather than once a step: a body whose direct factorization costs no
    /// more than [`SUBSTEP_FACTOR_WORK`] (a rope, a cable, a truss: a chain of particles factors
    /// in a few multiply-adds per particle). The step-start operator holds each element's
    /// direction and load where the step began, and a body that turns within the step (a rope
    /// swinging a weight) answers its constraints through a stiffness that has turned away from
    /// its elements: a heavy body attached to it is then pushed along that stale direction at
    /// every substep, and the swing gains energy until it flies apart.
    pub fn refactors_every_substep(&self, index: usize) -> bool {
        let entry = &self.active[index];
        // SAFETY: read-only access to the body's own system during its own stage.
        let system = unsafe { &*entry.system };
        system.direct_factor_work().is_some_and(|work| work <= SUBSTEP_FACTOR_WORK)
    }

    /// Reads `active[index]`'s particle state from the solver bodies.
    /// # Safety
    /// Same contract as [`Self::predict`].
    pub unsafe fn refresh_state(&self, index: usize, bodies: &SolverBodies) {
        let entry = &self.active[index];
        let system = unsafe { &mut *entry.system };
        system.refresh_state(bodies);
    }

    /// The implicit elastic step of `active[index]` (one substep).
    /// # Safety
    /// The caller must guarantee exclusive access to this index (one claiming worker per index
    /// per stage) and that the body's solver-body slots are not written concurrently.
    pub unsafe fn predict(
        &self,
        index: usize,
        bodies: &mut SolverBodies,
        params: &SoftFemParameters,
    ) {
        let entry = &self.active[index];
        let system = unsafe { &mut *entry.system };
        system.predict(
            bodies,
            entry.dt,
            params.linear_tolerance,
            params.max_linear_iterations,
        );
    }

    /// Writes the per-cell state kept across steps back to the soft bodies.
    ///
    /// # Safety
    /// Same contract as [`Self::predict`].
    pub unsafe fn writeback(&self, index: usize, step_dt: Real) {
        let entry = &self.active[index];
        let system = unsafe { &*entry.system };
        let sb = unsafe { &mut *entry.body };
        system.writeback(sb, step_dt);
    }

    pub fn num_active(&self) -> usize {
        self.active.len()
    }
}

/// `Jᵀ eₖ` of a cluster's angular axis `k` at a particle of mass `m` and lever arm `r` about the
/// cluster's centroid: `m (I⁻¹ eₖ) × r`, the impulse a unit angular impulse on the cluster puts on
/// it (the transpose of `ω = I⁻¹ Σ mᵢ rᵢ × vᵢ`).
#[cfg(feature = "dim3")]
fn cluster_angular_jacobian(inv_inertia: &AngularInertia, k: usize, m: Real, r: Vector) -> Vector {
    use crate::utils::AngularInertiaOps;
    let mut e = Vector::ZERO;
    e[k] = 1.0;
    inv_inertia.transform_vector(e).cross(r) * m
}

/// See the 3D overload: `m (I⁻¹) perp(r)`, the 2D angular axis being the scalar one.
#[cfg(feature = "dim2")]
fn cluster_angular_jacobian(inv_inertia: &AngularInertia, _k: usize, m: Real, r: Vector) -> Vector {
    Vector::new(-r.y, r.x) * (*inv_inertia * m)
}

/// The angular block of a cluster's `W = J A⁻¹ Jᵀ`, symmetrized.
#[cfg(feature = "dim3")]
fn cluster_angular_block(w: &[[Real; SPATIAL_DIM]; SPATIAL_DIM]) -> AngularInertia {
    let sym = |a: usize, b: usize| 0.5 * (w[DIM + a][DIM + b] + w[DIM + b][DIM + a]);
    parry::utils::SdpMatrix3 {
        m11: sym(0, 0),
        m12: sym(0, 1),
        m13: sym(0, 2),
        m22: sym(1, 1),
        m23: sym(1, 2),
        m33: sym(2, 2),
    }
}

/// See the 3D overload.
#[cfg(feature = "dim2")]
fn cluster_angular_block(w: &[[Real; SPATIAL_DIM]; SPATIAL_DIM]) -> AngularInertia {
    w[DIM][DIM]
}

/// The responses of a particle's `DIM` axes (`A⁻¹ e_p ⊗ axis`), into `out` (`DIM` vectors per
/// particle of the body, axis-major), and their values at the particle: the block `(A⁻¹)_pp`, the
/// augmented inverse mass of a constraint acting on that particle alone.
fn particle_response(
    system: &mut SoftFemSystem,
    particle: u32,
    out: &mut [Vector],
    params: &SoftFemParameters,
) -> Matrix {
    if let Some(block) = system.particle_block_from_columns(particle, out) {
        return block;
    }
    let n = system.num_particles();
    let mut inv_mass = Matrix::ZERO;
    for k in 0..DIM {
        let mut axis = Vector::ZERO;
        axis[k] = 1.0;
        let u = &mut out[k * n..(k + 1) * n];
        system.response_into(core::iter::once((particle, axis)), u, params);
        *inv_mass.col_mut(k) = u[particle as usize];
    }
    // The block is symmetric up to the solve's accuracy: symmetrize it so the constraint's inverse
    // stays consistent with the responses.
    (inv_mass + inv_mass.transpose()) * 0.5
}

/// The lumped path's lever test (`SoftContact::update`): a side whose free-particle weight is
/// a sliver of the pinned ones' cannot be moved by the constraint.
fn is_lever<const N: usize>(weights: &[Real; N], im: &[Real; N]) -> bool {
    let mut w2 = 0.0;
    let mut im_max: Real = 0.0;
    for k in 0..N {
        w2 += weights[k] * weights[k] * im[k];
        im_max = im_max.max(im[k]);
    }
    im_max > 0.0 && w2 < 0.04 * im_max
}

/// Largest velocity anywhere in the body per unit of velocity correction at a constraint's point
/// that a response may produce; past it the constraint is a lever (its point sits next to pinned
/// particles, or its anchor weights nearly cancel) and the side falls back to the lumped anchors.
pub(crate) const MAX_RESPONSE_AMPLIFICATION: Real = 8.0;

/// Whether every direction's response is usable: a positive gain, and the amplification
/// `max |u| / gain` within `MAX_RESPONSE_AMPLIFICATION`.
fn well_conditioned(responses: &[Vector], gains: &[Real; DIM], n: usize) -> bool {
    for (k, gain) in gains.iter().enumerate() {
        let u = &responses[k * n..(k + 1) * n];
        let u_max = u.iter().map(|v| v.length()).fold(0.0, Real::max);
        if u_max == 0.0 {
            continue;
        }
        if *gain <= 0.0 || u_max > MAX_RESPONSE_AMPLIFICATION * *gain {
            return false;
        }
    }
    true
}
