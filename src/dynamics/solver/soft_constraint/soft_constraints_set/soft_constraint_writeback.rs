//! Writeback of the solved state to the soft bodies: element impulses and plastic flow, particle poses and velocities, shape-matching, volume and attachment state.

use core::ops::Range;
use core::sync::atomic::Ordering;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use simba::scalar::{ComplexField as _, RealField as _};

use crate::alloc_prelude::*;
use crate::dynamics::solver::solver_body::SolverBodies;
use crate::dynamics::solver::velocity_solver::SubstepRecord;
use crate::math::{DIM, Real, Vector};
use crate::utils::RotationOps;

use super::super::soft_element_constraint::SoftScalarConstraintWriteback;
use super::*;

impl SoftConstraintsSet {
    /// Writes the accumulated impulses (and shape-matching rotations) back to the soft bodies,
    /// with the rest shapes the substeps' plastic flow left: the last substep flows here, at the
    /// particles' positions in `bodies` (each earlier substep flowed as the next one began, see
    /// `solve_color_constraint`), and each element states the work its flows took out of it. It
    /// also updates the elements' smoothed tear load (`dt`: the step length). SAFETY: `range`
    /// constraints write disjoint elements, one worker each.
    pub unsafe fn writeback_constraints(
        &self,
        range: Range<usize>,
        bodies: &SolverBodies,
        dt: Real,
    ) {
        // Virtual index space: the scalar constraints first, then the elastic-cell block constraints.
        let num_scalar_constraints = self.scalar_constraints.len();
        for idx in range {
            if idx < num_scalar_constraints {
                let constraint = &self.scalar_constraints[idx];
                let awake = &self.awake[constraint.soft_body as usize];
                let sb = unsafe { &mut *awake.ptr };
                let impulse = crate::utils::canonicalize_zero(constraint.impulse);
                // The last substep's creep, at the positions it ended at.
                let mut flowed = *constraint;
                if constraint.creep_yield > 0.0 {
                    flowed.creep_at(bodies, sb, awake.substep_dt);
                }
                match constraint.writeback {
                    SoftScalarConstraintWriteback::Edge => {
                        let ei = constraint.element as usize;
                        let material = *sb.element_material(sb.edges[ei].material);
                        let tears = material.tears();
                        let resistance = if tears {
                            let e = &sb.edges[ei];
                            sb.effective_tear_resistance(e.tear_resistance, &e.vertices)
                        } else {
                            1.0
                        };
                        let edge = &mut sb.edges[ei];
                        edge.impulse = crate::utils::canonicalize_zero(flowed.impulse);
                        edge.flow_work = flowed.flow_work;
                        edge.force = impulse * crate::utils::inv(awake.substep_dt);
                        // The rest length the substeps' flow left, at the yield force and past
                        // the strain yield.
                        if flowed.rest != edge.rest_length {
                            let initial = edge.initial_rest_length();
                            if (flowed.rest - edge.rest_length).abs() > 1.0e-4 * initial {
                                awake.plastic_flow.store(true, Ordering::Relaxed);
                            }
                            edge.rest_length = flowed.rest;
                            edge.plastic_strain = flowed.rest / initial - 1.0;
                        }
                        if tears {
                            // Length as seen at the last substep's update.
                            let len = (constraint.pos[1] - constraint.pos[0]).length();
                            // The force is the step's peak substep impulse over the substep length
                            // (a spike must not hide behind the last substep), the pull's for the
                            // tear force and the push's for the crush force; the stretch is
                            // measured against the initial rest length, so plastic flow counts.
                            let pull = constraint.peak_impulse.max(constraint.impulse);
                            let push = constraint.trough_impulse.min(constraint.impulse);
                            let inv_dt = crate::utils::inv(awake.substep_dt);
                            let initial = edge.initial_rest_length();
                            let load = material
                                .edge_tear_load(len, initial, pull * inv_dt)
                                .max(material.edge_tear_load(len, initial, push * inv_dt))
                                * crate::utils::inv(resistance);
                            edge.stress = material.smooth_stress(edge.stress, load, dt);
                            if edge.stress > 1.0 {
                                edge.torn = true;
                                awake.torn.store(true, Ordering::Relaxed);
                            }
                        }
                    }
                    #[cfg(feature = "dim3")]
                    SoftScalarConstraintWriteback::Dihedral => {
                        let dihedral = &mut sb.dihedrals[constraint.element as usize];
                        dihedral.impulse = crate::utils::canonicalize_zero(flowed.impulse);
                        dihedral.flow_work = flowed.flow_work;
                        // The rest angle the substeps' flow left.
                        if flowed.rest != dihedral.rest_angle {
                            if (flowed.rest - dihedral.rest_angle).abs() > 1.0e-4 {
                                awake.plastic_flow.store(true, Ordering::Relaxed);
                            }
                            let initial = dihedral.initial_rest_angle();
                            dihedral.rest_angle = flowed.rest;
                            dihedral.plastic_set = flowed.rest - initial;
                        }
                    }
                    SoftScalarConstraintWriteback::CellVolume => {
                        let cell = &mut sb.cells[constraint.element as usize];
                        cell.impulses[0] = impulse;
                        cell.flow_work = 0.0;
                    }
                }
            } else {
                let constraint = &self.elastic_constraints[idx - num_scalar_constraints];
                let end = self.elastic_end_state(idx - num_scalar_constraints);
                let mut strain_impulse = end.strain_impulse;
                let awake = &self.awake[constraint.soft_body as usize];
                let sb = unsafe { &mut *awake.ptr };
                // The last substep's flow, at the positions it ended at.
                let mut flowed = *constraint;
                if constraint.flow.is_some() {
                    if let Some(flow) = flow_elastic_cell(
                        &mut flowed,
                        sb,
                        |c| {
                            c.read_positions(bodies);
                            c.pos
                        },
                        &end.strain_impulse,
                        &end.rotation.to_mat(),
                        awake.substep_dt,
                        self.max_corrective_velocity,
                        false,
                    ) {
                        strain_impulse *= 1.0 - flow.share;
                    }
                }
                let material = *sb.element_material(sb.cells[constraint.element as usize].material);
                let resistance = if material.tear_strain.is_some() {
                    let c = &sb.cells[constraint.element as usize];
                    sb.effective_tear_resistance(c.tear_resistance, &c.vertices)
                } else {
                    1.0
                };
                let rest0: [Vector; DIM + 1] = {
                    let vertices = sb.cells[constraint.element as usize].vertices;
                    core::array::from_fn(|k| {
                        sb.particles[vertices[k] as usize].initial_rest_position
                    })
                };
                let cell = &mut sb.cells[constraint.element as usize];
                let impulses = (strain_impulse.iter()).chain(core::iter::once(&end.vol_impulse));
                for (dst, src) in cell.impulses.iter_mut().zip(impulses) {
                    *dst = crate::utils::canonicalize_zero(*src);
                }
                cell.rotation = end.rotation;
                cell.flow_work = 0.0;
                // The tear strain counts the plastic stretch the last substep's strain is read
                // against.
                let read_against = constraint
                    .flow
                    .map_or(cell.plastic_stretch, |flow| flow.plastic_stretch);
                let tensile = material
                    .tear_strain
                    .is_some()
                    .then(|| total_tensile_strain(&end.strain, &read_against));
                if let Some(flow) = flowed.flow {
                    if stretch_moved(&cell.plastic_stretch, &flow.plastic_stretch) {
                        awake.plastic_flow.store(true, Ordering::Relaxed);
                    }
                    if flow.plastic_stretch != cell.plastic_stretch {
                        let dm = flow.plastic_stretch
                            * crate::dynamics::SoftBody::cell_edge_matrix(rest0);
                        let factorial = if DIM == 2 { 2.0 } else { 6.0 };
                        cell.plastic_stretch = flow.plastic_stretch;
                        cell.inv_rest_matrix = flowed.inv_rest_matrix;
                        cell.rest_volume = dm.determinant() / factorial;
                    }
                    cell.flow_work = flow.work;
                }
                if let Some(tensile) = tensile {
                    let load = material.cell_tear_load(tensile) * crate::utils::inv(resistance);
                    cell.stress = material.smooth_stress(cell.stress, load, dt);
                    if cell.stress > 1.0 {
                        cell.torn = true;
                        awake.torn.store(true, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    /// Writes the solved poses and velocities of awake body `ai`'s particles back to the soft
    /// body (linear damping applied, a pinned particle snapped to its target), and the substep
    /// states `substep_vels` kept for the particles the body records
    /// (`SoftBody::substep_particles`). SAFETY: writes only that body's particles; the caller
    /// guarantees each awake body is written back by one worker.
    pub unsafe fn writeback_particles(
        &self,
        ai: usize,
        bodies: &SolverBodies,
        substep_vels: &[Vec<SubstepRecord>],
        step_dt: Real,
    ) {
        let awake = &self.awake[ai];
        let sb = unsafe { &mut *awake.ptr };
        let slots = &self.slots[awake.slot_start..awake.slot_start + awake.num_particles];
        let damping = crate::utils::inv(1.0 + step_dt * sb.particle_settings.linear_damping);
        for (p, &slot) in sb.particles.iter_mut().zip(slots) {
            let vel = bodies.get_vel(slot).linear;
            if p.inv_mass == 0.0 {
                // A target-driven pin lands exactly on its target (and stays in that mode: with
                // the target reached, the next step's velocity is zero unless a new one is set).
                p.velocity = vel;
                p.position = match p.next_position {
                    Some(target) => target,
                    None => bodies.get_pose(slot).translation,
                };
            } else {
                p.velocity = vel * damping;
                p.position = bodies.get_pose(slot).translation;
            }
        }
        // A body held to a plane leaves the step on it, moving along it.
        if let Some((normal, offset)) = sb.plane {
            for p in &mut sb.particles {
                p.velocity -= normal * normal.dot(p.velocity);
                p.position -= normal * (normal.dot(p.position) - offset);
            }
        }
        if !sb.recorded_particles.is_empty() {
            record_substep_particles(sb, slots, bodies, substep_vels, awake.substep_dt);
        }
    }

    /// Writes back the shape-matching, volume-piece and attachment state (serial).
    pub fn writeback_bodies(&self) {
        for constraint in &self.attachments {
            // SAFETY: serial stage, exclusive access.
            let ptr = self.awake[constraint.soft_body as usize].ptr;
            let sb = unsafe { &mut *ptr };
            if let Some(a) = sb.attachments.get_mut(constraint.attachment as usize) {
                a.impulse = constraint.impulse;
            }
        }
        for awake in &self.awake {
            // SAFETY: serial stage, exclusive access.
            let sb = unsafe { &mut *awake.ptr };
            if !awake.awake_clusters.is_empty() {
                for awake_cluster in &awake.awake_clusters {
                    let cluster = &mut sb.clusters[awake_cluster.cluster as usize];
                    cluster.rotation = awake_cluster.rotation;
                    cluster
                        .shape_impulses
                        .resize(cluster.particles.len(), Vector::ZERO);
                }
                for constraint in &self.shape_constraints[awake.shape_constraints.clone()] {
                    let cluster = &mut sb.clusters[constraint.cluster as usize];
                    if let Ok(k) = cluster.particles.binary_search(&constraint.particle) {
                        cluster.shape_impulses[k] = constraint.impulse;
                    }
                }
            }
            for vc in &self.volume_constraints[awake.volume_constraints.clone()] {
                sb.volume_pieces[vc.piece as usize].impulse =
                    crate::utils::canonicalize_zero(vc.impulse);
            }
            sb.plastic_flowing = awake.plastic_flow.load(Ordering::Relaxed);
            sb.tearing_pending |= awake.torn.load(Ordering::Relaxed);
            // One history entry per step: its first pass shifts the history, its later CCD
            // passes keep the largest approach speed.
            let speed = awake.contact_approach_speed;
            if sb.contact_approach_step_open {
                let entry = &mut sb.contact_approach_speeds[0];
                *entry = match (*entry, speed) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                };
            } else {
                sb.contact_approach_speeds = [speed, sb.contact_approach_speeds[0]];
                sb.contact_approach_step_open = true;
            }
            // Summed by the contact writeback that follows the solve.
            sb.contact_load = 0.0;
        }
    }
}

pub(super) use super::super::soft_element_constraint::outer_product as outer;

/// Copy the substep states the solver kept for `sb`'s recorded particles into
/// `sb.substep_particles`, substep-major, the last substep's the state each particle was just
/// written back with. The linear damping the writeback applies once, to the velocity the step
/// ends with, is spread evenly over the substeps: substep `k` of `n` carries `k / n` of it.
fn record_substep_particles(
    sb: &mut crate::dynamics::SoftBody,
    slots: &[u32],
    bodies: &SolverBodies,
    substep_vels: &[Vec<SubstepRecord>],
    substep_dt: Real,
) {
    let earlier = sb
        .recorded_particles
        .first()
        .and_then(|&particle| slots.get(particle as usize))
        .and_then(|&slot| substep_vels.get(slot as usize))
        .map_or(0, Vec::len);
    let substeps = (earlier + 1) as Real;
    let crate::dynamics::SoftBody { recorded_particles, substep_particles, particles, .. } = &mut *sb;
    substep_particles.clear();
    substep_particles.reserve((earlier + 1) * recorded_particles.len());
    for k in 0..=earlier {
        let share = (k + 1) as Real / substeps;
        for &particle in recorded_particles.iter() {
            let slot = slots.get(particle as usize).copied();
            let state = if k < earlier {
                slot.and_then(|slot| {
                    let (vel, position, _) = substep_vels.get(slot as usize)?.get(k)?;
                    let p = particles.get(particle as usize)?;
                    let damped = p.velocity - bodies.get_vel(slot).linear;
                    Some((vel.linear + damped * share, *position))
                })
            } else {
                particles.get(particle as usize).map(|p| (p.velocity, p.position))
            };
            substep_particles.push(state.unwrap_or((Vector::ZERO, Vector::ZERO)));
        }
    }
    sb.substep_particles_dt = substep_dt;
}
