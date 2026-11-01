//! A multibody's contacts solved together with the joint rows of the multibody that bear an
//! impulse.
//!
//! A contact on one link of a multibody moves every coordinate of the multibody, and a joint row
//! holding a link at its stop pushes on every coordinate too. Solved one after the other, the two
//! rows hand momentum back and forth between the light link the contact strikes and the heavy
//! multibody the row holds it to, and each exchange moves only about the light link's share of the
//! mass: a 1.5 kg forearm held at its stop on a 200 kg box closes about 5 % of the error between
//! the two rows per round. Solved as one block, the contact meets the multibody with those joint
//! rows holding, which is the mass it pushes against, and the block is exact in one solve.
//!
//! Every row of a block is written in one form: its error `f` (zero where the row holds) and the
//! rate `K[a][b]` at which an impulse on row `b` changes the error of row `a`. A contact's error is
//! its velocity error plus its softness term, and a joint row's is its softness term minus its
//! velocity error, so a positive impulse raises the error of the row it is applied on in both.
//!
//! `K[a][b]` sums, over every velocity the two rows both act on, the jacobian of `a` there times
//! the weighted jacobian of `b`: a multibody's generalized velocities, and a rigid body's velocity
//! where a row acts on one (a joint holding a rigid body to a link, a contact's rigid body). Two
//! rows that act on no velocity in common do not couple.

use crate::alloc_prelude::*;
use crate::dynamics::solver::GenericJointConstraint;
use crate::math::{DVector, Real, SPATIAL_DIM};
use na::DVectorView;

/// The velocities one side of a row acts on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SideVelocities {
    /// A multibody's generalized velocities, from this index of the generic solver velocities.
    Multibody(u32),
    /// A rigid body's velocity, the solver body at this index.
    Body(u32),
}

/// Where one side of a row reads its jacobian and its weighted jacobian (the jacobian times the
/// inverse mass).
#[derive(Copy, Clone, Debug)]
pub(crate) enum SideJacobian {
    /// In the row's jacobian buffer: `ndofs` entries from `j_id`, and as many from `wj_id`.
    Buffer {
        ndofs: usize,
        j_id: usize,
        wj_id: usize,
    },
    /// Held with the side, over a rigid body's linear then angular velocity.
    Inline {
        j: [Real; SPATIAL_DIM],
        wj: [Real; SPATIAL_DIM],
    },
}

impl SideJacobian {
    fn jacobian<'a>(&'a self, buffer: &'a DVector) -> &'a [Real] {
        match self {
            SideJacobian::Buffer { ndofs, j_id, .. } => &buffer.as_slice()[*j_id..*j_id + *ndofs],
            SideJacobian::Inline { j, .. } => &j[..],
        }
    }

    fn weighted<'a>(&'a self, buffer: &'a DVector) -> &'a [Real] {
        match self {
            SideJacobian::Buffer { ndofs, wj_id, .. } => {
                &buffer.as_slice()[*wj_id..*wj_id + *ndofs]
            }
            SideJacobian::Inline { wj, .. } => &wj[..],
        }
    }
}

/// One side of a row: the velocities it acts on and its jacobian there.
#[derive(Copy, Clone, Debug)]
pub(crate) struct RowSide {
    /// The velocities the side acts on.
    pub velocities: SideVelocities,
    /// Where the side's jacobian and weighted jacobian are read.
    pub jacobian: SideJacobian,
    /// The sign the side's jacobian times the velocities enters the row's error with.
    pub error_sign: Real,
    /// The sign a positive impulse on the row is applied to the side's velocities with.
    pub impulse_sign: Real,
}

impl RowSide {
    /// Whether the side acts on a multibody's generalized velocities.
    pub fn is_multibody(&self) -> bool {
        matches!(self.velocities, SideVelocities::Multibody(_))
    }
}

/// The rate at which an impulse on row `b` changes the error of row `a`, through every velocity
/// the two rows both act on. A side held in a buffer is read from the buffer of its own row.
pub(crate) fn coupling(
    a: &[Option<RowSide>; 2],
    a_jacobians: &DVector,
    b: &[Option<RowSide>; 2],
    b_jacobians: &DVector,
) -> Real {
    let mut k = 0.0;
    for sa in a.iter().flatten() {
        for sb in b.iter().flatten() {
            if sa.velocities == sb.velocities {
                let j = sa.jacobian.jacobian(a_jacobians);
                let wj = sb.jacobian.weighted(b_jacobians);
                let j = DVectorView::from_slice(j, j.len());
                let wj = DVectorView::from_slice(wj, wj.len());
                k += sa.error_sign * sb.impulse_sign * j.dot(&wj);
            }
        }
    }
    k
}

/// Whether two rows act on a same multibody.
fn share_a_multibody(a: &[Option<RowSide>; 2], b: &[Option<RowSide>; 2]) -> bool {
    a.iter()
        .flatten()
        .any(|sa| sa.is_multibody() && b.iter().flatten().any(|sb| sa.velocities == sb.velocities))
}

/// The multibody joint rows a pass's contacts are solved together with, and the buffers each
/// block is solved in. Reused from pass to pass.
#[derive(Default)]
pub(crate) struct CoupledRows {
    /// The index, among the generic joint constraints, of each joint row that acts on a
    /// multibody and bears an impulse strictly inside its bounds: a limit pushing on its stop,
    /// a motor not at its force bound, a coupling, a joint holding a body to a link.
    pub engaged: Vec<usize>,
    /// The sides of each engaged row, indexed like `engaged`.
    pub sides: Vec<[Option<RowSide>; 2]>,
    /// `K` between the engaged rows, row-major, `engaged.len()` square; the diagonal holds each
    /// row's own `1 / inv_lhs`, its softness included.
    pub coupling: Vec<Real>,
    /// The positions in `engaged` of the joint rows of the block being solved.
    pub block: Vec<usize>,
    /// For each joint row of the block and each contact row, in that order, `K` from the
    /// contact row to the joint row and from the joint row to the contact row.
    pub block_coupling: Vec<(Real, Real)>,
    /// The block's system: `K` plus each row's softness, `n` square.
    pub matrix: Vec<Real>,
    /// Each block row's error before the block is solved.
    pub error: Vec<Real>,
    /// Each block row's impulse before the block is solved.
    pub impulse: Vec<Real>,
    /// Each block row's impulse bounds.
    pub bounds: Vec<[Real; 2]>,
    /// The impulse the block applies on each of its rows.
    pub delta: Vec<Real>,
    /// Whether each block row is solved for (false while it is held at a bound).
    free: Vec<bool>,
    /// Which bound each held block row is held at: true for its upper bound.
    at_upper_bound: Vec<bool>,
    /// The free rows, the reduced system they are solved in, and its right-hand side.
    free_rows: Vec<usize>,
    reduced: Vec<Real>,
    reduced_rhs: Vec<Real>,
}

impl CoupledRows {
    /// Collects the engaged joint rows of `joints` and the coupling between them.
    pub fn gather(&mut self, joints: &[GenericJointConstraint], jacobians: &DVector) {
        self.engaged.clear();
        self.sides.clear();
        for (i, row) in joints.iter().enumerate() {
            let [lo, hi] = row.impulse_bounds;
            if row.impulse <= lo || row.impulse >= hi {
                continue;
            }
            let sides = row.sides();
            if !sides.iter().flatten().any(RowSide::is_multibody) {
                continue;
            }
            self.engaged.push(i);
            self.sides.push(sides);
        }
        let n = self.engaged.len();
        self.coupling.clear();
        self.coupling.resize(n * n, 0.0);
        for a in 0..n {
            for b in 0..n {
                self.coupling[a * n + b] = if a == b {
                    crate::utils::inv(joints[self.engaged[a]].inv_lhs)
                } else {
                    coupling(&self.sides[a], jacobians, &self.sides[b], jacobians)
                };
            }
        }
    }

    /// Chooses the engaged joint rows a point's contact rows are solved together with, into
    /// `block`, and their coupling with each contact row, into `block_coupling`: every engaged
    /// row on a multibody the contact rows act on. Each contact row is given by its sides, read
    /// from `jacobians`.
    pub fn select(
        &mut self,
        contact: &[[Option<RowSide>; 2]],
        jacobians: &DVector,
        joint_jacobians: &DVector,
    ) {
        self.block.clear();
        self.block_coupling.clear();
        for p in 0..self.engaged.len() {
            if !share_a_multibody(&self.sides[p], &contact[0]) {
                continue;
            }
            self.block.push(p);
            for sides in contact {
                let to_row = coupling(&self.sides[p], joint_jacobians, sides, jacobians);
                let to_contact = coupling(sides, jacobians, &self.sides[p], joint_jacobians);
                self.block_coupling.push((to_row, to_contact));
            }
        }
    }

    /// Prepares the block buffers for `n` rows.
    pub fn begin_block(&mut self, n: usize) {
        self.matrix.clear();
        self.matrix.resize(n * n, 0.0);
        for v in [&mut self.error, &mut self.impulse, &mut self.delta] {
            v.clear();
            v.resize(n, 0.0);
        }
        self.bounds.clear();
        self.bounds.resize(n, [0.0, 0.0]);
        self.free.clear();
        self.free.resize(n, true);
        self.at_upper_bound.clear();
        self.at_upper_bound.resize(n, false);
    }

    /// Frees every row of the block again, for a new [`Self::solve_block`] of the same rows
    /// with changed bounds.
    #[cfg(feature = "dim3")]
    pub fn release(&mut self) {
        self.free.fill(true);
        self.at_upper_bound.fill(false);
    }

    /// Solves the block prepared in `matrix`, `error`, `impulse` and `bounds` for the impulse
    /// each row applies, into `delta`: the impulses that zero the error of every row inside its
    /// bounds, with each other row held at a bound its error pushes it against.
    ///
    /// Each iteration solves the free rows with the held ones at their bounds, then changes the
    /// state of the first row that is wrong: a free row past a bound is held at it, and a held
    /// row whose error pulls it back inside its bounds is freed (least-index principal pivoting,
    /// which ends for a positive definite matrix such as a block's). Answers false when the block
    /// is singular or the pivoting does not settle within `4 n + 4` solves, leaving `delta`
    /// unspecified.
    pub fn solve_block(&mut self) -> bool {
        let n = self.error.len();
        for _ in 0..4 * n + 4 {
            let mut free = core::mem::take(&mut self.free_rows);
            free.clear();
            free.extend((0..n).filter(|i| self.free[*i]));
            let m = free.len();
            for a in 0..n {
                if !self.free[a] {
                    self.delta[a] = self.held_bound(a) - self.impulse[a];
                }
            }
            self.reduced.clear();
            self.reduced.resize(m * m, 0.0);
            self.reduced_rhs.clear();
            self.reduced_rhs.resize(m, 0.0);
            for (r, &a) in free.iter().enumerate() {
                let mut rhs = -self.error[a];
                for b in 0..n {
                    if !self.free[b] {
                        rhs -= self.matrix[a * n + b] * self.delta[b];
                    }
                }
                self.reduced_rhs[r] = rhs;
                for (c, &b) in free.iter().enumerate() {
                    self.reduced[r * m + c] = self.matrix[a * n + b];
                }
            }
            let solved = solve_dense(m, &mut self.reduced, &mut self.reduced_rhs);
            if solved {
                for (r, &a) in free.iter().enumerate() {
                    self.delta[a] = self.reduced_rhs[r];
                }
            }
            self.free_rows = free;
            if !solved {
                return false;
            }

            let mut changed = false;
            for a in 0..n {
                let [lo, hi] = self.bounds[a];
                if self.free[a] {
                    let new = self.impulse[a] + self.delta[a];
                    if new < lo || new > hi {
                        self.free[a] = false;
                        self.at_upper_bound[a] = new > hi;
                        changed = true;
                        break;
                    }
                } else {
                    let mut residual = self.error[a];
                    for b in 0..n {
                        residual += self.matrix[a * n + b] * self.delta[b];
                    }
                    // A positive impulse raises a row's error: held at its lower bound the row
                    // asks for less impulse while its residual is positive, and at its upper
                    // bound for more while it is negative.
                    let wants_inside = if self.at_upper_bound[a] {
                        residual > 0.0
                    } else {
                        residual < 0.0
                    };
                    if wants_inside {
                        self.free[a] = true;
                        changed = true;
                        break;
                    }
                }
            }
            if !changed {
                return true;
            }
        }
        false
    }

    /// The bound a held row is held at.
    fn held_bound(&self, a: usize) -> Real {
        let [lo, hi] = self.bounds[a];
        if self.at_upper_bound[a] { hi } else { lo }
    }
}

/// Solves the `n`-square system `matrix * x = rhs` (row-major) in place by Gaussian elimination
/// with partial pivoting, leaving `x` in `rhs`. Answers false when a pivot is not finite or
/// vanishes against the largest diagonal entry.
fn solve_dense(n: usize, matrix: &mut [Real], rhs: &mut [Real]) -> bool {
    let scale = (0..n).fold(0.0 as Real, |m, i| m.max(matrix[i * n + i].abs()));
    let tiny = scale * Real::EPSILON * 64.0;
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if matrix[row * n + col].abs() > matrix[pivot * n + col].abs() {
                pivot = row;
            }
        }
        let p = matrix[pivot * n + col];
        if !p.is_finite() || p.abs() <= tiny {
            return false;
        }
        if pivot != col {
            for k in 0..n {
                matrix.swap(pivot * n + k, col * n + k);
            }
            rhs.swap(pivot, col);
        }
        for row in col + 1..n {
            let factor = matrix[row * n + col] / p;
            if factor != 0.0 {
                for k in col..n {
                    matrix[row * n + k] -= factor * matrix[col * n + k];
                }
                rhs[row] -= factor * rhs[col];
            }
        }
    }
    for col in (0..n).rev() {
        let mut v = rhs[col];
        for k in col + 1..n {
            v -= matrix[col * n + k] * rhs[k];
        }
        rhs[col] = v / matrix[col * n + col];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_couple_through_a_rigid_body_they_both_act_on_and_through_nothing_else() {
        // A joint row's side on solver body 3, read from its buffer, and a contact row's side on
        // the same body, held with the side.
        let mut buffer = DVector::zeros(2 * SPATIAL_DIM);
        for k in 0..SPATIAL_DIM {
            buffer[k] = (k + 1) as Real;
            buffer[SPATIAL_DIM + k] = 0.5 * (k + 1) as Real;
        }
        let joint = [
            Some(RowSide {
                velocities: SideVelocities::Body(3),
                jacobian: SideJacobian::Buffer {
                    ndofs: SPATIAL_DIM,
                    j_id: 0,
                    wj_id: SPATIAL_DIM,
                },
                error_sign: -1.0,
                impulse_sign: -1.0,
            }),
            None,
        ];
        let contact_on = |velocities| {
            [
                Some(RowSide {
                    velocities,
                    jacobian: SideJacobian::Inline {
                        j: [1.0; SPATIAL_DIM],
                        wj: [2.0; SPATIAL_DIM],
                    },
                    error_sign: 1.0,
                    impulse_sign: 1.0,
                }),
                None,
            ]
        };
        let none = DVector::zeros(0);
        let sum: Real = (1..=SPATIAL_DIM).map(|k| k as Real).sum();
        let contact = contact_on(SideVelocities::Body(3));
        // The joint row's jacobian times the contact's weighted jacobian, and the contact's
        // jacobian times the joint row's weighted jacobian, each with the joint side's sign.
        assert_eq!(coupling(&joint, &buffer, &contact, &none), -2.0 * sum);
        assert_eq!(coupling(&contact, &none, &joint, &buffer), -0.5 * sum);
        // Another body, and a multibody whose generic velocities start at the same index.
        for velocities in [SideVelocities::Body(4), SideVelocities::Multibody(3)] {
            let other = contact_on(velocities);
            assert_eq!(coupling(&joint, &buffer, &other, &none), 0.0);
            assert_eq!(coupling(&other, &none, &joint, &buffer), 0.0);
        }
    }

    #[test]
    fn a_block_with_no_row_at_a_bound_zeroes_every_error() {
        let mut rows = CoupledRows::default();
        rows.begin_block(2);
        rows.matrix.copy_from_slice(&[2.0, 1.0, 1.0, 3.0]);
        rows.error.copy_from_slice(&[-3.0, -5.0]);
        rows.bounds
            .copy_from_slice(&[[0.0, Real::MAX], [0.0, Real::MAX]]);
        assert!(rows.solve_block());
        // 2a + b = 3, a + 3b = 5.
        assert!((rows.delta[0] - 0.8).abs() < 1.0e-5);
        assert!((rows.delta[1] - 1.4).abs() < 1.0e-5);
    }

    #[test]
    fn a_row_the_block_would_push_past_its_bound_is_held_there_and_the_rest_solved_around_it() {
        let mut rows = CoupledRows::default();
        rows.begin_block(2);
        rows.matrix.copy_from_slice(&[2.0, 1.0, 1.0, 3.0]);
        // Alone the block answers a = 2.6, b = -1.2: b cannot pull, so it is held at 0 and `a`
        // takes the whole error.
        rows.error.copy_from_slice(&[-4.0, 1.0]);
        rows.bounds
            .copy_from_slice(&[[0.0, Real::MAX], [0.0, Real::MAX]]);
        assert!(rows.solve_block());
        assert_eq!(rows.delta[1], 0.0);
        assert!((rows.delta[0] - 2.0).abs() < 1.0e-5);
    }

    #[test]
    fn a_row_pushed_out_only_by_another_row_that_cannot_pull_keeps_its_impulse() {
        // A limit holding its link (impulse 0.83, error 0) and a contact that separates (error
        // 3.58): solved free, the contact pulls and drags the limit's impulse below zero with it.
        // Held at zero, the contact leaves the limit exactly as it was.
        let mut rows = CoupledRows::default();
        rows.begin_block(2);
        rows.matrix.copy_from_slice(&[27.77, -4.95, -4.95, 1.16]);
        rows.error.copy_from_slice(&[0.0, 3.58]);
        rows.impulse.copy_from_slice(&[0.83, 0.0]);
        rows.bounds
            .copy_from_slice(&[[0.0, Real::MAX], [0.0, Real::MAX]]);
        assert!(rows.solve_block());
        assert_eq!(rows.delta[1], 0.0);
        assert!(rows.delta[0].abs() < 1.0e-6);
    }
}
