//! Impulse joints that join their bodies as a tree, solved exactly.
//!
//! A pass of Gauss-Seidel over the rows of a chain of joints moves each row's impulse to what
//! the row asks for with the impulses of its neighbours where the last sweep left them, so an
//! impulse travels one joint along the chain per sweep: one pass over a five-link chain leaves
//! most of the tension the substep needs unsolved, and the stabilization pass then takes the
//! velocity that tension would have turned out of the bodies' motion. A free-swinging chain of
//! five rods on ball joints lost 62% of its energy in 10 s at 120 substeps a second in one pass
//! of each kind, and 17% with every pass solved to convergence.
//!
//! The joints of an island whose bodies they join as a forest (no loop through the moving
//! bodies; any number of joints to fixed bodies) are taken out of the coloured sweeps and
//! solved here, one tree at a time, with every row of unbounded impulse (a locked axis, an
//! unbounded motor) solved exactly against every other: the system the rows' Gauss-Seidel
//! converges to, `(J M⁻¹ Jᵀ + C) Δ = J v + b - C λ`, factored once a substep and solved in
//! each pass. Eliminating the joints from the leaves towards the root fills no entry the
//! system does not already hold (the rows of a joint couple only to the rows of joints that
//! share one of its bodies, and once a joint's other body has no joint left but this one,
//! every row it still couples to shares the same body), so the factorization costs time
//! linear in the number of joints for chains and ragdolls alike. A row of bounded impulse (a
//! limit, a bounded motor) and a row the factorization finds no inverse mass left for are
//! solved by Gauss-Seidel after the exact solve, in the same stage. A joint that closes a
//! loop through the moving bodies stays in the coloured sweeps.

use crate::alloc_prelude::*;
use core::ops::Range;

use crate::dynamics::solver::joint_constraint::JointConstraint;
use crate::dynamics::solver::solver_body::SolverBodies;
use crate::dynamics::{JointGraphEdge, JointIndex, RigidBodySet};
use crate::math::{AngVector, Real, SPATIAL_DIM, Vector};
use crate::utils::DotProduct;

/// The most rows of one joint a tree solves exactly: one per degree of freedom.
const ROWS: usize = SPATIAL_DIM;
/// The reals one row-by-row block of two joints holds.
const BLOCK: usize = ROWS * ROWS;
/// A side of a joint that no moving body of the tree is on.
const NONE: u32 = u32::MAX;
/// A row is solved exactly while the share of its own inverse mass that the joints eliminated
/// before it leave it stays above this; below it the row is solved by Gauss-Seidel after the
/// exact solve. Bodies held from moving along or turning about an axis leave the rows of two
/// joints on one body dependent on each other (a hinge chain held to a plane: the rows across
/// the plane of two hinges on the thigh ask for the thigh's two free turns three times), and
/// solved exactly a row the others already nearly hold took an impulse that tore the hinge
/// 3 cm apart at a share of 1e-3. A row holding the axis of a chain of `n` like links hanging
/// from the world keeps about `1 / n` of its own: the top joint of a chain of up to about a
/// hundred links is solved exactly along it.
const PIVOT_TOLERANCE: Real = 1.0e-2;

/// The solver colour a joint solved in a tree carries: it is solved in no coloured sweep.
pub(crate) const SOLVER_COLOR_TREE: u8 = 200;

/// One joint of a tree, in the order the factorization eliminates them.
#[derive(Clone, Copy, Debug)]
struct TreeJoint {
    /// The joint's graph index while the trees are laid out, then its scalar builder's.
    builder: u32,
    /// The tree body on each side of the joint, [`NONE`] for a side no moving body of the tree
    /// is on.
    sides: [u32; 2],
    /// The tree body whose joints the elimination of this one couples: the side nearer the root.
    elim_body: u32,
    /// Where this joint stands among the joints of `elim_body`.
    elim_pos: u32,
}

/// One tree: its joints, its bodies, and where its numbers are kept.
#[derive(Clone, Debug)]
struct JointTree {
    /// The tree's joints, into [`JointTrees::joints`].
    joints: Range<usize>,
    /// The tree's bodies, into [`JointTrees::body_joints_at`] and [`JointTrees::body_pairs`].
    bodies: Range<usize>,
    /// The tree's matrices, into [`JointTrees::matrices`], in blocks: the joints' diagonal
    /// blocks, their inverses, the blocks of every two joints sharing a body, then the scratch
    /// of one elimination.
    matrices: Range<usize>,
    /// The number of joint pairs sharing a body, summed over the tree's bodies.
    pairs: usize,
}

/// The impulse joints of the islands solved as trees, laid out with the joint assembly.
#[derive(Default)]
pub(crate) struct JointTrees {
    trees: Vec<JointTree>,
    /// The first tree of each solve group, and one past the last group's last.
    group_starts: Vec<usize>,
    joints: Vec<TreeJoint>,
    /// The joints of every tree body, as tree-joint indices in elimination order.
    body_joints: Vec<u32>,
    /// Each tree body's run of `body_joints`.
    body_joints_at: Vec<Range<usize>>,
    /// Each tree body's first pair block, counted from the tree's first pair block.
    body_pairs: Vec<usize>,
    matrices: Vec<Real>,
    /// Per row slot (`ROWS` per joint): what the row reads of its two bodies' velocities and
    /// what a unit of its impulse adds to them, side 1 then side 2.
    row_vels: Vec<[RowVel; 4]>,
    /// Per row slot: the constraint row it is.
    row_ids: Vec<u32>,
    /// Per joint: how many of its rows are solved exactly.
    row_counts: Vec<u8>,
    /// Per row slot: whether the factorization left the row inverse mass to solve it with.
    row_kept: Vec<bool>,
    /// Per row slot: the row's own diagonal entry, before any elimination.
    row_diag: Vec<Real>,
    /// Per row slot: the residual the pass solves, then the impulse that solves it.
    rhs: Vec<Real>,
    delta: Vec<Real>,
}

/// A row's part on one side of a joint: what it reads of that body's velocity, or what a unit of
/// its impulse adds to it.
#[derive(Copy, Clone, Default)]
struct RowVel {
    linear: Vector,
    angular: AngVector,
}

/// The pair block of joints `x < y` among the `n` joints of one body, counted from the body's
/// first pair block.
fn pair_index(n: usize, x: usize, y: usize) -> usize {
    x * n - x * (x + 1) / 2 + (y - x - 1)
}

fn dot(a: &RowVel, b: &RowVel) -> Real {
    a.linear.gdot(b.linear) + a.angular.gdot(b.angular)
}

impl JointTrees {
    /// Takes out of `interactions` the joints that join their bodies as trees of two or more
    /// joints and lays the trees out, group by group. `slot_group` names each island body's
    /// solve group (empty for a single group). A joint qualifies when it locks an axis and
    /// every side of it that moves is an awake dynamic body of one group; the joints are taken
    /// in the order `interactions` holds them, and one that would close a loop through the
    /// moving bodies stays, as does one with a side on a kinematic body. Each taken joint's `builder` holds its graph index until
    /// [`Self::set_builder`] replaces it.
    pub(super) fn select(
        &mut self,
        interactions: &mut Vec<JointIndex>,
        impulse_joints: &mut [JointGraphEdge],
        bodies: &RigidBodySet,
        num_island_bodies: usize,
        slot_group: &[u16],
        num_groups: usize,
    ) {
        self.trees.clear();
        self.group_starts.clear();
        self.joints.clear();
        self.body_joints.clear();
        self.body_joints_at.clear();
        self.body_pairs.clear();

        let group_of = |slot: u32| -> usize { slot_group.get(slot as usize).copied().unwrap_or(0) as usize };
        // The moving sides of a joint: the solver slot of each side that is an awake dynamic
        // island body. `None` for a joint with a side on any other solver body (a kinematic or
        // a sleeping body's read-only slot), whose velocity slot other constraints share.
        let moving = |joint_i: JointIndex| -> Option<[u32; 2]> {
            let joint = &impulse_joints[joint_i].weight;
            let mut sides = [NONE; 2];
            for (s, handle) in [joint.body1, joint.body2].into_iter().enumerate() {
                let id = joint.solver_body_ids[s];
                if id == NONE {
                    continue;
                }
                if (id as usize) >= num_island_bodies || !bodies[handle].is_dynamic() {
                    return None;
                }
                sides[s] = id;
            }
            Some(sides)
        };

        // Union-find over the island's bodies, joined by the taken joints.
        let mut parent: Vec<u32> = (0..num_island_bodies as u32).collect();
        fn find(parent: &mut [u32], mut i: u32) -> u32 {
            while parent[i as usize] != i {
                let up = parent[parent[i as usize] as usize];
                parent[i as usize] = up;
                i = up;
            }
            i
        }
        let mut taken: Vec<(JointIndex, [u32; 2])> = Vec::new();
        for joint_i in interactions.iter() {
            let joint = &impulse_joints[*joint_i].weight;
            if joint.data.locked_axes.is_empty() {
                continue;
            }
            let Some(sides) = moving(*joint_i) else {
                continue;
            };
            match sides {
                [NONE, NONE] => continue,
                [a, NONE] | [NONE, a] => {
                    let _ = a;
                }
                [a, b] => {
                    if group_of(a) != group_of(b) {
                        continue;
                    }
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra == rb {
                        continue;
                    }
                    parent[ra.max(rb) as usize] = ra.min(rb);
                }
            }
            taken.push((*joint_i, sides));
        }

        // The trees: the taken joints by the root of their bodies, first appearance first.
        let root_of = |parent: &mut [u32], sides: [u32; 2]| -> u32 {
            let body = if sides[0] != NONE { sides[0] } else { sides[1] };
            find(parent, body)
        };
        let mut tree_of_root: Vec<u32> = alloc::vec![NONE; num_island_bodies];
        let mut members: Vec<Vec<usize>> = Vec::new();
        for (t, (_, sides)) in taken.iter().enumerate() {
            let root = root_of(&mut parent, *sides) as usize;
            if tree_of_root[root] == NONE {
                tree_of_root[root] = members.len() as u32;
                members.push(Vec::new());
            }
            members[tree_of_root[root] as usize].push(t);
        }
        let mut in_tree = alloc::vec![false; taken.len()];
        let mut groups_of_trees: Vec<(usize, usize)> = Vec::new();
        for (m, list) in members.iter().enumerate() {
            if list.len() >= 2 {
                for t in list {
                    in_tree[*t] = true;
                }
                let sides = taken[list[0]].1;
                let body = if sides[0] != NONE { sides[0] } else { sides[1] };
                groups_of_trees.push((group_of(body), m));
            }
        }
        // Group-major, first appearance within a group.
        groups_of_trees.sort_by_key(|(g, m)| (*g, *m));

        let mut local_of_slot: Vec<u32> = alloc::vec![NONE; num_island_bodies];
        let mut next_group = 0;
        for (group, m) in groups_of_trees {
            while next_group <= group {
                self.group_starts.push(self.trees.len());
                next_group += 1;
            }
            let list = &members[m];
            // The tree's bodies, numbered in the order its joints name them.
            let bodies_start = self.body_joints_at.len();
            let mut slots: Vec<u32> = Vec::new();
            for t in list {
                for side in taken[*t].1 {
                    if side != NONE && local_of_slot[side as usize] == NONE {
                        local_of_slot[side as usize] = slots.len() as u32;
                        slots.push(side);
                    }
                }
            }
            let num_bodies = slots.len();
            let local = |slot: u32| if slot == NONE { NONE } else { local_of_slot[slot as usize] };
            // Depths from the first body, over the joints between two moving bodies.
            let mut adjacency: Vec<Vec<u32>> = alloc::vec![Vec::new(); num_bodies];
            for t in list {
                let [a, b] = taken[*t].1;
                if a != NONE && b != NONE {
                    adjacency[local(a) as usize].push(local(b));
                    adjacency[local(b) as usize].push(local(a));
                }
            }
            let mut depth = alloc::vec![u32::MAX; num_bodies];
            let mut queue = alloc::vec![0u32];
            depth[0] = 0;
            let mut head = 0;
            while head < queue.len() {
                let body = queue[head] as usize;
                head += 1;
                for next in adjacency[body].clone() {
                    if depth[next as usize] == u32::MAX {
                        depth[next as usize] = depth[body] + 1;
                        queue.push(next);
                    }
                }
            }
            // Elimination order: deeper joints first; at one depth a body's joints to the world
            // before the joint above it.
            let mut order: Vec<(u32, u8, usize, [u32; 2], u32)> = list
                .iter()
                .map(|t| {
                    let sides = [local(taken[*t].1[0]), local(taken[*t].1[1])];
                    match sides {
                        [a, NONE] | [NONE, a] => (depth[a as usize], 0, *t, sides, a),
                        [a, b] => {
                            let near = if depth[a as usize] < depth[b as usize] { a } else { b };
                            let far = if near == a { b } else { a };
                            (depth[far as usize], 1, *t, sides, near)
                        }
                    }
                })
                .collect();
            order.sort_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));

            let joints_start = self.joints.len();
            for (_, _, t, sides, elim) in &order {
                self.joints.push(TreeJoint {
                    builder: taken[*t].0 as u32,
                    sides: *sides,
                    elim_body: *elim,
                    elim_pos: 0,
                });
            }
            // Each body's joints, in elimination order, and where each joint stands at the body
            // its elimination couples.
            let mut at_body: Vec<Vec<u32>> = alloc::vec![Vec::new(); num_bodies];
            for (j, joint) in self.joints[joints_start..].iter().enumerate() {
                for side in joint.sides {
                    if side != NONE {
                        at_body[side as usize].push(j as u32);
                    }
                }
            }
            let mut pairs = 0;
            let mut max_degree = 0;
            for list in &at_body {
                self.body_pairs.push(pairs);
                let start = self.body_joints.len();
                self.body_joints.extend_from_slice(list);
                self.body_joints_at.push(start..self.body_joints.len());
                pairs += list.len() * list.len().saturating_sub(1) / 2;
                max_degree = max_degree.max(list.len());
            }
            for j in 0..order.len() {
                let joint = &mut self.joints[joints_start + j];
                let list = &at_body[joint.elim_body as usize];
                joint.elim_pos = list.iter().position(|x| *x == j as u32).unwrap_or(0) as u32;
            }
            let num_joints = order.len();
            let blocks = 2 * num_joints + pairs + max_degree;
            let matrices_start = self.matrices.len();
            self.matrices.resize(matrices_start + blocks * BLOCK, 0.0);
            self.trees.push(JointTree {
                joints: joints_start..self.joints.len(),
                bodies: bodies_start..bodies_start + num_bodies,
                matrices: matrices_start..self.matrices.len(),
                pairs,
            });
            for slot in slots {
                local_of_slot[slot as usize] = NONE;
            }
        }
        while self.group_starts.len() <= num_groups {
            self.group_starts.push(self.trees.len());
        }

        let rows = self.joints.len() * ROWS;
        self.row_vels.clear();
        self.row_vels.resize(rows, [RowVel::default(); 4]);
        self.row_ids.clear();
        self.row_ids.resize(rows, 0);
        self.row_counts.clear();
        self.row_counts.resize(self.joints.len(), 0);
        self.row_kept.clear();
        self.row_kept.resize(rows, false);
        self.row_diag.clear();
        self.row_diag.resize(rows, 0.0);
        self.rhs.clear();
        self.rhs.resize(rows, 0.0);
        self.delta.clear();
        self.delta.resize(rows, 0.0);

        let mut tree_joint = alloc::vec![false; impulse_joints.len()];
        for (t, (joint_i, _)) in taken.iter().enumerate() {
            if in_tree[t] {
                impulse_joints[*joint_i].weight.solver_color = SOLVER_COLOR_TREE;
                tree_joint[*joint_i] = true;
            }
        }
        interactions.retain(|joint_i| !tree_joint[*joint_i]);
    }

    /// The trees of solve group `group`.
    pub(super) fn group_trees(&self, group: usize) -> Range<usize> {
        match (self.group_starts.get(group), self.group_starts.get(group + 1)) {
            (Some(start), Some(end)) => *start..*end,
            _ => 0..0,
        }
    }

    /// The number of joints the trees hold.
    pub(super) fn num_joints(&self) -> usize {
        self.joints.len()
    }

    /// Gives every joint of the trees of solve group `group` a scalar builder: `generate` lays
    /// out the builder of a joint (its graph index) and answers the builder's index.
    pub(super) fn generate_builders(
        &mut self,
        group: usize,
        mut generate: impl FnMut(JointIndex) -> usize,
    ) {
        for tree in self.group_trees(group) {
            for j in self.trees[tree].joints.clone() {
                let joint_i = self.joints[j].builder as JointIndex;
                self.joints[j].builder = generate(joint_i) as u32;
            }
        }
    }

    /// Solves the rows of tree `tree` for one pass. `wo_bias` is the stabilization pass,
    /// `first` the first iteration of a pass, `warmstart` applies the impulses the rows start
    /// from, and `factor` builds the substep's factorization first.
    ///
    /// # Safety
    /// The caller is the only one touching the tree's slices of `self` and the velocities of the
    /// tree's bodies.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn solve(
        this: *mut Self,
        tree: usize,
        rows: &mut [JointConstraint<Real, 1>],
        joint_rows: &[Range<usize>],
        solver_bodies: &mut SolverBodies,
        wo_bias: bool,
        first: bool,
        warmstart: bool,
        factor: bool,
    ) {
        let this = unsafe { &mut *this };
        let t = this.trees[tree].clone();
        let joints = &this.joints[t.joints.clone()];
        let row_base = t.joints.start * ROWS;
        let q = joints.len();

        // SAFETY: each tree's ranges of the per-row and per-joint arrays and of the matrices are
        // its own, and the caller holds the tree alone.
        let ptr = |v: &mut Vec<Real>, r: Range<usize>| unsafe {
            core::slice::from_raw_parts_mut(v.as_mut_ptr().add(r.start), r.len())
        };
        let m = ptr(&mut this.matrices, t.matrices.clone());
        let rows_n = q * ROWS;
        let rhs = ptr(&mut this.rhs, row_base..row_base + rows_n);
        let delta = ptr(&mut this.delta, row_base..row_base + rows_n);
        let row_vels = unsafe {
            core::slice::from_raw_parts_mut(this.row_vels.as_mut_ptr().add(row_base), rows_n)
        };
        let row_ids = unsafe {
            core::slice::from_raw_parts_mut(this.row_ids.as_mut_ptr().add(row_base), rows_n)
        };
        let row_kept = unsafe {
            core::slice::from_raw_parts_mut(this.row_kept.as_mut_ptr().add(row_base), rows_n)
        };
        let row_diag = ptr(&mut this.row_diag, row_base..row_base + rows_n);
        let row_counts = unsafe {
            core::slice::from_raw_parts_mut(this.row_counts.as_mut_ptr().add(t.joints.start), q)
        };
        let body_joints_at = &this.body_joints_at[t.bodies.clone()];
        let body_pairs = &this.body_pairs[t.bodies.clone()];
        let body_joints = &this.body_joints;

        // Every row's stabilization target reads the velocities the substep moved the bodies
        // with before any row is solved; bias goes in the stabilization pass, and the impulses
        // the rows start from go on before any is solved.
        for joint in joints {
            for r in joint_rows[joint.builder as usize].clone() {
                let c = &mut rows[r];
                if wo_bias {
                    if first {
                        c.carry_turn(solver_bodies);
                    }
                    c.remove_bias_from_rhs();
                }
            }
        }
        if warmstart {
            for joint in joints {
                for r in joint_rows[joint.builder as usize].clone() {
                    rows[r].warmstart(solver_bodies);
                }
            }
        }

        let (diag, rest) = m.split_at_mut(q * BLOCK);
        let (dinv, rest) = rest.split_at_mut(q * BLOCK);
        let (pairs, temp) = rest.split_at_mut(t.pairs * BLOCK);
        let pair_at = |body: usize, x: usize, y: usize| -> usize {
            let n = body_joints_at[body].len();
            (body_pairs[body] + pair_index(n, x, y)) * BLOCK
        };

        if factor {
            // The rows each joint solves exactly, and what each reads and moves.
            for (j, joint) in joints.iter().enumerate() {
                let mut k = 0;
                for r in joint_rows[joint.builder as usize].clone() {
                    let c = &rows[r];
                    if k < ROWS && c.impulse_bounds == [-Real::MAX, Real::MAX] {
                        let slot = j * ROWS + k;
                        row_ids[slot] = r as u32;
                        row_vels[slot] = [
                            RowVel { linear: -c.lin_jac, angular: -c.ang_jac1 },
                            RowVel { linear: c.lin_jac, angular: c.ang_jac2 },
                            RowVel { linear: c.lin_jac * c.im1, angular: c.ii_ang_jac1 },
                            RowVel { linear: -(c.lin_jac * c.im2), angular: -c.ii_ang_jac2 },
                        ];
                        k += 1;
                    }
                }
                row_counts[j] = k as u8;
            }
            // The diagonal blocks: each joint's rows against each other, through both bodies.
            for j in 0..q {
                let k = row_counts[j] as usize;
                let d = &mut diag[j * BLOCK..(j + 1) * BLOCK];
                d.fill(0.0);
                for a in 0..k {
                    let ra = &row_vels[j * ROWS + a];
                    for b in 0..k {
                        let rb = &row_vels[j * ROWS + b];
                        d[a * ROWS + b] = -(dot(&ra[0], &rb[2]) + dot(&ra[1], &rb[3]));
                    }
                    d[a * ROWS + a] += rows[row_ids[j * ROWS + a] as usize].cfm_gain;
                    row_diag[j * ROWS + a] = d[a * ROWS + a];
                }
            }
            // The blocks of every two joints sharing a body, through that body.
            for body in 0..body_joints_at.len() {
                let list = &body_joints[body_joints_at[body].clone()];
                let n = list.len();
                for x in 0..n {
                    let jx = list[x] as usize;
                    let sx = if joints[jx].sides[0] == body as u32 { 0 } else { 1 };
                    for y in x + 1..n {
                        let jy = list[y] as usize;
                        let sy = if joints[jy].sides[0] == body as u32 { 0 } else { 1 };
                        let at = pair_at(body, x, y);
                        let p = &mut pairs[at..at + BLOCK];
                        p.fill(0.0);
                        for a in 0..row_counts[jx] as usize {
                            let ra = &row_vels[jx * ROWS + a];
                            for b in 0..row_counts[jy] as usize {
                                let rb = &row_vels[jy * ROWS + b];
                                p[a * ROWS + b] = -dot(&ra[sx], &rb[2 + sy]);
                            }
                        }
                    }
                }
            }
            // Eliminate the joints in order.
            for (j, joint) in joints.iter().enumerate() {
                let kj = row_counts[j] as usize;
                invert(
                    &diag[j * BLOCK..(j + 1) * BLOCK],
                    &row_diag[j * ROWS..j * ROWS + kj],
                    kj,
                    &mut dinv[j * BLOCK..(j + 1) * BLOCK],
                    &mut row_kept[j * ROWS..j * ROWS + kj],
                );
                let body = joint.elim_body as usize;
                let list = &body_joints[body_joints_at[body].clone()];
                let n = list.len();
                let x = joint.elim_pos as usize;
                // Keep each block of this joint with a later one, and put in its place the
                // inverse of the joint's block times it.
                for y in x + 1..n {
                    let ky = row_counts[list[y] as usize] as usize;
                    let at = pair_at(body, x, y);
                    let tmp = &mut temp[(y - x - 1) * BLOCK..(y - x) * BLOCK];
                    tmp.copy_from_slice(&pairs[at..at + BLOCK]);
                    let p = &mut pairs[at..at + BLOCK];
                    let di = &dinv[j * BLOCK..(j + 1) * BLOCK];
                    for a in 0..kj {
                        for b in 0..ky {
                            let mut s = 0.0;
                            for c in 0..kj {
                                s += di[a * ROWS + c] * tmp[c * ROWS + b];
                            }
                            p[a * ROWS + b] = s;
                        }
                    }
                }
                // Take the joint out of the blocks of the later joints it shares the body with.
                for y in x + 1..n {
                    let jy = list[y] as usize;
                    let ky = row_counts[jy] as usize;
                    for w in y..n {
                        let jw = list[w] as usize;
                        let kw = row_counts[jw] as usize;
                        let z_at = pair_at(body, x, w);
                        let tmp_y = (y - x - 1) * BLOCK;
                        for a in 0..ky {
                            for b in 0..kw {
                                let mut s = 0.0;
                                for c in 0..kj {
                                    s += temp[tmp_y + c * ROWS + a] * pairs[z_at + c * ROWS + b];
                                }
                                if y == w {
                                    diag[jy * BLOCK + a * ROWS + b] -= s;
                                } else {
                                    let at = pair_at(body, y, w);
                                    pairs[at + a * ROWS + b] -= s;
                                }
                            }
                        }
                    }
                }
            }
        }

        // What each row still asks for, at the velocities the bodies hold now.
        for j in 0..q {
            for a in 0..row_counts[j] as usize {
                let c = &rows[row_ids[j * ROWS + a] as usize];
                let v1 = solver_bodies.get_vel(c.solver_vel1[0]);
                let v2 = solver_bodies.get_vel(c.solver_vel2[0]);
                let jv = c.lin_jac.gdot(v2.linear - v1.linear) + c.ang_jac2.gdot(v2.angular)
                    - c.ang_jac1.gdot(v1.angular);
                rhs[j * ROWS + a] = jv + c.rhs - c.cfm_gain * c.impulse;
            }
        }
        // Forward through the elimination order.
        for (j, joint) in joints.iter().enumerate() {
            let kj = row_counts[j] as usize;
            let body = joint.elim_body as usize;
            let list = &body_joints[body_joints_at[body].clone()];
            let x = joint.elim_pos as usize;
            for y in x + 1..list.len() {
                let jy = list[y] as usize;
                let at = pair_at(body, x, y);
                for b in 0..row_counts[jy] as usize {
                    let mut s = 0.0;
                    for a in 0..kj {
                        s += pairs[at + a * ROWS + b] * rhs[j * ROWS + a];
                    }
                    rhs[jy * ROWS + b] -= s;
                }
            }
        }
        // Back from the root.
        for (j, joint) in joints.iter().enumerate().rev() {
            let kj = row_counts[j] as usize;
            let body = joint.elim_body as usize;
            let list = &body_joints[body_joints_at[body].clone()];
            let x = joint.elim_pos as usize;
            for a in 0..kj {
                let mut s = 0.0;
                for c in 0..kj {
                    s += dinv[j * BLOCK + a * ROWS + c] * rhs[j * ROWS + c];
                }
                for y in x + 1..list.len() {
                    let jy = list[y] as usize;
                    let at = pair_at(body, x, y);
                    for b in 0..row_counts[jy] as usize {
                        s -= pairs[at + a * ROWS + b] * delta[jy * ROWS + b];
                    }
                }
                delta[j * ROWS + a] = s;
            }
        }
        // Apply the impulses.
        for j in 0..q {
            for a in 0..row_counts[j] as usize {
                let slot = j * ROWS + a;
                if !row_kept[slot] {
                    continue;
                }
                let d = delta[slot];
                let c = &mut rows[row_ids[slot] as usize];
                c.impulse += d;
                let mut v1 = solver_bodies.get_vel(c.solver_vel1[0]);
                let mut v2 = solver_bodies.get_vel(c.solver_vel2[0]);
                v1.linear += c.lin_jac * c.im1 * d;
                v1.angular += c.ii_ang_jac1 * d;
                v2.linear -= c.lin_jac * c.im2 * d;
                v2.angular -= c.ii_ang_jac2 * d;
                solver_bodies.set_vel(c.solver_vel1[0], v1);
                solver_bodies.set_vel(c.solver_vel2[0], v2);
            }
        }
        // The rows the exact solve leaves: bounded impulses, and rows left no inverse mass.
        for (j, joint) in joints.iter().enumerate() {
            let exact = &row_ids[j * ROWS..j * ROWS + row_counts[j] as usize];
            let kept = &row_kept[j * ROWS..j * ROWS + row_counts[j] as usize];
            for r in joint_rows[joint.builder as usize].clone() {
                let solved = exact
                    .iter()
                    .zip(kept)
                    .any(|(id, kept)| *id as usize == r && *kept);
                if !solved {
                    rows[r].solve(solver_bodies);
                }
            }
        }
    }
}

/// The inverse of the symmetric `k`-by-`k` block `a` (stride [`ROWS`]) into `out`, through its
/// `L D Lᵀ` factorization. A row whose pivot falls to [`PIVOT_TOLERANCE`] of its own diagonal
/// entry `own` (or below zero) is left out: its row and column of the inverse are zero, and
/// `kept` reads false for it.
fn invert(a: &[Real], own: &[Real], k: usize, out: &mut [Real], kept: &mut [bool]) {
    let mut l = [0.0; BLOCK];
    let mut d = [0.0; ROWS];
    for i in 0..k {
        let mut di = a[i * ROWS + i];
        for m in 0..i {
            di -= l[i * ROWS + m] * l[i * ROWS + m] * d[m];
        }
        kept[i] = own[i] > 0.0 && di > PIVOT_TOLERANCE * own[i];
        d[i] = if kept[i] { di } else { 0.0 };
        for r in i + 1..k {
            let mut s = a[r * ROWS + i];
            for m in 0..i {
                s -= l[r * ROWS + m] * l[i * ROWS + m] * d[m];
            }
            l[r * ROWS + i] = if kept[i] { s / d[i] } else { 0.0 };
        }
    }
    out.fill(0.0);
    for col in 0..k {
        // L y = e_col.
        let mut y = [0.0; ROWS];
        for i in 0..k {
            let mut s = if i == col { 1.0 } else { 0.0 };
            for m in 0..i {
                s -= l[i * ROWS + m] * y[m];
            }
            y[i] = s;
        }
        for i in 0..k {
            y[i] = if kept[i] { y[i] / d[i] } else { 0.0 };
        }
        // Lᵀ x = y.
        let mut x = [0.0; ROWS];
        for i in (0..k).rev() {
            let mut s = y[i];
            for m in i + 1..k {
                s -= l[m * ROWS + i] * x[m];
            }
            x[i] = s;
        }
        for i in 0..k {
            out[i * ROWS + col] = if kept[i] && kept[col] { x[i] } else { 0.0 };
        }
    }
}
