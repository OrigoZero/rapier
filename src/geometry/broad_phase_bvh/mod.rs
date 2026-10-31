use crate::alloc_prelude::*;
use crate::data::Coarena;
use crate::dynamics::IntegrationParameters;
use crate::geometry::{Aabb, ColliderHandle};
use crate::math::Real;
use parry::partitioning::{Bvh, BvhLeafUpdateStatus, BvhWorkspace};
use parry::utils::hashmap::HashMap;

mod update;

/// The broad-phase collision detector that quickly filters out distant object pairs.
///
/// The broad-phase is the "first pass" of collision detection. It uses a hierarchical
/// bounding volume tree (BVH) to quickly identify which collider pairs are close enough
/// to potentially collide, avoiding expensive narrow-phase checks for distant objects.
///
/// Think of it as a "spatial index" that answers: "Which objects are near each other?"
///
/// You typically don't interact with this directly - it's managed by [`PhysicsPipeline`](crate::pipeline::PhysicsPipeline).
/// However, you can use it to create a [`QueryPipeline`](crate::pipeline::QueryPipeline) for spatial queries.
#[derive(Default, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct BroadPhaseBvh {
    pub(crate) tree: Bvh,
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    workspace: BvhWorkspace,
    #[cfg_attr(
        feature = "serde-serialize",
        serde(
            serialize_with = "serialize_pairs",
            deserialize_with = "crate::utils::serde::deserialize_from_vec_tuple"
        )
    )]
    pairs: HashMap<(ColliderHandle, ColliderHandle), u32>,
    /// For each collider, the other colliders it currently forms a pair with. Lets
    /// stale-pair detection examine only pairs adjacent to changed colliders instead of
    /// re-scanning the whole `pairs` map (a pair can only stop overlapping if one side changed).
    pair_adjacency: Coarena<Vec<ColliderHandle>>,
    /// Workspace buffer holding the colliders whose AABB was updated in the tree
    /// during the last `update` call.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    updated_colliders: Vec<ColliderHandle>,
    /// Workspace buffer holding the leaf pairs reported by the tree traversal. Only the
    /// sequential traversal needs it (it reports through a closure); the parallel one
    /// returns its own vector.
    #[cfg(not(feature = "parallel"))]
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    candidates_workspace: Vec<(u32, u32)>,
    /// Workspace: per-collider "was updated this step" bit (collider arena index), so the
    /// stale-pair scan can visit a pair from one side only when both sides moved.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    updated_mask: Vec<bool>,
    /// Workspace buffer holding the stale pairs detected during `update`.
    ///
    /// The boolean indicates if a `DeletePair` event must be emitted for the pair.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    stale_pairs: Vec<(ColliderHandle, ColliderHandle, bool)>,
    /// Leaves updated at the previous `update` call — (a superset of) the leaves whose
    /// change flag the previous refit set; partial refitting needs it to clear those flags.
    ///
    /// Note that this needs to be serialized for determinism after snapshot restore.
    prev_updated_leaves: Vec<u32>,
    /// Leaves updated in the tree during the current `update` call.
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    curr_updated_leaves: Vec<u32>,
    /// Colliders whose tree leaf was updated through [`Self::set_aabb`] since the last
    /// `update` call (e.g. by the physics pipeline at the end of the previous step).
    /// They count as changed colliders for the next `update` call.
    pending_set_aabb: Vec<ColliderHandle>,
    /// Quality-degrading tree changes (in-place leaf updates, removals) since the last
    /// incremental optimization; re-inserted leaves don't count (SAH re-insertion is
    /// self-optimizing). Periodic optimization is skipped while small relative to tree size.
    changes_since_optimize: u32,
    /// True when the previous `update` saw few leaves change: that regime relocates moved leaves via
    /// SAH re-insertion (tree quality without an O(tree) optimizer/refit pass); bulk regimes keep cheaper
    /// in-place updates + the periodic optimizer. One step of hysteresis: `set_aabb` runs between updates.
    reinsert_leaf_updates: bool,
    /// Workspace buffer for the precomputed leaf updates of [`Self::update`].
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    update_workspace: Vec<(ColliderHandle, Aabb, Real)>,
    /// Workspace of the parallel leaf-update batches (the tree API takes raw
    /// leaf indices).
    #[cfg(feature = "parallel")]
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    update_batch_workspace: Vec<(Aabb, u32, Real)>,
    #[cfg(feature = "parallel")]
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    update_batch_statuses: Vec<BvhLeafUpdateStatus>,
    frame_index: u32,
    optimization_strategy: BvhOptimizationStrategy,
    /// If enabled, each tree leaf's change-detection margin adapts to the collider's size
    /// (12.5% of its smallest AABB extent, capped) instead of a fixed fraction of the
    /// length unit (default: `false`). Large shapes then keep their leaf and candidate
    /// pairs valid across bigger displacements — fewer tree updates and pair re-checks,
    /// at the cost of slightly fatter AABBs (more candidate pairs for the narrow-phase).
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    pub adaptive_change_detection_margin: bool,
    /// True when the last `update` deferred its (quality-only) BVH optimization pass
    /// so the physics pipeline can run it concurrently with the narrow phase and
    /// solver; consumed by [`Self::take_deferred_optimize`].
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    deferred_optimize_pending: bool,
    /// Colliders whose pairs were forgotten through [`Self::forget_pairs_of`] since the
    /// last `update` call, ordered by index. The next update deletes every pair each one
    /// forms and re-inserts its leaf, so the pairs it still overlaps are reported as new
    /// ones.
    #[cfg_attr(feature = "serde-serialize", serde(default))]
    forgotten_colliders: Vec<ColliderHandle>,
}

// TODO: would be interesting to try out:
// "Fast Insertion-Based Optimization of Bounding Volume Hierarchies"
// by Bittner et al.
/// Selection of strategies to maintain through time the broad-phase BVH in shape that remains
/// efficient for collision-detection and scene queries.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Default, PartialEq, Eq, Copy, Clone)]
pub enum BvhOptimizationStrategy {
    /// Different sub-trees of the BVH will be optimized at each frame.
    #[default]
    SubtreeOptimizer,
    /// Disables incremental BVH optimization (discouraged).
    ///
    /// This should not be used except for debugging purpose.
    None,
}

/// Runs the deferred (quality-only) optimization pass on `tree`.
///
/// The parallel and sequential refits produce the same nodes (parry pins that in
/// `refit_parallel_matches_sequential`), so which one runs is a pure execution choice.
pub(crate) fn run_bvh_optimize(tree: &mut Bvh, workspace: &mut BvhWorkspace) {
    tree.optimize_incremental(workspace);
    // Flag-preserving refit: the change-detection flags were already resolved by
    // the partial refit that ran before this step's pair traversal, and the next
    // step's traversal must see them untouched.
    #[cfg(feature = "parallel")]
    tree.refit_without_resolve_parallel(workspace);
    #[cfg(not(feature = "parallel"))]
    tree.refit_without_resolve(workspace);
}

/// A pending (quality-only) BVH optimization pass, extracted from the broad-phase so
/// it can run on another thread while the rest of the step doesn't touch the tree.
///
/// Deferred in every build, so every consumer sees the same tree at the same point of
/// the step: with a spare worker the pass runs concurrently, otherwise it runs inline
/// at the join point (see `PhysicsPipeline::join_deferred_bvh_optimize`). Running it
/// eagerly instead would optimize the tree *before* this step's pair traversal rather
/// than after — a different tree, hence different pairs.
pub(crate) struct DeferredBvhOptimize {
    tree: Bvh,
    workspace: BvhWorkspace,
}

impl DeferredBvhOptimize {
    pub(crate) fn run(&mut self) {
        run_bvh_optimize(&mut self.tree, &mut self.workspace);
    }
}

/// Serializes the pair map by collider index, so the bytes describe the pair *set* rather
/// than the map's insertion history (see `serialize_sorted_to_vec_tuple`).
#[cfg(feature = "serde-serialize")]
fn serialize_pairs<S: serde::Serializer>(
    pairs: &HashMap<(ColliderHandle, ColliderHandle), u32>,
    s: S,
) -> Result<S::Ok, S::Error> {
    crate::utils::serde::serialize_sorted_to_vec_tuple(
        pairs,
        |(a, b)| (a.into_raw_parts(), b.into_raw_parts()),
        s,
    )
}

impl BroadPhaseBvh {
    const CHANGE_DETECTION_ENABLED: bool = true;
    // Fraction of the length unit each tree leaf is fattened by (movement within the skin
    // leaves tree and pairs untouched; pairs appear up to `2 * factor` early). 0.04 keeps
    // broad-phase cost low without a measurable narrow-phase hit.
    const CHANGE_DETECTION_FACTOR: Real = 4.0e-2;
    /// Upper bound of the adaptive change-detection margin, as a fraction of the
    /// length unit (see [`Self::adaptive_change_detection_margin`]).
    const ADAPTIVE_CHANGE_DETECTION_CAP: Real = 0.25;

    /// Initializes a new empty broad-phase.
    pub fn new() -> Self {
        Self::default()
    }

    /// The change-detection margin (fat-AABB skin) for a leaf with the given AABB:
    /// a fixed fraction of the length unit, or, with
    /// [`Self::adaptive_change_detection_margin`], proportional to the shape's
    /// smallest extent and kept within [fixed margin, cap].
    fn change_detection_skin(&self, params: &IntegrationParameters, aabb: &Aabb) -> Real {
        if !Self::CHANGE_DETECTION_ENABLED {
            0.0
        } else if self.adaptive_change_detection_margin {
            let min_extent = aabb.extents().min_element();
            (min_extent * 0.125).clamp(
                Self::CHANGE_DETECTION_FACTOR * params.length_unit,
                Self::ADAPTIVE_CHANGE_DETECTION_CAP * params.length_unit,
            )
        } else {
            Self::CHANGE_DETECTION_FACTOR * params.length_unit
        }
    }

    /// Initializes a new empty broad-phase with the specified strategy for incremental
    /// BVH optimization.
    pub fn with_optimization_strategy(optimization_strategy: BvhOptimizationStrategy) -> Self {
        Self {
            optimization_strategy,
            ..Default::default()
        }
    }

    /// Extracts the deferred BVH optimization pass requested by the last [`Self::update`],
    /// if any, moving the tree out of the broad-phase. The tree MUST be handed back through
    /// [`Self::finish_deferred_optimize`] before anything else uses this broad-phase.
    pub(crate) fn take_deferred_optimize(&mut self) -> Option<DeferredBvhOptimize> {
        self.deferred_optimize_pending.then(|| {
            self.deferred_optimize_pending = false;
            DeferredBvhOptimize {
                tree: core::mem::replace(&mut self.tree, Bvh::new()),
                workspace: core::mem::take(&mut self.workspace),
            }
        })
    }

    /// Puts back the tree extracted by [`Self::take_deferred_optimize`].
    pub(crate) fn finish_deferred_optimize(&mut self, task: DeferredBvhOptimize) {
        self.tree = task.tree;
        self.workspace = task.workspace;
    }

    /// Forgets every pair the collider `handle` forms, as though it had just been added.
    ///
    /// The next [`Self::update`] reports each pair the collider formed as a
    /// [`DeletePair`](crate::geometry::BroadPhasePairEvent::DeletePair), so the narrow-phase drops the pair with its
    /// contacts and the impulses they carry into the next solve, and reports every
    /// collider the collider's AABB overlaps then as a [`AddPair`](crate::geometry::BroadPhasePairEvent::AddPair),
    /// which the narrow-phase builds afresh. The deletions come first, in the order of
    /// the colliders' indices, whatever order the colliders were forgotten in.
    ///
    /// A body placed somewhere by the user (a teleport, a level put back where it
    /// started) forgets the pairs of its colliders, so the step after the placement
    /// finds its contacts the way the first step after the body was added finds them:
    /// a scene whose every dynamic body is placed back steps as it did when it was built.
    ///
    /// A collider forgotten again before the next update is held once.
    pub fn forget_pairs_of(&mut self, handle: ColliderHandle) {
        if let Err(at) = self
            .forgotten_colliders
            .binary_search_by_key(&handle.into_raw_parts(), |h| h.into_raw_parts())
        {
            self.forgotten_colliders.insert(at, handle);
        }
    }

    /// Sets the AABB associated to the given collider.
    ///
    /// The change is immediately applied and propagated through the underlying BVH;
    /// change detection accounts for it during the next broad-phase update.
    pub fn set_aabb(&mut self, params: &IntegrationParameters, handle: ColliderHandle, aabb: Aabb) {
        let change_detection_skin = self.change_detection_skin(params, &aabb);
        let leaf_index = handle.into_raw_parts().0;
        // Same regime split as the `update` loop: small change volumes relocate
        // moved leaves through self-optimizing SAH re-insertion, bulk volumes use
        // in-place updates (and count toward the periodic optimizer).
        let status = if self.reinsert_leaf_updates {
            self.tree.reinsert_or_update_with_change_detection(
                aabb,
                leaf_index,
                change_detection_skin,
            )
        } else {
            self.tree
                .insert_with_change_detection(aabb, leaf_index, change_detection_skin)
        };
        match status {
            // The new AABB stayed within the leaf's fattened AABB: the tree was left
            // untouched, so the next `update` has nothing to refit or re-check for
            // this collider.
            BvhLeafUpdateStatus::Unchanged => {}
            BvhLeafUpdateStatus::UpdatedInPlace | BvhLeafUpdateStatus::Inserted => {
                if !self.reinsert_leaf_updates && status == BvhLeafUpdateStatus::UpdatedInPlace {
                    self.changes_since_optimize = self.changes_since_optimize.saturating_add(1);
                }
                self.pending_set_aabb.push(handle);
            }
        }
    }
}

#[cfg(test)]
#[cfg(all(feature = "dim3", feature = "f32"))]
mod test {
    #[allow(unused_imports)]
    use crate::alloc_prelude::*;
    use crate::math::Vector;
    use crate::prelude::{
        CCDSolver, ColliderBuilder, ColliderSet, DefaultBroadPhase, ImpulseJointSet,
        IntegrationParameters, IslandManager, MultibodyJointSet, NarrowPhase, PhysicsPipeline,
        RigidBodyBuilder, RigidBodySet, SoftBodySet,
    };

    /// With the adaptive change-detection margin enabled, collisions must still be
    /// detected and resolved like with the fixed margin (the margin only affects how
    /// often tree leaves are updated, not which pairs eventually collide).
    #[test]
    fn adaptive_change_detection_margin_smoke() {
        let mut final_ys = [0.0; 2];

        for (i, adaptive) in [false, true].into_iter().enumerate() {
            let mut bodies = RigidBodySet::new();
            let mut colliders = ColliderSet::new();
            let mut impulse_joints = ImpulseJointSet::new();
            let mut multibody_joints = MultibodyJointSet::new();
            let mut soft_bodies = SoftBodySet::new();
            let mut pipeline = PhysicsPipeline::new();
            let mut islands = IslandManager::new();
            let mut broad_phase = DefaultBroadPhase::new();
            broad_phase.adaptive_change_detection_margin = adaptive;
            let mut narrow_phase = NarrowPhase::new();
            let mut ccd = CCDSolver::new();

            colliders.insert(ColliderBuilder::cuboid(10.0, 0.5, 10.0));
            let ball =
                bodies.insert(RigidBodyBuilder::dynamic().translation(Vector::new(0.0, 4.0, 0.0)));
            colliders.insert_with_parent(ColliderBuilder::ball(0.5), ball, &mut bodies);

            let params = IntegrationParameters::default();
            for _ in 0..200 {
                pipeline.step(
                    Vector::new(0.0, -9.81, 0.0),
                    &params,
                    &mut islands,
                    &mut broad_phase,
                    &mut narrow_phase,
                    &mut bodies,
                    &mut colliders,
                    &mut impulse_joints,
                    &mut multibody_joints,
                    &mut soft_bodies,
                    &mut ccd,
                    &(),
                    &(),
                );
            }

            final_ys[i] = bodies[ball].translation().y;
        }

        // Both must rest on the floor (0.5 half-thickness + 0.5 radius).
        for y in final_ys {
            assert!(
                (y - 1.0).abs() < 0.02,
                "ball did not rest on the floor: y = {y}"
            );
        }
    }

    /// The bits of every dynamic body's pose and velocities, step after step, of a
    /// stack of boxes struck by a ball: once from the scene as it was built, then
    /// (in the same world) once more after every body was placed back where it
    /// started, with or without its colliders' pairs forgotten.
    fn stack_struck_twice(forget: bool) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut impulse_joints = ImpulseJointSet::new();
        let mut multibody_joints = MultibodyJointSet::new();
        let mut soft_bodies = SoftBodySet::new();
        let mut pipeline = PhysicsPipeline::new();
        let mut islands = IslandManager::new();
        let mut broad_phase = DefaultBroadPhase::new();
        let mut narrow_phase = NarrowPhase::new();
        let mut ccd = CCDSolver::new();

        colliders.insert(ColliderBuilder::cuboid(8.0, 0.5, 8.0).translation(Vector::new(0.0, -0.5, 0.0)));
        let mut start = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                let at = Vector::new(col as f32 * 1.02 - 1.02, 0.5 + row as f32, 0.0);
                let body = bodies.insert(RigidBodyBuilder::dynamic().translation(at).can_sleep(false));
                colliders.insert_with_parent(ColliderBuilder::cuboid(0.5, 0.5, 0.5), body, &mut bodies);
                start.push((body, at));
            }
        }
        let ball_at = Vector::new(0.0, 1.0, 4.0);
        let ball = bodies.insert(RigidBodyBuilder::dynamic().translation(ball_at).can_sleep(false));
        colliders.insert_with_parent(ColliderBuilder::ball(0.4), ball, &mut bodies);
        start.push((ball, ball_at));

        let mut params = IntegrationParameters::default();
        params.dt = 1.0 / 30.0;
        let mut run = |bodies: &mut RigidBodySet,
                       colliders: &mut ColliderSet,
                       broad_phase: &mut DefaultBroadPhase| {
            bodies[ball].set_linvel(Vector::new(0.0, 0.0, -8.0), true);
            let mut record = Vec::new();
            for _ in 0..90 {
                pipeline.step(
                    Vector::new(0.0, -9.81, 0.0),
                    &params,
                    &mut islands,
                    broad_phase,
                    &mut narrow_phase,
                    bodies,
                    colliders,
                    &mut impulse_joints,
                    &mut multibody_joints,
                    &mut soft_bodies,
                    &mut ccd,
                    &(),
                    &(),
                );
                let mut bits = Vec::new();
                for (body, _) in &start {
                    let rb = &bodies[*body];
                    let pose = rb.position();
                    let (t, r) = (pose.translation, pose.rotation);
                    for v in [t.x, t.y, t.z, r.x, r.y, r.z, r.w] {
                        bits.push(v.to_bits());
                    }
                    for v in [rb.linvel(), rb.angvel()] {
                        bits.extend([v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]);
                    }
                }
                record.push(bits);
            }
            record
        };

        let first = run(&mut bodies, &mut colliders, &mut broad_phase);
        for (body, at) in &start {
            let rb = &mut bodies[*body];
            rb.set_position(crate::math::Pose::from_translation(*at), true);
            rb.set_linvel(Vector::ZERO, true);
            rb.set_angvel(Vector::ZERO, true);
            if forget {
                for collider in rb.colliders() {
                    broad_phase.forget_pairs_of(*collider);
                }
            }
        }
        let second = run(&mut bodies, &mut colliders, &mut broad_phase);
        (first, second)
    }

    /// A scene whose every dynamic body is placed back where it started, with its
    /// colliders' pairs forgotten, steps to the same bits as it did when it was built:
    /// the contacts and their warm-start impulses left by the first run do not reach
    /// the second.
    #[test]
    fn a_scene_placed_back_with_its_pairs_forgotten_steps_as_it_was_built() {
        let (first, second) = stack_struck_twice(true);
        let parted = first.iter().zip(&second).position(|(a, b)| a != b);
        assert_eq!(parted, None, "the second run parted from the first at step {parted:?}");
    }

    /// The same scene placed back without forgetting its pairs carries the contacts
    /// of the first run into the second, which parts from the first: what the test
    /// above measures is the forgetting.
    #[test]
    fn a_scene_placed_back_keeping_its_pairs_steps_on_from_their_impulses() {
        let (first, second) = stack_struck_twice(false);
        assert_ne!(first, second);
    }
}
