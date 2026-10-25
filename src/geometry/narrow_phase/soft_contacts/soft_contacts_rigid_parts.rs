//! The manifolds of a soft surface against a rigid shape made of many parts (a triangle mesh, a
//! heightfield, a compound): each element of the surface meets the parts that hold it.
//!
//! A triangle mesh finer than the soft surface puts dozens of its triangles within reach of every
//! element (a bowl's triangles fan into its pole), and a manifold for every (element, triangle)
//! pair within reach multiplies the two counts. The contacts the soft-body solver keeps are one per
//! surface vertex and one per rigid feature reaching into an element, so an element meets:
//!
//! - the part closest to each of its vertices (the vertex's contact), and
//! - the parts reaching into it past the surface its vertices meet: the plane through the heights
//!   of the rigid surface under the element's vertices is where the vertex contacts hold the
//!   element, and a part in front of that plane over the element (a bump, a ridge, a spike) would
//!   pass through the element between its vertices. A part on a flat or hollow stretch lies on or
//!   behind that plane, where the vertex contacts already hold the element. Of the parts in front,
//!   the element meets the ones reaching deepest, as many as it has vertices; a part of another
//!   kind than a triangle (a compound's child shape) counts as reaching as deep as its bounding
//!   box.
//!
//! A wire's segment meets every part within reach of it.

use crate::alloc_prelude::*;

use crate::geometry::{ContactData, ContactManifold, ContactManifoldData};
use crate::math::{Pose, Real, Vector};
use crate::utils::{CrossProduct, DotProduct};
use parry::bounding_volume::{Aabb, BoundingVolume};
use parry::query::details::NormalConstraints;
use parry::query::{PersistentQueryDispatcher, PointQuery};
use parry::shape::{CompositeShape, HeightField, Shape, Triangle};
use parry::utils::hashmap::HashMap;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use simba::scalar::{ComplexField as _, RealField as _};

/// How many parts reaching into an element it meets beyond the parts closest to its vertices.
const PROTRUDING_PARTS_PER_ELEMENT: usize = 3;

/// The share of an element's longest edge a part must reach past the plane the vertex contacts hold
/// the element at to count as reaching into it: the plane's own rounding (a flat stretch of
/// triangles finer than the element) stays below it.
const PROTRUSION_TOLERANCE: Real = 0.01;

/// A rigid shape's parts, as the pass reads them.
#[derive(Copy, Clone)]
enum RigidParts<'a> {
    Composite(&'a dyn CompositeShape),
    HeightField(&'a HeightField),
}

impl<'a> RigidParts<'a> {
    fn of(shape: &'a dyn Shape) -> Option<Self> {
        if let Some(c) = shape.as_composite_shape() {
            Some(Self::Composite(c))
        } else {
            shape.as_heightfield().map(Self::HeightField)
        }
    }

    fn local_aabb(&self) -> Aabb {
        match self {
            Self::Composite(c) => c.bvh().root_aabb(),
            Self::HeightField(h) => *h.root_aabb(),
        }
    }

    /// Calls `f` with every part whose bounding box meets `aabb` (in the shape's frame).
    fn for_each_in_aabb(&self, aabb: &Aabb, mut f: impl FnMut(u32)) {
        match self {
            Self::Composite(c) => {
                for leaf in c.bvh().intersect_aabb(aabb) {
                    f(leaf);
                }
            }
            Self::HeightField(h) => h.map_elements_in_local_aabb(aabb, &mut |id, _| f(id)),
        }
    }

    /// Whether some part's bounding box meets `aabb` (in the shape's frame).
    fn any_in_aabb(&self, aabb: &Aabb) -> bool {
        match self {
            Self::Composite(c) => c.bvh().intersect_aabb(aabb).next().is_some(),
            Self::HeightField(h) => {
                let mut any = false;
                h.map_elements_in_local_aabb(aabb, &mut |_, _| any = true);
                any
            }
        }
    }

    /// Calls `f` with every part whose bounding box meets `aabb` (in the shape's frame) where
    /// `reaching` holds for the box, and for every box of the shape's hierarchy holding it.
    fn for_each_reaching(
        &self,
        aabb: &Aabb,
        reaching: impl Fn(&Aabb) -> bool,
        mut f: impl FnMut(u32),
    ) {
        match self {
            Self::Composite(c) => {
                for leaf in c.bvh().leaves(|node| {
                    let node_aabb = node.aabb();
                    node_aabb.intersects(aabb) && reaching(&node_aabb)
                }) {
                    f(leaf);
                }
            }
            Self::HeightField(h) => h.map_elements_in_local_aabb(aabb, &mut |id, triangle| {
                if reaching(&triangle.local_aabb()) {
                    f(id)
                }
            }),
        }
    }

    /// The bounding box of part `id` in the shape's frame.
    fn part_aabb(&self, id: u32) -> Option<Aabb> {
        let mut out = None;
        self.with_part(id, &mut |pose, shape, _| {
            out = Some(match pose {
                Some(pose) => shape.compute_aabb(pose),
                None => shape.compute_local_aabb(),
            });
        });
        out
    }

    /// Calls `f` with part `id`: its pose in the shape's frame (`None` for the identity), its shape
    /// and its normal constraints.
    fn with_part(
        &self,
        id: u32,
        f: &mut dyn FnMut(Option<&Pose>, &dyn Shape, Option<&dyn NormalConstraints>),
    ) {
        match self {
            Self::Composite(c) => c.map_part_at(id, f),
            Self::HeightField(h) => {
                if let Some(triangle) = h.triangle_at_id(id) {
                    let constraints = h.triangle_normal_constraints(id);
                    f(
                        None,
                        &triangle,
                        constraints.as_ref().map(|c| c as &dyn NormalConstraints),
                    );
                }
            }
        }
    }

    /// The triangle part `id` is, in the shape's frame (`None` for a part of another kind).
    fn triangle(&self, id: u32) -> Option<Triangle> {
        let mut out = None;
        self.with_part(id, &mut |pose, shape, _| {
            if let Some(t) = shape.as_triangle() {
                out = Some(match pose {
                    Some(pose) => t.transformed(pose),
                    None => *t,
                });
            }
        });
        out
    }

    /// The point of the shape closest to `point` within `max_dist` (in the shape's frame) and the
    /// part it lies on.
    fn closest(&self, point: Vector, max_dist: Real) -> Option<(u32, Vector)> {
        let project = |id: u32| {
            let mut out = None;
            self.with_part(id, &mut |pose, shape, _| {
                out = Some(match pose {
                    Some(pose) => shape.project_point(pose, point, false).point,
                    None => shape.project_local_point(point, false).point,
                });
            });
            out
        };
        match self {
            Self::Composite(c) => c
                .bvh()
                .find_best(
                    max_dist,
                    |node, _| node.aabb().distance_to_local_point(point, true),
                    |leaf, _| project(leaf).map(|p| ((p - point).length(), p)),
                )
                .map(|(leaf, (_, p))| (leaf, p)),
            Self::HeightField(_) => {
                let mut best: Option<(Real, u32, Vector)> = None;
                let aabb = Aabb::new(point, point).loosened(max_dist);
                self.for_each_in_aabb(&aabb, |id| {
                    if let Some(p) = project(id) {
                        let d = (p - point).length();
                        if d <= max_dist && best.is_none_or(|b| d < b.0) {
                            best = Some((d, id, p));
                        }
                    }
                });
                best.map(|(_, id, p)| (id, p))
            }
        }
    }
}

/// Whether the pair of `shape1` and `shape2` (`soft_first`: `shape1` is the soft surface) is a
/// soft surface against a rigid shape of many parts, whose manifolds [`soft_surface_part_manifolds`]
/// computes.
pub(crate) fn meets_rigid_parts(shape1: &dyn Shape, shape2: &dyn Shape, soft_first: bool) -> bool {
    let (soft, rigid) = if soft_first {
        (shape1, shape2)
    } else {
        (shape2, shape1)
    };
    (soft.as_trimesh().is_some() || soft.as_polyline().is_some()) && RigidParts::of(rigid).is_some()
}

/// A vertex's contact with the rigid shape: the part closest to it and the point of that part.
#[derive(Copy, Clone)]
struct VertexReach {
    part: u32,
    /// The closest point (rigid frame).
    point: Vector,
}

/// The manifolds of the soft surface (`shape1` when `soft_first`, else `shape2`) against the rigid
/// shape of many parts: one per (element, part) pair the module documentation describes, each
/// computed by `dispatcher` and warm-started from the manifold of the same pair in `manifolds`,
/// which it replaces. `pos12` is the pose of `shape2` in `shape1`'s frame and `prediction` the
/// pair's contact reach.
#[allow(clippy::too_many_arguments)]
pub(crate) fn soft_surface_part_manifolds(
    dispatcher: &dyn PersistentQueryDispatcher<ContactManifoldData, ContactData>,
    pos12: &Pose,
    shape1: &dyn Shape,
    shape2: &dyn Shape,
    soft_first: bool,
    prediction: Real,
    manifolds: &mut Vec<ContactManifold>,
) {
    let (soft_shape, rigid_shape) = if soft_first {
        (shape1, shape2)
    } else {
        (shape2, shape1)
    };
    let (Some(soft), Some(rigid)) = (soft_shape.as_composite_shape(), RigidParts::of(rigid_shape))
    else {
        manifolds.clear();
        return;
    };
    // The surface's vertices and elements (a wire's elements are segments).
    let (vertices, triangles, segments): (&[Vector], &[[u32; 3]], &[[u32; 2]]) =
        if let Some(mesh) = soft_shape.as_trimesh() {
            (mesh.vertices(), mesh.indices(), &[])
        } else if let Some(wire) = soft_shape.as_polyline() {
            (wire.vertices(), &[], wire.indices())
        } else {
            manifolds.clear();
            return;
        };
    // The rigid shape's pose in the soft surface's frame, and the reverse.
    let rigid_in_soft = if soft_first { *pos12 } else { pos12.inverse() };
    let soft_in_rigid = rigid_in_soft.inverse();

    // The previous manifolds, by pair, to warm-start the pairs met again.
    let mut old = core::mem::take(manifolds);
    let old_index: HashMap<(u32, u32), usize> = old
        .iter()
        .enumerate()
        .map(|(i, m)| ((m.subshape1, m.subshape2), i))
        .collect();

    let mut reach: HashMap<u32, Option<VertexReach>> = HashMap::default();
    let mut vertex_reach = |v: u32| -> Option<VertexReach> {
        *reach.entry(v).or_insert_with(|| {
            let p = soft_in_rigid * vertices[v as usize];
            rigid
                .closest(p, prediction)
                .map(|(part, point)| VertexReach { part, point })
        })
    };

    let rigid_aabb = rigid
        .local_aabb()
        .transform_by(&rigid_in_soft)
        .loosened(prediction);
    let mut parts: Vec<u32> = Vec::new();
    let mut protruding: Vec<(Real, u32)> = Vec::new();
    for element in soft.bvh().intersect_aabb(&rigid_aabb) {
        parts.clear();
        let e = element as usize;
        if let Some(tri) = triangles.get(e) {
            let points = tri.map(|v| soft_in_rigid * vertices[v as usize]);
            // An element no part comes within reach of meets none.
            if !rigid.any_in_aabb(&Aabb::from_points(points).loosened(prediction)) {
                continue;
            }
            let reaches = tri.map(|v| vertex_reach(v));
            for r in reaches.iter().flatten() {
                if !parts.contains(&r.part) {
                    parts.push(r.part);
                }
            }
            protruding_parts(&rigid, points, reaches, prediction, &parts, &mut protruding);
            for &(_, part) in protruding.iter().take(PROTRUDING_PARTS_PER_ELEMENT) {
                parts.push(part);
            }
        } else if segments.get(e).is_some() {
            // A wire's segment meets every part within reach.
            let mut aabb = Aabb::new_invalid();
            soft.map_part_at(element, &mut |pose, shape, _| {
                let pose = match pose {
                    Some(pose) => soft_in_rigid * *pose,
                    None => soft_in_rigid,
                };
                aabb = shape.compute_aabb(&pose);
            });
            rigid.for_each_in_aabb(&aabb.loosened(prediction), |part| parts.push(part));
        }

        for &part in &parts {
            let key = if soft_first {
                (element, part)
            } else {
                (part, element)
            };
            let mut manifold = match old_index.get(&key) {
                Some(&i) => old[i].take(),
                None => {
                    let mut m = ContactManifold::new();
                    m.subshape1 = key.0;
                    m.subshape2 = key.1;
                    m
                }
            };
            manifold.mark_shapes_deformed();
            soft.map_part_at(element, &mut |soft_pose, soft_part, soft_constraints| {
                rigid.with_part(part, &mut |rigid_pose, rigid_part, rigid_constraints| {
                    let soft_at = soft_pose.copied().unwrap_or(Pose::IDENTITY);
                    let rigid_at = rigid_pose.copied().unwrap_or(Pose::IDENTITY);
                    if soft_first {
                        manifold.set_subshape_pos1(soft_pose.copied());
                        manifold.set_subshape_pos2(rigid_pose.copied());
                        let _ = dispatcher.contact_manifold_convex_convex(
                            &(soft_at.inverse() * rigid_in_soft * rigid_at),
                            soft_part,
                            rigid_part,
                            soft_constraints,
                            rigid_constraints,
                            prediction,
                            &mut manifold,
                        );
                    } else {
                        manifold.set_subshape_pos1(rigid_pose.copied());
                        manifold.set_subshape_pos2(soft_pose.copied());
                        let _ = dispatcher.contact_manifold_convex_convex(
                            &(rigid_at.inverse() * soft_in_rigid * soft_at),
                            rigid_part,
                            soft_part,
                            rigid_constraints,
                            soft_constraints,
                            prediction,
                            &mut manifold,
                        );
                    }
                });
            });
            manifolds.push(manifold);
        }
    }
}

/// The parts reaching into the element of corners `points` (rigid frame) past the plane its vertex
/// contacts `reaches` hold it at, deepest first, into `out`; `selected` are the parts the element
/// meets already.
fn protruding_parts(
    rigid: &RigidParts,
    points: [Vector; 3],
    reaches: [Option<VertexReach>; 3],
    prediction: Real,
    selected: &[u32],
    out: &mut Vec<(Real, u32)>,
) {
    out.clear();
    let [a, b, c] = points;
    let Some(normal) = (b - a).gcross(c - a).try_normalize() else {
        return;
    };
    let longest = (b - a).length().max((c - b).length()).max((a - c).length());
    // The side the rigid shape is on: the vertices' contacts, else the centroid's (an element
    // whose vertices are all out of reach meets the part closest to its centroid).
    let mut toward: Real = reaches
        .iter()
        .zip(points.iter())
        .filter_map(|(r, p)| r.map(|r| (r.point - *p).gdot(normal)))
        .sum();
    if toward == 0.0 {
        let centroid = (a + b + c) / 3.0;
        let Some((part, point)) = rigid.closest(centroid, prediction + longest) else {
            return;
        };
        if !selected.contains(&part) {
            out.push((Real::MAX, part));
        }
        toward = (point - centroid).gdot(normal);
    }
    let n = if toward < 0.0 { -normal } else { normal };
    // The height along `n` of the rigid surface under each vertex: the tangent plane at its closest
    // point, the reach where nothing is within it.
    let heights: [Real; 3] = core::array::from_fn(|k| {
        reaches[k]
            .and_then(|r| {
                let u = r.point - points[k];
                let along = u.gdot(n);
                (along.abs() > 1.0e-6).then(|| (u.gdot(u) / along).min(prediction))
            })
            .unwrap_or(prediction)
    });
    // How far a point reaches past the plane through those heights, toward the element:
    // `heights[0] + (q - a) . gradient`, the plane's height over the element's barycentric frame
    // less the point's own height.
    let (e1, e2) = (b - a, c - a);
    let (d11, d12, d22) = (e1.gdot(e1), e1.gdot(e2), e2.gdot(e2));
    let det = d11 * d22 - d12 * d12;
    if det.abs() <= Real::EPSILON {
        return;
    }
    let gradient = ((e1 * d22 - e2 * d12) * (heights[1] - heights[0])
        + (e2 * d11 - e1 * d12) * (heights[2] - heights[0]))
        / det
        - n;
    let depth_at = |q: &Vector| heights[0] + (*q - a).gdot(gradient);
    // The deepest a box reaches: at the corner farthest along the gradient.
    let box_depth =
        |aabb: &Aabb| depth_at(&aabb.center()) + aabb.half_extents().gdot(gradient.abs());
    let tolerance = PROTRUSION_TOLERANCE * longest;
    // The element's side planes, facing in.
    let sides: [(Vector, Vector); 3] = [(a, b, c), (b, c, a), (c, a, b)].map(|(p, q, r)| {
        let w = n.gcross(q - p);
        (p, if w.gdot(r - p) < 0.0 { -w } else { w })
    });
    // The prism over the element as deep as the reach: a part outside it lies beside the element,
    // or reaches it no deeper than the vertex contacts do.
    let reach = n * prediction;
    let aabb = Aabb::from_points([
        a - reach,
        b - reach,
        c - reach,
        a + reach,
        b + reach,
        c + reach,
    ]);
    rigid.for_each_reaching(
        &aabb,
        |node| box_depth(node) > tolerance,
        |part| {
            if selected.contains(&part) || out.iter().any(|o| o.1 == part) {
                return;
            }
            // A part of another kind than a triangle reaches as deep as its box.
            let Some(triangle) = rigid.triangle(part) else {
                if let Some(part_aabb) = rigid.part_aabb(part) {
                    out.push((box_depth(&part_aabb), part));
                }
                return;
            };
            // The triangle's piece over the element: the depth is affine, so the piece's deepest
            // point is one of its corners, and none is deeper than the whole triangle's deepest.
            let corners = [triangle.a, triangle.b, triangle.c];
            if corners.iter().map(depth_at).fold(Real::MIN, Real::max) <= tolerance {
                return;
            }
            let mut poly: Polygon = corners.into_iter().collect();
            for (origin, inward) in sides {
                poly = clip(&poly, origin, inward);
                if poly.is_empty() {
                    return;
                }
            }
            let depth = poly.iter().map(depth_at).fold(Real::MIN, Real::max);
            if depth > tolerance {
                out.push((depth, part));
            }
        },
    );
    out.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
}

/// A triangle clipped by an element's three side planes (each plane adds at most one corner).
type Polygon = arrayvec::ArrayVec<Vector, 6>;

/// The part of the convex polygon `poly` on the inner side of the plane through `origin` facing
/// `inward`.
fn clip(poly: &[Vector], origin: Vector, inward: Vector) -> Polygon {
    let mut out = Polygon::new();
    for (i, &p) in poly.iter().enumerate() {
        let q = poly[(i + 1) % poly.len()];
        let (dp, dq) = ((p - origin).gdot(inward), (q - origin).gdot(inward));
        if dp >= 0.0 {
            out.push(p);
        }
        if (dp >= 0.0) != (dq >= 0.0) {
            out.push(p + (q - p) * (dp / (dp - dq)));
        }
    }
    out
}
