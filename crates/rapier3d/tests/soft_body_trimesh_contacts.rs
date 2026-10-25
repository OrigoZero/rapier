//! A soft body against a rigid shape of many parts (a triangle mesh, a heightfield): the
//! manifolds the narrow phase builds for it, and the body resting on the shape.

use rapier3d::prelude::*;

/// A hemispherical bowl of radius `radius` resting on y = 0: a lathe of `segments` around by
/// `rings` from the pole to the rim, whose triangles fan into the pole.
fn bowl(radius: Real, segments: u32, rings: u32) -> (Vec<Vector>, Vec<[u32; 3]>) {
    let mut vertices = Vec::new();
    for r in 0..=rings {
        let phi = r as Real / rings as Real * core::f32::consts::FRAC_PI_2;
        let (y, around) = (radius - radius * phi.cos(), radius * phi.sin());
        for s in 0..segments {
            let theta = s as Real / segments as Real * core::f32::consts::TAU;
            vertices.push(Vector::new(around * theta.cos(), y, around * theta.sin()));
        }
    }
    let mut indices = Vec::new();
    for r in 0..rings {
        for s in 0..segments {
            let a = r * segments + s;
            let b = r * segments + (s + 1) % segments;
            let c = (r + 1) * segments + s;
            let d = (r + 1) * segments + (s + 1) % segments;
            indices.push([a, c, b]);
            indices.push([b, c, d]);
        }
    }
    (vertices, indices)
}

/// The engine's `sphere` soft body at resolution 6: a 0.8 m icosphere filled with cells.
fn soft_sphere(center: Vector) -> SoftBodyBuilder {
    let surface = SoftBodyBuilder::sphere(Vector::ZERO, 0.4, 3);
    let vertices: Vec<Vector> = surface.positions.iter().map(|p| *p + center).collect();
    SoftBodyBuilder::volumetric(&vertices, &surface.surface, 0.8 / 5.0).unwrap()
}

/// What a drop measured: the most manifolds and points the soft surface held over the run, and
/// those it held at the end.
struct Drop {
    world: PhysicsWorld,
    body: SoftBodyHandle,
    peak_manifolds: usize,
    peak_points: usize,
    final_manifolds: usize,
}

fn drop_onto(ground: ColliderBuilder, body: SoftBodyBuilder, steps: usize) -> Drop {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / 30.0;
    world.insert(RigidBodyBuilder::fixed(), ground);
    let handle = world.insert_soft_body(body);
    let surface = world.soft_bodies[handle]
        .collision_mesh()
        .unwrap()
        .collider();
    let mut drop = Drop {
        world,
        body: handle,
        peak_manifolds: 0,
        peak_points: 0,
        final_manifolds: 0,
    };
    for _ in 0..steps {
        drop.world.step();
        let (mut manifolds, mut points) = (0, 0);
        for pair in drop.world.narrow_phase.contact_pairs_with(surface) {
            for m in pair.manifolds() {
                manifolds += 1;
                points += m.points.len();
            }
        }
        drop.peak_manifolds = drop.peak_manifolds.max(manifolds);
        drop.peak_points = drop.peak_points.max(points);
        drop.final_manifolds = manifolds;
    }
    drop
}

impl Drop {
    fn particles(&self) -> impl Iterator<Item = Vector> + '_ {
        self.world.soft_bodies[self.body].particle_positions()
    }

    fn skin(&self) -> Real {
        self.world.soft_bodies[self.body].particle_radius()
    }

    fn center(&self) -> Vector {
        self.world.soft_bodies[self.body].center_of_mass()
    }

    fn surface_elements(&self) -> usize {
        self.world.soft_bodies[self.body]
            .collision_mesh()
            .unwrap()
            .indices()
            .len()
    }
}

/// A soft sphere dropped into a 1536-triangle bowl, whose triangles fan into its pole far finer
/// than the sphere's elements, lands and rests at the bottom where the same sphere rests on a box,
/// with every particle inside the bowl, and the manifolds its surface holds stay within a few per
/// element it has: each element meets only the triangles holding it.
#[test]
fn a_soft_body_in_a_triangle_mesh_bowl_meets_the_triangles_holding_it() {
    let (vertices, indices) = bowl(2.0, 48, 16);
    assert_eq!(indices.len(), 1536);
    let mesh = drop_onto(
        ColliderBuilder::trimesh(vertices, indices).unwrap(),
        soft_sphere(Vector::new(0.0, 1.2, 0.0)),
        150,
    );
    let floor = drop_onto(
        ColliderBuilder::cuboid(1.2, 0.05, 1.2).translation(Vector::new(0.0, -0.05, 0.0)),
        soft_sphere(Vector::new(0.0, 1.2, 0.0)),
        150,
    );
    let elements = mesh.surface_elements();
    println!(
        "bowl: {} manifolds / {} points at the peak, {} at rest, {elements} surface elements",
        mesh.peak_manifolds, mesh.peak_points, mesh.final_manifolds
    );
    assert!(
        mesh.peak_manifolds <= 2 * elements,
        "{} manifolds at the peak for {elements} surface elements",
        mesh.peak_manifolds
    );
    assert!(
        mesh.peak_points <= 8 * elements,
        "{} contact points at the peak for {elements} surface elements",
        mesh.peak_points
    );
    assert!(
        mesh.final_manifolds > 0,
        "the sphere does not touch the bowl"
    );
    // Resting at the bottom of the bowl as on the floor (the sphere's fill is asymmetric, so it
    // settles a few centimetres off the pole).
    let (rest, on_floor) = (mesh.center(), floor.center());
    assert!(
        rest.x.abs() < 0.06 && rest.z.abs() < 0.06,
        "the sphere rests off the pole: {rest:?}"
    );
    assert!(
        (rest.y - on_floor.y).abs() < 0.05,
        "the sphere rests at {} in the bowl and at {} on the floor",
        rest.y,
        on_floor.y
    );
    // Every particle is held inside the bowl's sphere, a skin from its surface.
    let skin = mesh.skin();
    let centre = Vector::new(0.0, 2.0, 0.0);
    for p in mesh.particles() {
        let from_centre = (p - centre).length();
        assert!(
            from_centre < 2.0 - 0.5 * skin,
            "a particle at {p:?} passed into the bowl's surface ({from_centre})"
        );
    }
}

/// A soft sphere dropped onto a rigid ball made of 4608 triangles rests on top of it as on a
/// convex ball shape: the triangles reaching into its elements between their vertices (the ball
/// bulges into every flat element resting on it) hold the surface out of the ball.
#[test]
fn a_soft_body_rests_on_a_finely_tessellated_convex_mesh() {
    let (vertices, indices) = Ball::new(1.0).to_trimesh(48, 48);
    let mesh = drop_onto(
        ColliderBuilder::trimesh(vertices, indices).unwrap(),
        soft_sphere(Vector::new(0.0, 1.6, 0.0)),
        150,
    );
    let ball = drop_onto(
        ColliderBuilder::ball(1.0),
        soft_sphere(Vector::new(0.0, 1.6, 0.0)),
        150,
    );
    let elements = mesh.surface_elements();
    println!(
        "ball mesh: {} manifolds / {} points at the peak, {} at rest, {elements} surface elements",
        mesh.peak_manifolds, mesh.peak_points, mesh.final_manifolds
    );
    let skin = mesh.skin();
    for p in mesh.particles() {
        let from_centre = p.length();
        assert!(
            from_centre > 1.0 + 0.5 * skin,
            "a particle at {p:?} passed into the ball ({from_centre})"
        );
    }
    let (rest, on_ball) = (mesh.center(), ball.center());
    assert!(
        (rest - on_ball).length() < 0.05,
        "the sphere rests at {rest:?} on the mesh and at {on_ball:?} on the ball"
    );
    assert!(
        mesh.peak_manifolds <= 4 * elements,
        "{} manifolds at the peak for {elements} surface elements",
        mesh.peak_manifolds
    );
}

/// A cloth of coarse elements dropped across a roof of two triangles drapes over the ridge: the
/// ridge passes under the middle of its elements, far from their vertices, and holds them up.
#[test]
fn a_coarse_cloth_drapes_over_a_ridge_of_two_triangles() {
    let vertices = vec![
        Vector::new(-3.0, 0.0, -3.0),
        Vector::new(-3.0, 0.0, 3.0),
        Vector::new(0.0, 1.0, -3.0),
        Vector::new(0.0, 1.0, 3.0),
        Vector::new(3.0, 0.0, -3.0),
        Vector::new(3.0, 0.0, 3.0),
    ];
    let indices = vec![[0, 1, 3], [0, 3, 2], [2, 3, 5], [2, 5, 4]];
    // A 2.2 x 2 m cloth of 5 x 5 particles, no row of which lies over the ridge.
    let cloth = SoftBodyBuilder::cloth(
        Vector::new(-1.1, 1.5, -1.0),
        Vector::new(0.55, 0.0, 0.0),
        Vector::new(0.0, 0.0, 0.5),
        5,
        5,
    );
    let roof = drop_onto(
        ColliderBuilder::trimesh(vertices, indices).unwrap(),
        cloth,
        120,
    );
    let skin = roof.skin();
    for p in roof.particles() {
        let height = 1.0 - p.x.abs() / 3.0;
        assert!(
            p.y > height - 0.5 * skin,
            "a particle at {p:?} passed through the roof (its surface is at {height})"
        );
    }
    // The cloth stays held up on the ridge.
    let center = roof.center();
    assert!(center.y > 0.7, "the cloth fell off the ridge: {center:?}");
}

/// A soft sphere dropped onto a heightfield of rolling hills finer than its elements rests on
/// them without a particle passing below the ground.
#[test]
fn a_soft_body_rests_on_a_fine_heightfield() {
    let n = 64;
    let size = 6.0;
    let height_at = |x: Real, z: Real| 0.05 * (x * 6.0).sin() * (z * 6.0).cos();
    let heights = Array2::from_fn(n, n, |i, j| {
        let z = -size / 2.0 + size * i as Real / (n - 1) as Real;
        let x = -size / 2.0 + size * j as Real / (n - 1) as Real;
        height_at(x, z)
    });
    let ground = drop_onto(
        ColliderBuilder::heightfield(heights, Vector::new(size, 1.0, size)),
        soft_sphere(Vector::new(0.1, 1.0, 0.1)),
        150,
    );
    let elements = ground.surface_elements();
    println!(
        "heightfield: {} manifolds / {} points at the peak, {} at rest, {elements} surface elements",
        ground.peak_manifolds, ground.peak_points, ground.final_manifolds
    );
    let skin = ground.skin();
    for p in ground.particles() {
        assert!(
            p.y > height_at(p.x, p.z) - 0.5 * skin,
            "a particle at {p:?} passed below the ground ({})",
            height_at(p.x, p.z)
        );
    }
    assert!(
        ground.peak_manifolds <= 4 * elements,
        "{} manifolds at the peak for {elements} surface elements",
        ground.peak_manifolds
    );
    assert!(
        ground.final_manifolds > 0,
        "the sphere does not touch the ground"
    );
}
