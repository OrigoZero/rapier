//! `cargo run -p rapier3d --features "profiler" --release --example crash_bench [resolution] [hz]
//! [speed] [seconds] [young_modulus] [--profile] [--series]`:
//! a soft shell held to a rigid chassis driven into a narrow post, the scene of a car crash whose
//! dent is as narrow as the post. Reports the per-step time (average, median, 95th percentile,
//! worst), the stage split, the contacts and the dent the shell kept.

use rapier3d::prelude::*;
use std::time::Instant;

struct Scene {
    world: PhysicsWorld,
    shell: SoftBodyHandle,
    car: RigidBodyHandle,
}

fn scene(resolution: usize, hz: Real, speed: Real, young_modulus: Real) -> Scene {
    let mut world = PhysicsWorld::new();
    world.integration_parameters.dt = 1.0 / hz;
    world.integration_parameters.num_solver_iterations = 4;
    let chassis_group = Group::GROUP_1;
    let shell_group = Group::GROUP_2;
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.5, 0.0)),
        ColliderBuilder::cuboid(3.0, 0.5, 30.0),
    );
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, 1.0, 10.0)),
        ColliderBuilder::cuboid(0.1, 1.0, 0.1),
    );
    let (car, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, 0.15, -6.0))
            .linvel(Vector::new(0.0, 0.0, speed)),
        ColliderBuilder::cuboid(0.8, 0.15, 1.6)
            .mass(600.0)
            .friction(0.0)
            .collision_groups(InteractionGroups::new(
                chassis_group,
                !shell_group,
                InteractionTestMode::default(),
            )),
    );
    let extent = Vector::new(1.8, 0.8, 4.4);
    let longest = 4.4;
    let count = |axis: Real| -> usize {
        let cells = (resolution as Real - 1.0) * axis / longest;
        (cells.round() as usize + 1).max(2)
    };
    let center = Vector::new(0.0, 0.72, -6.0);
    let builder = SoftBodyBuilder::cuboid(
        center,
        extent * 0.5,
        count(extent.x),
        count(extent.y),
        count(extent.z),
    )
    .cell_model(SoftBodyCellModel::Corotational)
    .material(SoftBodyMaterial {
        young_modulus,
        poisson_ratio: 0.3,
        elastic_damping_ratio: 0.5,
        plastic_yield: 0.02,
        plastic_creep: 10.0,
        plastic_max: 0.5,
        edge_plastic_yield: 0.02,
        edge_plastic_creep: 10.0,
        edge_plastic_max: 0.5,
        ..Default::default()
    })
    .mass(40.0)
    .volume_preservation(true)
    .surface_collider(
        ColliderBuilder::ball(0.05)
            .density(0.0)
            .friction(0.6)
            .restitution(0.1)
            .collision_groups(InteractionGroups::new(
                shell_group,
                !chassis_group,
                InteractionTestMode::default(),
            )),
    );
    let floor_y = center.y - extent.y * 0.5 + extent.y * 0.1;
    let floor: Vec<usize> = builder
        .particle_positions()
        .iter()
        .enumerate()
        .filter(|(_, p)| p.y <= floor_y + 1.0e-4)
        .map(|(i, _)| i)
        .collect();
    let shell = world.insert_soft_body(builder);
    {
        let PhysicsWorld {
            soft_bodies,
            bodies,
            ..
        } = &mut world;
        let sb = &mut soft_bodies[shell];
        for i in 0..sb.num_particles() {
            sb.set_particle_velocity(i, Vector::new(0.0, 0.0, speed));
        }
        for i in floor {
            sb.attach_particle(i, car, bodies);
        }
    }
    Scene { world, shell, car }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() as f64 - 1.0) * q).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let resolution: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(24);
    let hz: Real = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let speed: Real = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(13.9);
    let seconds: Real = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let young_modulus: Real = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(20000.0);
    let mut sc = scene(resolution, hz, speed, young_modulus);
    sc.world.physics_pipeline.counters.enable();
    let steps = (seconds * hz) as usize;
    let mut times = Vec::with_capacity(steps);
    let mut stage = [0.0f64; 6];
    let mut worst_step = 0;
    let mut max_contacts = 0;
    let mut substeps = Vec::with_capacity(steps);
    let mut total_substeps = 0usize;
    for k in 0..steps {
        let root = sc.world.soft_bodies[sc.shell].root_body();
        let extra = sc.world.bodies[root].additional_solver_iterations();
        substeps.push(4 + extra);
        total_substeps += 4 + extra;
        let start = Instant::now();
        sc.world.step();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let c = &sc.world.physics_pipeline.counters;
        stage[0] += c.cd.broad_phase_time.time_ms();
        stage[1] += c.cd.narrow_phase_time.time_ms();
        stage[2] += c.stages.island_construction_time.time_ms();
        stage[3] += c.solver.velocity_assembly_time.time_ms();
        stage[4] += c.solver.velocity_resolution_time.time_ms();
        stage[5] += c.solver.velocity_writeback_time.time_ms();
        max_contacts = max_contacts.max(c.solver.ncontacts);
        if ms > times.iter().copied().fold(0.0, f64::max) {
            worst_step = k;
        }
        times.push(ms);
    }
    let sb = &sc.world.soft_bodies[sc.shell];
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // The dent the shell kept: each particle's permanent set (how far its rest position flowed),
    // the largest and the mean, and the front face's.
    let car = sc.world.bodies[sc.car].position();
    let n = sb.particles().len();
    let initial: Vec<Vector> = sb
        .particles()
        .iter()
        .map(|p| p.initial_rest_position())
        .collect();
    let max_z = initial.iter().map(|p| p.z).fold(Real::MIN, Real::max);
    let mut front: Vec<(Real, Real)> = Vec::new();
    let mut largest: Real = 0.0;
    let mut mean: Real = 0.0;
    for (i, p) in sb.particles().iter().enumerate() {
        let set = (p.rest_position() - initial[i]).length();
        largest = largest.max(set);
        mean += set / n as Real;
        if initial[i].z > max_z - 1.0e-3 {
            front.push((initial[i].x, set));
        }
    }
    // The front's profile across x: each column's mean permanent set over the front face. The
    // dent is the excess over the least-set column, and its width the span over which that excess
    // is at least half its peak (interpolated between columns).
    let mut cols: Vec<(Real, Real, usize)> = Vec::new();
    for (x, set) in &front {
        match cols.iter_mut().find(|c| (c.0 - x).abs() < 1.0e-3) {
            Some(c) => {
                c.1 += set;
                c.2 += 1;
            }
            None => cols.push((*x, *set, 1)),
        }
    }
    cols.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let profile: Vec<(Real, Real)> = cols.iter().map(|c| (c.0, c.1 / c.2 as Real)).collect();
    let base = profile.iter().map(|c| c.1).fold(Real::MAX, Real::min);
    let peak = profile.iter().map(|c| c.1 - base).fold(0.0, Real::max);
    let depth = peak;
    let half = 0.5 * peak;
    let mut lo = Real::MAX;
    let mut hi = Real::MIN;
    for k in 0..profile.len() {
        let (x, e) = (profile[k].0, profile[k].1 - base);
        if e >= half {
            lo = lo.min(x);
            hi = hi.max(x);
            if k > 0 {
                let (x0, e0) = (profile[k - 1].0, profile[k - 1].1 - base);
                if e0 < half {
                    lo = lo.min(x0 + (x - x0) * (half - e0) / (e - e0));
                }
            }
            if k + 1 < profile.len() {
                let (x1, e1) = (profile[k + 1].0, profile[k + 1].1 - base);
                if e1 < half {
                    hi = hi.max(x + (x1 - x) * (e - half) / (e - e1));
                }
            }
        }
    }
    let width = if peak > 0.0 { hi - lo } else { 0.0 };
    println!(
        "res {resolution} hz {hz} particles {n} cells {} steps {} | avg {avg:.2} med {:.2} p95 {:.2} worst {:.2} (step {worst_step}) | bp {:.2} np {:.2} isl {:.2} asm {:.2} res {:.2} wb {:.2} per step | contacts max {max_contacts} substeps {:.2} | dent depth {depth:.3} width {width:.3} set max {largest:.3} mean {mean:.4} car z {:.2}",
        sb.cells().len(),
        times.len(),
        percentile(&sorted, 0.5),
        percentile(&sorted, 0.95),
        sorted[sorted.len() - 1],
        stage[0] / steps as f64,
        stage[1] / steps as f64,
        stage[2] / steps as f64,
        stage[3] / steps as f64,
        stage[4] / steps as f64,
        stage[5] / steps as f64,
        total_substeps as f64 / steps as f64,
        car.translation.z,
    );
    if args.iter().any(|a| a == "--profile") {
        let mut cols: Vec<(i32, Real, usize)> = Vec::new();
        for (x, set) in &front {
            let key = (x * 1000.0).round() as i32;
            match cols.iter_mut().find(|c| c.0 == key) {
                Some(c) => {
                    c.1 += set;
                    c.2 += 1;
                }
                None => cols.push((key, *set, 1)),
            }
        }
        cols.sort_by_key(|c| c.0);
        for c in cols {
            print!("x {:.2}: {:.3}  ", c.0 as Real / 1000.0, c.1 / c.2 as Real);
        }
        println!();
        println!(
            "edges {} dihedrals {} cells {}",
            sb.edges().len(),
            sb.dihedrals().len(),
            sb.cells().len()
        );
    }
    if args.iter().any(|a| a == "--series") {
        for (k, t) in times.iter().enumerate() {
            println!("{k} {t:.2} {}", substeps[k]);
        }
    }
}
