//! A FEM soft body's elastic motion follows its material at every substep length: a body with no
//! damping released from a stretch keeps the energy of its oscillation for as long as it runs, and
//! a damped one loses it at the same rate whatever the substep length.

#![cfg(all(feature = "dim3", feature = "fem"))]

use rapier3d::prelude::*;

/// The largest kinetic energy the cube holds in each half second after it is released
/// stretched by 10% along x, with no gravity, no material damping and nothing to touch, the FEM
/// solver's stiffness damping at `stiffness_damping` seconds.
fn kinetic_peaks(rate: Real, substeps: usize, stiffness_damping: Real) -> Vec<Real> {
    let mut world = PhysicsWorld::new();
    world.gravity = Vector::ZERO;
    world.integration_parameters.dt = 1.0 / rate;
    world.integration_parameters.num_solver_iterations = substeps;
    world
        .integration_parameters
        .soft_bodies
        .fem
        .stiffness_damping = stiffness_damping;
    let mut builder = SoftBodyBuilder::cuboid(Vector::ZERO, Vector::splat(0.5), 4, 4, 4)
        .cell_model(SoftBodyCellModel::Corotational)
        .solver(SoftBodySolver::Fem)
        .mass(10.0)
        .volume_preservation(false)
        .can_sleep(false)
        .material(SoftBodyMaterial {
            edge_softness: SpringCoefficients::new(0.0, 0.0),
            bend_softness: SpringCoefficients::new(0.0, 0.0),
            volume_softness: SpringCoefficients::new(0.0, 0.0),
            young_modulus: 2.0e4,
            poisson_ratio: 0.3,
            elastic_damping_ratio: 0.0,
            ..Default::default()
        })
        .no_surface_collider();
    let h = world.insert_soft_body(builder.clone());
    for p in &mut builder.positions {
        p.x *= 1.1;
    }
    for (i, p) in builder.positions.iter().enumerate() {
        world.soft_bodies[h].set_particle_position(i, *p);
    }
    let half_second = (rate / 2.0) as usize;
    let mut peaks = Vec::new();
    let mut peak: Real = 0.0;
    for step in 0..4 * 2 * half_second {
        world.step();
        let particles = world.soft_bodies[h].particles();
        let mass = 10.0 / particles.len() as Real;
        let kinetic: Real = particles
            .iter()
            .map(|p| 0.5 * mass * p.velocity().length_squared())
            .sum();
        peak = peak.max(kinetic);
        if (step + 1) % half_second == 0 {
            peaks.push(peak);
            peak = 0.0;
        }
    }
    peaks
}

#[test]
fn an_undamped_fem_body_keeps_the_energy_of_its_oscillation_at_every_substep_length() {
    for (rate, substeps) in [(60.0, 1), (60.0, 2), (120.0, 2), (240.0, 4)] {
        let peaks = kinetic_peaks(rate, substeps, 0.0);
        let first = peaks[0];
        assert!(
            first > 1.0,
            "{rate} x {substeps}: the cube barely moved: {peaks:?}"
        );
        // The modes the stretch set going trade energy among themselves, so each half second's
        // peak wanders about the first; none of them decays away or grows.
        for (k, peak) in peaks.iter().enumerate() {
            assert!(
                *peak > 0.6 * first && *peak < 1.4 * first,
                "{rate} x {substeps}: half second {k} peaks at {peak} J against {first} J in the first: {peaks:?}"
            );
        }
    }
}

#[test]
fn a_damped_fem_body_loses_the_energy_of_its_oscillation_alike_at_every_substep_length() {
    let damping = SoftFemParameters::default().stiffness_damping;
    assert!(damping > 0.0);
    // Each rate's energy left after one, two and three seconds, as a share of its first half
    // second's: one material, one decay.
    let decays: Vec<Vec<Real>> = [(60.0, 1), (60.0, 2), (120.0, 2), (240.0, 4)]
        .iter()
        .map(|&(rate, substeps)| {
            let peaks = kinetic_peaks(rate, substeps, damping);
            assert!(
                peaks[0] > 1.0,
                "{rate} x {substeps}: the cube barely moved: {peaks:?}"
            );
            [1, 3, 5].iter().map(|&k| peaks[k] / peaks[0]).collect()
        })
        .collect();
    let reference = &decays[decays.len() - 1];
    for decay in &decays {
        for (share, expected) in decay.iter().zip(reference) {
            assert!(
                (share - expected).abs() < 0.1 * expected.max(0.01),
                "the energy left depends on the substep length: {decays:?}"
            );
        }
    }
    // The damping takes energy out: none of the rates keeps the first half second's.
    for decay in &decays {
        assert!(
            decay[2] < 0.9,
            "the damped cube kept its energy: {decays:?}"
        );
    }
}
