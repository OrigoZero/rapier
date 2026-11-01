//! Corotational elastic cells solved [`SIMD_WIDTH`] at a time: the cells of one color share no
//! particle, so the lanes of a batch gather, update, solve and scatter independently.
//!
//! A batch holds a step's whole solve of its cells, lane by lane: their particles' solver slots
//! and inverse masses, their rest shapes, strain coefficients and softened strain blocks, the
//! state each substep's update leaves (the rotation, the strain, the volumetric row's gradient,
//! gain and Schur complement) and the impulses the passes accumulate. The cells'
//! [`SoftElasticConstraint`]s are what the batch is packed from, and the write-back reads each
//! cell's end state from its lane.

use simba::simd::{SimdBool as _, SimdComplexField, SimdPartialOrd, SimdValue};

use super::{
    ElasticEndState, MAX_CONSTRAINT_PARTICLES, STRAIN_ROWS, SoftElasticConstraint,
    SoftElasticModel, StrainVector,
};
use crate::dynamics::solver::solver_body::SolverBodies;
use crate::math::{Real, Rotation, SIMD_WIDTH, SimdBool, SimdReal};

/// The independent entries of a symmetric `STRAIN_ROWS × STRAIN_ROWS` matrix.
const SYMMETRIC_ENTRIES: usize = STRAIN_ROWS * (STRAIN_ROWS + 1) / 2;

/// The index of entry `(i, j)` of a symmetric matrix packed by rows of its upper triangle.
#[inline(always)]
const fn packed(i: usize, j: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    i * STRAIN_ROWS - i * (i + 1) / 2 + j
}

/// A 3-vector of lanes.
type Lanes3 = [SimdReal; 3];
/// A 3×3 matrix of lanes, as its columns.
type Lanes33 = [Lanes3; 3];

#[inline(always)]
fn splat(x: Real) -> SimdReal {
    SimdReal::splat(x)
}

#[inline(always)]
fn dot3(a: &Lanes3, b: &Lanes3) -> SimdReal {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn cross3(a: &Lanes3, b: &Lanes3) -> Lanes3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn determinant(m: &Lanes33) -> SimdReal {
    dot3(&m[0], &cross3(&m[1], &m[2]))
}

/// The strain-row coordinates `(m₀₀, m₁₁, m₂₂, ½(m₀₁ + m₁₀), ½(m₀₂ + m₂₀), ½(m₁₂ + m₂₁))` of
/// `entry(i, j) = mᵢⱼ`.
#[inline(always)]
fn strain_rows(entry: impl Fn(usize, usize) -> SimdReal) -> [SimdReal; STRAIN_ROWS] {
    let half = splat(0.5);
    [
        entry(0, 0),
        entry(1, 1),
        entry(2, 2),
        half * (entry(0, 1) + entry(1, 0)),
        half * (entry(0, 2) + entry(2, 0)),
        half * (entry(1, 2) + entry(2, 1)),
    ]
}

/// The rotation matrix of the quaternions `q` (`x`, `y`, `z`, `w`), as its columns.
#[inline(always)]
fn rotation_matrix(q: &[SimdReal; 4]) -> Lanes33 {
    let [x, y, z, w] = *q;
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    let one = splat(1.0);
    [
        [one - (yy + zz), xy + wz, xz - wy],
        [xy - wz, one - (xx + zz), yz + wx],
        [xz + wy, yz - wx, one - (xx + yy)],
    ]
}

/// The rotation `R` of the polar decomposition `A = R S` of each lane's `a`, warm-started from
/// `q` and iterated until each lane's correction is below `tolerance` radians: the iteration of
/// [`crate::dynamics::soft_body::soft_body_shape_matching::extract_rotation_within`], lane by
/// lane.
fn extract_rotations(a: &Lanes33, q: &mut [SimdReal; 4], tolerance: Real) {
    const MAX_ITERS: usize = 8;
    let mut done = SimdBool::splat(false);
    for _ in 0..MAX_ITERS {
        let r = rotation_matrix(q);
        let mut omega = [splat(0.0); 3];
        let mut denom = splat(0.0);
        for c in 0..3 {
            let rc = cross3(&r[c], &a[c]);
            for k in 0..3 {
                omega[k] += rc[k];
            }
            denom += dot3(&r[c], &a[c]);
        }
        let scale = splat(1.0) / (denom.simd_abs() + splat(1.0e-9));
        let omega = omega.map(|o| o * scale);
        let w = dot3(&omega, &omega).simd_sqrt();
        done = done | w.simd_lt(splat(tolerance));
        if done.all() {
            break;
        }
        // The exact update `exp(ω/2) q` for a large step; a small one takes the first-order
        // quaternion `(1, ω/2)` (renormalized) instead of sin/cos.
        let half = splat(0.5);
        let mut step = [
            half * omega[0],
            half * omega[1],
            half * omega[2],
            splat(1.0),
        ];
        let large = w.simd_gt(splat(0.1));
        if large.any() {
            let (sin, cos) = (half * w).simd_sin_cos();
            let axis_scale = sin / w.simd_max(splat(Real::EPSILON));
            for k in 0..3 {
                step[k] = (omega[k] * axis_scale).select(large, step[k]);
            }
            step[3] = cos.select(large, step[3]);
        }
        // `step * q` (the step applied after `q`), normalized.
        let [sx, sy, sz, sw] = step;
        let [qx, qy, qz, qw] = *q;
        let mut next = [
            sw * qx + sx * qw + sy * qz - sz * qy,
            sw * qy - sx * qz + sy * qw + sz * qx,
            sw * qz + sx * qy - sy * qx + sz * qw,
            sw * qw - sx * qx - sy * qy - sz * qz,
        ];
        let norm = (next[0] * next[0] + next[1] * next[1] + next[2] * next[2] + next[3] * next[3])
            .simd_sqrt();
        let inv_norm = splat(1.0) / norm;
        for k in 0..4 {
            next[k] *= inv_norm;
            q[k] = q[k].select(done, next[k]);
        }
    }
}

/// Stages an empty lane of [`SoftElasticBatch::new`]: zeros, and the identity rotation (its
/// `w` is the second staged value from the end).
fn put_empty_lane<const FIELDS: usize>(staged: &mut [[Real; SIMD_WIDTH]; FIELDS], l: usize) {
    for field in staged.iter_mut() {
        field[l] = 0.0;
    }
    staged[FIELDS - 2][l] = 1.0;
}

/// [`SIMD_WIDTH`] corotational cells of one color, lane by lane.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SoftElasticBatch {
    /// Each lane's cell: its index in the constraints set's elastic constraints, `u32::MAX` for
    /// an empty lane.
    pub constraints: [u32; SIMD_WIDTH],
    /// The solver slot of each particle of each lane's cell, `u32::MAX` where there is none.
    ids: [[u32; SIMD_WIDTH]; MAX_CONSTRAINT_PARTICLES],
    /// Whether each particle of each lane is a solver body (its position is read from it).
    simulated: [SimdBool; MAX_CONSTRAINT_PARTICLES],
    im: [SimdReal; MAX_CONSTRAINT_PARTICLES],
    /// The particles' positions at the last update (the assembly's for those that are not
    /// solver bodies).
    pos: [Lanes3; MAX_CONSTRAINT_PARTICLES],
    /// The inverse rest edge matrix, as its columns.
    inv_rest: Lanes33,
    /// The cells' strain coefficients (`∂F/∂x`, see [`SoftElasticConstraint::coefficients`]).
    coeffs: [Lanes3; MAX_CONSTRAINT_PARTICLES],
    /// The inverse of the softened strain block, packed (see [`packed`]).
    inv_a: [SimdReal; SYMMETRIC_ENTRIES],
    erp_strain: [SimdReal; STRAIN_ROWS],
    cfm_strain: [SimdReal; STRAIN_ROWS],
    strain_cap: [SimdReal; STRAIN_ROWS],
    erp_vol: SimdReal,
    cfm_coeff_vol: SimdReal,
    vol_cap: SimdReal,
    /// Whether the volumetric row is active.
    volumetric: SimdBool,
    /// The cell rotation (quaternions `x`, `y`, `z`, `w`), warm start of each update.
    quaternion: [SimdReal; 4],
    /// The cell rotation of the substep, as columns.
    rotation: Lanes33,
    /// The strain of the last update (cell frame, clamped).
    strain: [SimdReal; STRAIN_ROWS],
    grad_vol: [Lanes3; MAX_CONSTRAINT_PARTICLES],
    rhs_vol: SimdReal,
    cfm_gain_vol: SimdReal,
    coupling: [SimdReal; STRAIN_ROWS],
    coupling_image: [SimdReal; STRAIN_ROWS],
    inv_schur: SimdReal,
    strain_impulse: [SimdReal; STRAIN_ROWS],
    vol_impulse: SimdReal,
}

impl SoftElasticBatch {
    /// Packs the corotational cells `lanes` (indices into `constraints`, `u32::MAX` for an empty
    /// lane) of one color, as the assembly left them.
    pub fn new(constraints: &[SoftElasticConstraint], lanes: [u32; SIMD_WIDTH]) -> Self {
        // Each lane's values, written field after field (in the order they are read back below),
        // then turned into lanes field by field.
        const FIELDS: usize = 4 * MAX_CONSTRAINT_PARTICLES
            + 3 * 3
            + 3 * MAX_CONSTRAINT_PARTICLES
            + SYMMETRIC_ENTRIES
            + 4 * STRAIN_ROWS
            + 3
            + 4
            + 1;
        let mut staged = [[0.0; SIMD_WIDTH]; FIELDS];
        let mut ids = [[u32::MAX; SIMD_WIDTH]; MAX_CONSTRAINT_PARTICLES];
        let mut simulated = [[false; SIMD_WIDTH]; MAX_CONSTRAINT_PARTICLES];
        let mut volumetric = [false; SIMD_WIDTH];
        for (l, &ci) in lanes.iter().enumerate() {
            let mut f = 0;
            let mut put = |value: Real| {
                staged[f][l] = value;
                f += 1;
            };
            if ci == u32::MAX {
                // An empty lane: zeros, and the identity rotation.
                put_empty_lane(&mut staged, l);
                continue;
            }
            let c = &constraints[ci as usize];
            let SoftElasticModel::Corotational(model) = &c.model else {
                put_empty_lane(&mut staged, l);
                continue;
            };
            for p in 0..MAX_CONSTRAINT_PARTICLES {
                ids[p][l] = c.solver_ids[p];
                simulated[p][l] = c.solver_ids[p] != u32::MAX;
                put(c.im[p]);
                for k in 0..3 {
                    put(c.pos[p][k]);
                }
            }
            for j in 0..3 {
                let column = c.inv_rest_matrix.col(j);
                for k in 0..3 {
                    put(column[k]);
                }
            }
            for p in 0..MAX_CONSTRAINT_PARTICLES {
                for k in 0..3 {
                    put(c.coeffs[p][k]);
                }
            }
            for i in 0..STRAIN_ROWS {
                for j in i..STRAIN_ROWS {
                    put(c.inv_a[(i, j)]);
                }
            }
            for r in 0..STRAIN_ROWS {
                put(model.erp_strain[r]);
                put(model.cfm_strain[r]);
                put(c.strain_cap[r]);
                put(c.strain_impulse[r]);
            }
            put(model.erp_vol);
            put(model.cfm_coeff_vol);
            put(model.vol_cap);
            put(c.rotation.x);
            put(c.rotation.y);
            put(c.rotation.z);
            put(c.rotation.w);
            put(c.vol_impulse);
            volumetric[l] = model.volumetric;
        }
        let mut fields = staged.iter().map(|lanes| SimdReal::from(*lanes));
        let mut next = || fields.next().unwrap_or_else(|| splat(0.0));
        let mut im = [splat(0.0); MAX_CONSTRAINT_PARTICLES];
        let mut pos = [[splat(0.0); 3]; MAX_CONSTRAINT_PARTICLES];
        for p in 0..MAX_CONSTRAINT_PARTICLES {
            im[p] = next();
            pos[p] = [next(), next(), next()];
        }
        let inv_rest: Lanes33 = core::array::from_fn(|_| [next(), next(), next()]);
        let coeffs: [Lanes3; MAX_CONSTRAINT_PARTICLES] =
            core::array::from_fn(|_| [next(), next(), next()]);
        let inv_a: [SimdReal; SYMMETRIC_ENTRIES] = core::array::from_fn(|_| next());
        let mut erp_strain = [splat(0.0); STRAIN_ROWS];
        let mut cfm_strain = [splat(0.0); STRAIN_ROWS];
        let mut strain_cap = [splat(0.0); STRAIN_ROWS];
        let mut strain_impulse = [splat(0.0); STRAIN_ROWS];
        for r in 0..STRAIN_ROWS {
            erp_strain[r] = next();
            cfm_strain[r] = next();
            strain_cap[r] = next();
            strain_impulse[r] = next();
        }
        let erp_vol = next();
        let cfm_coeff_vol = next();
        let vol_cap = next();
        let quaternion = [next(), next(), next(), next()];
        let vol_impulse = next();
        let zero = splat(0.0);
        let zero3 = [zero; 3];
        Self {
            constraints: lanes,
            ids,
            simulated: simulated.map(SimdBool::from),
            im,
            pos,
            inv_rest,
            coeffs,
            inv_a,
            erp_strain,
            cfm_strain,
            strain_cap,
            erp_vol,
            cfm_coeff_vol,
            vol_cap,
            volumetric: SimdBool::from(volumetric),
            quaternion,
            rotation: [zero3; 3],
            strain: [zero; STRAIN_ROWS],
            grad_vol: [zero3; MAX_CONSTRAINT_PARTICLES],
            rhs_vol: zero,
            cfm_gain_vol: zero,
            coupling: [zero; STRAIN_ROWS],
            coupling_image: [zero; STRAIN_ROWS],
            inv_schur: zero,
            strain_impulse,
            vol_impulse,
        }
    }

    /// Takes lane `l`'s rest shape, strain coefficients and rows from its cell `c` after the cell
    /// flowed plastically, its strain rows keeping `retained` of their impulses (see
    /// `flow_elastic_cell`).
    pub fn refit_lane(&mut self, l: usize, c: &SoftElasticConstraint, retained: Real) {
        let SoftElasticModel::Corotational(model) = &c.model else {
            return;
        };
        for j in 0..3 {
            let column = c.inv_rest_matrix.col(j);
            for k in 0..3 {
                self.inv_rest[j][k].replace(l, column[k]);
            }
        }
        for p in 0..MAX_CONSTRAINT_PARTICLES {
            for k in 0..3 {
                self.coeffs[p][k].replace(l, c.coeffs[p][k]);
            }
        }
        for i in 0..STRAIN_ROWS {
            for j in i..STRAIN_ROWS {
                self.inv_a[packed(i, j)].replace(l, c.inv_a[(i, j)]);
            }
        }
        for r in 0..STRAIN_ROWS {
            self.erp_strain[r].replace(l, model.erp_strain[r]);
            self.cfm_strain[r].replace(l, model.cfm_strain[r]);
            self.strain_cap[r].replace(l, c.strain_cap[r]);
            let impulse = self.strain_impulse[r].extract(l);
            self.strain_impulse[r].replace(l, impulse * retained);
        }
        self.erp_vol.replace(l, model.erp_vol);
        self.cfm_coeff_vol.replace(l, model.cfm_coeff_vol);
        self.vol_cap.replace(l, model.vol_cap);
        self.volumetric.replace(l, model.volumetric);
    }

    /// The substep's update of every lane's cell from the particles' positions: its rotation,
    /// strain, volumetric gradient, gain and Schur complement (see
    /// [`SoftElasticConstraint::update`]). Returns the lanes strained past `min_strain`.
    pub fn update(
        &mut self,
        bodies: &SolverBodies,
        rotation_tolerance: Real,
        min_strain: Real,
    ) -> SimdBool {
        for p in 0..MAX_CONSTRAINT_PARTICLES {
            let gathered = bodies.gather_translations(self.ids[p]);
            for k in 0..3 {
                self.pos[p][k] = gathered[k].select(self.simulated[p], self.pos[p][k]);
            }
        }
        let edges: Lanes33 =
            core::array::from_fn(|e| core::array::from_fn(|k| self.pos[e + 1][k] - self.pos[0][k]));
        // F = Ds Dm⁻¹, column by column.
        let f: Lanes33 = core::array::from_fn(|j| {
            core::array::from_fn(|k| {
                edges[0][k] * self.inv_rest[j][0]
                    + edges[1][k] * self.inv_rest[j][1]
                    + edges[2][k] * self.inv_rest[j][2]
            })
        });
        extract_rotations(&f, &mut self.quaternion, rotation_tolerance);
        let r = rotation_matrix(&self.quaternion);
        self.rotation = r;
        let det = determinant(&f);
        let inverted = det.simd_lt(splat(0.0));
        let one = splat(1.0);
        // `Rᵀ F - I` in strain rows, clamped: at the snap-back caps while the cell is inverted,
        // at 100 % strain (or the cap when larger) otherwise.
        let raw = strain_rows(|i, j| {
            let s = dot3(&r[i], &f[j]);
            if i == j { s - one } else { s }
        });
        let mut strained = SimdBool::splat(false);
        let min_strain = splat(min_strain);
        for row in 0..STRAIN_ROWS {
            let cap = self.strain_cap[row].select(inverted, self.strain_cap[row].simd_max(one));
            let strain = raw[row].simd_clamp(-cap, cap);
            self.strain[row] = strain;
            strained = strained | strain.simd_abs().simd_gt(min_strain);
        }

        // The volumetric row: `C_V = det(F) - 1`, its gradient through `F = Ds Dm⁻¹`.
        let cofactor = [
            cross3(&f[1], &f[2]),
            cross3(&f[2], &f[0]),
            cross3(&f[0], &f[1]),
        ];
        let mut g0 = [splat(0.0); 3];
        for j in 0..3 {
            let column: Lanes3 = core::array::from_fn(|k| {
                cofactor[0][k] * self.inv_rest[0][j]
                    + cofactor[1][k] * self.inv_rest[1][j]
                    + cofactor[2][k] * self.inv_rest[2][j]
            });
            for k in 0..3 {
                g0[k] -= column[k];
            }
            self.grad_vol[j + 1] = column;
        }
        self.grad_vol[0] = g0;
        let vol_cap = self.vol_cap.select(inverted, self.vol_cap.simd_max(one));
        let c_vol = (det - one).simd_clamp(-vol_cap, vol_cap);

        // The volumetric effective mass, and its coupling with the strain rows (cell frame):
        // `bᵣ = Σₚ wₚ gᵣ[p]·(Rᵀ∇C_V)ₚ`, the strain rows of `Rᵀ Σₚ wₚ ∇C_Vₚ cₚᵀ`.
        let mut a_vv = splat(0.0);
        let mut weighted: Lanes33 = [[splat(0.0); 3]; 3];
        for p in 0..MAX_CONSTRAINT_PARTICLES {
            let g = self.grad_vol[p];
            let im = self.im[p];
            a_vv += im * dot3(&g, &g);
            for (j, column) in weighted.iter_mut().enumerate() {
                let c = self.coeffs[p][j] * im;
                for k in 0..3 {
                    column[k] += g[k] * c;
                }
            }
        }
        let b = strain_rows(|i, j| dot3(&r[i], &weighted[j]));
        self.cfm_gain_vol = a_vv * self.cfm_coeff_vol;
        let z: [SimdReal; STRAIN_ROWS] = core::array::from_fn(|i| {
            let mut sum = splat(0.0);
            for (j, b) in b.iter().enumerate() {
                sum += self.inv_a[packed(i, j)] * *b;
            }
            sum
        });
        let mut bz = splat(0.0);
        for row in 0..STRAIN_ROWS {
            bz += b[row] * z[row];
        }
        let schur = a_vv + self.cfm_gain_vol - bz;
        let zero = splat(0.0);
        let usable = self.volumetric & schur.simd_ne(zero);
        self.inv_schur = (one / schur).select(usable, zero);
        self.coupling = b;
        self.coupling_image = z;
        self.rhs_vol = c_vol * self.erp_vol;
        let valid = SimdBool::from(array![|l| self.constraints[l] != u32::MAX]);
        strained & valid
    }

    /// Scales the accumulated impulses by `coefficient` and applies them: the warm start of a
    /// substep's first pass.
    pub fn warmstart(&mut self, bodies: &mut SolverBodies, coefficient: Real) {
        let coefficient = splat(coefficient);
        for r in 0..STRAIN_ROWS {
            self.strain_impulse[r] *= coefficient;
        }
        self.vol_impulse *= coefficient;
        let (strain, vol) = (self.strain_impulse, self.vol_impulse);
        let (mut vels, words) = self.gather(bodies);
        self.scatter_impulses(bodies, &mut vels, &words, &strain, vol);
    }

    /// One block Gauss-Seidel iteration of every lane's cell, or of the lanes `mask` holds.
    pub fn solve(&mut self, bodies: &mut SolverBodies, mask: Option<SimdBool>) {
        let (mut vels, words) = self.gather(bodies);
        // The velocity gradient `L = Σₚ vₚ cₚᵀ` (columns), seen in the cell frame: `Rᵀ L`.
        let mut gradient: Lanes33 = [[splat(0.0); 3]; 3];
        for (j, column) in gradient.iter_mut().enumerate() {
            for p in 0..MAX_CONSTRAINT_PARTICLES {
                let c = self.coeffs[p][j];
                for k in 0..3 {
                    column[k] += vels[p][k] * c;
                }
            }
        }
        let rate = strain_rows(|i, j| dot3(&self.rotation[i], &gradient[j]));
        let mut vol_rate = splat(0.0);
        for p in 0..MAX_CONSTRAINT_PARTICLES {
            vol_rate += dot3(&self.grad_vol[p], &vels[p]);
        }
        let res: [SimdReal; STRAIN_ROWS] = core::array::from_fn(|r| {
            rate[r] + self.erp_strain[r] * self.strain[r]
                - self.cfm_strain[r] * self.strain_impulse[r]
        });
        let r_vol = self.rhs_vol - self.cfm_gain_vol * self.vol_impulse + vol_rate;
        let y: [SimdReal; STRAIN_ROWS] = core::array::from_fn(|i| {
            let mut sum = splat(0.0);
            for (j, res) in res.iter().enumerate() {
                sum += self.inv_a[packed(i, j)] * *res;
            }
            sum
        });
        // Schur complement: the volumetric row first, the strain rows corrected by the coupling.
        let mut coupled = splat(0.0);
        for r in 0..STRAIN_ROWS {
            coupled += self.coupling[r] * y[r];
        }
        let mut d_vol = self.inv_schur * (r_vol - coupled);
        let mut d_strain: [SimdReal; STRAIN_ROWS] =
            core::array::from_fn(|r| y[r] - self.coupling_image[r] * d_vol);
        if let Some(mask) = mask {
            let zero = splat(0.0);
            d_vol = d_vol.select(mask, zero);
            for d in &mut d_strain {
                *d = d.select(mask, zero);
            }
        }
        for r in 0..STRAIN_ROWS {
            self.strain_impulse[r] += d_strain[r];
        }
        self.vol_impulse += d_vol;
        self.scatter_impulses(bodies, &mut vels, &words, &d_strain, d_vol);
    }

    /// The particles' linear velocities, and the words sharing their blocks.
    #[inline(always)]
    fn gather(
        &self,
        bodies: &SolverBodies,
    ) -> (
        [Lanes3; MAX_CONSTRAINT_PARTICLES],
        [SimdReal; MAX_CONSTRAINT_PARTICLES],
    ) {
        let mut vels = [[splat(0.0); 3]; MAX_CONSTRAINT_PARTICLES];
        let mut words = [splat(0.0); MAX_CONSTRAINT_PARTICLES];
        for p in 0..MAX_CONSTRAINT_PARTICLES {
            let [x, y, z, w] = bodies.gather_linear_vels(self.ids[p]);
            vels[p] = [x, y, z];
            words[p] = w;
        }
        (vels, words)
    }

    /// Takes the velocity change of strain impulses `strain` (cell frame) and a volumetric
    /// impulse `vol` off `vels`, and writes them back.
    #[inline(always)]
    fn scatter_impulses(
        &self,
        bodies: &mut SolverBodies,
        vels: &mut [Lanes3; MAX_CONSTRAINT_PARTICLES],
        words: &[SimdReal; MAX_CONSTRAINT_PARTICLES],
        strain: &[SimdReal; STRAIN_ROWS],
        vol: SimdReal,
    ) {
        // The symmetric strain tensor of the impulses (`½ λ` on the off-diagonal entries), turned
        // into the world frame: `R T`, column by column.
        let half = splat(0.5);
        let tensor = [
            [strain[0], half * strain[3], half * strain[4]],
            [half * strain[3], strain[1], half * strain[5]],
            [half * strain[4], half * strain[5], strain[2]],
        ];
        let world: Lanes33 = core::array::from_fn(|j| {
            core::array::from_fn(|k| {
                self.rotation[0][k] * tensor[j][0]
                    + self.rotation[1][k] * tensor[j][1]
                    + self.rotation[2][k] * tensor[j][2]
            })
        });
        for p in 0..MAX_CONSTRAINT_PARTICLES {
            let c = &self.coeffs[p];
            let im = self.im[p];
            for k in 0..3 {
                let dv = world[0][k] * c[0]
                    + world[1][k] * c[1]
                    + world[2][k] * c[2]
                    + self.grad_vol[p][k] * vol;
                vels[p][k] -= dv * im;
            }
            let [x, y, z] = vels[p];
            bodies.scatter_linear_vels(self.ids[p], [x, y, z, words[p]]);
        }
    }

    /// The lanes whose cells `strained` flags, if any.
    pub fn mask(&self, strained: impl Fn(u32) -> bool) -> Option<SimdBool> {
        let lanes: [bool; SIMD_WIDTH] = core::array::from_fn(|l| {
            self.constraints[l] != u32::MAX && strained(self.constraints[l])
        });
        lanes.iter().any(|s| *s).then(|| SimdBool::from(lanes))
    }

    /// Lane `l`'s cell as the step left it.
    pub fn lane(&self, l: usize) -> ElasticEndState {
        let q = self.quaternion.map(|q| q.extract(l));
        ElasticEndState {
            rotation: Rotation::from_xyzw(q[0], q[1], q[2], q[3]),
            strain: StrainVector::from_fn(|r, _| self.strain[r].extract(l)),
            strain_impulse: StrainVector::from_fn(|r, _| self.strain_impulse[r].extract(l)),
            vol_impulse: self.vol_impulse.extract(l),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc_prelude::*;
    use crate::dynamics::SoftBody;
    use crate::dynamics::solver::soft_constraint::soft_element_constraint::{
        CELL_ROTATION_TOLERANCE, CorotationalConstraint,
    };
    use crate::math::{Matrix, Vector};

    /// A corotational cell over particles `first..first + 4`, deformed and turned away from its
    /// rest shape, with impulses to warm-start from; the solver bodies hold its particles moving.
    fn cell(first: u32, seed: Real, bodies: &mut SolverBodies) -> SoftElasticConstraint {
        let rest = [
            Vector::new(0.0, 0.0, 0.0),
            Vector::new(0.3, 0.0, 0.05),
            Vector::new(0.02, 0.25, 0.0),
            Vector::new(0.0, 0.04, 0.28),
        ];
        let inv_rest_matrix = SoftBody::cell_edge_matrix(rest).inverse();
        let coeffs = SoftElasticConstraint::coefficients(&inv_rest_matrix);
        let im = [3.0, 2.5, 4.0 + seed, 0.0];
        let turn = Rotation::from_scaled_axis(Vector::new(0.3 * seed, -0.2, 0.5));
        let solver_ids = core::array::from_fn(|p| first + p as u32);
        for p in 0..MAX_CONSTRAINT_PARTICLES {
            let squash = Vector::new(1.0 + 0.1 * seed, 0.92, 1.05);
            let pose = &mut bodies.poses[(first + p as u32) as usize];
            pose.translation = turn * (rest[p] * squash) + Vector::new(seed, 1.0, -2.0);
            let vel = &mut bodies.vels[(first + p as u32) as usize];
            vel.linear = Vector::new(
                0.4 * seed - p as Real * 0.3,
                1.3 - p as Real,
                0.7 * p as Real,
            );
        }
        let block = SoftElasticConstraint::strain_block(&coeffs, &im);
        let cfm_strain = StrainVector::from_fn(|r, _| 0.05 * block[(r, r)]);
        let mut softened = block;
        for r in 0..STRAIN_ROWS {
            softened[(r, r)] += cfm_strain[r];
        }
        let inv_a = SoftElasticConstraint::invert_block(softened, [false; STRAIN_ROWS]);
        SoftElasticConstraint {
            soft_body: 0,
            element: first,
            solver_ids,
            pos: rest,
            im,
            coeffs,
            inv_rest_matrix,
            rotation: Rotation::from_scaled_axis(Vector::new(0.25 * seed, -0.15, 0.45)),
            rot_mat: Matrix::IDENTITY,
            inv_a,
            strain_cap: StrainVector::repeat(0.4),
            strain: StrainVector::zeros(),
            inverted: false,
            strain_impulse: StrainVector::from_fn(|r, _| 0.01 * (r as Real - 2.0) * seed),
            strain_per_impulse: StrainVector::zeros(),
            vol_impulse: -0.02 * seed,
            model: SoftElasticModel::Corotational(CorotationalConstraint {
                erp_strain: StrainVector::repeat(12.0),
                cfm_strain,
                erp_vol: 9.0,
                cfm_coeff_vol: 0.03,
                volumetric: true,
                vol_cap: 0.6,
                grad_vol: [Vector::ZERO; MAX_CONSTRAINT_PARTICLES],
                cfm_gain_vol: 0.0,
                rhs_vol: 0.0,
                b: StrainVector::zeros(),
                z: StrainVector::zeros(),
                inv_s: 0.0,
            }),
            flow: None,
        }
    }

    fn close(a: Real, b: Real, what: &str) {
        assert!(
            (a - b).abs() <= 2.0e-4 * (1.0 + a.abs().max(b.abs())),
            "{what}: scalar {a} vs batch {b}"
        );
    }

    /// A batch's update, warm start and passes leave its cells' particles, impulses, rotations
    /// and strains where the cells solved one at a time leave them, and an empty lane touches
    /// nothing.
    #[test]
    fn a_batch_solves_its_cells_as_the_cells_solve_alone() {
        let lanes = SIMD_WIDTH - 1;
        let mut scalar_bodies = SolverBodies::default();
        scalar_bodies.resize(4 * SIMD_WIDTH + 1);
        let mut constraints = Vec::new();
        for l in 0..lanes {
            constraints.push(cell(
                4 * l as u32,
                0.5 + l as Real * 0.37,
                &mut scalar_bodies,
            ));
        }
        let mut batch_bodies = SolverBodies::default();
        batch_bodies.resize(4 * SIMD_WIDTH + 1);
        batch_bodies.vels.copy_from_slice(&scalar_bodies.vels);
        batch_bodies.poses.copy_from_slice(&scalar_bodies.poses);
        let untouched = 4 * SIMD_WIDTH;
        scalar_bodies.vels[untouched].linear = Vector::new(5.0, 6.0, 7.0);
        batch_bodies.vels[untouched].linear = Vector::new(5.0, 6.0, 7.0);

        let ids = core::array::from_fn(|l| if l < lanes { l as u32 } else { u32::MAX });
        let mut batch = SoftElasticBatch::new(&constraints, ids);
        let min_strain = 0.05;
        let mut scalar_strained = [false; SIMD_WIDTH];
        for (l, c) in constraints.iter_mut().enumerate() {
            c.update(&scalar_bodies);
            scalar_strained[l] = c.is_strained(min_strain);
            c.strain_impulse *= 0.9;
            c.vol_impulse *= 0.9;
            c.warmstart(&mut scalar_bodies);
            for _ in 0..4 {
                c.solve(&mut scalar_bodies);
            }
        }
        let strained = batch.update(&batch_bodies, CELL_ROTATION_TOLERANCE, min_strain);
        batch.warmstart(&mut batch_bodies, 0.9);
        for _ in 0..4 {
            batch.solve(&mut batch_bodies, None);
        }

        for (l, c) in constraints.iter().enumerate() {
            assert_eq!(strained.extract(l), scalar_strained[l], "lane {l} strained");
            let end = batch.lane(l);
            for r in 0..STRAIN_ROWS {
                close(c.strain[r], end.strain[r], "strain");
                close(c.strain_impulse[r], end.strain_impulse[r], "strain impulse");
            }
            close(c.vol_impulse, end.vol_impulse, "volumetric impulse");
            assert!(
                c.rotation.dot(end.rotation).abs() > 1.0 - 1.0e-5,
                "lane {l} rotation"
            );
        }
        for (a, b) in scalar_bodies.vels.iter().zip(&batch_bodies.vels) {
            for k in 0..3 {
                close(a.linear[k], b.linear[k], "velocity");
                close(a.angular[k], b.angular[k], "angular velocity");
            }
        }
        assert!(
            !strained.extract(SIMD_WIDTH - 1),
            "an empty lane is never strained"
        );
        assert_eq!(
            batch_bodies.vels[untouched].linear,
            Vector::new(5.0, 6.0, 7.0)
        );

        // A masked pass moves only the lanes it holds.
        let before = batch_bodies.vels.clone();
        let mask = batch.mask(|c| c == 1).unwrap();
        batch.solve(&mut batch_bodies, Some(mask));
        for p in 0..4 {
            assert_eq!(
                before[p].linear, batch_bodies.vels[p].linear,
                "lane 0 is outside the mask"
            );
        }
    }
}
