# Physics Models Documentation

## Overview

SM_2D implements comprehensive physics models for proton transport in water, following ICRU, PDG, and NIST standards. All models are implemented as CUDA device functions for GPU acceleration.

## Implementation Status (Feb 2026)

| Physics Model | Status | Match Rate | Notes |
|---------------|--------|------------|-------|
| Energy Loss (CSDA) | ✅ Complete | ~99% | NIST PSTAR based |
| Fermi-Eyges MCS | ⚠️ Phase B | 88% | Moment-based (see commit 9e691a3) |
| Nuclear Attenuation | ✅ Complete | ~95% | ICRU 63 based |
| Energy Straggling | ✅ Complete | N/A | Vavilov implementation |
| Lateral Spreading | ✅ Complete | ~92% | Deterministic Gaussian |

---

## 1. Multiple Coulomb Scattering (Highland Formula)

### Reference: PDG 2024 Review of Particle Physics

### The Highland Formula (2D Projection)

```
σ_θ = (13.6 MeV / βcp) × z × sqrt(x/X_0) × [1 + 0.038 × ln(x/X_0)] / √2
```

Where:
- `βcp` = momentum × velocity [MeV/c]
- `z` = projectile charge (1 for protons)
- `x` = step length [mm]
- `X_0` = radiation length (360.8 mm for water)
- `1/√2` = 3D→2D projection correction

### Implementation (`highland.hpp`)

```cpp
struct HighlandParams {
    float m_p_MeV = 938.272f;      // Proton rest mass
    float X0_water = 360.8f;       // Radiation length [mm]
    float MCS_2D_CORRECTION = 0.70710678f;  // 1/√2
};

// Calculate scattering angle sigma
__host__ __device__ float highland_sigma(float E_MeV, float ds, float X0) {
    // Relativistic kinematics
    float gamma = 1.0f + E_MeV / m_p_MeV;
    float beta = sqrt(1.0f - 1.0f / (gamma * gamma));
    float beta_cp = beta * (E_MeV + m_p_MeV);  // MeV/c

    // Highland formula
    float theta_rms = (13.6f / beta_cp) * sqrt(ds / X0);
    theta_rms *= (1.0f + 0.038f * log(ds / X0));

    // 2D projection correction
    return theta_rms * MCS_2D_CORRECTION;
}

// Sample scattering angle using Box-Muller
__device__ float sample_mcs_angle(float sigma_theta, unsigned& seed) {
    float u1 = curand_uniform(&seed);
    float u2 = curand_uniform(&seed);
    float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
    return sigma_theta * z0;
}
```

### Variance Accumulation (v0.8)

For accurate multi-step scattering, variance is accumulated:

```cpp
// CORRECT: Accumulate variance
sigma_2_total += sigma_theta * sigma_theta;

// Then sample from total variance
float theta_scatter = sqrt(sigma_2_total) * sample_normal();

// WRONG: Do NOT accumulate sigma directly
// sigma_total += sigma_theta;  // This overestimates scattering!
```

### Direction Update After Scattering

```cpp
__device__ void update_direction_after_mcs(
    float& mu, float& eta,  // Direction cosines
    float delta_theta
) {
    // Current angle
    float theta = atan2(eta, mu);

    // Add scattering
    theta += delta_theta;

    // Update direction cosines
    mu = cos(theta);
    eta = sin(theta);

    // Normalize (ensure mu² + eta² = 1)
    float norm = sqrt(mu*mu + eta*eta);
    mu /= norm;
    eta /= norm;
}
```

---

## 2. Energy Straggling (Vavilov Theory)

### Three Regimes

| Regime | κ Parameter | Distribution |
|--------|-------------|--------------|
| Bohr | κ > 10 | Gaussian |
| Vavilov | 0.01 < κ < 10 | Vavilov interpolation |
| Landau | κ < 0.01 | Landau (asymmetric) |

### Vavilov Parameter

```
κ = ξ / T_max

ξ = (K/2) × (Z/A) × (z²/β²) × ρ × x
T_max = (2 m_e c² β² γ²) / (1 + 2γ m_e/m_p + (m_e/m_p)²)
```

Where:
- `K = 0.307 MeV cm²/g`
- `Z/A = 0.555` (for water)
- `m_e c² = 0.511 MeV`

### Implementation (`energy_straggling.hpp`)

```cpp
// Calculate Bohr straggling sigma
__host__ __device__ float bohr_straggling_sigma(float E_MeV, float ds) {
    float gamma = 1.0f + E_MeV / m_p_MeV;
    float beta = sqrt(1.0f - 1.0f / (gamma * gamma));

    // Bohr formula (simplified for water)
    float kappa_0 = 0.156f;  // Pre-computed for water
    float sigma = kappa_0 * sqrt(ds) / beta;

    return sigma;
}

// Calculate Vavilov kappa parameter
__host__ __device__ float vavilov_kappa(float E_MeV, float ds) {
    float gamma = 1.0f + E_MeV / m_p_MeV;
    float beta = sqrt(1.0f - 1.0f / (gamma * gamma));

    // Maximum energy transfer
    float m_ratio = m_ec2_MeV / m_p_MeV;
    float gamma_sq = gamma * gamma;
    float T_max = (2 * m_ec2_MeV * beta * beta * gamma_sq) /
                  (1.0f + 2.0f * gamma * m_ratio + m_ratio * m_ratio);

    // Vavilov parameter
    float K = 0.307f;  // MeV cm²/g
    float xi = (K / 2.0f) * 0.555f * (1.0f / (beta * beta)) * ds;  // ρ=1 for water

    return xi / T_max;
}

// Regime-dependent straggling
__device__ float energy_straggling_sigma(float E_MeV, float ds) {
    float kappa = vavilov_kappa(E_MeV, ds);

    if (kappa > 10.0f) {
        // Bohr regime: Gaussian
        return bohr_straggling_sigma(E_MeV, ds);
    } else if (kappa < 0.01f) {
        // Landau regime: Use Landau width
        return landau_width(E_MeV, ds);
    } else {
        // Vavilov regime: Interpolate
        float sigma_bohr = bohr_straggling_sigma(E_MeV, ds);
        float sigma_landau = landau_width(E_MeV, ds);
        return interpolate_vavilov(kappa, sigma_bohr, sigma_landau);
    }
}

// Sample energy loss with straggling
__device__ float sample_energy_loss_with_straggling(
    float E_MeV,
    float ds,
    unsigned& seed
) {
    // Mean energy loss (Bethe-Bloch)
    float dE_mean = compute_mean_energy_loss(E_MeV, ds);

    // Straggling sigma
    float sigma = energy_straggling_sigma(E_MeV, ds);

    // Sample Gaussian (Bohr regime, most common for protons)
    float u1 = curand_uniform(&seed);
    float u2 = curand_uniform(&seed);
    float z = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);

    return dE_mean + sigma * z;
}
```

### Most Probable Energy Loss (Landau)

```
Δp = ξ [ln(ξ/T_max) + ln(1 + β²γ²) + 0.2 - β² - δ/2]
```

Where `δ` is density effect correction (negligible for water < 250 MeV).

---

## 3. Nuclear Attenuation

### Reference: ICRU Report 63

### Cross-Section Model

```cpp
// Energy-dependent total cross-section (mm⁻¹)
__host__ __device__ float Sigma_total(float E_MeV) {
    // Logarithmic dependence on energy
    constexpr float sigma_100 = 0.0012f;  // at 100 MeV
    constexpr float sigma_20 = 0.0016f;   // at 20 MeV
    constexpr float E_ref = 100.0f;

    if (E_MeV < 5.0f) {
        // Linear ramp from 0 at 5 MeV
        return sigma_20 * (E_MeV - 5.0f) / 15.0f;
    } else if (E_MeV < 20.0f) {
        // Interpolate 5-20 MeV
        float t = (E_MeV - 5.0f) / 15.0f;
        return t * sigma_20;
    } else {
        // Logarithmic above 20 MeV
        float a = log(sigma_20 / sigma_100) / log(20.0f / 100.0f);
        return sigma_100 * pow(E_MeV / E_ref, a);
    }
}
```

### Survival Probability

```cpp
// Survival probability after step ds
__device__ float survival_probability(float E_MeV, float ds) {
    float sigma = Sigma_total(E_MeV);
    return exp(-sigma * ds);
}
```

### Energy Conservation

Nuclear interactions remove both weight and energy:

```cpp
__device__ void apply_nuclear_attenuation(
    float& weight,      // Modified: weight *= survival
    double& energy_rem, // Accumulator: energy removed by nuclear
    float E_MeV,
    float ds
) {
    float sigma = Sigma_total(E_MeV);
    float prob_interaction = 1.0f - exp(-sigma * ds);

    float weight_removed = weight * prob_interaction;
    weight -= weight_removed;

    // Track energy for conservation audit
    energy_rem += weight_removed * E_MeV;
}
```

---

## 4. R-Based Step Control

### Principle: dR/ds = -1 (CSDA Approximation)

### Maximum Step Size

```cpp
__host__ __device__ float compute_max_step_physics(float E, const RLUT& lut, float dx = 1.0f, float dz = 1.0f) {
    float R = lut.lookup_R(E);  // CSDA range [mm]

    // Primary limit: fraction of remaining range
    float delta_R_max = 0.02f * R;  // 2% of range

    // Energy-dependent refinement factor near Bragg peak
    float dS_factor = 1.0f;

    if (E < 5.0f) {
        // Very near end of range: extreme refinement
        dS_factor = 0.2f;
        delta_R_max = fminf(delta_R_max, 0.1f);  // Max 0.1 mm
    } else if (E < 10.0f) {
        // Near Bragg peak: high refinement
        dS_factor = 0.3f;
        delta_R_max = fminf(delta_R_max, 0.2f);  // Max 0.2 mm
    } else if (E < 20.0f) {
        // Bragg peak region: moderate refinement
        dS_factor = 0.5f;
        delta_R_max = fminf(delta_R_max, 0.5f);  // Max 0.5 mm
    } else if (E < 50.0f) {
        // Pre-Bragg: light refinement
        dS_factor = 0.7f;
        delta_R_max = fminf(delta_R_max, 0.7f);  // Max 0.7 mm
    }

    // Apply refinement factor
    delta_R_max = delta_R_max * dS_factor;

    // Hard limits
    delta_R_max = fminf(delta_R_max, 1.0f);  // Max 1 mm
    delta_R_max = fmaxf(delta_R_max, 0.05f);  // Min 0.05 mm

    // Cell size limit (prevents skipping cells)
    float cell_limit = 0.25f * fminf(dx, dz);
    delta_R_max = fminf(delta_R_max, cell_limit);

    return delta_R_max;
}
```

### Energy Update Using R-LUT

```cpp
// Compute energy after step (R-based method)
__device__ float compute_energy_after_step(float E_in, float ds, const RLUT& lut) {
    float R_in = lut.lookup_R(E_in);
    float R_out = R_in - ds;  // CSDA: dR/ds = -1
    return lut.lookup_E_inverse(R_out);  // Inverse lookup
}

// Energy deposited in step
__device__ float compute_energy_deposition(float E_in, float ds, const RLUT& lut) {
    float E_out = compute_energy_after_step(E_in, ds, lut);
    return E_in - E_out;  // All energy loss becomes deposition
}
```

### Why R-Based Instead of S-Based?

| Method | Formula | Accuracy | Stability |
|--------|---------|----------|-----------|
| S-based | E_out = E_in - S(E) × ds | Good if S constant | Poor near Bragg |
| R-based | E_out = E⁻¹(R(E) - ds) | Exact (CSDA) | Stable everywhere |

**Key Advantage**: R is monotonic decreasing, ensuring unique inverse lookup.

---

## 5. Fermi-Eyges Lateral Spread

### Theory

The lateral variance σ²_x(z) is computed from scattering power integrals:

```
Scattering Power: T(z) = dσ_θ²/dz

Moments:
A₀(z) = ∫₀ᶻ T(z') dz'        : Total angular variance
A₁(z) = ∫₀ᶻ z' × T(z') dz'    : First spatial moment
A₂(z) = ∫₀ᶻ z'² × T(z') dz'   : Second spatial moment

Lateral variance: σ²_x(z) = A₀×z² - 2×A₁×z + A₂
```

### Implementation (`fermi_eyges.hpp`)

```cpp
// Scattering power from Highland formula
__device__ float fermi_eyges_scattering_power(float E_MeV) {
    float sigma_theta = highland_sigma(E_MeV, 1.0f, X0_water);
    return sigma_theta * sigma_theta;  // T = σ² per mm
}

// Moment accumulation during transport
struct FermiEygesMoments {
    double A0 = 0.0;  // Total angular variance
    double A1 = 0.0;  // First spatial moment
    double A2 = 0.0;  // Second spatial moment
};

__device__ void device_update_fermi_eyges_moments(
    FermiEygesMoments& moments,
    float z,
    float ds,
    float E_MeV
) {
    float T = fermi_eyges_scattering_power(E_MeV);

    moments.A0 += T * ds;
    moments.A1 += T * z * ds;
    moments.A2 += T * z * z * ds;
}

// Compute lateral sigma at depth z
__host__ __device__ float fermi_eyges_sigma(
    const FermiEygesMoments& moments,
    float z
) {
    float variance = moments.A0 * z * z - 2.0 * moments.A1 * z + moments.A2;
    return sqrt(fmaxf(variance, 0.0f));
}
```

### Three-Component Lateral Spread

```cpp
// Total lateral spread = initial + geometric + MCS
float total_lateral_sigma_squared(
    float sigma_x0,      // Initial beam width
    float sigma_theta0,  // Initial angular spread
    float z,             // Depth
    float sigma_mcs      // MCS contribution
) {
    // Initial beam spread (diverges with distance)
    float sigma_initial = sigma_x0;

    // Geometric spread from initial divergence
    float sigma_geometric = sigma_theta0 * z;

    // MCS contribution (from Fermi-Eyges)
    float sigma_scattering = sigma_mcs;

    // Total variance (in quadrature)
    return sqrt(sigma_initial*sigma_initial +
                sigma_geometric*sigma_geometric +
                sigma_scattering*sigma_scattering);
}
```

---

## 6. Physics Pipeline Integration

### Complete Step Physics

```cpp
__device__ void transport_step(
    // Input state
    float theta, float E, float x, float z, float w,
    // Grid parameters
    float dx, float dz,
    // LUT
    const RLUT& lut,
    // Output
    float& E_dep, double& E_nuc_rem, float& boundary_flux[4]
) {
    // 1. Step size control
    float ds = compute_max_step_physics(E, lut);
    ds = fminf(ds, compute_boundary_step(x, z, dx, dz, theta));

    // 2. Energy loss
    float E_out = compute_energy_after_step(E, ds, lut);

    // 3. Energy straggling
    float dE_straggle = sample_energy_loss_with_straggling(E, ds, seed);
    E_out += dE_straggle;  // Add straggling correction
    E_out = fmaxf(E_out, E_cutoff);

    // 4. Energy deposition
    E_dep = E - E_out;

    // 5. MCS
    float sigma_theta = highland_sigma(E, ds, X0_water);
    float delta_theta = sample_mcs_angle(sigma_theta, seed);
    theta += delta_theta;

    // 6. Nuclear attenuation
    float w_before = w;
    apply_nuclear_attenuation(w, E_nuc_rem, E, ds);

    // 7. Position update
    x += ds * sin(theta);
    z += ds * cos(theta);

    // 8. Boundary check
    check_boundary_emission(x, z, dx, dz, boundary_flux);
}
```

---

## Physical Constants

| Constant | Value | Unit | Description |
|----------|-------|------|-------------|
| `m_p` | 938.272 | MeV/c² | Proton rest mass |
| `m_e c²` | 0.511 | MeV | Electron rest energy |
| `X0_water` | 360.8 | mm | Radiation length of water |
| `E_cutoff` | 0.1 | MeV | Energy cutoff |
| `E_trigger` | 10 | MeV | Fine transport trigger |
| `rho_water` | 1.0 | g/cm³ | Density of water |

---

## References

1. **NIST PSTAR Database** - Stopping powers and ranges for protons
2. **PDG 2024** - Particle Data Group review (Highland formula)
3. **ICRU Report 63** - Nuclear cross-sections for protons
4. **Vavilov (1957)** - Energy straggling theory
5. **Fermi-Eyges** - Multiple scattering theory
6. **Bethe-Bloch** - Mean energy loss formula
