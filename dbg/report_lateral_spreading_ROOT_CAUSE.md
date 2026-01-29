# Debug Report: Lateral Spreading Issue - ROOT CAUSE FOUND

**Date**: 2026-01-29
**Status**: Root Cause Identified - GPU Ignores Gaussian Beam Profile
**Investigation Method**: MPDBGER Multi-Path Debugging + Runtime Data Analysis

---

## ROOT CAUSE

**Location**: `src/cuda/gpu_transport_wrapper.cu:168-177`

**Issue**: GPU transport injects a **single pencil beam particle** at `(x0=0, theta0=0)` with full weight, completely **ignoring** the Gaussian beam parameters `sigma_x` and `sigma_theta` specified in the configuration.

---

## Evidence

### Configuration (sim.ini)
```ini
[beam]
profile = gaussian  # ← Specifies Gaussian beam

[spatial]
sigma_x_mm = 3.0    # ← 3mm initial lateral spread

[angular]
sigma_theta_rad = 0.001  # ← Initial angular divergence

[sampling]
n_samples = 1000     # ← Should sample 1000 particles
```

### GPU Transport (ACTUAL BEHAVIOR)
```cpp
// src/cuda/gpu_transport_wrapper.cu:168-177
inject_source_kernel<<<1, 1>>>(
    psi_in,
    source_cell,
    theta0, E0, W_total,  // ← Single particle, NOT sampled!
    x_in_cell, z_in_cell,
    ...
);
```

### CPU GaussianSource (EXPECTED BEHAVIOR)
```cpp
// src/source/gaussian_source.cpp:22-25
for (int i = 0; i < src.n_samples; ++i) {
    float x = x_dist(rng);         // ← Sampled from N(x0, sigma_x)
    float theta = theta_dist(rng); // ← Sampled from N(theta0, sigma_theta)
    float E = E_dist(rng);
    float w_per_sample = src.W_total / src.n_samples;
    // Inject particle...
}
```

---

## Impact

| Metric | Expected (Gaussian) | Actual (Pencil) | Error |
|--------|-------------------|----------------|-------|
| Initial x spread | ~3 mm (sigma_x) | 0 mm | -100% |
| Initial theta spread | ~0.001 rad | 0 rad | -100% |
| Lateral FWHM at depth | Should broaden with depth | 0 mm | -100% |
| Number of source particles | 1000 | 1 | -99.9% |

**Runtime Verification**:
- Dose analysis shows **ALL dose in x=0 column only**
- Lateral FWHM = 0 mm at ALL depths
- Total dose in x=0 column: 136.85 Gy
- Dose in all other x columns: 0 Gy

---

## Why Lateral Spreading is Zero

1. **No initial lateral spread**: All particles start at x=0
2. **No initial angular divergence**: All particles have theta=0
3. **MCS too weak**: Even with scattering, particles don't move enough to cross x-cell boundaries in a single step

For 150 MeV protons:
- Scattering angle per step: ~0.001 rad
- Step size: ~0.5-5 mm
- Lateral displacement per step: ~0.0005-0.005 mm
- Cell size: 0.5 mm
- **Particles never reach neighboring x-cells**

---

## Required Fix

### Option 1: Implement GPU Gaussian Sampling (Recommended)

Modify `src/cuda/gpu_transport_wrapper.cu` to sample multiple particles like the CPU code:

```cuda
__global__ void inject_gaussian_source_kernel(
    DevicePsiC psi,
    int source_cell_base,
    float x0, float z0, float theta0, float E0, float W_total,
    float sigma_x, float sigma_theta, float sigma_E,
    int n_samples,
    unsigned int seed,
    ...
) {
    // Each thread injects one sample
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;

    // Sample from Gaussian distributions
    curandState state;
    curand_init(seed, i, 0, &state);

    float x = x0 + sigma_x * curand_normal(&state);
    float theta = theta0 + sigma_theta * curand_normal(&state);
    float E = fmaxf(0.1f, E0 + sigma_E * curand_normal(&state));
    float w = W_total / n_samples;

    // Calculate cell for this sample
    int ix = static_cast<int>((x - x_min) / dx);
    int iz = 0;  // All at z=0
    int cell = iz * Nx + ix;

    // Inject...
}
```

### Option 2: Use CPU Preamble

For smaller simulations, run the Gaussian source injection on CPU then copy to GPU.

---

## Relationship to Previous Findings

The "energy-dependent scattering reduction factors" (0.3x at 150 MeV) found in the initial investigation are **NOT the root cause**. They would reduce lateral spread by 70%, but we're seeing 100% reduction (zero spread).

The real issue is that:
1. The Gaussian beam parameters are being ignored
2. The simulation is using a pencil beam instead
3. With zero initial spread, even reduced scattering can't create visible lateral spread

---

## Verification

To confirm this fix:
1. Implement GPU Gaussian sampling
2. Run simulation with sigma_x=3.0mm, sigma_theta=0.001rad
3. Verify dose is distributed across multiple x columns
4. Verify lateral FWHM increases with depth as expected

---

## References

- **GPU bug location**: `src/cuda/gpu_transport_wrapper.cu:168-177`
- **CPU reference**: `src/source/gaussian_source.cpp`
- **Configuration**: `sim.ini` (beam profile = gaussian)
- **Dose data**: `results/dose_2d.txt` (all dose in x=0 column)

---

## Agent Links

- Investigation 1 (Static): agentId: a5192a2
- Investigation 2 (Physics): agentId: a6659a2
- Investigation 3 (Scaffold): agentId: a7909cb
- Investigation 4 (Runtime): agentId: a6bde10
- **ROOT CAUSE**: Runtime dose analysis + code comparison
