# PLAN_MCS_K2: MCS Physics Revision for K2 Coarse Transport Kernel

**Date**: 2026-02-02
**Status**: DRAFT
**Kernel**: K2_CoarseTransport
**Focus**: 2D Multiple Coulomb Scattering (MCS) Implementation

---

## 1. Overview

This plan addresses the physics implementation of Multiple Coulomb Scattering (MCS) in the K2 coarse transport kernel, based on PDG Highland formula and proper 2D scattering theory for proton transport in water.

**Scope**: K2 kernel (`src/cuda/kernels/k2_coarsetransport.cu`)

---

## 2. Current K2 MCS Implementation

### 2.1 Code Location

**File**: `src/cuda/kernels/k2_coarsetransport.cu`
**Lines**: 180-240

### 2.2 Current Implementation

```cpp
// PLAN_fix_scattering: Stochastic Interpolation for Lateral Spreading
// ========================================================================
// Instead of accumulating theta (which causes beam shifting),
// we use Gaussian weight distribution across multiple cells for lateral spreading.
//
// Key changes from previous implementation:
// 1. Theta is NOT accumulated: theta_new = theta (not theta + theta_scatter)
// 2. Lateral spread is calculated: sigma_x = sin(sigma_theta) * step
// 3. Weight is distributed across multiple cells using Gaussian CDF
// ========================================================================

// Calculate MCS scattering angle (RMS)
float sigma_theta = device_highland_sigma(E, coarse_range_step);

// PLAN_fix_scattering: DO NOT accumulate theta
// Theta remains unchanged; lateral spreading is handled by weight distribution
float theta_new = theta;  // No theta accumulation!

float mu_new = cosf(theta_new);
float eta_new = sinf(theta_new);

// Calculate lateral spread sigma_x from scattering angle
// sigma_x represents the standard deviation of lateral displacement
float sigma_x = device_lateral_spread_sigma(sigma_theta, coarse_step_limited);

// Position update (use limited step to avoid boundary crossing)
float x_new = x_cell + eta_new * coarse_step_limited;
float z_new = z_cell + mu_new * coarse_step_limited;
```

### 2.3 Lateral Spreading Function

**File**: `src/cuda/device/device_physics.cuh:320-325`

```cpp
// Calculate lateral spread sigma_x from scattering angle
// sigma_x = sin(sigma_theta) * step
// For small angles: sin(sigma_theta) ≈ sigma_theta
__device__ inline float device_lateral_spread_sigma(float sigma_theta, float step) {
    return sinf(fminf(sigma_theta, 1.57f)) * step;  // sin(theta), theta < pi/2
}
```

---

## 3. Physics Verification

### 3.1 Highland Formula (Current)

**File**: `src/cuda/device/device_physics.cuh:52-75`

```cpp
__device__ inline float device_highland_sigma(float E_MeV, float ds, float X0 = DEVICE_X0_water) {
    constexpr float z = 1.0f;  // Proton charge

    // Relativistic kinematics
    float gamma = (E_MeV + DEVICE_m_p_MeV) / DEVICE_m_p_MeV;
    float beta2 = 1.0f - 1.0f / (gamma * gamma);
    float beta = sqrtf(fmaxf(beta2, 0.0f));
    float p_MeV = sqrtf(fmaxf((E_MeV + DEVICE_m_p_MeV) * (E_MeV + DEVICE_m_p_MeV) -
                             DEVICE_m_p_MeV * DEVICE_m_p_MeV, 0.0f));

    float t = ds / X0;
    if (t < 1e-6f) return 0.0f;

    // Highland correction factor (PDG 2024: bracket >= 0.25)
    float ln_t = logf(t);
    float bracket = 1.0f + 0.038f * ln_t;
    bracket = fmaxf(bracket, 0.25f);

    // P2 FIX: Apply 2D projection correction
    float sigma_3d = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
    return sigma_3d * DEVICE_MCS_2D_CORRECTION;  // <-- 1/√2 APPLIED
}
```

**Where**:
- `DEVICE_X0_water = 360.8f` (mm) - Correct for water
- `DEVICE_MCS_2D_CORRECTION = 0.70710678f` (1/√2) - **May be incorrect**

### 3.2 PDG Highland Formula Reference

```
θ₀ = (13.6 MeV / βcp) × z × √(x/X₀) × [1 + 0.038 × ln(x/X₀)]
```

**Definition**: θ₀ is the RMS **"projected" scattering angle** for one plane.

**Key Question**: Does the 2D simulation's θ variable need the 1/√2 correction?

| Scenario | Correction Needed | Reason |
|----------|-------------------|--------|
| θ is 3D total scattering angle | Yes, divide by √2 | Convert 3D → 2D projection |
| θ is 1D projected angle | No | θ₀ already is projected |
| θ is x-z plane angle (SM_2D case) | **No** | θ₀ already is the projected RMS |

### 3.3 K2 Coordinate System

```cpp
// Direction
float mu = cosf(theta);  // Direction cosine in z (beam direction)
float eta = sinf(theta);  // Direction cosine in x (transverse)
```

**Geometry**: 2D x-z plane with single polar angle θ
- θ = 0 → beam along +z axis
- θ ≠ 0 → beam deflected in x-z plane

**Conclusion**: θ IS the 1D projected angle for the x-z plane. The Highland formula θ₀ gives exactly this quantity.

---

## 4. Issues Identified

### 4.1 Issue 1: Potentially Incorrect 1/√2 Correction

**Severity**: HIGH

**Problem**: Code applies `DEVICE_MCS_2D_CORRECTION = 1/√2` to Highland formula.

**Analysis**:
- Highland θ₀ is explicitly defined as the **projected** scattering angle (PDG)
- For 2D x-z plane simulation, θ represents this projected angle
- No additional correction should be needed

**Impact**:
- Current sigma is ~30% smaller than it should be
- Lateral spreading is underestimated
- Beam may be too narrow

**Recommendation**:
- Remove `DEVICE_MCS_2D_CORRECTION` from `device_highland_sigma()`
- Update documentation to clarify θ₀ is already the projected angle

### 4.2 Issue 2: Lateral Spread Calculation

**Severity**: MEDIUM

**Current Code**:
```cpp
float sigma_x = device_lateral_spread_sigma(sigma_theta, coarse_step_limited);
// = sin(sigma_theta) * step
```

**Analysis**:
- For small angles: sin(σ_θ) ≈ σ_θ (correct)
- For larger angles: sin(σ_θ) < σ_θ (conservative)
- Maximum clamping at π/2 is appropriate

**Status**: Acceptable for small-angle approximation

### 4.3 Issue 3: Step Size Units

**Severity**: LOW

**Current Code**:
```cpp
float coarse_range_step = coarse_step_limited / mu_abs;  // Path length
float sigma_theta = device_highland_sigma(E, coarse_range_step);  // Uses path length
float sigma_x = device_lateral_spread_sigma(sigma_theta, coarse_step_limited);  // Uses geometric
```

**Analysis**:
- Highland formula uses path length (ds = geometric / |cos θ|)
- Lateral spread uses geometric distance (correct)
- Units are consistent

**Status**: Correct

### 4.4 Issue 4: No Variance Accumulation (K2 Specific)

**Severity**: LOW for K2 (HIGH for K3)

**K2 Behavior**:
- Theta is NOT accumulated: `theta_new = theta`
- Lateral spreading is via weight distribution only
- This is CORRECT for coarse transport (high energy, minimal scattering)

**K3 Behavior** (for comparison):
- Theta IS accumulated with random sampling each step
- This causes excessive lateral spreading in fine transport
- Needs variance accumulation fix (separate plan)

---

## 5. Proposed Changes for K2

### 5.1 Change 1: Remove 1/√2 Correction (HIGH Priority)

**File**: `src/cuda/device/device_physics.cuh`

**Before**:
```cpp
// Line 39
constexpr float DEVICE_MCS_2D_CORRECTION = 0.70710678f;

// Line 73-74
float sigma_3d = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
return sigma_3d * DEVICE_MCS_2D_CORRECTION;
```

**After**:
```cpp
// Line 39
// REMOVED: Highland theta_0 IS the projected scattering angle (PDG 2024)
// No 2D correction needed for x-z plane simulation
// constexpr float DEVICE_MCS_2D_CORRECTION = 0.70710678f;  // DEPRECATED

// Line 73-74
float sigma = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
return sigma;  // Highland sigma already is the projected angle RMS
```

**Also Update**: `src/include/physics/highland.hpp` (CPU version)

### 5.2 Change 2: Update Documentation (MEDIUM Priority)

**File**: `src/cuda/device/device_physics.cuh` (comments)

**Before**:
```cpp
// ============================================================================
// MCS 2D Projection Correction (Physical Analysis Applied)
// ============================================================================
// This simulation uses a 2D geometry (x-z plane).
// The Highland formula gives the 3D scattering angle sigma_3D.
//
// PHYSICS CORRECTION (2026-01):
// For 3D isotropic scattering projected onto a 2D plane:
//   - The azimuthal angle φ is uniformly distributed in [0, 2π]
//   - Projected variance: σ_2D² = σ_3D² / 2 (variance splits equally)
//   - Therefore: σ_2D = σ_3D / √2 ≈ 0.707 × σ_3D
```

**After**:
```cpp
// ============================================================================
// Highland Formula for MCS (PDG 2024)
// ============================================================================
// This simulation uses a 2D geometry (x-z plane) with a single polar angle θ.
//
// The Highland formula gives θ₀, the RMS "projected" scattering angle for
// one plane (e.g., the x-z plane). Since SM_2D directly simulates the x-z
// plane with θ as the deflection angle, no additional 2D correction is needed.
//
// Formula: θ₀ = (13.6 MeV / βcp) × z × √(x/X₀) × [1 + 0.038 × ln(x/X₀)]
//
// where:
//   θ₀ = RMS projected scattering angle (for x-z plane)
//   x = step length (path length along trajectory)
//   X₀ = radiation length (360.8 mm for water)
```

### 5.3 Change 3: Verify Lateral Spread Emission (LOW Priority)

**File**: `src/cuda/kernels/k2_coarsetransport.cu:228-240`

**Current Code**:
```cpp
if (exit_face == FACE_Z_PLUS && sigma_x > dx * 0.01f) {
    // Multi-cell emission with lateral spreading
    device_emit_lateral_spread(
        OutflowBuckets, cell, iz_target,
        theta_new, E_new, w_new,
        x_new, sigma_x, dx, Nx, Nz,
        theta_edges, E_edges,
        N_theta, N_E, N_theta_local, N_E_local,
        10  // Spread across 10 cells
    );
}
```

**Status**: This approach is correct - uses Gaussian weight distribution for lateral spreading without accumulating theta.

**No changes needed**, but verify `device_emit_lateral_spread` implementation is correct.

---

## 6. Verification Plan

### 6.1 Unit Tests

1. **Highland sigma calculation**:
   ```cpp
   // Test case: 150 MeV proton, 1 mm water
   // Expected: θ₀ ≈ 1.5 mrad (from PDG table)
   float sigma = device_highland_sigma(150.0f, 1.0f);
   assert(fabs(sigma - 0.0015) < 0.0002);
   ```

2. **Energy dependence**:
   | Energy (MeV) | Expected θ₀ (mrad) |
   |--------------|-------------------|
   | 70 | ~4.1 |
   | 100 | ~2.9 |
   | 150 | ~1.5 |
   | 200 | ~1.5 |

3. **Step dependence**:
   - Verify √(x/X₀) dependence
   - Test with x = 0.5, 1.0, 2.0 mm

### 6.2 Integration Tests

1. **K2 coarse transport only**:
   - Set E_trigger = 0.05 MeV (force coarse-only)
   - Run 150 MeV pencil beam
   - Verify forward penetration and lateral spread

2. **Compare lateral spread**:
   - Measure σₓ at various depths
   - Compare with Fermi-Eyges theory

### 6.3 Regression Tests

| Metric | Before Fix | After Fix | Expected |
|--------|------------|-----------|----------|
| Sigma at 150 MeV, 1mm | ~1.0 mrad | ~1.5 mrad | ~1.5 mrad |
| Lateral σ at 80mm | Current | ~1.4× larger | Fermi-Eyges |
| Forward range | Current | Similar | ~158 mm |

---

## 7. Implementation Steps

### Step 1: Prepare Test Baseline
```bash
# Run current implementation
cd build
make run_simulation
./run_simulation
cp results/dose_2d.txt results/before_fix_dose.txt
```

### Step 2: Implement Changes
```bash
# Edit files
# src/cuda/device/device_physics.cuh
# src/include/physics/highland.hpp
```

### Step 3: Rebuild and Test
```bash
cd build
make -j$(nproc)
./run_simulation
cp results/dose_2d.txt results/after_fix_dose.txt
```

### Step 4: Compare Results
```bash
python3 << 'EOF'
import numpy as np

before = np.loadtxt('results/before_fix_dose.txt')
after = np.loadtxt('results/after_fix_dose.txt')

print(f"Before: Max dose = {before[:,3].max():.2f} Gy at z = {before[np.argmax(before[:,3]), 2]:.1f} mm")
print(f"After:  Max dose = {after[:,3].max():.2f} Gy at z = {after[np.argmax(after[:,3]), 2]:.1f} mm")

# Compare lateral profiles at mid-range
mid_depth = 80  # mm
before_lat = before[abs(before[:,2] - mid_depth) < 1]
after_lat = after[abs(after[:,2] - mid_depth) < 1]

if len(before_lat) > 0 and len(after_lat) > 0:
    sigma_before = np.sqrt(np.sum(before_lat[:,1]**2 * before_lat[:,3]) / np.sum(before_lat[:,3]))
    sigma_after = np.sqrt(np.sum(after_lat[:,1]**2 * after_lat[:,3]) / np.sum(after_lat[:,3]))
    print(f"Lateral σ at {mid_depth}mm: Before = {sigma_before:.2f} mm, After = {sigma_after:.2f} mm")
EOF
```

### Step 5: Validation
- Verify increased lateral spread (~1.4× larger)
- Verify Bragg peak position unchanged
- Verify total energy deposition conserved

---

## 8. Rollback Plan

If results are worse after the fix:

```bash
git checkout HEAD -- src/cuda/device/device_physics.cuh
git checkout HEAD -- src/include/physics/highland.hpp
cd build && make -j$(nproc)
```

Document findings and re-evaluate the physics interpretation.

---

## 9. Related Documents

- `docs/MCS_revision_plan_2D.md` - Overall MCS revision plan
- `dbg/H6_MCS_investigation_report.md` - Previous MCS investigation
- `docs/detailed/physics/01_physics_models.md` - Physics documentation

---

## 10. Decision Checklist

Before committing changes:

- [ ] Unit tests pass (Highland formula validation)
- [ ] Integration tests pass (coarse transport)
- [ ] Lateral spread increased by expected amount (~√2)
- [ ] Forward penetration maintained
- [ ] Energy conserved
- [ ] Documentation updated
- [ ] Code comments clarified

---

**END OF PLAN_MCS_K2**
