# PLAN_MCS: K2 MCS Revision Implementation Plan

**Date**: 2026-02-02
**Status**: READY FOR IMPLEMENTATION
**Scope**: `K2_CoarseTransport` + Common Physics Functions + Outflow Spreading
**Version**: v1.0

---

## Executive Summary

This plan implements a **two-phase approach** to revise the Multiple Coulomb Scattering (MCS) physics in the K2 coarse transport kernel:

1. **Phase A (Option 1)**: Hotfix to achieve a "verifiable state"
   - Remove redundant 1/√2 correction from Highland formula
   - Fix sigma_x mapping with /√3 correction
   - Change spread radius to sigma-based (remove fixed 10)
   - Add weight/variance conservation measurements

2. **Phase B (Option 2)**: Full Fermi-Eyges moment-based upgrade
   - Implement A=⟨θ²⟩, B=⟨xθ⟩, C=⟨x²⟩ moment tracking
   - Proper T=θ₀²/ds scattering power calculation
   - O(n^(3/2)) scaling matching Fermi-Eyges theory

---

## Table of Contents

- [0. Common Goals and Metrics](#0-common-goals-and-metrics)
- [1. Phase A: Option 1 (Hotfix)](#1-phase-a-option-1-hotfix)
- [2. Phase B: Option 2 (Full Upgrade)](#2-phase-b-option-2-full-upgrade)
- [3. Execution Checklist](#3-execution-checklist)
- [4. Risk Mitigation](#4-risk-mitigation)

---

## 0. Common Goals and Metrics

### 0.1 Common Goals

| ID | Goal | Description |
|----|------|-------------|
| G1 | Highland θ₀ definition consistency | K2/K3/CPU/GPU use same interpretation (plane-projected RMS) |
| G2 | K2 diffusion numerical conservation | Weight conservation + variance loss measurement |
| G3 | K2 scale normalization (Option 2) | σₓ ∝ z^(3/2) (Fermi-Eyges) |

### 0.2 Common Measurement Metrics

Fixed metrics for regression/validation:

| Metric | Description | Target |
|--------|-------------|--------|
| `theta0(E, ds)` | Unit test | Verify Highland formula |
| `Σ weight` | Conservation check | Step/emission level |
| `σx(z)` | Depth profile | Minimum 3 depths |
| Bragg peak position | Range check | Minimal change |

---

## 1. Phase A: Option 1 (Hotfix)

**Goal**: Achieve a "verifiable state" by removing clearly incorrect physics and adding measurement capabilities.

### 1.1 A-1: Baseline Capture (REQUIRED)

**Purpose**: Create reference for comparison after changes.

```bash
cd /workspaces/SM_2D/build
make -j$(nproc)
./run_simulation > output_message.txt 2>&1

mkdir -p results/baseline
cp results/dose_2d.txt results/baseline/dose_before.txt
cp output_message.txt results/baseline/output_before.txt
```

**Saved Files**:
- `results/baseline/dose_before.txt`
- `results/baseline/output_before.txt`

### 1.2 A-2: Highland 1/√2 Removal

**Files to Modify**:
- `src/cuda/device/device_physics.cuh`
- `src/include/physics/highland.hpp`

**Changes**:

#### device_physics.cuh

**Before**:
```cpp
// Line ~39
constexpr float DEVICE_MCS_2D_CORRECTION = 0.70710678f;

// Line ~73-74
float sigma_3d = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
return sigma_3d * DEVICE_MCS_2D_CORRECTION;
```

**After**:
```cpp
// REMOVED: Highland theta_0 IS the projected RMS (PDG 2024)
// No 2D correction needed for x-z plane simulation
// constexpr float DEVICE_MCS_2D_CORRECTION = 0.70710678f;  // DEPRECATED

float sigma = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
return sigma;  // Highland sigma already is the projected angle RMS
```

#### highland.hpp

**Before**:
```cpp
// Line ~46
constexpr float MCS_2D_CORRECTION = 0.70710678f;

// Line ~228
return sigma_3d * MCS_2D_CORRECTION;
```

**After**:
```cpp
// REMOVED: Highland theta_0 IS the projected RMS (PDG 2024)
// constexpr float MCS_2D_CORRECTION = 0.70710678f;  // DEPRECATED

float sigma = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
return sigma;
```

**Acceptance Criteria**:
- [ ] Compiles without errors
- [ ] Unit test shows `theta0(150MeV,1mm)` increased by ×√2 (~1.4 mrad → ~2.0 mrad)

### 1.3 A-3: K2 sigma_x Transformation Correction

**Current Implementation**:
```cpp
// device_lateral_spread_sigma() in device_physics.cuh
return sinf(min(sigma_theta, 1.57f)) * step;
```

**Recommended Correction (apply /√3)**:
```cpp
__device__ inline float device_lateral_spread_sigma(float sigma_theta, float step) {
    // Apply sqrt(3) correction for continuous scattering within step
    // For small angles: sin(σ_θ) ≈ σ_θ
    // sigma_x = sigma_theta * step / sqrt(3)
    float sin_theta = sinf(fminf(sigma_theta, 1.57f));
    return sin_theta * step / 1.7320508f;  // Divide by √3
}
```

**Rationale**:
- Matches "continuous scattering distributed within step" model
- Reduces short-distance over-spreading tendency
- Prepares for transition to Option 2

**Acceptance Criteria**:
- [ ] σₓ reduced to ~0.577× of previous for same sigma_theta
- [ ] Single-step lateral spread more physically reasonable

### 1.4 A-4: Sigma-Based Spread Radius

**Current Implementation** (in `k2_coarsetransport.cu`):
```cpp
device_emit_lateral_spread(
    /* ... */,
    10  // Fixed radius: SPREAD ACROSS 10 CELLS
);
```

**Issue**: Fixed radius=10 causes:
- Tail mass truncation when σₓ is large
- Re-normalization causes variance loss
- Weight distribution artifacts

**Recommended Fix**:
```cpp
// Calculate sigma-based radius
float k_sigma = 3.0f;  // Cover ±3σ (99.7% of Gaussian)
float radius_sigma = sigma_x / dx;
int radius = static_cast<int>(ceilf(k_sigma * radius_sigma));

// Safety clamps
radius = max(radius, 1);  // Minimum radius 1
radius = min(radius, min(Nx/2, 50));  // Upper clamp

device_emit_lateral_spread(
    /* ... */,
    radius  // Sigma-based radius
);
```

**Acceptance Criteria**:
- [ ] Weight conservation improves
- [ ] No sudden profile narrowing at large σₓ
- [ ] Performance acceptable with variable radius

### 1.5 A-5: Debug Measurements

**Purpose**: Distinguish physics issues from truncation/normalization issues.

**Recommended Measurements** (debug macro on/off):

```cpp
#define DEBUG_MCS_CONSERVATION 0  // Set to 1 for debugging

#if DEBUG_MCS_CONSERVATION
// After emission
atomicAdd(&debug_weight_sum, w_out_sum);
atomicAdd(&debug_weight_in, w_in);

// Variance tracking (if feasible)
// Σ w * x and Σ w * x² for outflow bins
// Var = <x²> - <x>²
#endif
```

**Acceptance Criteria**:
- [ ] `|w_out_sum - w_in| / w_in < 1e-6` per step
- [ ] No abnormal variance reduction
- [ ] Large σₓ doesn't cause performance collapse

### 1.6 A-6: Option 1 Test Suite

#### (1) Unit: Highland θ₀

```cpp
// Test case: 150 MeV proton, 1 mm water
float sigma = device_highland_sigma(150.0f, 1.0f);
// Expected: ~2.0 mrad (×√2 from before)
assert(fabs(sigma - 0.0020) < 0.0003);
```

#### (2) Integration: Coarse-Only

```python
# Force K2 only by setting E_trigger very low
# Run and check σx(z) monotonic increase
# Verify no abnormal sudden drops
```

#### (3) Regression: Bragg Peak

```python
# Energy loss unchanged → peak position should be stable
# Peak amplitude may decrease (wider spread)
# But position ~158 mm should be maintained
```

### 1.7 A-7: Phase A Commit/PR Criteria

**Commit 1**:
```
fix(mcs): remove redundant 1/sqrt(2) correction

The Highland formula θ₀ is defined as the RMS "projected"
scattering angle for one plane (PDG 2024). No additional
2D correction is needed for x-z plane simulation.

- Remove DEVICE_MCS_2D_CORRECTION from device_physics.cuh
- Remove MCS_2D_CORRECTION from highland.hpp
- Update documentation
```

**Commit 2**:
```
fix(k2): correct sigma_x mapping and sigma-based spread radius

- Apply /√3 correction to lateral spread sigma_x
- Change spread radius from fixed 10 to sigma-based (k=3σ)
- Add debug measurements for weight/variance conservation
```

---

## 2. Phase B: Option 2 (Full Upgrade)

**Goal**: Upgrade K2 to Fermi-Eyges moment-based evolution for correct O(z^(3/2)) scaling.

### 2.1 B-1: Moment State Design

**Moments to Track**:
- **A = ⟨θ²⟩**: Angular variance
- **B = ⟨xθ⟩**: Position-angle covariance
- **C = ⟨x²⟩**: Position variance

**Existing State** (reuse):
- `x_mean`, `theta_mean`: Centroids (already exist)

**Storage Location**:

```cpp
// In coarse packet/bucket state structure
struct K2MomentState {
    float A;  // ⟨θ²⟩ angular variance
    float B;  // ⟨xθ⟩ covariance
    float C;  // ⟨x²⟩ position variance
};

// Or add to existing OutflowBucket structure
struct OutflowBucket {
    // ... existing fields ...
    float moment_A;  // ⟨θ²⟩
    float moment_B;  // ⟨xθ⟩
    float moment_C;  // ⟨x²⟩
};
```

**Memory Layout Options**:
1. **Recommended**: Add to existing struct (coherent access)
2. **Alternative**: Separate SoA buffers for better caching
3. **Fallback**: A and C only (reconstruct B ≈ 0 for small angles)

### 2.2 B-2: Scattering Power T Calculation

```cpp
// In device_physics.cuh or K2 kernel
__device__ inline float device_scattering_power_T(float E_MeV, float ds, float X0 = DEVICE_X0_water) {
    // Highland theta0 at step middle energy
    float theta0 = device_highland_sigma(E_MeV, ds, X0);

    // Scattering power T = θ₀² / ds
    // Guard against ds too small
    if (ds < 1e-6f) return 0.0f;

    float T = theta0 * theta0 / ds;
    return T;
}

// Usage in K2 kernel:
float E_mid = E - 0.5f * dE;  // Mid-step energy
float T = device_scattering_power_T(E_mid, coarse_range_step);
```

**Key Details**:
- Use `E_mid = E - dE/2` for step
- Add `ds` stability check: `if ds < eps: T=0`
- T has units rad²/mm

### 2.3 B-3: Fermi-Eyges Moment Update

```cpp
// In K2 kernel, per coarse step
// Note: Update in order using old values

void fermi_eyges_step(
    float& A, float& B, float& C,  // Moments (in/out)
    float T, float ds               // Scattering power, step size
) {
    // Store old values
    float A_old = A;
    float B_old = B;

    // Fermi-Eyges moment evolution equations
    // d⟨θ²⟩/dz = T
    A = A_old + T * ds;

    // d⟨xθ⟩/dz = ⟨θ²⟩
    B = B_old + A_old * ds + 0.5f * T * ds * ds;

    // d⟨x²⟩/dz = 2⟨xθ⟩
    C = C + 2.0f * B_old * ds + A_old * ds * ds + (1.0f / 3.0f) * T * ds * ds * ds;
}
```

**Usage in K2**:
```cpp
// After energy loss calculation
float E_mid = E - 0.5f * dE;
float T = device_scattering_power_T(E_mid, coarse_range_step);

// Update moments
fermi_eyges_step(moment_A, moment_B, moment_C, T, coarse_range_step);

// Now sigma_x = sqrt(C) for lateral spreading
float sigma_x = sqrtf(fmaxf(moment_C, 0.0f));
```

### 2.4 B-4: Outflow Spreading with Moment-Based sigma_x

**Current K2**: `sigma_x = sin(sigma_theta) * step` (recreated each step)

**New K2 (Option 2)**: `sigma_x = sqrt(C)` (accumulated over path)

```cpp
// In K2 kernel, at cell exit
float sigma_theta_accumulated = sqrtf(fmaxf(moment_A, 0.0f));
float sigma_x_accumulated = sqrtf(fmaxf(moment_C, 0.0f));

// For lateral spread emission
device_emit_lateral_spread(
    /* ... */,
    sigma_x_accumulated,  // Use accumulated C moment
    /* ... */
);
```

**Optional: Angular Spreading**
```cpp
// If theta-bin spreading is desired in K2
float sigma_theta_for_bin = sqrtf(fmaxf(moment_A, 0.0f));
// Use for theta-bin weight distribution
```

### 2.5 B-5: K2→K3 Transition Criteria (Moment-Based)

**Current Criteria**:
```cpp
if (sigma_theta > 0.02f) {  // 20 mrad threshold
    // Trigger K3
}
```

**Recommended Moment-Based Criteria**:
```cpp
float sqrt_A = sqrtf(fmaxf(moment_A, 0.0f));
float sqrt_C_over_dx = sqrtf(fmaxf(moment_C, 0.0f)) / dx;

bool k2_valid =
    (sqrt_A < 0.02f) &&                    // θ_RMS < 20 mrad
    (sqrt_C_over_dx < 3.0f) &&             // σₓ < 3 bins
    (sqrt_A * coarse_range_step < 0.1f);   // Small-angle approximation valid

if (!k2_valid) {
    // Transfer to K3 fine transport
}
```

### 2.6 B-6: Phase B Test Suite

#### (1) Unit: Moment Scaling Test

```cpp
// Fixed E, fixed ds, n steps
// Expected: A ~ n, C ~ n³
// Therefore: sqrt(C) ~ n^(3/2)

for (int n = 1; n <= 10; n++) {
    float A = 0, B = 0, C = 0;
    for (int i = 0; i < n; i++) {
        fermi_eyges_step(A, B, C, T, ds);
    }
    // Check scaling
    assert(fabs(A - n * T * ds) < 1e-6);
    assert(fabs(C / (n*n*n) - T * ds*ds*ds / 3.0) < 1e-4);
}
```

#### (2) Integration: σₓ(z) Profile

```python
# Measure σₓ at 3-5 depths
# Check log-log slope ~1.5 (z^(3/2) scaling)
depths = [40, 80, 120, 160]
sigmas = [measure_sigma_x(d) for d in depths]
# Fit log(σ) vs log(z): slope should be ~1.5
```

#### (3) Regression: Energy/Bragg/Weight

```python
# Energy loss: unchanged → Bragg peak position stable
# Spreading changed → peak amplitude may decrease
# Total weight: conserved
```

### 2.7 B-7: Phase B Commit/PR Criteria

**Commit**:
```
feat(k2): implement Fermi-Eyges moment evolution for MCS

- Add A=⟨θ²⟩, B=⟨xθ⟩, C=⟨x²⟩ moments to K2 state
- Implement T=θ₀²/ds scattering power calculation
- Add Fermi-Eyges moment update equations
- Use sigma_x = sqrt(C) for lateral spreading
- Update K2→K3 transition criteria to moment-based

Expected: σₓ ∝ z^(3/2) scaling matching Fermi-Eyges theory
```

---

## 3. Execution Checklist

### Phase A (Option 1)

- [ ] **A-1**: Save baseline results
- [ ] **A-2**: Remove 1/√2 correction (GPU/CPU)
- [ ] **A-3**: Apply /√3 to sigma_x
- [ ] **A-4**: Change spread radius to sigma-based
- [ ] **A-5**: Add weight/variance measurements
- [ ] **A-6**: Run unit/integration/regression tests
- [ ] **A-7**: Create PR for Phase A

### Phase B (Option 2)

- [ ] **B-1**: Add A,B,C to packet structure
- [ ] **B-2**: Implement T=θ₀²/ds with E_mid
- [ ] **B-3**: Insert moment update equations
- [ ] **B-4**: Change sigma_x to sqrt(C)
- [ ] **B-5**: Update K2→K3 criteria
- [ ] **B-6**: Run scaling test (n^(3/2)) + integration + regression
- [ ] **B-7**: Create PR for Phase B

---

## 4. Risk Mitigation

### 4.1 Memory/Performance (Option 2)

**Risk**: Adding 3 floats per packet may be expensive.

**Mitigation**:
1. **Full mode**: Store A, B, C (recommended)
2. **Lite mode**: Store A, C only; approximate B ≈ 0
3. **On-the-fly**: Reconstruct from accumulated path (less accurate)

### 4.2 Large Result Changes

**Risk**: Option 2 will significantly change lateral spread.

**Mitigation**:
- Compare to **Fermi-Eyges theory** not old code
- Use Geant4 or experimental data as ground truth
- Expected change is in "correct direction" (more spread)

### 4.3 Numerical Stability

**Risk**: Moment updates may accumulate errors.

**Mitigation**:
- Add `fmaxf(value, 0.0f)` before sqrt
- Check for NaN/inf in debug mode
- Consider renormalization for very long paths

---

## 5. File Reference

### Files Modified

| Phase | File | Changes |
|-------|------|---------|
| A | `src/cuda/device/device_physics.cuh` | Remove 1/√2, add /√3, T function |
| A | `src/include/physics/highland.hpp` | Remove 1/√2 correction |
| A | `src/cuda/kernels/k2_coarsetransport.cu` | Sigma-based radius, debug measurements |
| B | K2 state structures | Add A, B, C moments |
| B | `src/cuda/kernels/k2_coarsetransport.cu` | Moment updates, sigma_x = sqrt(C) |

### Key Function Locations

| Function | File | Lines |
|----------|------|-------|
| `device_highland_sigma` | device_physics.cuh | ~52-75 |
| `device_lateral_spread_sigma` | device_physics.cuh | ~320-325 |
| `highland_sigma` | highland.hpp | ~60-81 |
| K2 MCS logic | k2_coarsetransport.cu | ~180-240 |
| `device_emit_lateral_spread` | k2_coarsetransport.cu | ~call at 228-240 |

---

## 6. Expected Outcomes

### Phase A (Option 1)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Highland θ₀ (150 MeV, 1mm) | ~1.4 mrad | ~2.0 mrad | ×√2 |
| sigma_x (single step) | current | ×0.577 | /√3 |
| Spread radius | fixed 10 | σ-based | variable |
| Lateral scaling | O(√n) | O(√n) | unchanged |

### Phase B (Option 2)

| Metric | Before (K2) | After (Option 2) | Theory |
|--------|-------------|------------------|---------|
| ⟨θ²⟩ scaling | - | O(n) | O(n) |
| ⟨x²⟩ scaling | O(n) | O(n³) | O(n³) |
| σₓ scaling | O(√n) | O(n^(3/2)) | O(z^(3/2)) |
| Match to Fermi-Eyges | Partial | Full | ✓ |

---

**END OF PLAN_MCS**
