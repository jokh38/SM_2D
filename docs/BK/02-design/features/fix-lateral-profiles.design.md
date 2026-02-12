# Fix Lateral Profiles - Design Document

> **Summary**: Design for smooth lateral profile implementation in deterministic proton transport
>
> **Project**: SM_2D Proton Therapy Simulation
> **Version**: 1.0
> **Author**: Claude (Sisyphus Mode)
> **Date**: 2026-02-04
> **Status**: Draft
> **Planning Doc**: [fix-lateral-profiles.plan.md](../01-plan/features/fix-lateral-profiles.plan.md)

---

## 1. Overview

### 1.1 Design Goals

1. **Smoothness**: Lateral profiles should exhibit smooth Gaussian-like distributions at all depths
2. **Physical Accuracy**: Lateral spread should follow Fermi-Eyges theory: σ_x²(z) ∝ z^(3/2)
3. **Performance**: Runtime increase limited to < 20%
4. **Backward Compatibility**: Depth-dose curves should remain unchanged

### 1.2 Design Principles

- **Minimal changes**: Modify only necessary code paths
- **Incremental**: Each fix can be tested independently
- **SPEC compliance**: All changes must align with SPEC.md requirements

---

## 2. Root Cause Analysis Summary

### 2.1 Identified Issues

| Issue | Location | Impact | Fix |
|-------|----------|--------|-----|
| Coarse x_sub discretization | `local_bins.hpp:21` | Only 4 positions/cell | Increase to 8-16 |
| Per-step sigma_x | `k3_finetransport.cu:254-255` | No accumulated spread | Track cumulative σ_x |
| Moment reset | `k2_coarsetransport.cu:208-225` | Moments not persisted | Carry forward |
| Hard clamping | `k3:428`, `k2:423` | Boundary artifacts | Soft boundaries |

### 2.2 Expected Behavior (Fermi-Eyges Theory)

For a proton beam with initial RMS width σ_0:
```
σ_x²(z) = σ_0² + ⟨θ²⟩ · z² / 3 + ... (higher order terms)
```

At depth z=150mm with σ_0=6mm and σ_θ≈2mrad:
```
σ_x(150mm) ≈ √(36 + 0.01) ≈ 6.0008 mm  (should be noticeably wider!)
```

---

## 3. Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    K1-K6 Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │    K1   │───▶│    K2   │───▶│    K3   │───▶│    K4   │  │
│  │ Active  │    │ Coarse  │    │  Fine   │    │Transfer │  │
│  │  Mask   │    │Transport│    │Transport│    │         │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                       │              │                     │
│                       │    ┌─────────┴─────────┐           │
│                       │    │  Lateral Spreading │          │
│                       │    │  (CHANGES NEEDED)  │          │
│                       │    └────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow (Lateral Spreading)

```
Current (BROKEN):
  sigma_x ← f(step)  [per-step, tiny value]
  Spread to 4 x_sub bins with integer binning

Fixed:
  sigma_x_cumulative ← sigma_x_cumulative + f(step)  [accumulated]
  Spread to 8-16 x_sub bins with proper interpolation
```

---

## 4. Data Model Changes

### 4.1 Local Bins Structure

**Current** (`local_bins.hpp`):
```cpp
constexpr int N_x_sub = 4;        // 4 lateral positions per cell
constexpr int N_z_sub = 4;        // 4 depth positions per cell
constexpr int LOCAL_BINS = 128;   // 4 × 2 × 4 × 4 = 128
```

**Proposed**:
```cpp
constexpr int N_x_sub = 8;        // 8 lateral positions per cell
constexpr int N_z_sub = 4;        // 4 depth positions per cell (unchanged)
constexpr int LOCAL_BINS = 256;   // 4 × 2 × 8 × 4 = 256
```

**Memory Impact**:
- Per-cell storage: 128 → 256 bins (2x increase)
- Total for 128,000 cells: ~500 MB → ~1 GB (acceptable for RTX 2080)

### 4.2 Cumulative Sigma Tracking

**New state variable** (add to K3 kernel):
```cpp
// Track cumulative lateral spread from depth
__device__ float sigma_x_cumulative = sigma_x_initial;  // ~6mm at surface
// During transport:
sigma_x_cumulative += device_lateral_spread_sigma(sigma_theta, step);
```

---

## 5. Implementation Specification

### 5.1 Modification 1: Increase x_sub Resolution

**File**: `src/include/core/local_bins.hpp`

```cpp
// CHANGE:
constexpr int N_x_sub = 4;
// TO:
constexpr int N_x_sub = 8;  // Or 16 for production

// UPDATE comment:
// Sub-bin centers: -0.4375*dx, -0.3125*dx, -0.1875*dx, -0.0625*dx,
//                   +0.0625*dx, +0.1875*dx, +0.3125*dx, +0.4375*dx
```

**Files requiring update due to LOCAL_BINS change**:
- `src/cuda/kernels/k1_activemask.cu` (if hardcodes 128)
- `src/cuda/kernels/k3_finetransport.cu` (N_x_spread)
- `src/cuda/kernels/k2_coarsetransport.cu` (N_x_spread)
- Memory allocation in `gpu_transport_runner.cpp`

### 5.2 Modification 2: Cumulative Sigma Tracking

**File**: `src/cuda/kernels/k3_finetransport.cu`

**Current** (lines 253-256):
```cpp
float sigma_theta = device_highland_sigma(E, actual_range_step);
float sigma_x = device_lateral_spread_sigma(sigma_theta, actual_range_step);
sigma_x = fmaxf(sigma_x, LATERAL_SPREAD_MIN_SIGMA);
```

**Proposed**:
```cpp
// Calculate per-step contribution
float sigma_theta = device_highland_sigma(E, actual_range_step);
float sigma_x_step = device_lateral_spread_sigma(sigma_theta, actual_range_step);

// Accumulate from previous iterations (need to store per-particle)
// For now: use depth-based formula as approximation
float depth_from_surface = cell_z * dz + z_sub * dz / N_z_sub;
float sigma_x = sqrtf(sigma_x_initial * sigma_x_initial +
                      sigma_theta_step * sigma_theta_step * depth_from_surface * depth_from_surface / 3.0f);
```

### 5.3 Modification 3: Fix K2 Fermi-Eyges Moment Persistence

**File**: `src/cuda/kernels/k2_coarsetransport.cu`

**Current** (lines 208-225) - resets to zero:
```cpp
float moment_A = 0.0f;  // ⟨θ²⟩
float moment_B = 0.0f;  // ⟨xθ⟩
float moment_C = 0.0f;  // ⟨x²⟩
// ... calculate for this step only
```

**Proposed** - need to track per-depth:
```cpp
// Note: This requires architectural change - moments must be stored per depth slice
// For interim fix: use analytical Fermi-Eyges solution
float z_cm = (cell_z * dz) / 10.0f;  // Convert mm to cm
float theta_ms = get_highland_theta_ms_R80(R_R80);  // From spec
float sigma_x_FE = sqrtf(2.0f) * theta_ms * z_cm * powf(z_cm, 0.5f) / 3.0f;
```

### 5.4 Modification 4: Remove Hard Clamping

**File**: `src/cuda/kernels/k3_finetransport.cu`

**Current** (line 428):
```cpp
x_sub_spread = fmaxf(0, fminf(x_sub_spread, 3));  // Hard clamp to [0,3]
```

**Proposed**:
```cpp
// Use soft boundary with reflection or wrap-around
if (x_sub_spread < 0) {
    x_sub_spread = -x_sub_spread - 1;  // Reflect
    if (x_sub_spread < 0) x_sub_spread = 0;  // Final fallback
}
if (x_sub_spread >= N_x_sub) {
    x_sub_spread = 2 * N_x_sub - x_sub_spread - 1;  // Reflect
    if (x_sub_spread >= N_x_sub) x_sub_spread = N_x_sub - 1;  // Final fallback
}
```

---

## 6. Test Plan

### 6.1 Verification Method

| Test | Method | Success Criteria |
|------|--------|------------------|
| Visual inspection | Run simulation + visualize.py | Smooth profiles, no spikes |
| Lateral variance | Extract σ_x at depths | Matches ~ σ² ∝ z^(3/2) |
| Depth-dose comparison | Compare PDD before/after | < 1% shift, < 5% amplitude change |
| Memory usage | nvidia-smi during run | < 2GB GPU memory |
| Runtime | time ./run_simulation | < 20% increase |

### 6.2 Test Cases

1. **Baseline**: Run current version, save results
2. **Fix 1 (x_sub=8)**: Apply only Modification 1
3. **Fix 1+2**: Add cumulative sigma
4. **Fix 1+2+3**: Add K2 moment fix
5. **Fix 1+2+3+4**: Add soft boundaries
6. **Production (x_sub=16)**: Test with higher resolution

---

## 7. Implementation Order

1. **Phase 1**: Increase N_x_sub to 8 (local_bins.hpp)
2. **Phase 2**: Add cumulative sigma_x approximation (k3_finetransport.cu)
3. **Phase 3**: Fix K2 moment handling (k2_coarsetransport.cu)
4. **Phase 4**: Remove hard clamping (k3_finetransport.cu)
5. **Phase 5**: Test with N_x_sub=16 for production
6. **Phase 6**: Final verification and benchmarking

---

## 8. Rollback Plan

If any fix breaks the simulation:
1. Git revert the specific commit
2. Document the failure in dbg/debug_history.md
3. Proceed with next fix independently

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-04 | Initial draft | Claude (Sisyphus) |
