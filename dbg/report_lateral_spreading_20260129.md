# Debug Report: Lateral Spreading Issue in Proton Transport

**Date**: 2026-01-29
**Status**: Root Cause Identified
**Investigation Method**: MPDBGER Multi-Path Debugging + Ultrawork Parallel Analysis

---

## Executive Summary

**Finding**: Energy-dependent scattering reduction factors artificially suppress Multiple Coulomb Scattering (MCS) at high energies, causing lateral spreading to be only 30% of physical expectation for 150 MeV protons.

**Impact**: HIGH - Lateral dose distribution is too narrow, compromising clinical treatment planning accuracy

**Root Cause**: Arbitrary scattering reduction factors (0.3x, 0.5x, 0.7x) applied based on energy thresholds - NOT based on real physics

---

## Issue Summary

### Symptom
Lateral spreading in proton therapy simulation does not correctly reflect real physics. For 150 MeV protons:
- **Expected lateral σₓ at Bragg peak**: ~5.6 mm (Highland formula + Fermi-Eyges theory)
- **Actual**: Likely < 2 mm (estimated from 30% scattering reduction)
- **Error**: >60% under-prediction

### Expected vs Actual

| Metric | Expected (Theory) | Actual (Estimated) | Error |
|--------|-------------------|-------------------|-------|
| Lateral σₓ at 160 mm | ~5.6 mm | ~1.7 mm | -70% |
| Scattering at 150 MeV | 100% (full Highland) | 30% | -70% |
| Scattering at 70 MeV | 100% | 50% | -50% |
| Scattering at 20 MeV | 100% | 100% | 0% |

---

## Evidence Map (4-Path Analysis)

### Path 1: Static Analysis (Agent: a5192a2)

| Evidence | Location | Interpretation |
|----------|----------|----------------|
| Energy-dependent scattering reduction factors | `src/cuda/kernels/k3_finetransport.cu:42-45` | Artificial suppression of scattering |
| SCATTER_REDUCTION_HIGH_E = 0.3f | Line 42 | 70% reduction for E > 100 MeV |
| SCATTER_REDUCTION_MID_HIGH = 0.5f | Line 43 | 50% reduction for E > 50 MeV |
| SCATTER_REDUCTION_MID_LOW = 0.7f | Line 44 | 30% reduction for E > 20 MeV |
| Highland formula correctly implemented | `src/include/physics/highland.hpp:60-81` | Physics formula is valid |

### Path 2: Physics Validation (Agent: a6659a2)

| Evidence | Location | Interpretation |
|----------|----------|----------------|
| Expected σₓ ≈ 5.6 mm at 160 mm | Theory (Highland + Fermi-Eyges) | Proper lateral spread should be ~5.6 mm |
| Highland formula with 2D projection | `highland.hpp` | Physics implementation is correct |
| 2D correction factor = 1/√2 ≈ 0.707 | `device_physics.cuh` | Correct for x-z plane projection |

**Theoretical Calculation for 150 MeV at 160 mm depth**:
```
sigma_theta = (13.6 / (beta * p)) * sqrt(z/X0) * [1 + 0.038*ln(z/X0)]
            = (13.6 / 150) * sqrt(160/36.08) * [1 + 0.038*ln(160/36.08)]
            ≈ 0.0607 rad

sigma_x = sigma_theta * z / sqrt(3)  (Fermi-Eyges)
        = 0.0607 * 160 / 1.732
        ≈ 5.6 mm
```

### Path 3: Scaffold Detection (Agent: a7909cb)

| Evidence | Location | Interpretation |
|----------|----------|----------------|
| "no random scattering, just energy loss" | `k2_coarsetransport.cu:181` | Coarse transport explicitly disables scattering |
| "to maintain forward penetration" | `k3_finetransport.cu:40` | Reduction is intentional workaround |
| PhysicsConfig ignored | `k1k6_pipeline.cu:471-473` | Hardcoded flags instead of config structure |
| Coarse transport never runs | `gpu_transport_wrapper.cu:74` | E_trigger = 300 MeV > max energy |

### Path 4: Runtime Analysis (Agent: a6bde10)

| Evidence | Location | Interpretation |
|----------|----------|----------------|
| All particles use K3 fine transport | `gpu_transport_wrapper.cu:74` | E_trigger = 300 MeV, so K2 never runs |
| K3 applies reduced scattering | `k3_finetransport.cu:254-266` | 30% scattering at 150 MeV |
| Bragg peak correct at 159.5 mm | Runtime output | Depth physics works correctly |
| N_theta = 36 angular bins | `gpu_transport_runner.cpp` | May limit visualization resolution |

---

## Root Cause

### #1: Energy-Dependent Scattering Reduction Factors ⭐⭐⭐⭐⭐

**Score**: 24/25 (Highest Confidence)

**Location**: `src/cuda/kernels/k3_finetransport.cu:42-66`

**Code**:
```cpp
// Lines 42-45: Scattering reduction factors
constexpr float SCATTER_REDUCTION_HIGH_E = 0.3f;     // E > 100 MeV
constexpr float SCATTER_REDUCTION_MID_HIGH = 0.5f;   // E > 50 MeV  
constexpr float SCATTER_REDUCTION_MID_LOW = 0.7f;    // E > 20 MeV
constexpr float SCATTER_REDUCTION_LOW_E = 1.0f;      // E <= 20 MeV

// Lines 254-266: Application
if (E > ENERGY_HIGH_THRESHOLD) {
    scattering_reduction_factor = SCATTER_REDUCTION_HIGH_E;
} else if (E > ENERGY_MID_HIGH_THRESHOLD) {
    scattering_reduction_factor = SCATTER_REDUCTION_MID_HIGH;
} else if (E > ENERGY_MID_LOW_THRESHOLD) {
    scattering_reduction_factor = SCATTER_REDUCTION_MID_LOW;
} else {
    scattering_reduction_factor = SCATTER_REDUCTION_LOW_E;
}
sigma_mcs *= scattering_reduction_factor;
```

**Mechanism**:
- 150 MeV protons (E > 100 MeV): scattering reduced to **30%** of physical value
- 70 MeV protons (E > 50 MeV): scattering reduced to **50%** of physical value
- Only protons ≤20 MeV get full scattering

**Why This Exists** (from comment at line 40):
```cpp
// Scattering reduction factors (to maintain forward penetration)
```
This appears to be a workaround added to compensate for another issue (possibly the variance accumulation issue from H6_MCS_investigation_report.md).

---

## Hypothesis Verification

### Experiment: Remove Scattering Reduction

**File**: `src/cuda/kernels/k3_finetransport.cu`

**Proposed Change**:
```cpp
// BEFORE (lines 254-266):
if (E > ENERGY_HIGH_THRESHOLD) {
    scattering_reduction_factor = SCATTER_REDUCTION_HIGH_E;  // 0.3
} else if (E > ENERGY_MID_HIGH_THRESHOLD) {
    scattering_reduction_factor = SCATTER_REDUCTION_MID_HIGH;  // 0.5
} else if (E > ENERGY_MID_LOW_THRESHOLD) {
    scattering_reduction_factor = SCATTER_REDUCTION_MID_LOW;  // 0.7
} else {
    scattering_reduction_factor = SCATTER_REDUCTION_LOW_E;  // 1.0
}
sigma_mcs *= scattering_reduction_factor;

// AFTER:
// Remove energy-dependent reduction - use full physical scattering
// scattering_reduction_factor = 1.0f;  // Always full scattering
// sigma_mcs *= scattering_reduction_factor;
// Or simply delete the reduction logic entirely
```

### Expected Result

| Metric | Before Fix | After Fix | Expected (Theory) |
|--------|-----------|-----------|-------------------|
| Lateral σₓ at 160 mm | ~1.7 mm | ~5.6 mm | ~5.6 mm |
| Dose distribution width | Narrow | Wide | Wide (Fermi-Eyges) |
| Bragg peak depth | 159.5 mm | ~159 mm | ~158 mm |

### Refutation Test

If lateral spread does NOT increase after removing reduction factors:
1. Check device_physics.cuh for additional suppression
2. Verify enable_mcs flag is actually true
3. Examine K4 bucket transfer for lateral movement
4. Add debug output for sigma_mcs values at each step

---

## Additional Findings

### #2: Coarse Transport Scattering Disabled (Lower Priority)

**Location**: `src/cuda/kernels/k2_coarsetransport.cu:182`
```cpp
float theta_new = theta;  // Coarse: no random scattering, just energy loss
```

**Impact**: Currently NOT an issue because K2 never runs (E_trigger = 300 MeV > max 250 MeV)

**If fixed**: Would need to apply proper MCS in K2 or document why it's intentionally disabled

### #3: Angular Bin Resolution (Minor Issue)

**Location**: `src/gpu/gpu_transport_runner.cpp`
```cpp
const int N_theta = 36;  // Number of angular bins
```

**Impact**: May cause quantization artifacts in angular distribution, but does NOT cause systematic under-prediction of lateral spread

---

## Relationship to Previous H6 Investigation

The H6_MCS_investigation_report.md identified an issue where scattering was applied at EVERY STEP (excessive), causing particles to stop at 84mm instead of 158mm.

**Current Status**:
- Depth physics is now correct (Bragg peak at 159.5 mm) ✓
- BUT lateral spreading is artificially suppressed (70% reduction at 150 MeV) ✗

**Interpretation**:
The scattering reduction factors (0.3, 0.5, 0.7) were likely added as a workaround to compensate for the excessive scattering from the "every step" approach. Now that depth physics is working, these reduction factors are causing lateral spreading to be too small.

**Two-Step Fix May Be Needed**:
1. Remove the scattering reduction factors (this report)
2. Implement proper variance-based accumulation (H6 report recommendation)

---

## Next Step Plan

### Priority 1: Verify #1 Hypothesis

- [ ] Modify `k3_finetransport.cu` to remove scattering reduction factors
- [ ] Rebuild: `cd build && cmake .. -DUSE_CUDA=ON && make`
- [ ] Run simulation with same config (150 MeV)
- [ ] Compare lateral spread before/after
- [ ] Check if Bragg peak depth remains ~159 mm

### Priority 2: If #1 Verified

- [ ] Remove unused SCATTER_REDUCTION constants
- [ ] Add validation test for lateral spread
- [ ] Compare against Python reference (validation/proton_transport_water.py)
- [ ] Document physics in code comments

### Priority 3: Consider H6 Fix

If removing reduction factors causes depth penetration issues:
- [ ] Implement variance-based accumulation from H6_MCS_investigation_report.md
- [ ] Replace per-step random sampling with periodic 7-point quadrature
- [ ] Test both depth penetration and lateral spread

---

## References

- **Code**: `src/cuda/kernels/k3_finetransport.cu:42-66` (scattering reduction)
- **Code**: `src/cuda/kernels/k2_coarsetransport.cu:182` (coarse transport)
- **Physics**: `src/include/physics/highland.hpp:60-81` (Highland formula)
- **GPU**: `src/cuda/device/device_physics.cuh:54-75` (device physics)
- **Validation**: `validation/proton_transport_water.py` (Python reference)
- **Previous**: `dbg/H6_MCS_investigation_report.md` (variance accumulation issue)

---

## Agent Links

- Path 1 (Static): agentId: a5192a2
- Path 2 (Physics): agentId: a6659a2
- Path 3 (Scaffold): agentId: a7909cb
- Path 4 (Runtime): agentId: a6bde10
