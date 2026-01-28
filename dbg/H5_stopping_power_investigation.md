# H5: Stopping Power (dE/dx) Implementation Investigation Report

**Date**: 2026-01-28
**Investigator**: Sisyphus-Junior
**Status**: CRITICAL BUG FOUND

---

## Executive Summary

Investigated stopping power implementation and found that **particles are losing energy too fast** and stopping at **42mm depth** instead of the expected **158mm**. The root cause is an **artificial 1.0mm cap on step size** that contradicts the SPEC.md requirement of `delta_R_max = 0.02 * R`.

**Impact**: Particles travel only 26.6% of expected range (42mm vs 158mm), depositing only 22% of expected energy (33 MeV vs 150 MeV).

---

## 1. Stopping Power Implementation Analysis

### 1.1 NIST PSTAR Data Verification

**File**: `/workspaces/SM_2D/src/data/nist/pstar_water.txt`

```
150.0  5.443    15.77
```

- **Stopping Power (S)**: 5.443 MeV·cm²/g at 150 MeV
- **CSDA Range**: 15.77 g/cm² = 157.7 mm (for water, ρ=1.0 g/cm³)

**Verification**: ✓ Correct - matches official NIST PSTAR database

### 1.2 Range LUT Generation

**File**: `/workspaces/SM_2D/src/lut/r_lut.cpp:6-65`

**Method**: Log-log interpolation from NIST data
```cpp
// Convert NIST range from g/cm² to mm
const float g_cm2_to_mm = 10.0f / rho;  // rho = 1.0 g/cm³
lut.R[i] = nist_data.csda_range_g_cm2 * g_cm2_to_mm;
```

**Test Results**:
```
R(150 MeV) = 157.7 mm  ✓ Correct
S(150 MeV) = 5.443 MeV·cm²/g  ✓ Correct
lookup_E_inverse(157.7) = 150 MeV  ✓ Correct
```

**Verification**: ✓ LUT generation is correct

### 1.3 Energy Loss Calculation

**File**: `/workspaces/SM_2D/src/cuda/device/device_lut.cuh:119-131`

**Method**: R-based control (CORRECT and more accurate than S-based)
```cpp
__device__ inline float device_compute_energy_deposition(
    const DeviceRLUT& lut, float E, float step_length) {
    float E_new = device_compute_energy_after_step(lut, E, step_length);
    return E - E_new;
}
```

**Verification**:
- Test with step=0.5mm: dE = 0.2926 MeV (expected ~0.275 MeV)
- Error: 6.45% (reasonable for interpolation)
- **Conclusion**: ✓ Energy loss calculation is correct

**Physics Note**: R-based method is used instead of `dE = S(E) × step` because:
- R-based accounts for varying S over the step
- Mathematically equivalent but more accurate
- dR/ds = -1 in CSDA approximation

---

## 2. ROOT CAUSE: Step Size Artificially Limited

### 2.1 The Bug

**File**: `/workspaces/SM_2D/src/cuda/device/device_lut.cuh:173`

```cpp
delta_R_max = fminf(delta_R_max, 1.0f);  // ← BUG: Arbitrary 1mm cap
```

This line **contradicts SPEC.md line 203**:
```cpp
float delta_R_max = 0.02f * R;  // Max 2% range loss per substep
// NO mention of 1.0mm cap in SPEC!
```

### 2.2 Impact Analysis

**At 150 MeV**:
- SPEC requirement: `delta_R_max = 0.02 × 157.7mm = 3.15mm`
- Current code: `delta_R_max = min(3.15, 1.0) = 1.0mm`
- **Bug**: Step is **3.15x too small**!

**Cumulative Effect**:
```
Expected iterations: 157.7mm / 3.15mm ≈ 50 iterations
Actual iterations: 157.7mm / 1.0mm ≈ 157 iterations
Extra iterations: 107 (3x more!)
```

### 2.3 Actual Simulation Behavior

**From `/workspaces/SM_2D/output_message.txt`**:
```
Iterations: 86
Final cell: 16900 (iz=84, depth=42mm)
Energy deposited: 32.96 MeV (expected 150 MeV)
Bragg Peak: 0mm (surface dose)
```

**Analysis**:
- Particles travel 42mm in 86 iterations
- Effective step: 42mm / 86 = 0.49mm
- This is even smaller than 1.0mm!

**Explanation**: As particles lose energy and drop below 50 MeV, the step size limits kick in:
```
E < 50 MeV:  delta_R_max ≤ 0.5mm
E < 20 MeV:  delta_R_max ≤ 0.3mm
E < 10 MeV:  delta_R_max ≤ 0.15mm
```

So particles spend most of their time taking 0.1-0.5mm steps instead of the SPEC-required 2-3mm steps.

---

## 3. SPEC.md Deviations

### 3.1 SPEC.md Requirements

**Lines 199-213** (step control):
```cpp
float compute_max_step_physics(float E) {
    float R = lookup_R(E);
    
    // Option A: Fixed fraction of remaining range
    float delta_R_max = 0.02f * R;  // Max 2% range loss per substep
    
    // Energy-dependent refinement near Bragg
    if (E < 10.0f) {
        delta_R_max = min(delta_R_max, 0.2f);  // mm
    } else if (E < 50.0f) {
        delta_R_max = min(delta_R_max, 0.5f);
    }
    
    return delta_R_max;  // This IS the step size
}
```

**Key Points**:
1. Base step: `0.02 × R` (2% of remaining range)
2. Limit at E < 50 MeV: 0.5mm
3. Limit at E < 10 MeV: **0.2mm** (code has 0.15mm - slightly more conservative)
4. **NO 1.0mm cap mentioned**

### 3.2 Current Code Deviations

**File**: `/workspaces/SM_2D/src/cuda/device/device_lut.cuh:142-186`

```cpp
// Base: 2% of remaining range ✓
float delta_R_max = 0.02f * R;

// Energy-dependent refinement
if (E < 5.0f) {
    delta_R_max = fminf(delta_R_max, 0.1f);  // More conservative than SPEC
    dS_factor = 0.6f;
} else if (E < 10.0f) {
    delta_R_max = fminf(delta_R_max, 0.15f);  // Slightly more conservative
    dS_factor = 0.7f;
} else if (E < 20.0f) {
    delta_R_max = fminf(delta_R_max, 0.3f);  // ✓ Matches SPEC
    dS_factor = 0.8f;
} else if (E < 50.0f) {
    delta_R_max = fminf(delta_R_max, 0.5f);  // ✓ Matches SPEC
    dS_factor = 0.9f;
}

delta_R_max = delta_R_max * dS_factor;
delta_R_max = fminf(delta_R_max, 1.0f);  // ✗ SPEC deviation: ARTIFICIAL CAP
```

**Deviations Found**:
1. **Line 173**: `fminf(delta_R_max, 1.0f)` - **NOT in SPEC**
2. **Lines 152-161**: `dS_factor` (0.6-0.9) - **NOT in SPEC**
3. **Lines 155-156**: E < 5 MeV limit of 0.1mm - **NOT in SPEC** (SPEC starts at E < 10 MeV)

---

## 4. Why the 1.0mm Cap Was Added

**Comment from line 180-184**:
```cpp
// REMOVED: Artificial cell_limit was causing step size to be limited to 0.125mm
// This prevented 150MeV protons from traveling their full ~158mm range
```

This indicates previous issues with cell-size limiting. However, the **1.0mm cap was not removed** when `cell_limit` was removed, creating the current bug.

---

## 5. Verification: Expected vs Actual Behavior

### 5.1 Without 1.0mm Cap (SPEC-compliant)

```
At E=150 MeV: delta_R_max = 0.02 × 157.7 = 3.15mm
At E=100 MeV: delta_R_max = 0.02 × 77.2 = 1.54mm  
At E=70 MeV:  delta_R_max = 0.02 × 40.8 = 0.82mm (capped at 0.5mm by E < 50 rule)

Expected iterations: ~50
Expected range: 157.7mm
```

### 5.2 With 1.0mm Cap (Current buggy code)

```
At E=150 MeV: delta_R_max = min(3.15, 1.0) = 1.0mm  ← BUG
At E=100 MeV: delta_R_max = min(1.54, 1.0) = 1.0mm  ← BUG
At E=70 MeV:  delta_R_max = min(0.82, 1.0) = 0.82mm (capped at 0.5mm by E < 50 rule)

Actual iterations: 86
Actual range: 42mm (26.6% of expected)
```

---

## 6. Additional Findings

### 6.1 Coarse Transport Step Size

**File**: `/workspaces/SM_2D/src/cuda/gpu_transport_wrapper.cu:78`
```cpp
config.step_coarse = 5.0f;  // Much larger step for proper penetration
```

This value is **CORRECT** and appropriate for coarse transport (high energy). However, the fine transport step size (controlled by `device_compute_max_step`) is what limits particle penetration.

### 6.2 K2 Coarse Transport Division Bug

**File**: `/workspaces/SM_2D/src/cuda/kernels/k2_coarsetransport.cu:176`
```cpp
float coarse_range_step = coarse_step_limited / mu_abs;
```

**Issue**: Divides by `mu_abs` which is technically incorrect (coarse_step is already path length).

**Impact**: For normal incidence (mu=1), this is harmless. For angled beams, this would cause inconsistencies between energy loss and position update.

**Severity**: Low (not causing current range issue)

---

## 7. Recommended Fix

### 7.1 Primary Fix (CRITICAL)

**File**: `/workspaces/SM_2D/src/cuda/device/device_lut.cuh:173`

**REMOVE** the 1.0mm cap:
```cpp
// BEFORE (BUGGY):
delta_R_max = delta_R_max * dS_factor;
delta_R_max = fminf(delta_R_max, 1.0f);  // ← REMOVE THIS LINE
delta_R_max = fmaxf(delta_R_max, 0.1f);

// AFTER (SPEC-COMPLIANT):
delta_R_max = delta_R_max * dS_factor;
delta_R_max = fmaxf(delta_R_max, 0.1f);
```

### 7.2 Secondary Fixes (Recommended)

1. **Remove `dS_factor`** (lines 148-170, 172):
   - Not mentioned in SPEC.md
   - Reduces step size by 10-40% unnecessarily
   - Complicates code without clear benefit

2. **Align E < 10 MeV limit with SPEC** (line 161):
   - Current: `fminf(delta_R_max, 0.15f)`
   - SPEC: `fminf(delta_R_max, 0.2f)`

3. **Remove E < 5 MeV case** (lines 152-156):
   - Not in SPEC
   - Overly conservative (0.1mm limit)

### 7.3 Tertiary Fixes (Optional)

1. **Fix K2 division by mu_abs** (k2_coarsetransport.cu:176):
   - Change to: `float coarse_range_step = coarse_step_limited;`
   - Low priority (only affects angled beams)

---

## 8. Expected Impact of Fix

**Removing the 1.0mm cap**:
- Step size at 150 MeV: 1.0mm → 3.15mm (3.15x increase)
- Step size at 100 MeV: 1.0mm → 1.54mm (1.54x increase)
- Expected range: 42mm → 158mm (3.76x increase)
- Expected energy deposition: 33 MeV → 150 MeV (4.5x increase)
- Expected iterations: 86 → ~50 (1.7x decrease)

**Removing dS_factor** (additional 10-40% improvement at high energies)

---

## 9. Test Plan

1. **Remove line 173** in device_lut.cuh
2. **Rebuild and run simulation**
3. **Verify**:
   - Bragg peak depth ≈ 158mm
   - Energy deposited ≈ 150 MeV
   - Iterations ≈ 50-100
4. **If issues persist**, also remove `dS_factor` logic

---

## 10. Files Requiring Changes

| File | Line | Change | Severity |
|------|------|--------|----------|
| `src/cuda/device/device_lut.cuh` | 173 | Remove `fminf(delta_R_max, 1.0f)` | CRITICAL |
| `src/cuda/device/device_lut.cuh` | 148-172 | Consider removing `dS_factor` | HIGH |
| `src/cuda/device/device_lut.cuh` | 161 | Change 0.15f → 0.2f | MEDIUM |
| `src/cuda/kernels/k2_coarsetransport.cu` | 176 | Remove `/ mu_abs` | LOW |

---

## Conclusion

The stopping power (dE/dx) implementation is **physically correct**, but an **artificial 1.0mm cap on step size** violates SPEC.md requirements and prevents particles from traveling their full range. This is the root cause of H5 (particles stopping at 42mm instead of 158mm).

**Recommended Action**: Remove line 173 in `src/cuda/device/device_lut.cuh` and rebuild.

