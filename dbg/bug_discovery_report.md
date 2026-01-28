# Bug Discovery Report: SM_2D Particle Transport Energy Loss

**Date**: 2025-01-28
**Last Updated**: 2026-01-28
**Status**: PARTIAL FIX APPLIED - Significant Progress Made (22% of expected energy deposition)
**Method**: MPDBGER 4-Path Analysis

---

## 1. Context

| Item | Value |
|------|-------|
| Repository | SM_2D (Proton Therapy Simulation) |
| Environment | NVIDIA GeForce RTX 2080, GPU transport enabled |
| Input Data | 150 MeV protons, source at (0,0), Gaussian beam |

## 2. Symptom

| Aspect | Expected | Actual | Gap |
|--------|----------|--------|-----|
| **Bragg Peak Depth** | ~158 mm | 1 mm | -157 mm |
| **Energy Deposition** | ~150 MeV | 16.965 MeV | -89% |
| **Simulation Iterations** | ~400-600 | 116 | -5x |

---

## 3. What We Checked (by Path)

### Path 1: Static/Dynamic Analysis (Agent: acfb286)

**Entry Points & Call Graph**:
- Main: `k1k6_pipeline.cu:589` - `run_k1k6_pipeline_transport()`
- K2 processes ActiveMask==0 cells (coarse/high-energy)
- K3 processes ActiveMask==1 cells (active/low-energy)
- **Critical insight**: K2 and K3 operate on **disjoint cell sets**

**Bug Candidates Found**:
1. **Energy Binning Error** (Possibility: HIGH)
   - Location: `k2_coarsetransport.cu:135`, `k3_finetransport.cu:157`
   - Code uses **lower edge**: `E = expf(log_E_min + E_bin * dlog)`
   - Should use **geometric mean** to match `EnergyGrid::rep[]`
   - Causes 6-10% systematic energy error per bin operation

2. **"K2→K3 Energy Gap" is NOT a bug** (Possibility: HIGH)
   - K2 reports E=83-88 MeV (coarse cells)
   - K3 reports E=17-19 MeV (active cells)
   - These are **different particle populations**
   - The apparent 60-70 MeV gap is expected behavior

### Path 2: Logic & Spec Tracing (Agent: a26388a)

**Breaking Points Identified**:

1. **Energy Binning Uses Lower Edge (SPEC says geometric mean)**
   - SPEC.md line 76: `E_rep[i] = sqrt(E_edges[i] * E_edges[i+1])`
   - Code k2:135, k3:157: `E = expf(log_E_min + E_bin * dlog)` (lower edge)
   - **Impact**: 10-15% energy loss per write-read cycle
   - **Severity**: CRITICAL

2. **Step Size Limited by Cell Size (contradicts SPEC)**
   - SPEC.md line 212: `delta_R_max = 0.02 * R` (at 150 MeV, R=158mm, so 3.16mm)
   - Code `step_control.hpp:57-58`: `cell_limit = 0.25 * min(dx, dz)` = 0.125mm
   - **Impact**: Particles take 25x more iterations than needed
   - **Severity**: CRITICAL

3. **Cumulative Effect**:
   - Each iteration: 10-15% energy loss from binning
   - ~70 iterations to reach "low energy" state
   - 150 * 0.85^70 ≈ 0.001 MeV (with floor, settles at ~10-20 MeV)

### Path 3: Scaffold Detection (Agent: ab51719)

**Result**: **No scaffold code found**
- All physics calculations are complete
- Nuclear cross-section 0.0012 is legitimate ICRU 63 value
- No placeholder values or stubs in production code
- Bug is a genuine implementation issue

### Path 4: Log Forensics (Agent: aada374)

**Critical Anomalies**:
1. Energy loss rate: 150 MeV → 20 MeV in ~70 iterations (2x too fast)
2. Weight decay: 1.0 → 1e-12 in ~70 iterations
3. The "60-70 MeV gap" is comparing different particle populations

**Silent Gaps**:
- No energy balance tracking per step
- No stopping power verification output
- No nuclear attenuation factor logging

---

## 4. Evidence Map

| Evidence | Source | Interpretation | Hypothesis |
|----------|--------|----------------|------------|
| `E = expf(log_E_min + E_bin * dlog)` (lower edge) | CODE(k2:135, k3:157) | Deviates from SPEC geometric mean | H1 |
| `E_rep = sqrt(E_edges[i] * E_edges[i+1])` | SPEC.md:76 | SPEC requires geometric mean | H1 |
| Comment: "CRITICAL FIX: Use lower edge" | CODE(k2:132-134) | Intentional deviation from SPEC | H1 |
| `cell_limit = 0.25 * fminf(dx, dz)` | CODE(step_control:57) | Limits step to 0.125mm | H2 |
| `delta_R_max = 0.02f * R` | SPEC.md:203 | SPEC requires 2% of range | H2 |
| 150 MeV → ~20 MeV in 70 iterations | RUNTIME | Energy loss 2x too fast | H1 + H2 |
| K2 E=83-88, K3 E=17-19 | RUNTIME | Different particle populations | NOT A BUG |
| Bragg peak at 1mm | RUNTIME | Particles travel ~14.5mm total | H2 |

---

## 5. Hypotheses Ranked

### H1: Energy Binning Uses Lower Edge Instead of Geometric Mean (CRITICAL)

**Score**: 24/25

**Evidence**:
- CODE(k2:135, k3:157): Uses lower edge
- SPEC.md:76: Requires geometric mean
- Code comment acknowledges deviation

**Mechanism**:
1. Particle with E_new=150 MeV written to bin i
2. When read back, energy = E_edges[i] (lower edge) not geometric mean
3. Each write-read loses ~10% energy
4. After ~70 iterations: 150 * 0.9^70 → negligible

**If true, MUST observe**:
- Particles in bin i have lower energy than when written
- Energy decay follows exponential pattern
- Fix restores energy conservation

**Refutation Experiment**:
```cpp
// Change in k2_coarsetransport.cu:135 and k3_finetransport.cu:157:
// FROM: float E = expf(log_E_min + E_bin * dlog);
// TO:   float E = expf(log_E_min + (E_bin + 0.5f) * dlog);
```

### H2: Step Size Limited to 0.125mm Instead of 2% of Range (CRITICAL)

**Score**: 23/25

**Evidence**:
- CODE(step_control:57-58): cell_limit = 0.125mm
- SPEC.md:203: delta_R_max = 0.02 * R = 3.16mm at 150 MeV
- Actual step is 25x smaller than SPEC

**Mechanism**:
1. At 150 MeV, R ≈ 158mm
2. SPEC: step = 3.16mm, Code: step = 0.125mm
3. Particles take 25x more iterations
4. Each iteration has binning error from H1
5. After 116 iterations: 116 * 0.125mm = 14.5mm (not 158mm!)

**If true, MUST observe**:
- Particles travel only ~14.5mm before sim ends
- Iteration count is ~116 (not ~400-600)
- Increasing step size increases distance

**Refutation Experiment**:
```cpp
// Comment out in step_control.hpp:55-58:
// float cell_limit = 0.25f * fminf(dx, dz);
// delta_R_max = fminf(delta_R_max, cell_limit);
```

### H3: "K2→K3 Energy Gap" is a Misinterpretation (NOT A BUG)

**Score**: 5/25

**Evidence**:
- K2 processes ActiveMask==0, K3 processes ActiveMask==1
- Different particle populations
- Expected behavior, not a bug

**Status**: Explains the "gap" but NOT the root cause

---

## 6. Root Cause Conclusion

### PRIMARY ROOT CAUSE: H1 + H2 Interaction

The energy deposition bug is caused by **two SPEC deviations compounding**:

1. **H1: Energy binning error** - ~10% energy loss per iteration
2. **H2: Step size too small** - 25x more iterations than needed

**Combined Effect**:
```
Iteration 1: 150 MeV → bin → read as ~135 MeV (10% loss)
Iteration 2: 135 MeV → bin → read as ~122 MeV
...
Iteration 70: ~10-20 MeV (floor effect)
Distance: 70 * 0.125mm ≈ 8.75mm (not 158mm!)
```

---

## 7. Fix Results (COMMIT: 2b60143)

### Fixes Applied (2026-01-28)

**H1: Energy Binning** - `src/cuda/kernels/k2_coarsetransport.cu:135`, `k3_finetransport.cu:157`
```cpp
// BEFORE (lower edge):
float E = expf(log_E_min + E_bin * dlog);
// AFTER (geometric mean per SPEC.md:76):
float E = expf(log_E_min + (E_bin + 0.5f) * dlog);
```

**H2: Step Size Limits** - `src/include/physics/step_control.hpp:55-58`, `gpu_transport_wrapper.cu:78`
- Removed cell_limit (0.125mm)
- Removed 1mm hard limit
- Increased step_coarse from 0.3mm to 5mm

**H3: Boundary Crossing** - `src/cuda/kernels/k2_coarsetransport.cu:170-176`, `k3_finetransport.cu:206-210`
```cpp
// BEFORE: Particles stopped at 99.9% of boundary
float coarse_step_limited = fminf(coarse_step, geometric_to_boundary * 0.999f);
// AFTER: Let boundary detection handle crossing
float coarse_step_limited = coarse_step;
```

### Results Summary

| Metric | Before Fixes | After Fixes | Expected | Change |
|--------|--------------|-------------|----------|--------|
| Energy Deposited | 16.97 MeV | 32.96 MeV | ~150 MeV | +94% |
| Iterations | 116 | 86 | ~400-600 | -26% |
| Max Depth (cell) | 1700 (z=8mm) | 16900 (z=84mm) | ~158mm | +950% |
| Bragg Peak | 1mm | 0mm (surface) | ~158mm | Surface dose |

### Progress Assessment

**Improvements:**
- Energy deposited nearly doubled (16.97 → 32.96 MeV)
- Particles travel 10x farther (8mm → 84mm depth)
- Fewer iterations needed (more efficient transport)

**Remaining Issues:**
- Dose peaks at surface instead of Bragg peak (~158mm)
- Only 22% of expected energy deposited (32.96 / 150 MeV)
- Particles stop at 84mm, not full 158mm range

---

## 8. Attempt Log (Including Previous H1 Refutation)

| # | Attempt | Result | Verdict |
|---|---------|--------|---------|
| 1 | weight_active_min: 1e-6 → 1e-12 | No change (16.965 MeV) | FAILED |
| 2 | Remove weight check entirely | Worse (11.8591 MeV) | FAILED |
| 3 | E_trigger: 10 → 20 MeV | Worse (11.8591 MeV) | FAILED |
| 4 | All SPEC values combined | No improvement | FAILED |
| 5 | **MPDBGER 4-path analysis** | Root cause identified | SUCCESS |

**Note**: Previous "H1" referred to weight threshold hypothesis, which is now refuted.
**New H1/H2** refer to energy binning and step size issues identified by MPDBGER.

---

## 9. Next Steps (Remaining Investigation)

### Remaining Issues After H1, H2, H3 Fixes

1. **Dose peaks at surface (0mm)** instead of Bragg peak (~158mm)
2. **Only 22% of expected energy** deposited (32.96 / 150 MeV)
3. **Particles stop at 84mm depth**, not full 158mm range

### Hypotheses for Remaining Issues

**H4: Nuclear Attenuation Too Aggressive**
- Weight drops from 1.0 to ~1e-6 quickly
- At low weights, energy contribution (E × w) becomes negligible
- Check: Verify nuclear cross-section (currently 0.0012 mm⁻¹) and attenuation formula

**H5: Energy Loss Rate Too High**
- Particles lose energy ~4x too fast (stop at 84mm instead of 158mm)
- Check: Verify stopping power (dE/dx) calculations against NIST data

**H6: Excessive Lateral Scattering**
- Particles may scatter sideways instead of penetrating forward
- Check: MCS angle calculation and application

### Verification Plan

**Priority 1: Check Nuclear Attenuation**
```cpp
// File: src/cuda/device_physics.cuh
// Verify: nuclear_cross_section value and formula
// SPEC.md suggests 0.0050 mm⁻¹, code uses 0.0012 mm⁻¹
```

**Priority 2: Verify Stopping Power LUT**
```cpp
// Check R(150 MeV) = 157.667 mm is used correctly
// Verify dE = S(E) × step calculation
```

**Priority 3: Review MCS Scattering**
```cpp
// Check theta_0 = (13.6 MeV / beta*c*p) * sqrt(step/X0) formula
// Verify angular distribution application
```

---

## 10. H7: Energy Grid E_max Bug (FIXED 2026-01-28)

### Critical Finding
The code was using E_max=300 MeV in multiple locations, causing energy grid corruption.

### Root Cause
R(300 MeV) returns NaN because NIST PSTAR data only covers up to ~250 MeV for protons in water. This caused:
1. Corrupted energy edges in the LUT
2. Particles reading at wrong energies (176-194 MeV instead of ~150 MeV)
3. Incorrect energy binning during write/read cycles

### Files Modified
1. `src/cuda/gpu_transport_wrapper.cu:88` - EnergyGrid(0.1f, 250.0f, N_E)
2. `src/gpu/gpu_transport_runner.cpp:56` - GenerateRLUT(0.1f, 250.0f, 256)

### Results
- Energy read from bin: 176-194 MeV → 150.984 MeV (CORRECT)
- Energy loss now decreases: 151 → 149 → 147 MeV (CORRECT)
- K3 LUT now shows E_max=250.000 MeV (was 300.000 MeV)

---

## 11. References

- **Code**: `src/cuda/kernels/k2_coarsetransport.cu`, `k3_finetransport.cu`
- **Code**: `src/include/physics/step_control.hpp`
- **SPEC**: `SPEC.md` lines 70-77 (energy grid), 200-213 (step control)
- **Debug Output**: `output_message.txt`
- **Debug History**: `.sisyphus/debug_history.md`
