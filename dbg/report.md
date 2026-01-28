# Debug Report: Particle Transport Energy Loss

## Issue Summary

**Date**: 2026-01-28
**Last Updated**: 2026-01-28
**Status**: PARTIAL FIX APPLIED - Significant Progress Made
**Severity**: Critical (simulation produces 22% of expected energy deposition)
**Commit**: 2b60143

## Symptom - Before Fixes

| Metric | Expected | Actual | Error |
|--------|----------|--------|-------|
| Bragg Peak Depth | ~158 mm | 1 mm | -99% |
| Energy Deposited | ~150 MeV | 16.965 MeV | -89% |
| Simulation Iterations | ~400-600 | 116 | -77% |

## Symptom - After Fixes

| Metric | Expected | Actual | Error |
|--------|----------|--------|-------|
| Bragg Peak Depth | ~158 mm | 0 mm (surface) | -100% |
| Energy Deposited | ~150 MeV | 32.96 MeV | -78% |
| Simulation Iterations | ~400-600 | 86 | -79% |
| Max Depth Reached | ~158 mm | 84 mm | -47% |
| Bragg Peak Dose | ~100+ Gy | 3.41 Gy | -96% |

**Improvement**: Energy deposited increased from 16.97 MeV to 32.96 MeV (+94%)

## Root Causes Identified and Fixed

### H1: Energy Binning Uses Lower Edge Instead of Geometric Mean (FIXED)
- Location: `src/cuda/kernels/k2_coarsetransport.cu:135`, `k3_finetransport.cu:157`
- Code: `float E = expf(log_E_min + E_bin * dlog);` (lower edge)
- SPEC.md:76 requires: `E_rep = sqrt(E_edges[i] * E_edges[i+1])` (geometric mean)
- Fix: `float E = expf(log_E_min + (E_bin + 0.5f) * dlog);`
- Impact: Reduced energy loss per binning operation

### H2: Step Size Limited by Multiple Constraints (PARTIALLY FIXED)
1. **cell_limit** in `step_control.hpp`: Limited step to 0.125mm (FIXED - removed)
2. **1mm hard limit** in `step_control.hpp`: Capped step at 1mm (FIXED - removed)
3. **step_coarse** limited by cell size: 0.5mm effective limit (FIXED - now 5mm)
4. **step_coarse** in `gpu_transport_wrapper.cu`: Was 0.3mm (FIXED - now 5mm)

### H3: Boundary Crossing Prevented by 99.9% Step Limit (FIXED)
- Location: `src/cuda/kernels/k2_coarsetransport.cu:170-176`
- Code: `coarse_step_limited = fminf(coarse_step, geometric_to_boundary * 0.999f);`
- Problem: Particles reached 99.9% of boundary but never crossed
- Fix: Removed limit, let boundary detection handle crossing
- Impact: Particles now travel 84mm vs 8mm before (10.5x improvement)

## Results of Fixes

### Progress Made
- Energy deposited: 16.97 → 32.96 MeV (+94%)
- Max depth reached: 8mm → 84mm (+950%)
- Iteration efficiency: 116 → 86 iterations (-26%)

### Remaining Issues
1. Dose peaks at surface (0mm) instead of Bragg peak (~158mm)
2. Energy deposited still only 22% of expected (32.96 / 150 MeV)
3. Particles only penetrate to 42mm, not full 158mm range

## Possible Remaining Root Causes

### Hypothesis A: Nuclear Attenuation Too Aggressive
- Weight drops from 1.0 to ~1e-6 quickly
- At low weights, energy contribution (E × w) becomes negligible
- Check: Verify nuclear cross-section and attenuation formula

### Hypothesis B: Energy Loss Rate Too High
- Particles lose energy too fast, stopping at 42mm instead of 158mm
- Suggests dE/dx is ~4x too high
- Check: Verify stopping power calculations against NIST data

### Hypothesis C: Excessive Lateral Scattering
- Particles may scatter sideways instead of penetrating forward
- Check: MCS angle calculation and application

## Files Modified

1. `src/cuda/kernels/k2_coarsetransport.cu`
   - Line 135: Energy binning (H1)
   - Line 94: step_coarse limit removed (H2)
   - Lines 170-176: Boundary crossing fix (H3)

2. `src/cuda/kernels/k3_finetransport.cu`
   - Line 157: Energy binning (H1)
   - Lines 206-210: Boundary crossing fix (H3)

3. `src/include/physics/step_control.hpp`
   - Line 50: Removed 1mm hard limit (H2)
   - Lines 55-58: Removed cell_limit (H2)

4. `src/cuda/gpu_transport_wrapper.cu`
   - Line 78: Increased step_coarse from 0.3mm to 5mm (H2)

## References

- Full analysis: `dbg/bug_discovery_report.md`
- Debug history: `dbg/debug_history.md`
- SPEC: `SPEC.md:76` (energy grid), `SPEC.md:203` (step control)

---

## 2026-01-28 Update: H7 Fix Applied

### Energy Grid E_max Bug (FIXED)
Fixed critical bug where E_max=300 MeV was causing NaN values in the range LUT.

**Files Modified:**
- `src/cuda/gpu_transport_wrapper.cu:88` - Changed E_max from 300.0 to 250.0 MeV
- `src/gpu/gpu_transport_runner.cpp:56` - Changed GenerateRLUT from 300.0 to 250.0 MeV

**Results:**
- Energy read from bin: 176-194 MeV → 150.984 MeV (CORRECT)
- Energy loss now decreases correctly: 151 → 149 → 147 MeV
- K3 LUT shows E_max=250.000 MeV (was 300.000 MeV)

**Current Status (after H1, H2, H3, H5, H7 fixes):**
- Energy deposited: 29.78 MeV (~20% of expected 150 MeV)
- Bragg Peak: 2.5mm depth (vs expected ~158mm)
- Iterations: 44 (vs expected ~400-600)

**Remaining Issue:**
Particles still stop at ~2.5mm instead of 158mm. This suggests either:
1. Fine transport (K3) has different step size limits than coarse transport (K2)
2. Energy loss rate is too high
3. Particles switch to fine transport too early
