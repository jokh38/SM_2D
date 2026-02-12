# Bug Report: Double Division by mu_init in Step Limiting

**Date**: 2026-01-26
**Status**: Partially Fixed - Major Issue Remains
**Priority**: Critical

## Summary

The K2 and K3 transport kernels had a bug where `step_to_boundary` (already a path length) was being divided by `mu_init` (direction cosine) again, making particles always cross cell boundaries instead of remaining in cells.

## Fixes Applied

### Fix 1: Removed Redundant mu_init Division (Completed)

**K3 (k3_finetransport.cu:187-188)**
Changed from:
```cpp
float path_length_to_boundary = (mu_init > 1e-6f) ? step_to_boundary / mu_init : step_to_boundary;
float path_length_to_boundary_safe = path_length_to_boundary * 0.999f;
float actual_range_step = fminf(step_phys, path_length_to_boundary_safe);
```
To:
```cpp
// step_to_boundary is already the path length - don't divide by mu_init again!
float step_to_boundary_safe = step_to_boundary * 0.999f;
float actual_range_step = fminf(step_phys, step_to_boundary_safe);
```

**K2 (k2_coarsetransport.cu:174-183)**
Changed from:
```cpp
float coarse_step_limited = fminf(coarse_step, step_to_boundary * 0.999f);
float mu_abs = fmaxf(fabsf(mu), 1e-6f);
float coarse_range_step = coarse_step_limited / mu_abs;
```
To:
```cpp
// Convert path length to geometric distance for comparison with coarse_step
float mu_abs = fmaxf(fabsf(mu), 1e-6f);
float geometric_to_boundary = step_to_boundary * mu_abs;
float coarse_step_limited = fminf(coarse_step, geometric_to_boundary * 0.999f);
float coarse_range_step = coarse_step_limited / mu_abs;
```

### Fix 2: Energy Bin Center vs Lower Edge (Completed)

**Problem**: When reading energy from phase space bins, the code used bin center `(E_bin + 0.5) * dlog` which gave E=160.5 MeV for a 150 MeV particle.

**Root Cause**: Inconsistency between writing (uses `floor()`) and reading (uses `E_bin + 0.5` center).

**Fix (k2_coarsetransport.cu:147, k3_finetransport.cu:148)**:
```cpp
// OLD: float E = expf(log_E_min + (E_bin + 0.5f) * dlog);  // Center caused 150->160 MeV error
float E = expf(log_E_min + E_bin * dlog);  // Lower edge for consistency
```

**Result**: Energy now correctly decreases (141.6 → 141.3 → 110.2 → ...)

### Fix 3: Boundary Detection Epsilon (Completed)

**Problem**: Particles at z_new=0.2495 mm (just below 0.25 mm boundary) were not detected as crossing because `0.2495 > 0.25` was false.

**Root Cause**: 0.999 safety margin + exact boundary check prevented crossings.

**Fix (device_bucket.cuh:571-584)**:
```cpp
// CRITICAL FIX: Add epsilon tolerance to detect boundary crossings
// even when step is limited by safety margin (0.999 factor)
constexpr float BOUNDARY_EPSILON = 0.001f;  // 0.001 mm tolerance
float half_dz = dz * 0.5f;
float half_dx = dx * 0.5f;

if (z_new > half_dz - BOUNDARY_EPSILON) return FACE_Z_PLUS;
if (z_new < -half_dz + BOUNDARY_EPSILON) return FACE_Z_MINUS;
// ... similar for x boundaries
```

**Result**: Particles now cross boundaries (exit_face=0 instead of -1)

## Current Issues

### CRITICAL: Weight Doubling Bug

**Symptoms**:
- Weight doubles each iteration: 0.9995 → 1.998 → 3.996 → 7.990 → 15.255
- Total energy deposited: 1.46e6 MeV (expected ~150 MeV)
- Bragg Peak still at z=0 mm (expected ~158 mm)

**Debug Observations**:
```
Iteration 1: psi_out weight=0.9995, K4: cell=300 received 0.9995
Iteration 2: psi_out weight=1.998,  K4: cell=300 received 0.9995, cell=500 received 0.999
Iteration 3: psi_out weight=3.996,  K4: cell=300 received 0.9995, cell=500 received 1.998
Iteration 4: psi_out weight=7.990,  K4: cell=300 received 0.9995, cell=500 received 2.997
```

The psi_out weight is approximately **double** what individual cells report receiving!

**Hypothesis**: Particles are being written to psi_out multiple times:
1. K3 writes particles to psi_out (for those "remaining in cell")
2. K4 adds particles from buckets to psi_out
3. These might be the SAME particles being counted twice!

**Evidence**: In iterations 1-5, "K3: Writing particle to psi_out" does NOT appear, meaning all particles go to buckets. But psi_out still doubles...

**Next Investigation Steps**:
1. Check if psi_out is properly cleared before K4
2. Verify K3 and K4 write to different cells (no overlap)
3. Add debug to track exactly which cells/bins K3 writes to
4. Check if K4 is reading from correct bucket indices

## PDD Output

All dose is at z=0 mm surface - particles are not propagating into the phantom despite crossing cell boundaries.

```
# Depth-Dose Distribution (PDD)
0.0000	3.8956	1.0000  # All dose at surface
0.5000	0.0000	0.0000
1.0000	0.0000	0.0000
...
```

## Files Modified

1. `/workspaces/SM_2D/src/cuda/kernels/k3_finetransport.cu` - Removed redundant mu_init division, fixed energy bin lower edge
2. `/workspaces/SM_2D/src/cuda/kernels/k2_coarsetransport.cu` - Fixed geometric/path length comparison, fixed energy bin lower edge
3. `/workspaces/SM_2D/src/cuda/device/device_bucket.cuh` - Added boundary epsilon tolerance
4. `/workspaces/SM_2D/src/cuda/kernels/k4_transfer.cu` - Added debug output

## References

- NIST PSTAR data: 150 MeV protons should have ~158 mm range in water
- Previous fix: K6 buffer clearing bug (particles destroyed after swap)
- Previous fix: Boundary detection order (check before clamping)
