# SM_2D Physics Issues - GPU Investigation Update

**Date**: 2026-02-07  
**Method**: GPU reruns + validator reruns + CUDA code inspection  
**Status**: Core handoff/injection bugs remain fixed and the strong `-x` beam-bending bug is now corrected, but transport is still physically incorrect (Bragg too shallow, high transport-drop channels, and non-closing energy channels)

---

## Executive Summary

1. **K2->K3 handoff was broken at threshold (fixed).**
   - Before fix, active cells stayed zero even in deep iterations.
   - After fix, K3 activates and transport completes.

2. **Source injection had a slot-allocation race (fixed).**
   - Before fix, ~60% source energy was dropped at injection.
   - After fix, source slot drop is 0%.

3. **Lateral spreading is still incorrect (open, critical).**
   - Follow-up patches removed the fully flat sigma trend, but depth behavior is still non-physical.
   - Current profiles narrow at deep depth (`sigma_140 < sigma_100`), which is opposite expected MCS broadening.

4. **Energy accounting is still inconsistent (open, critical).**
   - Transport channels still exceed source energy in verbose runs.
   - K5 weight/energy audits remain unstable through late iterations.

5. **SPEC cross-reference had stale "missing implementation" claims (corrected).**
   - Hysteresis is implemented in K1 (`E_fine_on/E_fine_off` with previous-mask hold).
   - Crossing guard is implemented in K2 (step split at `E_fine_on` crossing).
   - K5 energy audit is implemented (`K5_ConservationAudit` with `E_error/E_pass`).

6. **Follow-up transport patch set is incomplete (open, critical).**
   - Beam-direction sampling, depth spread, and energy/boundary channels are still not jointly consistent.
   - A targeted physics-coupled fix pass is required before further tuning.

---

## Follow-up Session Update (Current Handoff, 2026-02-07)

### Fresh artifacts for next session

- `validation/latest_gpu_run_sigma38_600_current.log`
- `validation/latest_gpu_run_sigma38_verbose_current.log`
- `validation/latest_comparison_sigma38_current.log`

### Current validation snapshot (`validation/gpu_compare_sigma38_600.ini`)

- Iteration progression from `validation/latest_gpu_run_sigma38_600_current.log`:
  - `Iteration 50: 0 active, 26 coarse cells`
  - `Iteration 100: 0 active, 48 coarse cells`
  - `Iteration 150: 0 active, 43 coarse cells`
  - `Transport complete after 194 iterations`
- Bragg peak from same run:
  - `Bragg Peak: 109 mm`
- MOQUI comparison from `validation/latest_comparison_sigma38_current.log`:
  - Bragg: SM_2D `109 mm`, MOQUI `154 mm` (error `-45 mm`, `-29.22%`)
  - Relative dose: `20 mm = 0.748`, `100 mm = 0.958`, `140 mm = 0.234`
  - Lateral sigma: `20 mm = 4.246 mm`, `100 mm = 5.096 mm`, `140 mm = 2.972 mm`
  - Key failure: deep-depth narrowing (`sigma_140` much smaller than MOQUI `6.369 mm`)

### Current verbose energy channels (`validation/latest_gpu_run_sigma38_verbose_current.log`)

- `Source Energy (total): 150.002 MeV`
- `Energy Deposited: 85.3472 MeV`
- `Cutoff Energy Deposited: 3.49934e-06 MeV`
- `Nuclear Energy Deposited: 15.9599 MeV`
- `Boundary Loss Energy: 27.4601 MeV`
- `Transport Drop Energy: 23.9333 MeV`
- `Total Accounted Energy (transport only): 152.7 MeV`
- `Transport complete after 195 iterations`

### Follow-up code changes already applied (still insufficient)

1. Gaussian CDF stabilization and clamping:
   - `src/cuda/device/device_physics.cuh`
2. Multi-cell emission weight non-negativity + renormalization:
   - `src/cuda/device/device_bucket.cuh`
3. K2 energy-step path consistency change:
   - `src/cuda/kernels/k2_coarsetransport.cu`
4. K2/K3 switched to incremental (delta-variance) spread estimate:
   - `src/cuda/kernels/k2_coarsetransport.cu`
   - `src/cuda/kernels/k3_finetransport.cu`
5. Runtime angular grid narrowed from `[-pi/2, +pi/2]` to `[-0.35, +0.35] rad` for `N_theta=36`:
   - `src/gpu/gpu_transport_runner.cpp`

### Additional regression evidence

- `build/tests/sm2d_tests`: `88` passed, `1` failed.
- Failing test:
  - `EnergyLossOnlyTest.FullPhysics` (`tests/gpu/test_energy_loss_only_gpu.cu`)
  - Mid-depth spread is too low (`sigma_100` below test threshold).

---

## Follow-up Session Update (Current Handoff, 2026-02-07, Late Patch Pass)

### Fresh artifacts for next session

- `validation/latest_gpu_run_sigma38_600_fix_slots.log`
- `validation/latest_gpu_run_sigma38_verbose_fix_slots.log`
- `validation/latest_comparison_sigma38_fix_slots.log`
- `validation/latest_gpu_run_sigma38_600_fix_drift.log`
- `validation/latest_comparison_sigma38_fix_drift.log`

### Root causes confirmed and fixed in this pass

1. **Systematic `-x` drift in forward lateral spreading (fixed):**
   - Multi-cell spread used an even-sized window with asymmetric integer offsets (`-5..+4`) in `device_emit_lateral_spread`, biasing every `+z` transfer to the left.
   - Fixes:
     - centered Gaussian window construction in `device_gaussian_spread_weights`,
     - odd spread-cell enforcement and symmetric offset mapping in `device_emit_lateral_spread`.
   - Files:
     - `src/cuda/device/device_physics.cuh`
     - `src/cuda/device/device_bucket.cuh`

2. **Direction/transport mismatch in K2/K3 (fixed):**
   - K2/K3 sampled `theta_new` but advected position with stale `mu/eta`.
   - Fix:
     - advection now uses `mu_new = cos(theta_new)`, `eta_new = sin(theta_new)` in both kernels.
   - Files:
     - `src/cuda/kernels/k2_coarsetransport.cu`
     - `src/cuda/kernels/k3_finetransport.cu`

3. **Concurrent slot-claim race pattern in bucket/output allocation (partially fixed):**
   - Allocation paths now accept CAS outcomes `old==EMPTY || old==bid` to avoid false misses under contention.
   - Files:
     - `src/cuda/device/device_bucket.cuh`
     - `src/cuda/kernels/k2_coarsetransport.cu`
     - `src/cuda/kernels/k3_finetransport.cu`
     - `src/cuda/kernels/k4_transfer.cu`

### Validation snapshot after fixes (`validation/gpu_compare_sigma38_600.ini`)

- From `validation/latest_gpu_run_sigma38_600_fix_slots.log`:
  - `Iteration 50: 0 active, 18 coarse cells`
  - `Iteration 100: 0 active, 38 coarse cells`
  - `Iteration 150: 0 active, 89 coarse cells`
  - `Transport complete after 204 iterations`
  - `Bragg Peak: 111 mm`
- From `validation/latest_comparison_sigma38_fix_slots.log`:
  - Bragg: SM_2D `111 mm`, MOQUI `154 mm` (error `-43 mm`, `-27.92%`)
  - Relative dose: `20 mm = 0.738`, `100 mm = 0.946`, `140 mm = 0.459`
  - Lateral sigma: `20 mm = 4.246 mm`, `100 mm = 4.671 mm`, `140 mm = 5.096 mm`

### Beam-centroid check (drift diagnosis)

- Parsed from current `results/dose_2d.txt`:
  - `mu_x(20 mm) = +0.023 mm`
  - `mu_x(100 mm) = +0.023 mm`
  - `mu_x(140 mm) = -1.024 mm`
  - Depth-wise centroid stats: mean `-0.123 mm`, min `-1.100 mm`, max `+1.124 mm`
- Interpretation:
  - Strong prior one-sided drift (tens of mm toward `-x`) is removed.

### Current verbose energy channels (still failing closure)

- From `validation/latest_gpu_run_sigma38_verbose_fix_slots.log`:
  - `Source Energy (total): 150.002 MeV`
  - `Energy Deposited: 90.8394 MeV`
  - `Nuclear Energy Deposited: 16.0816 MeV`
  - `Boundary Loss Energy: 9.3226e-07 MeV`
  - `Transport Drop Energy: 45.9643 MeV`
  - `Total Accounted Energy (transport only): 152.885 MeV`
  - `Transport complete after 211 iterations`

### Regression status

- `build/tests/sm2d_tests`: still `88` passed, `1` failed.
- Failing test unchanged:
  - `EnergyLossOnlyTest.FullPhysics`
  - `sigma_100 = 0.112 mm` (expected `> 0.5 mm`).

---

## Reproduction Artifacts

### Pre-fix evidence

- `validation/latest_gpu_run_sigma38_600.log`
  - `Iteration 200: 0 active, 1071 coarse cells`
  - `Iteration 250: 0 active, 1393 coarse cells`
  - K3 never engaged.

- `validation/latest_gpu_run_sigma38_verbose.log`
  - Source injection accounting:
    - Injected in-grid: `0.395998` (39.5998%)
    - Slot dropped: `0.603996` (60.3996%)
    - Injected in-grid energy: `59.4105 MeV`
    - Slot dropped energy: `90.5911 MeV`

### Post-fix evidence

- `validation/latest_gpu_run_sigma38_600_fixed.log`
  - `Iteration 200: 288 active, 0 coarse cells`
  - `Iteration 250: 32 active, 0 coarse cells`
  - `Transport complete after 262 iterations`

- `validation/latest_gpu_run_sigma38_verbose_fixed2.log`
  - Source injection accounting:
    - Injected in-grid: `0.999991` (99.9991%)
    - Slot dropped: `0` (0%)
    - Injected in-grid energy: `150.002 MeV` (100%)
    - Slot dropped energy: `0 MeV` (0%)

- Comparison (latest):
  - `validation/latest_comparison_sigma38_verbose_fixed2.log`
  - Lateral sigma:
    - 20 mm: SM_2D `4.246 mm`, MOQUI `5.520 mm`
    - 100 mm: SM_2D `4.246 mm`, MOQUI `5.520 mm`
    - 140 mm: SM_2D `4.246 mm`, MOQUI `6.369 mm`

---

## Findings (Historical Snapshot Before Follow-up Patch Set)

## 1) K2->K3 threshold activation bug (fixed)

### Root cause
- K1 fine activation used strict `<` block-threshold checks.
- Particles clamped near `E_fine_on` could remain coarse indefinitely.

### Fix applied
- Changed to inclusive `<=` threshold checks in:
  - `src/cuda/kernels/k1_activemask.cu`

### Result
- Active list now transitions to K3 in late transport.

---

## 2) Source injection slot race (fixed)

### Root cause
- `device_psic_find_or_allocate_slot(...)` had contention behavior that could fail or clobber under parallel insertion.
- Per-claim zeroing inside allocation path was unsafe with concurrent writers.

### Fix applied
- Made allocation contention-safe (`CAS old==EMPTY || old==bid`) and removed claim-time zeroing:
  - `src/cuda/device/device_psic.cuh`

### Result
- Slot-dropped source energy reduced from ~60% to 0%.

---

## 3) Lateral spreading model remains incorrect (open, critical)

### Observed behavior
- Lateral sigma is nearly constant with depth in current runs.
- This is inconsistent with expected MCS-driven broadening.

### Code-level causes

1. **Direction is effectively frozen**
- K2 and K3 keep `theta_new = theta` (no progressive angular diffusion):
  - `src/cuda/kernels/k2_coarsetransport.cu`
  - `src/cuda/kernels/k3_finetransport.cu`

2. **No true incremental Fermi-Eyges accumulation in transport kernels**
- Kernels use depth-based absolute sigma heuristics each step.
- Available Fermi-Eyges helpers are not integrated into K2/K3 state evolution:
  - `src/cuda/device/device_physics.cuh` (`device_fermi_eyges_step`, `device_hybrid_sigma_x`)

3. **Spread emission is mostly local**
- Current in-cell + immediate-tail pattern does not implement robust multi-cell Gaussian transport at large sigma:
  - K2/K3 spreading blocks in `src/cuda/kernels/k2_coarsetransport.cu` and `src/cuda/kernels/k3_finetransport.cu`
- A dedicated multi-cell emitter exists but is not the main path:
  - `src/cuda/device/device_bucket.cuh` (`device_emit_lateral_spread`)

---

## 4) Energy accounting still inconsistent (open, critical)

### Observed behavior
- Example from `validation/latest_gpu_run_sigma38_verbose_fixed2.log`:
  - Source in-grid energy: `150.002 MeV`
  - Energy deposited: `152.973 MeV`
  - Nuclear deposited: `17.4854 MeV`
  - Total accounted (transport only): `170.465 MeV`
- Accounted energy exceeding source indicates channel overlap/double counting.

### Likely direct cause
- Nuclear removed energy appears counted in both `edep` and nuclear channels:
  - K2:
    - `edep += E_rem`
    - `cell_E_nuclear += E_rem`
  - K3:
    - `edep += E_rem`
    - `cell_E_nuclear += E_rem`

---

## 5) Transition/audit implementation status (confirmed present)

### 5.1 K1 hysteresis (implemented)
- K1 uses both `b_E_fine_on` and `b_E_fine_off`, and references `ActiveMaskPrev`:
  - `src/cuda/kernels/k1_activemask.cu`
  - `src/cuda/k1k6_pipeline.cu`
- Runtime config exposes hysteresis parameters with validation:
  - `src/include/core/incident_particle_config.hpp`
  - `src/include/core/config_loader.hpp`

### 5.2 K2 crossing guard (implemented)
- K2 detects within-step crossing of `E_fine_on` and splits coarse transport at threshold:
  - `src/cuda/kernels/k2_coarsetransport.cu`

### 5.3 K5 energy audit (implemented)
- `K5_ConservationAudit` includes weight + energy terms and computes `W_error/W_pass`, `E_error/E_pass`:
  - `src/cuda/kernels/k5_audit.cuh`
  - `src/cuda/kernels/k5_audit.cu`
- Pipeline executes this audit each iteration:
  - `src/cuda/k1k6_pipeline.cu`

---

## Changes Applied In This Session

1. Inclusive K1 activation thresholds:
   - `src/cuda/kernels/k1_activemask.cu`

2. Removed legacy K2 2x sigma inflation heuristic:
   - `src/cuda/kernels/k2_coarsetransport.cu`

3. Fixed source slot allocation race:
   - `src/cuda/device/device_psic.cuh`

4. Validator robustness fixes (data parsing and reshape correctness):
   - `validation/compare_sm2d_moqui.py`

5. Documentation correction:
   - Prior cross-reference report claims that hysteresis/crossing-guard/K5 energy audit were missing are not valid for current code.

6. Follow-up transport patch attempt (still failing physics validation):
   - `src/cuda/device/device_physics.cuh`
   - `src/cuda/device/device_bucket.cuh`
   - `src/cuda/kernels/k2_coarsetransport.cu`
   - `src/cuda/kernels/k3_finetransport.cu`
   - `src/gpu/gpu_transport_runner.cpp`

7. Fixed one-sided lateral-transfer bias (`-x` bend) by centering spread window and using odd spread-cell count:
   - `src/cuda/device/device_physics.cuh`
   - `src/cuda/device/device_bucket.cuh`

8. Coupled same-step transport direction to scattered angle in both transport kernels:
   - `src/cuda/kernels/k2_coarsetransport.cu`
   - `src/cuda/kernels/k3_finetransport.cu`

9. Hardened bucket/output slot claims under contention (`CAS old==EMPTY || old==bid`):
   - `src/cuda/device/device_bucket.cuh`
   - `src/cuda/kernels/k2_coarsetransport.cu`
   - `src/cuda/kernels/k3_finetransport.cu`
   - `src/cuda/kernels/k4_transfer.cu`

---

## Required Next Fixes

1. **Reduce transport-drop saturation (highest priority).**
   - Current verbose run still loses ~`45.96 MeV` to transport drops.
   - Investigate K2 bucket overflow and K3/K4 slot saturation under high phase-space occupancy.
2. **Reconcile energy channels so transport totals close to source energy.**
   - Current total accounted energy remains above source (`152.885 MeV` vs `150.002 MeV`).
   - Revisit overlap between continuous `edep` and nuclear removal accounting.
3. **Recover physically correct depth-dose range (Bragg still shallow).**
   - Current Bragg remains at `111 mm` vs MOQUI `154 mm`.
4. **Improve mid-depth lateral spread without reintroducing directional bias.**
   - Keep centroid near zero while raising `sigma_100` toward expected values.
   - `EnergyLossOnlyTest.FullPhysics` still fails (`sigma_100 = 0.112 mm`).
5. Keep K5/transition behaviors under regression protection (implemented, must not regress):
   - hysteresis behavior with `E_fine_on/E_fine_off`,
   - coarse-step crossing guard at `E_fine_on`,
   - K5 energy pass/fail semantics (`E_error`, `E_pass`).
6. Add targeted regressions:
   - depth-dependent sigma growth check (`sigma_20 < sigma_100 < sigma_140` for 150 MeV),
   - lateral-centroid symmetry check (`|mu_x(z)|` bounded for centered source),
   - energy conservation check (`|accounted-source|/source < tolerance`),
   - K2->K3 handoff regression near `E_fine_on`,
   - full-physics mid-depth lateral spread floor (existing GPU test should pass again).

---

## Notes On Validation Setup

- Full default `sim.ini` does not fit 8 GB RTX 2080 in this build due memory footprint.
- Comparison reruns used 8 GB-fit config (`validation/gpu_compare*.ini`) to produce fresh `results/*` on GPU.
