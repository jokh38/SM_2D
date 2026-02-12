# Root Cause Analysis & Fix Plan

**Date**: 2026-02-07
**Scope**: All open items from `docs/remaining.md` and `docs/issues.md`
**Test Baseline**: 93/93 pass (verified current workspace state)

---

## Table of Contents

1. [P1 Architecture Shift Incomplete](#1-p1-architecture-shift-incomplete)
2. [Preflight Estimator Assumes Dense Full-Grid Buckets](#2-preflight-estimator-assumes-dense-full-grid-buckets)
3. [No Explicit Prolongation/Restriction Path](#3-no-explicit-prolongationrestriction-path)
4. [High K2/K3 Duplication](#4-high-k2k3-duplication)
5. [Mid-Depth Lateral Spread Below MOQUI Reference](#5-mid-depth-lateral-spread-below-moqui-reference)
6. [Remaining Regression Test Gaps](#6-remaining-regression-test-gaps)

---

## 1. P1 Architecture Shift Incomplete

**Source**: `remaining.md` line 24

### What the spec requires

The target architecture is **coarse-persistent + fine-scratch**: only the coarse-grid phase-space state persists between iterations, while fine-grid state is temporary scratch within a batch.

### What was already done

Outflow bucket storage was successfully moved to batch-local scratch (`d_BucketScratch` with per-batch `CellToBucketBase` mapping). This removed the dense full-grid bucket allocation from `K1K6PipelineState`.

Evidence of completion:
- `src/cuda/k1k6_pipeline.cu:808` -- `d_CellToBucketBase` allocated per-grid but rebuilt per-batch
- `src/cuda/k1k6_pipeline.cu:699` -- `prepare_bucket_scratch_for_batch()` allocates and clears batch-local buckets

### Root cause of persistence

**The `psi_in` and `psi_out` buffers are still allocated as dense full-grid arrays spanning all `Nx * Nz` cells.**

In `src/cuda/gpu_transport_wrapper.cu:252-266`:

```cpp
DevicePsiC psi_in, psi_out;
device_psic_init(psi_in, Nx, Nz);   // Allocates Nx*Nz*Kb slots
device_psic_init(psi_out, Nx, Nz);  // Allocates Nx*Nz*Kb slots
```

Each `DevicePsiC` allocates `N_cells * Kb * (sizeof(uint32_t) + LOCAL_BINS * sizeof(float))` bytes. For typical grid sizes this is hundreds of MB per buffer.

The pipeline iterates by swapping `psi_in` and `psi_out` pointers each iteration (`k1k6_pipeline.cu`), which means the *entire* grid phase-space is persistent across every iteration. This is the dense full-grid pattern the spec wants to eliminate.

### Why it persists

Replacing `psi_in/psi_out` with coarse-persistent + fine-scratch requires:

1. **A coarse-grid PsiC buffer** that holds only the coarse-resolution cells (fewer cells, or fewer slots per cell) across iterations.
2. **A fine-grid scratch PsiC buffer** allocated per-batch for the active fine cells in that batch only.
3. **Prolongation logic** (coarse to fine) at the start of each fine batch.
4. **Restriction logic** (fine to coarse) at the end of each fine batch.

None of these components exist yet. The K4 transfer kernel writes directly into the full `psi_out` array (see issue 3). The pipeline swap assumes both buffers cover the same grid. Changing this is a structural refactor of the iteration loop.

### Fix plan

1. Define a `CoarsePsiC` type holding only coarse cells (or a sparse subset).
2. Add a `FineScratchPsiC` type allocated to batch size, not full grid.
3. At batch start: prolong coarse to fine-scratch for active cells.
4. Run K2/K3/K4 on fine-scratch.
5. At batch end: restrict fine-scratch to coarse, depositing accumulated changes.
6. Remove the two full-grid `DevicePsiC` allocations from `gpu_transport_wrapper.cu`.

### Dependency

Blocked by issue 3 (prolongation/restriction operators).

---

## 2. Preflight Estimator Assumes Dense Full-Grid Buckets

**Source**: `remaining.md` line 35

### What the preflight does

`src/perf/memory_preflight.cpp` estimates GPU memory needed before the run starts. It computes:

- `psi_buffers_bytes = n_cells * psic_single_cell_bytes * 2`  (line ~63)
- `outflow_buckets_bytes = n_cells * kBucketFacesPerCell * bucket_face_bytes`  (line ~65)

Both use `n_cells = Nx * Nz` (the full grid).

### Root cause of persistence

**The estimator was written before the batch-local bucket migration and has not been updated.**

The runtime now allocates `d_BucketScratch` sized to the *batch* cell count, not the full grid. But the preflight still reports dense `n_cells * 4` bucket bytes. This makes the preflight:

- **Over-estimate bucket memory** (reports full-grid when runtime uses batch-local).
- **Over-estimate psi memory** (reports two full-grid PsiC buffers, which is currently correct but will become wrong after P1 completion).
- **Fine-batch planner uses stale per-cell costs** (`bytes_per_dense_cell` includes full-grid bucket bytes), so batch sizing may be suboptimal.

Evidence in `memory_preflight.cpp:57-65`:

```cpp
est.outflow_buckets_bytes = n_cells * kBucketFacesPerCell * bucket_face_bytes;
```

This does not account for `prepare_bucket_scratch_for_batch()` which allocates only `batch_cells * 4 * bucket_face_bytes`.

### Fix plan

1. Split `estimate_dense_k1k6_memory` into:
   - `estimate_persistent_memory(Nx, Nz, N_theta, N_E)` -- coarse PsiC + pipeline scalars + LUT
   - `estimate_batch_scratch_memory(batch_cells, ...)` -- fine-scratch PsiC + batch-local buckets
2. Update `run_memory_preflight` to compute `max_batch_cells` from:
   `(usable_bytes - persistent_bytes) / scratch_bytes_per_cell`
3. Remove `outflow_buckets_bytes` from the persistent estimate (it is now scratch).
4. Update `bytes_per_dense_cell` to reflect the actual per-batch-cell cost.

### Dependency

Should be updated after P1 architecture shift so the persistent/scratch split matches the actual runtime allocation.

---

## 3. No Explicit Prolongation/Restriction Path

**Source**: `remaining.md` line 46

### What should exist

A two-level multigrid transport requires explicit operators:

- **Prolongation** (coarse to fine): interpolate coarse phase-space state onto the fine grid before a fine transport batch.
- **Restriction** (fine to coarse): aggregate fine-grid results back onto the coarse grid after a fine batch, conserving energy and weight.

These operators must satisfy a **conservation gate**: total weight and energy before prolongation must equal total weight and energy after restriction (within floating-point tolerance).

### What exists instead

K4 (`src/cuda/kernels/k4_transfer.cu:42-108`) is a **bucket-to-PsiC transfer kernel**, not a prolongation/restriction operator. It:

1. Iterates over all cells.
2. For each cell, reads inflow buckets from the 4 neighbor faces.
3. Directly writes bucket contents into `values_out` / `block_ids_out` (the output PsiC buffer).

This is a flat single-level transfer: outflow buckets emitted by K2/K3 are moved into the next-iteration PsiC. There is no concept of coarse vs. fine resolution levels in K4.

### Root cause

The original codebase was designed as a single-level transport (K2 for coarse step, K3 for fine step, but operating on the *same* grid). The K2/K3 distinction is about step-size control and energy thresholds, not spatial resolution. Prolongation and restriction are new requirements from the P1 spec that were never implemented.

### Fix plan

1. **Define grid-level metadata**: which cells are "coarse" and which are "fine" (the active mask from K1 already classifies cells; this can be reused).
2. **Implement `K_Prolong` kernel**:
   - Input: coarse PsiC buffer, active fine cell list
   - Output: fine-scratch PsiC buffer
   - Method: for each fine cell, find the overlapping coarse cell(s) and interpolate/copy phase-space components
3. **Implement `K_Restrict` kernel**:
   - Input: fine-scratch PsiC buffer (post-transport), active fine cell list
   - Output: coarse PsiC buffer (updated)
   - Method: for each fine cell, aggregate its phase-space back into the parent coarse cell, summing weights
4. **Add conservation gate test**:
   - Run prolong then identity transport (no physics) then restrict
   - Assert `|W_out - W_in| / W_in < epsilon` and `|E_out - E_in| / E_in < epsilon`
5. **Integrate into pipeline loop** between K1 (mask) and K2/K3 (transport).

### Dependency

This is a prerequisite for P1 completion (issue 1).

---

## 4. High K2/K3 Duplication

**Source**: `remaining.md` line 56

### What is duplicated

K2 (`src/cuda/kernels/k2_coarsetransport.cu`, 871 lines) and K3 (`src/cuda/kernels/k3_finetransport.cu`, 830 lines) contain near-identical blocks for:

| Block | K2 lines | K3 lines | Description |
|-------|----------|----------|-------------|
| Fermi-Eyges reconstruction | 267-282 | 310-325 | Reconstruct A/B/C moments from depth |
| Sigma computation | 282-302 | 325-345 | Compute effective sigma_x from C |
| Forward lateral spread emission | 373-408 | 393-430 | `device_emit_lateral_spread` call |
| Non-forward bucket emission | 410-450 | 440-480 | `device_emit_component_to_bucket_4d` call |
| In-cell Gaussian spreading | 465-590 | 492-618 | Sub-cell weight distribution |
| Slot allocation with fallback | 498-510 | 528-540 | `device_psic_find_or_allocate_slot_with_fallback` |
| Drop channel accounting | various | various | `atomicAdd` to `g_k{2,3}_*_drop_*` |

Diff analysis shows the core transport/spreading logic is structurally identical between the two kernels. The meaningful differences are:

1. **Step size computation**: K2 uses `coarse_range_step` with a limiter; K3 uses `actual_range_step` (CSDA step).
2. **Energy cutoff constant**: K2 uses literal `0.1f`; K3 uses `ENERGY_CUTOFF_MEV`.
3. **E-rebinning closure**: K3 computes `E_emit_transport` and deposits the `E_new - E_emit` mismatch locally; K2 does not (a divergence that may itself be a latent bug).
4. **Drop counter names**: `g_k2_*` vs `g_k3_*`.
5. **Moment validity check**: K2 has the `k2_moments_valid` profiling block; K3 does not.

### Root cause of persistence

Extraction of a shared helper was deferred because:

1. The kernels evolved rapidly during the physics fix pass, making refactoring risky mid-session.
2. Each kernel has slightly different control flow around step limiting and crossing guards that makes a naive extraction non-trivial.
3. There was no test-driven refactoring framework (the test suite covers physics behavior, not code structure).

### Regression risk

Any future physics fix (e.g., lateral spread improvement from issue 5) must be applied identically to both kernels. With the current duplication, a fix applied to K2 but not K3 (or vice versa) will silently introduce a physics inconsistency.

**Example of existing divergence**: K3 deposits the E-rebinning residual (`E_new - E_emit_transport`) back into `cell_edep` for transported weight, but K2 does not. This means energy accounting may differ between coarse and fine transport paths -- a potential source of the ~0.08% closure error seen in the audit residual channel.

### Fix plan

1. **Extract `device_transport_post_step` helper** (device function in a shared `.cuh` header):
   - Inputs: position, direction, energy, weight, sigma_x, grid params, bucket/PsiC pointers, drop counters
   - Handles: exit-face detection, cutoff absorption, lateral spread emission, bucket emission, in-cell spreading, slot allocation
   - Parameterized by: step type enum (coarse/fine), drop counter references
2. **Unify the E-rebinning closure** so both K2 and K3 deposit the `E_new - E_emit` mismatch consistently.
3. **Unify the cutoff constant** to use `ENERGY_CUTOFF_MEV` in both paths.
4. **Reduce K2/K3 to**: step-size computation + physics (Bethe-Bloch, straggling, nuclear) + call to shared post-step helper.
5. **Verify** with full test suite after extraction.

### Dependency

Should be done *before* the lateral spread fix (issue 5) to avoid applying that fix in two places.

---

## 5. Mid-Depth Lateral Spread Below MOQUI Reference

**Source**: `issues.md` line 401, 417

### Current state

`EnergyLossOnlyTest.FullPhysics` **passes** (93/93). Current `sigma_100 = 1.29 mm`, which clears the test threshold of 0.5 mm. However, the MOQUI reference at 100 mm depth is ~5.5 mm, so the lateral spread is still ~4x too narrow compared to the physics benchmark. The test passes, but the physics gap remains.

### Root cause analysis

The sigma_x computation in both K2 and K3 follows this pattern (K2 shown, K3 identical):

```cpp
// Reconstruct Fermi-Eyges moments from scratch at current depth
float path_start_mm = iz * dz + (z_cell + 0.5*dz);
float sigma_theta_start = device_highland_sigma(E, path_start_mm);
float A_old = sigma_theta_start * sigma_theta_start;
float C_old = sigma_x_initial*sigma_x_initial + (A_old * path_start_mm*path_start_mm) / 3.0;

// Evolve one step
device_fermi_eyges_step(A_new, B_new, C_new, T_step, step_mm);

// Use only the SINGLE-STEP INCREMENT plus small fraction of accumulated
float delta_C = C_new - C_old;
float C_effective = delta_C + 0.04 * C_old;
float sigma_x = sqrt(C_effective);
```

**Three compounding problems limit the spread:**

#### Problem A: Single-step increment, not accumulated spread

`delta_C = C_new - C_old` is the C-moment increment from a *single transport step* (typically 1-2 mm). For a 2 mm step at 100 mm depth in water with 150 MeV protons:

- `T_step ~ 0.001 rad^2/mm` (scattering power)
- `delta_C ~ T * ds^3 / 3 ~ 0.001 * 8 / 3 ~ 0.003 mm^2`
- `sigma_x_step = sqrt(0.003) ~ 0.05 mm`

This is the spread from *one step*, not the accumulated spread from the entire path. The accumulated C at 100 mm depth should be ~25-36 mm^2 (giving sigma_x ~5-6 mm).

The `UNRESOLVED_C_FRACTION = 0.04` term adds back 4% of `C_old`, which for `C_old ~ 30 mm^2` gives `0.04 * 30 = 1.2 mm^2`, yielding `sigma_x ~ sqrt(1.2) ~ 1.1 mm`. This partial correction is what brings the current result to ~1.29 mm -- enough to pass the test threshold but far short of the physics reference.

#### Problem B: No persistent moment accumulation between iterations

Each kernel invocation reconstructs `A_old`, `B_old`, `C_old` from scratch using the Highland formula at the component's current depth. This is an *analytic approximation* of the accumulated moments, not actual tracking. Between iterations:

- Component moments are not stored in the phase-space representation. PsiC stores weight, not moments.
- The `device_hybrid_sigma_x` function and `d_C_array` infrastructure in `device_physics.cuh` exist but are **never called** from K2 or K3 (confirmed by grep -- zero references in kernel files).
- After K4 transfer, the receiving cell has no knowledge of the accumulated scattering history of the transferred component.

This means every component at every step sees only a locally-reconstructed moment estimate, not a true path-integrated value.

#### Problem C: Spreading is applied per-step but does not compound across steps

Even if `sigma_x` were computed correctly, the spreading implementation distributes weight within the current cell or to immediate neighbors. This is correct for a single step. But the problem is that subsequent steps start from the *redistributed* sub-cell positions -- except the sub-cell position is coarsely quantized (8 bins per cell), so spatial information is lost. The Gaussian tails that escaped to neighbors are re-centered in their new cell. Over many steps, this quantization prevents the geometric growth of the beam envelope.

### Fix plan

The fundamental fix requires choosing between two approaches:

#### Approach A: Accumulated C-moment storage (recommended)

1. Add a per-cell, per-energy-block accumulated C-moment array to the pipeline state.
2. On each transport step, update `C[cell][E_bin] += C_add` where `C_add = T * ds^3/3`.
3. When computing sigma_x for spreading, use `sqrt(C_accumulated[cell][E_bin])` instead of `sqrt(delta_C + 0.04*C_old)`.
4. In K4 transfer, propagate C from source cell to destination cell (weighted average).
5. The `device_hybrid_sigma_x` and `device_update_moment_C` functions already exist in `device_physics.cuh` -- wire them into K2/K3.

Memory cost: `Nx * Nz * N_E_blocks * sizeof(float)` -- small relative to PsiC buffers.

#### Approach B: Use full accumulated C from reconstruction

1. Change `C_effective` from `delta_C + 0.04 * C_old` to simply `C_new` (the fully accumulated C at the end of the step).
2. This uses the analytically reconstructed accumulated spread instead of just the increment.
3. Simpler but less accurate than true moment tracking because the Highland reconstruction is approximate.

Either approach should raise `sigma_100` from ~1.3 mm toward the expected ~5 mm range.

### Dependency

Should be preceded by K2/K3 deduplication (issue 4) to apply the fix once.

---

## 6. Remaining Regression Test Gaps

**Source**: `issues.md` line 424

### What already exists

Several of the regression tests identified in `issues.md` are already implemented and passing:

| Test | Location | Status |
|------|----------|--------|
| Energy closure | `tests/gpu/test_energy_loss_only_gpu.cu:685` (`rel_energy_error < 0.08`) | Passing |
| Sigma growth shallow-to-mid | `tests/gpu/test_energy_loss_only_gpu.cu:687` (`sigma_20 < sigma_100`) | Passing |
| Mid-depth spread floor | `tests/gpu/test_energy_loss_only_gpu.cu:688` (`sigma_100 > 0.5`) | Passing |
| K5 drop channel audit | `tests/gpu/test_k5_drop_channels_gpu.cu:120` (W/E pass/fail, source/transport drops) | Passing |

### What is still missing

The following tests from the `issues.md` list do not yet have dedicated coverage:

| Test | Assertion | Protects Against |
|------|-----------|------------------|
| PDD continuity | No mid-depth collapse + distal re-rise in 150 MeV case | Transport loss causing PDD discontinuity |
| Sigma growth mid-to-deep | `sigma_100 < sigma_140` for 150 MeV | Deep-depth narrowing regression |
| Centroid symmetry | `abs(mu_x(z))` bounded for centered source | Directional bias regression |
| K2 to K3 handoff | Active list transitions near `E_fine_on` | Threshold activation regression |
| Runtime drop closure | K2/K3/K4 slot+bucket drops == 0 under standard config | Slot saturation in full pipeline (distinct from K5 unit audit) |

Note: The existing `FullPhysics` test has a conditional `sigma_140 >= sigma_100 * 0.7` check (`test_energy_loss_only_gpu.cu:691`), but it only fires when `sigma_140 > 0` and uses a relaxed 0.7x factor rather than asserting strict growth. This does not fully cover the mid-to-deep growth requirement.

### Fix plan

1. **Add new test cases** to `EnergyLossOnlyTest`:
   - `PDDContinuity` -- assert monotonic dose increase up to Bragg peak
   - `CentroidSymmetry` -- assert `abs(mu_x) < threshold` at each depth
   - `K2K3Handoff` -- run with `E_fine_on` threshold, assert K3 activates
2. **Add a full-pipeline drop closure test** that checks runtime K2/K3/K4 drop counters (not the K5 audit unit test).
3. **Tighten the sigma growth assertion**: add a strict `sigma_100 < sigma_140` check that is not gated by the `sigma_140 > 0` conditional.

### Dependency

None of the missing tests depend on the lateral spread fix. They can be added now.

---

## Suggested Execution Order

```
Phase 1 -- Reduce risk, enable safe changes
  #4  Extract shared K2/K3 post-step helper (deduplication)
  #6  Add missing regression tests (PDD continuity, centroid symmetry, K2-K3 handoff)

Phase 2 -- Improve lateral spread physics
  #5  Implement accumulated C-moment tracking in K2/K3
      Tighten sigma growth assertions after spread improves

Phase 3 -- Complete the architecture shift
  #3  Implement prolongation/restriction kernels + conservation gate test
  #1  Replace dense psi_in/psi_out with coarse-persistent + fine-scratch
  #2  Update preflight estimator to scratch-aware accounting
```

Each phase builds on the previous: deduplication makes the spread fix safer, the spread fix validates the transport physics, and the architecture shift can then proceed on a solid foundation.

---

## Summary Table

| # | Issue | Severity | Root Cause | Status |
|---|-------|----------|------------|--------|
| 1 | Dense psi_in/psi_out | Critical | P1 refactor not started for phase-space buffers | Open |
| 2 | Stale preflight estimator | High | Written pre-batch-migration, never updated | Open |
| 3 | No prolong/restrict | Medium | Single-level transport by original design | Open |
| 4 | K2/K3 duplication | Medium | Rapid physics fixes deferred extraction | Open |
| 5 | sigma_100 below MOQUI ref (~1.3 mm vs ~5.5 mm) | Medium | Single-step delta_C + 4% C_old instead of accumulated C; moment tracking infra exists but unused | Open (test passes, physics gap remains) |
| 6 | Remaining regression gaps | Medium | 4 of 7 identified tests already exist; 3 truly missing (PDD continuity, centroid symmetry, K2-K3 handoff) | Open |
