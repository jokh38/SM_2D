# Remaining Code Issues (Session Verification)

Date: 2026-02-07

This note captures what is still unresolved in the codebase based on direct checks performed in this session.

## Current Baseline (Verified)

- Test status in this workspace: `93/93` passed (`build/tests/sm2d_tests --gtest_brief=1`).
- Runtime preflight + fine-batch planning are active in runner path.
- K5 audit fail-fast is active when `validation_mode=true` (or `fail_fast_on_audit=true`).
- P1 progress: outflow buckets are now batch-local scratch (`d_BucketScratch`) with per-batch `CellToBucketBase` mapping; persistent full-grid bucket allocation was removed from pipeline state.

Evidence:
- `src/gpu/gpu_transport_runner.cpp:111`
- `src/gpu/gpu_transport_runner.cpp:137`
- `src/cuda/gpu_transport_wrapper.cu:236`
- `src/cuda/k1k6_pipeline.cu:808`
- `src/cuda/k1k6_pipeline.cu:1461`
- `src/cuda/k1k6_pipeline.cu:1539`

## Remaining Issues

## 1) P1 Architecture Shift Still Incomplete (Critical)

- Runtime still allocates full-grid `psi_in` and `psi_out` device phase-space buffers.
- SPEC target architecture (`coarse persistent + fine scratch`) requires replacing dense full-grid phase-space persistence, not only bucket storage.

Evidence:
- `src/cuda/gpu_transport_wrapper.cu:252`
- `src/cuda/gpu_transport_wrapper.cu:261`
- `src/cuda/gpu_transport_wrapper.cu:266`
- `src/cuda/device/device_psic.cuh:77`

## 2) Preflight Estimator Still Assumes Dense Full-Grid Buckets (High)

- Memory preflight still uses dense `N_cells * 4` outflow bucket estimation and dense bytes-per-cell planning.
- With batch-local bucket scratch now in runtime pipeline, preflight is conservative/misaligned and should be updated to scratch-aware budgeting.

Evidence:
- `src/perf/memory_preflight.cpp:43`
- `src/perf/memory_preflight.cpp:57`
- `src/perf/memory_preflight.cpp:63`
- `src/perf/memory_preflight.cpp:125`

## 3) No Explicit Prolongation/Restriction Operator Path (Medium)

- Current K4 path still transfers bucket contents directly into `psi_out`.
- No first-class `coarse -> fine -> coarse` prolong/restrict operator implementation and no dedicated conservation gate test for that path.

Evidence:
- `src/cuda/kernels/k4_transfer.cu:42`
- `src/cuda/kernels/k4_transfer.cu:108`
- `tests/gpu` (no dedicated prolong/restrict conservation test)

## 4) High K2/K3 Duplication (Maintainability Risk)

- K2 and K3 still contain large duplicated blocks for spreading, rebinning, bucket emission, and tail handling.
- This keeps regression risk high for future physics fixes.

Evidence:
- `src/cuda/kernels/k2_coarsetransport.cu:360`
- `src/cuda/kernels/k3_finetransport.cu:383`
- `src/cuda/kernels/k2_coarsetransport.cu:501`
- `src/cuda/kernels/k3_finetransport.cu:531`

## Suggested Next-Session Order

1. Finish P1 architecture shift: replace dense full-grid `psi_in/psi_out` persistence with coarse-persistent + fine-scratch phase-space path.
2. Implement explicit prolongation/restriction operator path and add its conservation test.
3. Update preflight memory estimator/planner to scratch-aware accounting.
4. Extract shared K2/K3 post-step helper to remove duplicated logic.
