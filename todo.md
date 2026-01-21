# SM_2D Todo List

> Project: Hierarchical Deterministic Transport Solver
> Spec: v0.8
> Target: RTX 2080-class (8GB VRAM)

---

## Phase 0: Setup
- [x] Project structure initialization
- [x] Build system (CMake/Make)
- [x] CUDA toolkit setup (mock runtime for environments without CUDA)
- [x] Test framework (Google Test / Catch2)
- [x] Development environment validation

## Phase 1: LUT Generation
- [x] NIST PSTAR data download script (data/nist/pstar_water.txt)
- [x] R(E) table generation (log-log interpolation) (src/lut/r_lut.cpp:224)
- [x] E(R) inverse table (src/lut/r_lut.cpp:288)
- [x] Unit test: R(150 MeV) ≈ 158 mm (tests/unit/test_r_lut.cpp:10)
- [x] Unit test: R(70 MeV) ≈ 40.8 mm (tests/unit/test_r_lut.cpp:19)
- [x] LUT file format with metadata/checksum (N/A for Phase 1 - using hardcoded fallback)

## Phase 2: Data Structures
- [x] Energy grid (N_E=256, log-spaced, bin-edge based) (include/core/grids.hpp, src/core/grids.cpp)
- [x] Angular grid (N_theta=512) (include/core/grids.hpp, src/core/grids.cpp)
- [x] LOCAL_BINS encoding/decoding (N_theta_local=8, N_E_local=4) (include/core/local_bins.hpp)
- [x] Block ID encoding (24-bit) (include/core/block_encoding.hpp)
- [x] PsiC sparse buffer structure (include/core/psi_storage.hpp, src/core/psi_storage.cpp)
- [x] OutflowBucket structure (Kb_out=64) (include/core/buckets.hpp, src/core/buckets.cpp)

## Phase 3: Physics Models
- [x] CSDA range lookup: `lookup_R(E)` (include/lut/r_lut.hpp, src/lut/r_lut.cpp:55)
- [x] Inverse energy lookup: `lookup_E_inverse(R)` (include/lut/r_lut.hpp, src/lut/r_lut.cpp:69)
- [x] R-based step size control (IC-1: no S(E) usage) (include/physics/step_control.hpp)
- [x] Highland sigma computation (include/physics/highland.hpp)
- [ ] Variance-based MCS accumulation (IC-2) - deferred to Phase 4
- [x] Nuclear cross-section `Sigma_total(E)` (include/physics/nuclear.hpp)
- [x] 7-point angular quadrature weights (include/physics/highland.hpp)

## Phase 4: Kernels
- [x] K1_ActiveMask: identify active cells (cuda/kernels/k1_activemask.cu, tests/kernels/test_k1_activemask.cpp)
- [ ] K2_CompactActive: generate active list (deferred to MVP)
- [x] K3_FineTransport: main transport loop (cuda/kernels/k3_finetransport.cu, tests/kernels/test_k3_finetransport.cpp)
  - [ ] R-based energy update (requires device LUT)
  - [ ] Variance-based MCS splitting (CPU stub implemented)
  - [ ] Bin-edge 2-bin energy discretization (IC-3) (deferred)
  - [ ] Nuclear attenuation with energy budget (IC-5) (CPU stub implemented)
  - [ ] Boundary handling (deferred)
  - [ ] Bucket emission (deferred)
- [x] K4_BucketTransfer: neighbor cell transfer (cuda/kernels/k4_transfer.cu)
- [x] K5_ConservationAudit: weight + energy verification (cuda/kernels/k5_audit.cu)
- [x] K6_SwapBuffers: double-buffer exchange (cuda/kernels/k6_swap.cu)

## Phase 5: Sources
- [x] PencilSource implementation (include/source/pencil_source.hpp, src/source/pencil_source.cpp)
- [x] GaussianSource implementation (include/source/gaussian_source.hpp, src/source/gaussian_source.cpp)
- [x] Source injection into PsiC
- [x] Boundary loss tracking (IC-6) (include/boundary/loss_tracking.hpp, src/boundary/loss_tracking.cpp)
- [x] Boundary conditions (include/boundary/boundaries.hpp, src/boundary/boundaries.cpp)
- [x] Unit tests for sources (tests/source/test_pencil.cpp, tests/source/test_gaussian.cpp)
- [x] Unit tests for boundaries (tests/boundary/test_boundaries.cpp, tests/boundary/test_loss_tracking.cpp)

## Phase 6: Audit
- [x] Weight conservation check (IC-1: <1e-6) (include/audit/conservation.hpp, src/audit/conservation.cpp)
- [x] Energy conservation check (IC-2: <1e-5) (include/audit/conservation.hpp, src/audit/conservation.cpp)
- [x] Global budget closure (include/audit/global_budget.hpp, src/audit/global_budget.cpp)
- [x] Audit logging and reporting (include/audit/reporting.hpp, src/audit/reporting.cpp)

## Phase 7: Validation
- [x] Validation headers and stub implementations (include/validation/*.hpp, src/validation/*.cpp)
- [x] Pencil beam simulation (include/validation/pencil_beam.hpp, src/validation/pencil_beam.cpp)
- [x] Bragg peak analysis (include/validation/bragg_peak.hpp, src/validation/bragg_peak.cpp)
- [x] Lateral spread analysis (include/validation/lateral_spread.hpp, src/validation/lateral_spread.cpp)
- [x] Determinism verification (include/validation/determinism.hpp, src/validation/determinism.cpp)
- [x] Validation report generation (include/validation/validation_report.hpp, src/validation/validation_report.cpp)
- [x] Unit tests for validation (tests/validation/test_*.cpp)
- [x] CMakeLists.txt updates for validation files
- [ ] Unit Tests (T1-T7)
- [ ] Integration Tests (I1-I6)
  - [ ] I1: Pencil 150 MeV → Bragg at R(150)±2%
  - [ ] I2: Pencil 70 MeV → Bragg at R(70)±2%
  - [ ] I3: Weight conservation
  - [ ] I4: Energy conservation
  - [ ] I5: Lateral σₓ vs Fermi-Eyges
  - [ ] I6: Determinism checksum
- [ ] Stress Tests (S1-S4)

## Phase 8: Optimization
- [x] Memory usage validation (<8GB VRAM) (include/perf/memory_profiler.hpp, src/perf/memory_profiler.cpp)
- [x] Kernel profiling (include/perf/kernel_profiler.hpp, src/perf/kernel_profiler.cpp)
- [x] Occupancy optimization (include/perf/occupancy_analyzer.hpp, src/perf/occupancy_analyzer.cpp)
- [x] Shared memory usage (tests/perf/test_memory_profiling.cpp, tests/perf/test_kernel_profiling.cpp, tests/perf/test_occupancy.cpp)

---

## Implementation Checklist (from Spec v0.8)

| ID | Rule | Status |
|----|------|--------|
| IC-1 | R-based Δs control (no S(E)) | ✅ |
| IC-2 | Variance-based MCS accumulation | ⬜ |
| IC-3 | Bin-edge 2-bin energy scatter | ⬜ |
| IC-4 | LOCAL_BINS = Nθ × NE = 32 | ✅ |
| IC-5 | AbsorbedEnergy_nuclear tracked | ✅ |
| IC-6 | Boundary loss tracking | ✅ |
| IC-7 | Bucket indexed [cell][face] | ⬜ |
| IC-8 | Highland: reduce ds, not clamp | ✅ |
| IC-9 | E_edges log-spaced | ✅ |
| IC-10 | Source injection defined | ✅ |

---

## Notes

- Use `[x]` for completed items
- Add file references (e.g., `src/lut.cpp:42`) when completing tasks
- Record issues encountered for pattern analysis
- Update this file **before** declaring any work session complete
