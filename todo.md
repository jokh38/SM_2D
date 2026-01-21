# SM_2D Todo List

> Project: Hierarchical Deterministic Transport Solver
> Spec: v0.8
> Target: RTX 2080-class (8GB VRAM)

---

## Phase 0: Setup
- [ ] Project structure initialization
- [ ] Build system (CMake/Make)
- [ ] CUDA toolkit setup
- [ ] Test framework (Google Test / Catch2)
- [ ] Development environment validation

## Phase 1: LUT Generation
- [ ] NIST PSTAR data download script
- [ ] R(E) table generation (log-log interpolation)
- [ ] E(R) inverse table
- [ ] Unit test: R(150 MeV) ≈ 158 mm
- [ ] Unit test: R(70 MeV) ≈ 40.8 mm
- [ ] LUT file format with metadata/checksum

## Phase 2: Data Structures
- [ ] Energy grid (N_E=256, log-spaced, bin-edge based)
- [ ] Angular grid (N_theta=512)
- [ ] LOCAL_BINS encoding/decoding (N_theta_local=8, N_E_local=4)
- [ ] Block ID encoding (24-bit)
- [ ] PsiC sparse buffer structure
- [ ] OutflowBucket structure (Kb_out=64)

## Phase 3: Physics Models
- [ ] CSDA range lookup: `lookup_R(E)`
- [ ] Inverse energy lookup: `lookup_E_inverse(R)`
- [ ] R-based step size control (IC-1: no S(E) usage)
- [ ] Highland sigma computation
- [ ] Variance-based MCS accumulation (IC-2)
- [ ] Nuclear cross-section `Sigma_total(E)`
- [ ] 7-point angular quadrature weights

## Phase 4: Kernels
- [ ] K1_ActiveMask: identify active cells
- [ ] K2_CompactActive: generate active list
- [ ] K3_FineTransport: main transport loop
  - [ ] R-based energy update
  - [ ] Variance-based MCS splitting
  - [ ] Bin-edge 2-bin energy discretization (IC-3)
  - [ ] Nuclear attenuation with energy budget (IC-5)
  - [ ] Boundary handling
  - [ ] Bucket emission
- [ ] K4_BucketTransfer: neighbor cell transfer
- [ ] K5_ConservationAudit: weight + energy verification
- [ ] K6_SwapBuffers: double-buffer exchange

## Phase 5: Sources
- [ ] PencilSource implementation
- [ ] GaussianSource implementation
- [ ] Source injection into PsiC
- [ ] Boundary loss tracking (IC-6)

## Phase 6: Audit
- [ ] Weight conservation check (IC-1: <1e-6)
- [ ] Energy conservation check (IC-2: <1e-5)
- [ ] Global budget closure
- [ ] Overflow detection and logging

## Phase 7: Validation
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
- [ ] Memory usage validation (<8GB VRAM)
- [ ] Kernel profiling
- [ ] Occupancy optimization
- [ ] Shared memory usage

---

## Implementation Checklist (from Spec v0.8)

| ID | Rule | Status |
|----|------|--------|
| IC-1 | R-based Δs control (no S(E)) | ⬜ |
| IC-2 | Variance-based MCS accumulation | ⬜ |
| IC-3 | Bin-edge 2-bin energy scatter | ⬜ |
| IC-4 | LOCAL_BINS = Nθ × NE = 32 | ⬜ |
| IC-5 | AbsorbedEnergy_nuclear tracked | ⬜ |
| IC-6 | Boundary loss tracking | ⬜ |
| IC-7 | Bucket indexed [cell][face] | ⬜ |
| IC-8 | Highland: reduce ds, not clamp | ⬜ |
| IC-9 | E_edges log-spaced | ⬜ |
| IC-10 | Source injection defined | ⬜ |

---

## Notes

- Use `[x]` for completed items
- Add file references (e.g., `src/lut.cpp:42`) when completing tasks
- Record issues encountered for pattern analysis
- Update this file **before** declaring any work session complete
