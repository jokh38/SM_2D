# Fix Lateral Profiles - Planning Document

> **Summary**: Fix jagged lateral profiles in deterministic proton therapy transport solver
>
> **Project**: SM_2D Proton Therapy Simulation
> **Version**: 1.0
> **Author**: Claude (Sisyphus Mode)
> **Date**: 2026-02-04
> **Status**: Draft

---

## 1. Overview

### 1.1 Purpose

Fix the jagged lateral profiles observed at all depths in the deterministic proton transport simulation. The simulation currently shows discrete spikes and discontinuities in lateral dose profiles instead of smooth Gaussian distributions.

### 1.2 Background

The SM_2D deterministic transport solver uses a K1-K6 kernel pipeline with phase-space bins (x, z, E, theta). Lateral profiles at 20mm, middle depth, and Bragg peak all show jagged, non-physical artifacts. This is NOT a Monte Carlo simulation - the jaggedness indicates numerical discretization problems.

**Key Observations:**
- Only 97 active cells out of 128,000 total cells
- High boundary loss: 29,126 MeV vs 137 MeV deposited
- Jagged profiles at ALL depths (surface, middle, Bragg peak)
- Raw dose data shows discrete jumps: 0.0094 → 0.0112 → 0.0093 Gy

### 1.3 Related Documents

- SPEC.md: Physics specification for proton transport
- docs/SPEC.md: MCS implementation requirements
- Recent commits: Fermi-Eyges moment tracking (mcs2-phase-b)

---

## 2. Scope

### 2.1 In Scope

- [ ] Fix coarse x_sub discretization (N_x_sub = 4 → 8 or 16)
- [ ] Implement accumulated lateral spreading (per-step → cumulative)
- [ ] Fix Fermi-Eyges moment persistence in K2
- [ ] Remove hard boundary clamping at cell edges
- [ ] Verify smooth lateral profiles after fixes

### 2.2 Out of Scope

- Complete rewrite of transport algorithm
- Changing from deterministic to Monte Carlo
- Performance optimization (unless related to fixes)
- Angular binning modifications

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | Increase x_sub resolution from 4 to at least 8 bins per cell | High | Pending |
| FR-02 | Implement cumulative sigma_x tracking from depth z=0 | High | Pending |
| FR-03 | Fix K2 Fermi-Eyges moments to persist across iterations | High | Pending |
| FR-04 | Remove hard clamping at x_sub boundaries | Medium | Pending |
| FR-05 | Verify lateral profiles are smooth at all depths | High | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| Correctness | Lateral profiles match Fermi-Eyges theory | Visual inspection + quantitative analysis |
| Performance | No more than 20% runtime increase | Benchmark before/after |
| Stability | All existing tests pass | make sm2d_tests |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] All functional requirements implemented
- [ ] Lateral profiles show smooth Gaussian-like distribution
- [ ] No discrete spikes at cell boundaries
- [ ] Simulation run verified with output visualization
- [ ] Code follows existing conventions

### 4.2 Quality Criteria

- [ ] Lateral profile variance matches theoretical σ²(z) ∝ z^(3/2)
- [ ] Bragg peak position unchanged (< 1mm shift)
- [ ] Peak dose within 5% of original value
- [ ] Zero regression in depth-dose curve

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Increased memory usage from more x_sub bins | Medium | High | Start with N_x_sub=8, monitor memory |
| Runtime performance degradation | Medium | Medium | Benchmark at each step, optimize if needed |
| Breaking existing physics | High | Low | Compare depth-dose curves before/after |
| Moment accumulation bugs | High | Medium | Add debug output for moment values |

---

## 6. Architecture Considerations

### 6.1 Project Level Selection

This is a **scientific computing project** using CUDA for GPU acceleration. The architecture is:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Transport Kernel | CUDA (K1-K6) | GPU phase-space evolution |
| CPU Wrapper | C++ | Host orchestration |
| Physics | Device functions | Energy loss, MCS, straggling |
| Visualization | Python (matplotlib) | Result plotting |

### 6.2 Key Architectural Decisions

| Decision | Current | Proposed | Rationale |
|----------|---------|----------|-----------|
| x_sub bins | 4 per cell | 8-16 per cell | Smoother lateral distribution |
| Lateral spread | Per-step sigma_x | Cumulative sigma_x | Correct Fermi-Eyges behavior |
| Moment persistence | Reset each iteration | Carry forward | Physical accuracy |

---

## 7. Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `src/include/core/local_bins.hpp` | 21 | N_x_sub: 4 → 8 or 16 |
| `src/cuda/kernels/k3_finetransport.cu` | 254-255 | Add cumulative sigma_x |
| `src/cuda/kernels/k3_finetransport.cu` | 400-428 | Increase N_x_spread, remove clamp |
| `src/cuda/kernels/k2_coarsetransport.cu` | 208-225 | Persist moments across iterations |
| `src/cuda/kernels/k2_coarsetransport.cu` | 400-423 | Match K3 changes |
| `src/cuda/device/device_physics.cuh` | 316-327 | Consider cumulative version |

---

## 8. Next Steps

1. [ ] Write design document (`fix-lateral-profiles.design.md`)
2. [ ] Review design with SPEC.md requirements
3. [ ] Start implementation (Do phase)

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-02-04 | Initial draft | Claude (Sisyphus) |
