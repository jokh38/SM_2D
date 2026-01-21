# Hierarchical Deterministic Transport Solver - Development Plan

**Version**: 1.0
**Based on**: SPEC.md v0.8
**Target Hardware**: RTX 2080 (8GB VRAM)
**Development Methodology**: Test-Driven Development (TDD)

---

## Overview

This plan implements a deterministic proton transport solver for 2D water phantom using CUDA/C++. The solver implements CSDA energy loss, Multiple Coulomb Scattering (Highland formula), and nuclear attenuation with conservation enforcement.

### Development Principles

1. **TDD Workflow**: Write tests BEFORE implementation for every component
2. **Physics-First**: Validate each physics model against NIST PSTAR data
3. **Incremental Integration**: Each phase produces testable, verifiable artifacts
4. **Conservation Enforcement**: Weight and energy conservation tested at every step
5. **Memory Awareness**: All allocations mindful of 8GB VRAM constraint

---

## Phase Structure

| Phase | File | Duration | Key Deliverables |
|-------|------|----------|------------------|
| **0** | [phase_0_setup.md](docs/phases/phase_0_setup.md) | Foundation | Build system, test framework, CUDA environment |
| **1** | [phase_1_lut.md](docs/phases/phase_1_lut.md) | 1-2 days | NIST LUT generation, R(E) validation |
| **2** | [phase_2_data_structures.md](docs/phases/phase_2_data_structures.md) | 2-3 days | Grids, sparse storage, buckets, encoding |
| **3** | [phase_3_physics.md](docs/phases/phase_3_physics.md) | 3-4 days | R-based steps, Highland, nuclear, 2-bin |
| **4** | [phase_4_kernels.md](docs/phases/phase_4_kernels.md) | 4-5 days | K1-K6 pipeline implementation |
| **5** | [phase_5_sources.md](docs/phases/phase_5_sources.md) | 2-3 days | Pencil/Gaussian sources, boundary conditions |
| **6** | [phase_6_audit.md](docs/phases/phase_6_audit.md) | 2 days | Conservation audit implementation |
| **7** | [phase_7_validation.md](docs/phases/phase_7_validation.md) | 3-4 days | Physics validation vs NIST |
| **8** | [phase_8_optimization.md](docs/phases/phase_8_optimization.md) | 2-3 days | Profiling and performance tuning |

**Total Estimated Duration**: 19-27 days (3-4 weeks)

---

## Phase Dependencies

```
Phase 0 (Setup)
    |
    v
Phase 1 (LUT) ---------> Phase 2 (Data Structures)
    |                         |
    v                         v
Phase 3 (Physics) <---------+
    |
    v
Phase 4 (Kernels) ---------> Phase 5 (Sources/Boundaries)
    |                         |
    v                         v
Phase 6 (Audit) <-----------+
    |
    v
Phase 7 (Validation) ------> Phase 8 (Optimization)
```

- **Phase 0 must complete** before any other phase
- **Phases 1-2** can develop in parallel after Phase 0
- **Phase 3** depends on Phase 1 (LUT) and Phase 2 (data structures)
- **Phase 4** depends on all prior phases
- **Phases 5-6** integrate with Phase 4
- **Phase 7** validates the complete system
- **Phase 8** runs in parallel with validation

---

## TDD Workflow Checklist

For each phase, follow this workflow:

1. **RED**: Write failing test(s)
   ```cpp
   TEST_F(FeatureTest, NotYetImplemented) {
       EXPECT_TRUE(false);  // Fails
   }
   ```

2. **GREEN**: Make test pass (minimal implementation)
   ```cpp
   bool feature() {
       return true;  // Passes
   }
   ```

3. **REFACTOR**: Improve implementation while keeping tests green
   ```cpp
   bool feature() {
       // Proper implementation
       return compute_correct_result();
   }
   ```

4. **DOCUMENT**: Add comments/docs if needed

5. **REPEAT** for next test/feature

---

## Exit Criteria Summary

| Phase | Critical Exit Criteria |
|-------|------------------------|
| 0 | All targets compile, CUDA works, test suite runs |
| 1 | R(150MeV) within ±1.3% of NIST (158mm) |
| 2 | Encoding round-trips, weight conserved in buckets |
| 3 | All physics unit tests pass |
| 4 | Full pipeline executes, K5 detects violations |
| 5 | Source injection correct, boundaries tracked |
| 6 | Cell weight error < 1e-6, energy error < 1e-5 |
| 7 | Bragg peak ±2%, lateral σ ±15% |
| 8 | Peak memory < 7GB, kernel times documented |

---

## Quick Reference

### Physical Constants
```cpp
const float E_min = 0.1f;       // MeV (cutoff)
const float E_max = 250.0f;     // MeV
const float E_cutoff = 0.1f;    // MeV
const float E_trigger = 10.0f;  // MeV
const int N_E = 256;
const int N_theta = 512;
const int LOCAL_BINS = 32;      // 8 × 4
const float X0_water = 360.8f;  // mm (radiation length)
const float m_p = 938.272f;     // MeV/c² (proton mass)
```

### NIST Reference Values (Water)
| Energy | CSDA Range |
|--------|------------|
| 150 MeV | ~158 mm |
| 70 MeV | ~40.8 mm |
| 10 MeV | ~1.2 mm |

### Conservation Tolerances
| Quantity | Tolerance |
|----------|-----------|
| Weight (cell) | < 1e-6 relative |
| Energy (cell) | < 1e-5 relative |
| Bragg peak position | ±2% of range |
| Lateral σ at mid-range | ±15% of Fermi-Eyges |

---

## Getting Started

To begin implementation:

```bash
# Start with Phase 0
cat docs/phases/phase_0_setup.md

# After completing Phase 0, proceed to Phases 1-2 in parallel
cat docs/phases/phase_1_lut.md
cat docs/phases/phase_2_data_structures.md
```

Each phase file contains:
- Objectives
- Tests to write first (RED phase)
- Implementation guidance (GREEN phase)
- Exit criteria checklist
