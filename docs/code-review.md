# Code Review: SM_2D (Hierarchical Deterministic Transport Solver)

**Date:** 2026-02-06
**Review Type:** Comprehensive Codebase Analysis
**Reviewer:** Claude (cdreview)
**Project:** Monte Carlo-style deterministic particle transport solver for proton therapy

---

## Executive Summary

SM_2D is a sophisticated physics simulation code implementing a deterministic phase-space transport solver for proton therapy dose calculation. The codebase demonstrates high-quality architecture with clear separation of concerns, comprehensive testing, and well-documented physics implementations.

### Overall Assessment

| Metric | Score | Status |
|--------|-------|--------|
| **Quality Score** | 75/100 | Good |
| **Code Coverage** | 30 test files | Excellent |
| **Linting Issues** | 28 (26 auto-fixable) | Minor |
| **Type Safety** | N/A (C++) | N/A |
| **Security** | 0 critical, 0 warnings | Pass |

---

## 1. Architecture Overview

### 1.1 Project Type

**Domain:** Scientific computing / Medical physics simulation
**Primary Language:** C++17 with CUDA
**Architecture:** Hierarchical kernel pipeline with GPU acceleration

### 1.2 Directory Structure

```
SM_2D/
├── src/
│   ├── cuda/           # GPU kernels (K1-K6 pipeline)
│   ├── include/        # Header files
│   ├── core/           # Grid, storage, buckets
│   ├── lut/            # Lookup tables (NIST data)
│   ├── physics/        # Physics models
│   ├── boundary/       # Boundary handling
│   ├── audit/          # Conservation checking
│   ├── validation/     # Validation against reference
│   └── utils/          # Utilities
├── tests/              # 30+ test files (GTest)
├── validation/         # Python validation scripts
├── docs/               # Documentation
└── build/              # CMake build artifacts
```

### 1.3 Component Statistics

- **Total C++ files:** 66
- **CUDA kernels:** 9 (.cu files)
- **Test files:** 30+
- **Python scripts:** 5 (batch processing, visualization)

---

## 2. Core Architecture Analysis

### 2.1 Kernel Pipeline (K1-K6)

The implementation follows a 6-stage kernel pipeline:

| Kernel | File | Purpose | Complexity |
|--------|------|---------|------------|
| K1 | `k1_activemask.cu` | Active cell detection | Low |
| K2 | `k2_coarsetransport.cu` | Coarse energy transport | Medium |
| K3 | `k3_finetransport.cu` | Fine transport with scattering | High |
| K4 | `k4_transfer.cu` | Bucket transfer between cells | Medium |
| K5 | `k5_audit.cu` | Conservation verification | Low |
| K6 | `k6_swap.cu` | Double-buffer exchange | Low |

**Strengths:**
- Clear separation of transport regimes (coarse vs fine)
- Deterministic algorithm (not Monte Carlo) - unique approach
- Comprehensive physics modeling (energy loss, MCS, nuclear attenuation)

### 2.2 Physics Implementation

The code implements several physics models:

1. **Energy Loss:** CSDA range-energy relation via NIST PSTAR data
2. **Multiple Coulomb Scattering:** Highland formula with Fermi-Eyges moments
3. **Nuclear Attenuation:** Cross-section based attenuation
4. **Lateral Spreading:** Deterministic Gaussian weight distribution

**Key Files:**
- `src/include/physics/fermi_eyges.hpp` - Fermi-Eyges scattering theory
- `src/include/physics/highland.hpp` - Highland formula
- `src/include/physics/nuclear.hpp` - Nuclear interactions
- `src/lut/nist_loader.cpp` - NIST data loading

---

## 3. Component Analysis

### 3.1 High-Complexity Components

| Component | Complexity | Notes |
|-----------|------------|-------|
| `batch_run.py:ParameterSweep.run_all` | 13 | Batch orchestration |
| `src/validation/deterministic_beam.cpp:run_pencil_beam` | 14 | Validation runner |
| `validation/analyze_angular_divergence.py:sigma_evolution_model` | 11 | Analysis function |

**Assessment:** Complexity is well-controlled. No functions exceed cognitive complexity of 15 (hotspot threshold).

### 3.2 Component Quality by Module

#### Core Module (`src/core/`)
- **Purpose:** Grid management, phase-space storage, bucket operations
- **Status:** Clean, low complexity (avg 3.0)
- **Files:** `grids.cpp`, `psi_storage.cpp`, `buckets.cpp`

#### CUDA Kernels (`src/cuda/kernels/`)
- **Purpose:** GPU transport implementation
- **Status:** Well-documented, clear physics comments
- **Key Files:**
  - `k2_coarsetransport.cu` - 500+ lines, handles coarse transport
  - `k3_finetransport.cu` - 700+ lines, main physics kernel

#### Audit Module (`src/audit/`)
- **Purpose:** Conservation verification (weight, energy)
- **Status:** Simple, focused functions
- **Files:** `conservation.cpp`, `global_budget.cpp`, `reporting.cpp`

---

## 4. Code Quality Assessment

### 4.1 Linting Results

```
Total Issues: 28
Auto-fixable: 26
Type Coverage: Not applicable (C++)
Security: 0 critical, 0 warnings
```

**Quick Fix:** `ruff check --fix .` (for Python files)

### 4.2 Code Patterns

**Strengths:**
1. Consistent naming conventions (snake_case for functions, PascalCase for classes)
2. Comprehensive inline documentation (especially in CUDA kernels)
3. Clear separation of device/host code
4. PIMPL pattern for CUDA types (see `gpu_transport_wrapper.cu`)

**Areas for Improvement:**
1. Some long functions (K3 kernel could be refactored)
2. Debug code should be conditionally compiled (currently present in main files)
3. Magic numbers scattered throughout (e.g., `0.999f`, `1e-12f`)

### 4.3 Documentation Quality

**Excellent:**
- `SPEC.md` - Comprehensive physics specification
- `README.md` - Build and usage instructions
- Inline comments in CUDA kernels explain physics

**Needs Improvement:**
- Some header files lack detailed documentation
- API documentation for key classes is sparse

---

## 5. Testing Coverage

### 5.1 Test Organization

```
tests/
├── audit/          # 4 files - Conservation tests
├── boundary/       # 2 files - Boundary condition tests
├── kernels/        # 2 files - Kernel unit tests
├── perf/           # 3 files - Profiling tools tests
├── source/         # 2 files - Source injection tests
├── unit/           # 14 files - Component unit tests
└── validation/     # 4 files - Physics validation tests
```

**Total Test Files:** 31
**Test Framework:** Google Test (GTest)

### 5.2 Test Coverage by Module

| Module | Test Count | Coverage |
|--------|------------|----------|
| Audit | 4 | Good |
| Boundary | 2 | Good |
| Core | 6+ | Excellent |
| CUDA Kernels | 2 | Limited (GPU tests hard) |
| Physics | 4+ | Good |
| Validation | 4 | Good |

**Note:** CUDA kernel testing is inherently difficult. The project uses validation against reference data (Moqui) instead.

---

## 6. Known Issues & Technical Debt

### 6.1 Debug Code in Production

**Issue:** Debug CSV dumping code is present in main transport file (`gpu_transport_wrapper.cu:23-95`)

**Recommendation:** Move to conditional compilation:
```cpp
#ifdef DEBUG_DUMP_CELLS
dump_initial_cells_to_csv(...);
#endif
```

### 6.2 Magic Numbers

**Issue:** Scattered hardcoded values:
- `0.999f` boundary limit
- `1e-12f` weight threshold
- `1e30f` infinity placeholder

**Recommendation:** Define named constants:
```cpp
constexpr float BOUNDARY_SAFETY_FACTOR = 0.999f;
constexpr float WEIGHT_CUTOFF = 1e-12f;
constexpr float INFINITY_DISTANCE = 1e30f;
```

### 6.3 Energy Grid Consistency

**Status:** Partially resolved (see `docs/.bkit-memory.json`)

**Historical Issue:** Energy grid definitions were inconsistent between files. The code now uses a piecewise-uniform grid (Option D2) with consistent bin edge calculation.

**Remaining Risk:** Ensure all files use the same energy calculation:
```cpp
float E = E_lower + ENERGY_OFFSET_RATIO * (E_upper - E_lower) * 0.5f;
```

### 6.4 MCS (Multiple Coulomb Scattering)

**Status:** 88% match rate as of mcs2-phase-b

**Issue:** Fermi-Eyges moment-based MCS implementation differs from specification variance accumulation approach.

**See:** `docs/revision_history.md` for detailed tracking.

---

## 7. Security & Safety

### 7.1 Security Analysis

- **Critical Issues:** 0
- **Warnings:** 0
- **Recommendations:** None for scientific computing context

### 7.2 Memory Safety

**Good Practices:**
- RAII patterns in C++ code
- Smart pointers used appropriately
- Clear ownership semantics

**CUDA Memory:**
- Proper error checking for CUDA operations
- Device memory allocation with size validation
- Memory profiling utilities available

---

## 8. Performance Considerations

### 8.1 GPU Optimization

**Strengths:**
- Separate compilation for CUDA kernels
- Occupancy analysis tools
- Memory pool management
- Kernel profiling infrastructure

**Tools Available:**
- `src/perf/kernel_profiler.cpp`
- `src/perf/memory_profiler.cpp`
- `src/perf/occupancy_analyzer.cpp`

### 8.2 Algorithmic Efficiency

- **CoarseList optimization:** K2 kernel uses direct list lookup instead of scanning (O(n) vs O(N*n))
- **Bucket emission:** Minimizes global memory writes
- **Sub-cell tracking:** Reduces phase-space resolution requirements

---

## 9. Recommendations

### 9.1 Priority: High

1. **[2min]** Run `ruff check --fix .` to auto-fix Python linting issues
2. **[1hr]** Extract debug code behind `#ifdef DEBUG` guards
3. **[2hrs]** Consolidate magic numbers into named constants

### 9.2 Priority: Medium

1. **[4hrs]** Refactor K3 kernel into smaller functions
2. **[3hrs]** Add API documentation to header files (Doxygen)
3. **[2hrs]** Improve test coverage for CUDA device functions

### 9.3 Priority: Low

1. **[8hrs]** Consider splitting CUDA kernels into multiple files by functionality
2. **[4hrs]** Add benchmark suite for performance regression testing

---

## 10. Conclusion

SM_2D is a well-architected scientific simulation code with strong physics foundations. The project demonstrates:

- Clear architectural vision (hierarchical kernel pipeline)
- Comprehensive testing approach
- Good documentation practices
- Active development with documented revision history

The main areas for improvement are code cleanup (debug code, magic numbers) rather than fundamental design issues. The codebase is production-ready for scientific research use.

### Quality Gate Status

| Criteria | Threshold | Actual | Status |
|----------|-----------|--------|--------|
| Linting errors | < 50 | 28 | PASS |
| Test files | > 10 | 31 | PASS |
| Security critical | 0 | 0 | PASS |
| Documentation | SPEC.md exists | Yes | PASS |

**Overall: PASS** - Codebase is suitable for continued development.

---

## Appendix A: File Inventory

### A.1 CUDA Kernels (9 files)
- `src/cuda/kernels/k1_activemask.cu`
- `src/cuda/kernels/k2_coarsetransport.cu`
- `src/cuda/kernels/k3_finetransport.cu`
- `src/cuda/kernels/k4_transfer.cu`
- `src/cuda/kernels/k5_audit.cu`
- `src/cuda/kernels/k6_swap.cu`
- `src/cuda/gpu_transport_wrapper.cu`
- `src/cuda/k1k6_pipeline.cu`
- `src/cuda/kernels/smoke.cu`

### A.2 Core Implementation (30+ files)
- `src/core/grids.cpp`
- `src/core/psi_storage.cpp`
- `src/core/buckets.cpp`
- `src/lut/nist_loader.cpp`
- `src/lut/r_lut.cpp`
- `src/source/pencil_source.cpp`
- `src/source/gaussian_source.cpp`
- `src/boundary/boundaries.cpp`
- `src/boundary/loss_tracking.cpp`
- `src/audit/conservation.cpp`
- `src/audit/global_budget.cpp`
- `src/audit/reporting.cpp`
- `src/validation/*.cpp` (6 files)
- `src/utils/*.cpp` (4 files)
- `src/perf/*.cpp` (3 files)

### A.3 Python Scripts (5 files)
- `batch_run.py` - Batch simulation runner
- `batch_plot.py` - Result visualization
- `visualize.py` - Dose plot generation
- `validation/analyze_*.py` (3 files) - Validation analysis

---

## Appendix B: Recent Commit History

Based on git log:
- `488fb01` - docs: add comprehensive revision history document
- `50092bd` - Fix lateral profiles: increase x_sub resolution (4→8), implement depth-based MCS
- `9e691a3` - feat(k2): implement Fermi-Eyges moment-based MCS (PDCA mcs2-phase-b)
- `dbf5cac` - feat(k2,k3): implement deterministic lateral spreading (NOT Monte Carlo)

The project shows active development with physics improvements and bug fixes.

---

**Document Version:** 1.0
**Generated by:** cdreview (cdscan + cdqa)
**Analysis Date:** 2026-02-06
