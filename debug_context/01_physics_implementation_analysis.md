# Physics Implementation Analysis

## Overview
This document analyzes the physics implementation in SM_2D (Proton Therapy Monte Carlo Simulation).

## ⚠️ CRITICAL BUG WARNING (2026-01-27)

**The current HEAD (commit 3859085) produces INCORRECT physics results:**

| Metric | Expected (150 MeV) | Actual | Status |
|--------|-------------------|--------|--------|
| Bragg Peak Position | ~157 mm | **2 mm** | ❌ FAIL |
| Energy Deposited | ~150 MeV | **6.9 MeV** | ❌ FAIL |

The batch results in `results/batch/batch_mean150/` (from Jan 26 08:14) show correct results with Bragg peak at 153.5mm. This indicates a regression was introduced in one of the late-evening commits on Jan 26.

**Until this bug is fixed, the simulation results should NOT be trusted for physics validation.**

---

## 1. Multiple Coulomb Scattering (MCS)

### Implementation Locations
- **CPU**: `src/include/physics/highland.hpp`
- **GPU**: `src/cuda/device/device_physics.cuh`

### Formula Used
```
σ_θ = (13.6 MeV / βcp) * z * sqrt(x/X_0) * [1 + 0.038 * ln(x/X_0)]
```

### Physics Correctness: ✓ CORRECT
1. **Relativistic kinematics properly implemented**:
   - γ = (E + m_p) / m_p
   - β = sqrt(1 - 1/γ²)
   - p = sqrt((E + m_p)² - m_p²)

2. **PDG 2024 compliance**:
   - Uses 13.6 MeV constant
   - Correction factor bracket ≥ 0.25 (PDG 2024 recommendation)
   - Valid for 1e-5 < t < 100 where t = x/X_0

3. **2D Projection Correction** (2026-01 fix):
   - σ_2D = σ_3D / √2 ≈ 0.707 × σ_3D
   - Correct for variance-based projection
   - Previous error: Used 2/π ≈ 0.637 (E[|cos(φ)|] for displacement, not angle variance)

### ⚠️ Documentation Issue
**Misleading Comment in GPU Code**: `src/cuda/device/device_physics.cuh` line 53 contains a comment that says "(2/π)" but the actual code correctly uses 0.70710678 (1/√2). The code is correct, but the comment should be updated to reflect the actual correction factor.

### Reference Values
| Constant | Value | Source |
|----------|-------|--------|
| X0_water | 360.8 mm | PDG |
| m_p | 938.272 MeV/c² | PDG |

---

## 2. Energy Straggling

### Implementation Locations
- **CPU**: `src/include/physics/energy_straggling.hpp`
- **GPU**: `src/cuda/device/device_physics.cuh`

### Models Implemented

### A. Vavilov Parameter κ
```
κ = ξ / T_max
where:
  ξ = (K/2) * (Z/A) * (z²/β²) * ρ * ds
  T_max = (2 m_e c² β² γ²) / (1 + 2γ m_e/m_p + (m_e/m_p)²)
  K = 0.307 MeV cm²/g
```

### B. Regime-Dependent Straggling
| Regime | Condition | Formula |
|--------|-----------|---------|
| Bohr (Gaussian) | κ >> 1 | σ_E = κ_0 * sqrt(ρ*ds) / β |
| Landau | κ << 1 | σ_eff = 4ξ / 2.355 |
| Vavilov | 0.01 < κ < 10 | Interpolated |

### Physics Correctness: ✓ CORRECT
1. **Full Vavilov regime handling** implemented
2. **Bohr straggling includes 1/β correction**
3. **Landau regime**: Uses FWHM/2.355 approximation (noted limitation)
4. **Vavilov interpolation**: Smooth weighting based on κ

### Note on Limitation
> "For therapeutic protons (50-250 MeV), kappa is typically in the Vavilov regime (0.01 < κ < 10), so Landau approximation has limited impact."

---

## 3. Nuclear Interactions

### Implementation Locations
- **CPU**: `src/include/physics/nuclear.hpp`
- **GPU**: `src/cuda/device/device_physics.cuh`

### Model
```cpp
σ_total(E) = σ_100 * [1 - 0.15 * ln(E/100)]  for E ≥ 20 MeV
```

### Reference Values (ICRU 63)
| Energy | σ [mm⁻¹] |
|--------|----------|
| 10 MeV | ~0.005 |
| 50 MeV | ~0.004 |
| 100 MeV | 0.0012 |
| 200 MeV | ~0.0025 |

### Physics Correctness: ✓ MOSTLY CORRECT
1. **ICRU 63 cross-sections** used as reference
2. **Energy-dependent logarithmic behavior**
3. **Coulomb barrier**: σ = 0 for E < 5 MeV

### Known Limitation
> "This simplified model treats all nuclear-removed energy as locally deposited. In reality, secondary particles transport ~70-80% of this energy away from the primary track, causing ~1-2% local dose overestimate."

---

## 4. Energy Loss (Stopping Power)

### Implementation Locations
- **CPU**: `src/include/physics/step_control.hpp`
- **GPU**: `src/cuda/device/device_lut.cuh`
- **Data**: NIST PSTAR (`src/lut/nist_loader.cpp`)

### R-Based Step Control
```cpp
R(E) = ∫_0^E dE'/(S(E')·ρ)  // CSDA range
dR/ds = -1  // In CSDA approximation

Step size control:
  δ_R_max = 0.02 * R  // Base: 2% of remaining range
  δ_R_max *= f(E)     // Energy-dependent refinement
```

### Energy-Dependent Refinement
| Energy Range | Refinement Factor | Max Step |
|--------------|-------------------|----------|
| E < 5 MeV | 0.2 | 0.1 mm |
| 5-10 MeV | 0.3 | 0.2 mm |
| 10-20 MeV | 0.5 | 0.5 mm |
| 20-50 MeV | 0.7 | 0.7 mm |
| E > 50 MeV | 1.0 | 1.0 mm |

### Physics Correctness: ✓ CORRECT
1. **R-based control** properly implemented
2. **Energy deposition**: dE = E - E(R - step)
3. **Alternative stopping power method**: dE = S(E) * ρ * ds / 10
4. **Proper handling** of R_new ≤ 0

---

## 5. Range Lookup Table (LUT)

### Implementation Locations
- **Header**: `src/include/lut/r_lut.hpp`
- **Source**: `src/lut/r_lut.cpp`
- **Data**: NIST PSTAR for protons in water

### Structure
```cpp
struct RLUT {
    EnergyGrid grid;
    vector<float> R;         // CSDA range [mm]
    vector<float> S;         // Stopping power [MeV cm²/g]
    vector<float> log_E;     // Pre-computed log(E)
    vector<float> log_R;     // Pre-computed log(R)
    vector<float> log_S;     // Pre-computed log(S)
};
```

### Interpolation Method
- **Log-log interpolation** for both R(E) and S(E)
- **Inverse lookup**: Binary search for R → E
- **GPU version**: Device-accessible structure in `src/cuda/device/device_lut.cuh`

### Physics Correctness: ✓ CORRECT

---

## Summary

| Physics Component | Status | Notes |
|-------------------|--------|-------|
| MCS (Highland) | ✓ CORRECT | PDG 2024 compliant, 2D projection corrected |
| Energy Straggling | ✓ CORRECT | Full Vavilov regime handling |
| Nuclear Interactions | ✓ CORRECT | ICRU 63, simplified local deposition |
| Energy Loss | ✓ CORRECT | R-based control with NIST data |
| Range LUT | ✓ CORRECT | Log-log interpolation, proper inverse |

### Known Approximations
1. Nuclear energy: All treated as local (1-2% dose overestimate)
2. Landau straggling: FWHM/2.355 approximation (limited impact for therapeutic protons)
3. 2D geometry: Assumes azimuthal symmetry (1/√2 projection correction)
