# MCS (Multiple Coulomb Scattering) Revision Plan for 2D Case

**Date**: 2026-02-02
**Status**: DRAFT
**Priority**: HIGH

---

## Executive Summary

This revision plan addresses the physics implementation of Multiple Coulomb Scattering (MCS) for the 2D proton transport solver (SM_2D), based on the PDG Highland formula and proper 2D scattering theory.

**Critical Finding**: The current implementation applies a 1/√2 correction factor that may be **INCORRECT** for the 2D x-z plane simulation geometry used in SM_2D.

---

## 1. Current Implementation Analysis

### 1.1 Highland Formula (Current)

**File**: `src/include/physics/highland.hpp:60-81`

```cpp
__host__ __device__ inline float highland_sigma(float E_MeV, float ds, float X0 = 360.8f) {
    constexpr float z = 1.0f;  // Proton charge

    // Relativistic kinematics
    float gamma = (E_MeV + m_p_MeV) / m_p_MeV;
    float beta = sqrtf(fmaxf(1.0f - 1.0f / (gamma * gamma), 0.0f));
    float p_MeV = sqrtf(fmaxf((E_MeV + m_p_MeV) * (E_MeV + m_p_MeV) - m_p_MeV * m_p_MeV, 0.0f));

    float t = ds / X0;
    if (t < 1e-6f) return 0.0f;

    // Highland correction factor
    float ln_t = logf(t);
    float bracket = 1.0f + 0.038f * ln_t;
    bracket = fmaxf(bracket, 0.25f);  // PDG 2024 recommendation

    // P2 FIX: Apply 2D projection correction
    float sigma_3d = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
    return sigma_3d * MCS_2D_CORRECTION;  // <-- 1/√2 APPLIED HERE
}
```

**Where**: `MCS_2D_CORRECTION = 0.70710678f;  // 1/√2`

### 1.2 Coordinate System

**File**: `src/cuda/kernels/k3_finetransport.cu`

```cpp
// Initial direction (before MCS)
float mu_init = cosf(theta);  // Direction cosine in z
float eta_init = sinf(theta);  // Direction cosine in x
```

**Geometry**: The simulation uses a **single polar angle θ** in the x-z plane (2D geometry).

---

## 2. Physics Review (PDG Highland Formula)

### 2.1 The Highland Formula (PDG 2024)

```
θ₀ = (13.6 MeV / βcp) × z × √(x/X₀) × [1 + 0.038 × ln(x/X₀)]
```

**Where θ₀ IS**: The RMS "projected" scattering angle for **one plane** (e.g., x-z plane).

**Key Definitions**:
- `θ₀` = RMS **projected** scattering angle (1D, for one plane)
- `x` = step length (material thickness)
- `X₀` = radiation length (360.8 mm for water)

### 2.2 1D vs 2D vs 3D Scattering

| Case | Definition | RMS Value |
|------|------------|-----------|
| **1D Projected Angle** | Scattering in ONE plane (x-z) | σ₁D = θ₀ |
| **2D Components** | θₓ, θᵧ (both with RMS θ₀) | σₓ = θ₀, σᵧ = θ₀ |
| **3D Total Angle** | θ = √(θₓ² + θᵧ²) | σ₃D = √2 × θ₀ |
| **2D Plane Projection** | x-z plane projection of 3D | σ₂D = σ₃D / √2 = θ₀ |

**CRITICAL INSIGHT**:
- If your simulation uses θ as the **1D projected angle in x-z plane**, then σ = θ₀ directly
- The 1/√2 correction is ONLY needed when converting FROM 3D scattering TO 2D projection
- Since SM_2D directly simulates the x-z plane with a single θ, it IS the projected angle!

---

## 3. Critical Issue: Incorrect 1/√2 Application?

### 3.1 The Question

**Does SM_2D's θ variable represent:**

A) The 1D projected scattering angle in the x-z plane?
   - **If YES**: σ = θ₀ (NO 1/√2 correction needed)

B) A component of 3D scattering that needs projection?
   - **If YES**: σ = θ₀ (still - because θ₀ is already projected!)

### 3.2 Analysis

Looking at the code:
```cpp
float mu = cosf(theta);  // Direction cosine in z (beam direction)
float eta = sinf(theta);  // Direction cosine in x (transverse)
```

This is a **2D x-z plane simulation** with:
- θ = angle from z-axis (beam direction)
- theta represents deflection in the x-z plane

**Conclusion**: θ IS the 1D projected angle for the x-z plane. Therefore:
- **σ should equal θ₀ directly**
- **The 1/√2 correction should NOT be applied**

### 3.3 Evidence from Code Comments

From `src/include/physics/highland.hpp:28-43`:
```cpp
// ============================================================================
// MCS 2D Projection Correction (Physical Analysis Applied)
// ============================================================================
// This simulation uses a 2D geometry (x-z plane).
// The Highland formula gives the 3D scattering angle sigma_3D.
//
// PHYSICS CORRECTION (2026-01):
// For 3D isotropic scattering projected onto a 2D plane:
//   - The azimuthal angle φ is uniformly distributed in [0, 2π]
//   - Projected variance: σ_2D² = σ_3D² / 2 (variance splits equally)
//   - Therefore: σ_2D = σ_3D / √2 ≈ 0.707 × σ_3D
```

**Issue**: This comment assumes Highland formula gives "3D scattering angle", but:
- The Highland formula θ₀ is explicitly defined as the **PROJECTED** angle (PDG 2024)
- "θ₀ is the RMS projected scattering angle" - PDG wording
- No additional correction is needed for 2D plane simulation

---

## 4. Additional Issues from Debug History

### 4.1 Missing Variance Accumulation (H6 Report)

**Current Code** (k3_finetransport.cu:243-265):
```cpp
// H6 FIX: Variance-based MCS accumulation (simplified implementation)
float theta_scatter = 0.0f;
if (enable_mcs) {
    float sigma_mcs = device_highland_sigma(E, actual_range_step);

    // Energy-dependent scattering reduction
    float scattering_reduction_factor = ...; // 0.3 to 1.0

    if (sigma_mcs > 0.0f) {
        theta_scatter = device_sample_mcs_angle(sigma_mcs * scattering_reduction_factor, seed);
    }
}
float theta_new = theta + theta_scatter;  // <-- Random walk in angle space!
```

**Problem**: Applies random scattering at EVERY step → excessive lateral spread

**Correct Approach** (per physics):
```cpp
// Accumulate variance
float var_accumulated = 0.0f;
for (each step) {
    float sigma_step = highland_sigma(E, ds);
    var_accumulated += sigma_step * sigma_step;  // <-- VARIANCE accumulates

    // Check threshold before applying
    float rms = sqrt(var_accumulated);
    if (rms > threshold) {
        // Apply 7-point quadrature or sample
        theta_new = theta + sample_normal() * rms;
        var_accumulated = 0.0f;  // Reset
    }
}
```

### 4.2 Scattering Reduction Factors (Unphysical?)

**File**: `k3_finetransport.cu:38-41, 249-260`

```cpp
// Scattering reduction factors (to maintain forward penetration)
constexpr float SCATTER_REDUCTION_HIGH_E = 0.3f;     // E > 100 MeV
constexpr float SCATTER_REDUCTION_MID_HIGH = 0.5f;   // E > 50 MeV
constexpr float SCATTER_REDUCTION_MID_LOW = 0.7f;    // E > 20 MeV
constexpr float SCATTER_REDUCTION_LOW_E = 1.0f;      // E <= 20 MeV
```

**Issue**: These reduction factors are ad-hoc and not based on physics!
- The Highland formula already accounts for energy dependence
- Reducing sigma further violates PDG physics

### 4.3 Lateral Spreading Method (K2 Kernel)

**File**: `k2_coarsetransport.cu:180-240`

The K2 kernel uses a **stochastic interpolation** method that does NOT accumulate theta:
```cpp
// PLAN_fix_scattering: Stochastic Interpolation for Lateral Spreading
// Instead of accumulating theta (which causes beam shifting),
// we use Gaussian weight distribution across multiple cells for lateral spreading.

float theta_new = theta;  // No theta accumulation!

// Calculate lateral spread sigma_x from scattering angle
float sigma_x = device_lateral_spread_sigma(sigma_theta, coarse_step_limited);
```

This approach is more physically correct than the random walk in K3!

---

## 5. Proposed Revision Plan

### Phase 1: Verify 2D Correction Factor

**Task**: Confirm whether MCS_2D_CORRECTION should be removed

**Action Items**:
1. Review PDG 2024 definition of θ₀
2. Confirm simulation geometry (2D x-z plane with single θ)
3. Decision: Remove 1/√2 if θ is the projected angle

**Files to Modify**:
- `src/include/physics/highland.hpp:46, 80`
- `src/cuda/device/device_physics.cuh:39, 74`

**Change**:
```cpp
// BEFORE:
constexpr float MCS_2D_CORRECTION = 0.70710678f;
return sigma_3d * MCS_2D_CORRECTION;

// AFTER (if confirmed):
// No correction needed - Highland theta_0 IS the projected angle
return sigma_3d;  // Highland sigma already is projected angle
```

### Phase 2: Remove Ad-hoc Scattering Reduction

**Task**: Remove unphysical scattering reduction factors

**Files to Modify**:
- `src/cuda/kernels/k3_finetransport.cu:37-46, 249-260`

**Change**:
```cpp
// REMOVE these lines:
constexpr float SCATTER_REDUCTION_HIGH_E = 0.3f;
constexpr float SCATTER_REDUCTION_MID_HIGH = 0.5f;
// ... etc ...

// REPLACE with:
if (sigma_mcs > 0.0f) {
    theta_scatter = device_sample_mcs_angle(sigma_mcs, seed);
}
```

### Phase 3: Implement Proper Variance Accumulation

**Task**: Implement variance-based MCS with threshold checking

**Approach Options**:

**Option A**: Full 7-point quadrature (per SPEC v0.8)
- Accumulate variance until threshold exceeded
- Split into 7 angular components when threshold reached
- More accurate, but complex to implement

**Option B**: Variance tracking with periodic sampling
- Accumulate variance across steps
- Sample from accumulated distribution at each step
- Simpler, still maintains correct variance

**Option C**: Keep current random sampling but with corrected sigma
- Remove 1/√2 correction (if applicable)
- Remove scattering reduction factors
- Let each step contribute correct variance

**Recommendation**: Start with Option C for baseline correctness, then consider Option B.

### Phase 4: Update Documentation

**Files to Update**:
- `docs/detailed/physics/01_physics_models.md`
- `docs/phases/phase_3_physics.md`
- Code comments in highland.hpp

**Required Updates**:
1. Clarify that Highland θ₀ IS the projected angle
2. Remove incorrect 2D correction explanation
3. Document the variance accumulation method

---

## 6. Verification Plan

### 6.1 Unit Tests

1. **Highland formula validation**:
   - Compare against PDG table values for water
   - Verify energy dependence (70, 100, 150, 200 MeV)
   - Check step dependence (√(x/X₀))

2. **Variance accumulation test**:
   - Verify σ²_total = Σ σ²_step (not σ_total = Σ σ_step)

### 6.2 Integration Tests

1. **Pencil beam in water**:
   - 160 MeV protons, verify Bragg peak at ~158 mm
   - Compare lateral spread with Fermi-Eyges theory
   - Check forward penetration

2. **Energy dependence**:
   - Run at 70, 100, 150, 200 MeV
   - Verify scattering decreases with energy

### 6.3 Benchmark Comparisons

| Metric | Current | After Fix | Expected |
|--------|---------|-----------|----------|
| Lateral σ at 80mm | ? | ? | Fermi-Eyges |
| Bragg peak position | ? | ~158mm | ~158mm |
| Forward penetration | ? | ~100% | ~100% |

---

## 7. Implementation Priority

| Phase | Priority | Complexity | Impact |
|-------|----------|------------|--------|
| 1: Verify 2D correction | CRITICAL | Low | HIGH (correct sigma) |
| 2: Remove reduction factors | HIGH | Low | HIGH (physics accuracy) |
| 3: Variance accumulation | HIGH | Medium | MEDIUM (variance correctness) |
| 4: Documentation | MEDIUM | Low | Low (maintenance) |

**Recommended Order**: 1 → 2 → 3 → 4

---

## 8. References

1. **PDG 2024**: "Passage of particles through matter"
   - Highland formula definition
   - θ₀ as "projected" scattering angle

2. **NIST PSTAR**: Water stopping powers and ranges
   - X₀ = 36.08 g/cm² = 360.8 mm (for ρ=1 g/cm³)

3. **Debug History**:
   - `dbg/H6_MCS_investigation_report.md`
   - `dbg/debug_history.md`

4. **Current Implementation**:
   - `src/include/physics/highland.hpp`
   - `src/cuda/device/device_physics.cuh`
   - `src/cuda/kernels/k3_finetransport.cu`

---

## 9. Decision Checklist

Before implementing, verify:

- [ ] PDG 2024 definition of θ₀ confirms it IS the projected angle
- [ ] SM_2D θ variable represents 1D angle in x-z plane
- [ ] Removing 1/√2 gives correct lateral spread
- [ ] Variance accumulation is properly implemented
- [ ] Tests pass after changes

---

**END OF REVISION PLAN**
