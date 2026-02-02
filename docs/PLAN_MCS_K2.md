# PLAN_MCS_K2: MCS Physics Revision for K2 Coarse Transport Kernel

**Date**: 2026-02-02
**Status**: DRAFT
**Kernel**: K2_CoarseTransport
**Focus**: 2D Multiple Coulomb Scattering (MCS) Implementation

---

## 1. Overview

This plan addresses the physics implementation of Multiple Coulomb Scattering (MCS) in the K2 coarse transport kernel, based on PDG Highland formula and proper 2D scattering theory for proton transport in water.

**Scope**: K2 kernel (`src/cuda/kernels/k2_coarsetransport.cu`)

---

## 2. Current K2 MCS Implementation

### 2.1 Code Location

**File**: `src/cuda/kernels/k2_coarsetransport.cu`
**Lines**: 180-240

### 2.2 Current Implementation

```cpp
// PLAN_fix_scattering: Stochastic Interpolation for Lateral Spreading
// ========================================================================
// Instead of accumulating theta (which causes beam shifting),
// we use Gaussian weight distribution across multiple cells for lateral spreading.
//
// Key changes from previous implementation:
// 1. Theta is NOT accumulated: theta_new = theta (not theta + theta_scatter)
// 2. Lateral spread is calculated: sigma_x = sin(sigma_theta) * step
// 3. Weight is distributed across multiple cells using Gaussian CDF
// ========================================================================

// Calculate MCS scattering angle (RMS)
float sigma_theta = device_highland_sigma(E, coarse_range_step);

// PLAN_fix_scattering: DO NOT accumulate theta
// Theta remains unchanged; lateral spreading is handled by weight distribution
float theta_new = theta;  // No theta accumulation!

float mu_new = cosf(theta_new);
float eta_new = sinf(theta_new);

// Calculate lateral spread sigma_x from scattering angle
// sigma_x represents the standard deviation of lateral displacement
float sigma_x = device_lateral_spread_sigma(sigma_theta, coarse_step_limited);

// Position update (use limited step to avoid boundary crossing)
float x_new = x_cell + eta_new * coarse_step_limited;
float z_new = z_cell + mu_new * coarse_step_limited;
```

### 2.3 Lateral Spreading Function

**File**: `src/cuda/device/device_physics.cuh:320-325`

```cpp
// Calculate lateral spread sigma_x from scattering angle
// sigma_x = sin(sigma_theta) * step
// For small angles: sin(sigma_theta) ≈ sigma_theta
__device__ inline float device_lateral_spread_sigma(float sigma_theta, float step) {
    return sinf(fminf(sigma_theta, 1.57f)) * step;  // sin(theta), theta < pi/2
}
```

---

## 3. Physics Verification

### 3.1 Highland Formula (Current)

**File**: `src/cuda/device/device_physics.cuh:52-75`

```cpp
__device__ inline float device_highland_sigma(float E_MeV, float ds, float X0 = DEVICE_X0_water) {
    constexpr float z = 1.0f;  // Proton charge

    // Relativistic kinematics
    float gamma = (E_MeV + DEVICE_m_p_MeV) / DEVICE_m_p_MeV;
    float beta2 = 1.0f - 1.0f / (gamma * gamma);
    float beta = sqrtf(fmaxf(beta2, 0.0f));
    float p_MeV = sqrtf(fmaxf((E_MeV + DEVICE_m_p_MeV) * (E_MeV + DEVICE_m_p_MeV) -
                             DEVICE_m_p_MeV * DEVICE_m_p_MeV, 0.0f));

    float t = ds / X0;
    if (t < 1e-6f) return 0.0f;

    // Highland correction factor (PDG 2024: bracket >= 0.25)
    float ln_t = logf(t);
    float bracket = 1.0f + 0.038f * ln_t;
    bracket = fmaxf(bracket, 0.25f);

    // P2 FIX: Apply 2D projection correction
    float sigma_3d = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
    return sigma_3d * DEVICE_MCS_2D_CORRECTION;  // <-- 1/√2 APPLIED
}
```

**Where**:
- `DEVICE_X0_water = 360.8f` (mm) - Correct for water
- `DEVICE_MCS_2D_CORRECTION = 0.70710678f` (1/√2) - **May be incorrect**

### 3.2 PDG Highland Formula Reference

```
θ₀ = (13.6 MeV / βcp) × z × √(x/X₀) × [1 + 0.038 × ln(x/X₀)]
```

**Definition**: θ₀ is the RMS **"projected" scattering angle** for one plane.

**Key Question**: Does the 2D simulation's θ variable need the 1/√2 correction?

| Scenario | Correction Needed | Reason |
|----------|-------------------|--------|
| θ is 3D total scattering angle | Yes, divide by √2 | Convert 3D → 2D projection |
| θ is 1D projected angle | No | θ₀ already is projected |
| θ is x-z plane angle (SM_2D case) | **No** | θ₀ already is the projected RMS |

### 3.3 K2 Coordinate System

```cpp
// Direction
float mu = cosf(theta);  // Direction cosine in z (beam direction)
float eta = sinf(theta);  // Direction cosine in x (transverse)
```

**Geometry**: 2D x-z plane with single polar angle θ
- θ = 0 → beam along +z axis
- θ ≠ 0 → beam deflected in x-z plane

**Conclusion**: θ IS the 1D projected angle for the x-z plane. The Highland formula θ₀ gives exactly this quantity.

---

## 4. Issues Identified

### 4.1 Issue 1: Potentially Incorrect 1/√2 Correction

**Severity**: HIGH

**Problem**: Code applies `DEVICE_MCS_2D_CORRECTION = 1/√2` to Highland formula.

**Analysis**:
- Highland θ₀ is explicitly defined as the **projected** scattering angle (PDG)
- For 2D x-z plane simulation, θ represents this projected angle
- No additional correction should be needed

**Impact**:
- Current sigma is ~30% smaller than it should be
- Lateral spreading is underestimated
- Beam may be too narrow

**Recommendation**:
- Remove `DEVICE_MCS_2D_CORRECTION` from `device_highland_sigma()`
- Update documentation to clarify θ₀ is already the projected angle

### 4.2 Issue 2: Lateral Spread Calculation

**Severity**: MEDIUM

**Current Code**:
```cpp
float sigma_x = device_lateral_spread_sigma(sigma_theta, coarse_step_limited);
// = sin(sigma_theta) * step
```

**Analysis**:
- For small angles: sin(σ_θ) ≈ σ_θ (correct)
- σ_θ is the RMS of angle distribution, not the angle itself
- The sin(σ_θ) form is an empirical approximation, not an exact transformation

**Status**: Conditionally acceptable for small-angle approximation
- K2 should enforce: σ_θ < 20 mrad for this approximation to be valid
- If σ_θ exceeds this threshold, particle should be handled by K3 instead

### 4.3 Issue 3: Step Size Units

**Severity**: LOW

**Current Code**:
```cpp
float coarse_range_step = coarse_step_limited / mu_abs;  // Path length
float sigma_theta = device_highland_sigma(E, coarse_range_step);  // Uses path length
float sigma_x = device_lateral_spread_sigma(sigma_theta, coarse_step_limited);  // Uses geometric
```

**Analysis**:
- Highland formula uses path length (ds = geometric / |cos θ|)
- Lateral spread uses geometric distance (correct)
- Units are consistent

**Status**: Correct

### 4.4 Issue 4: No Variance Accumulation (K2 Specific)

**Severity**: MEDIUM for K2 (HIGH for K3)

**K2 Behavior**:
- Theta is NOT accumulated: `theta_new = theta`
- Lateral spreading is via weight distribution only
- Scaling: O(√n) for lateral spread (underestimates Fermi-Eyges O(n^(3/2)))

**K3 Behavior** (for comparison):
- Theta IS accumulated with random sampling each step
- Should give O(n^(3/2)) scaling to match Fermi-Eyges theory
- Proper variance accumulation is required for K3

**Analysis**:
- Weight distribution (K2) is a compromise: O(√n) vs O(n^(3/2))
- For small number of coarse steps (~100), the difference may be acceptable
- For accurate physics, K2 should ideally use Fermi-Eyges moment evolution

---

## 5. Proposed Changes for K2

### 5.1 Change 1: Remove 1/√2 Correction (HIGH Priority)

**File**: `src/cuda/device/device_physics.cuh`

**Before**:
```cpp
// Line 39
constexpr float DEVICE_MCS_2D_CORRECTION = 0.70710678f;

// Line 73-74
float sigma_3d = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
return sigma_3d * DEVICE_MCS_2D_CORRECTION;
```

**After**:
```cpp
// Line 39
// REMOVED: Highland theta_0 IS the projected scattering angle (PDG 2024)
// No 2D correction needed for x-z plane simulation
// constexpr float DEVICE_MCS_2D_CORRECTION = 0.70710678f;  // DEPRECATED

// Line 73-74
float sigma = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
return sigma;  // Highland sigma already is the projected angle RMS
```

**Also Update**: `src/include/physics/highland.hpp` (CPU version)

### 5.2 Change 2: Update Documentation (MEDIUM Priority)

**File**: `src/cuda/device/device_physics.cuh` (comments)

**Before**:
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

**After**:
```cpp
// ============================================================================
// Highland Formula for MCS (PDG 2024)
// ============================================================================
// This simulation uses a 2D geometry (x-z plane) with a single polar angle θ.
//
// The Highland formula gives θ₀, the RMS "projected" scattering angle for
// one plane (e.g., the x-z plane). Since SM_2D directly simulates the x-z
// plane with θ as the deflection angle, no additional 2D correction is needed.
//
// Formula: θ₀ = (13.6 MeV / βcp) × z × √(x/X₀) × [1 + 0.038 × ln(x/X₀)]
//
// where:
//   θ₀ = RMS projected scattering angle (for x-z plane)
//   x = step length (path length along trajectory)
//   X₀ = radiation length (360.8 mm for water)
```

### 5.3 Change 3: Verify Lateral Spread Emission (LOW Priority)

**File**: `src/cuda/kernels/k2_coarsetransport.cu:228-240`

**Current Code**:
```cpp
if (exit_face == FACE_Z_PLUS && sigma_x > dx * 0.01f) {
    // Multi-cell emission with lateral spreading
    device_emit_lateral_spread(
        OutflowBuckets, cell, iz_target,
        theta_new, E_new, w_new,
        x_new, sigma_x, dx, Nx, Nz,
        theta_edges, E_edges,
        N_theta, N_E, N_theta_local, N_E_local,
        10  // Spread across 10 cells
    );
}
```

**Status**: This approach is correct - uses Gaussian weight distribution for lateral spreading without accumulating theta.

**No changes needed**, but verify `device_emit_lateral_spread` implementation is correct.

---

## 6. Verification Plan

### 6.1 Unit Tests

1. **Highland sigma calculation**:
   ```cpp
   // Test case: 150 MeV proton, 1 mm water
   // Expected: θ₀ ≈ 2.0 mrad (from PDG Highland formula)
   // Water X0 = 360.8 mm, t = 1/360.8 ≈ 0.00277
   float sigma = device_highland_sigma(150.0f, 1.0f);
   assert(fabs(sigma - 0.0020) < 0.0003);
   ```

2. **Energy dependence** (corrected values from PDG Highland):
   | Energy (MeV) | Expected θ₀ (mrad) | Calculation |
   |--------------|-------------------|-------------|
   | 70 | ~4.11 | Higher scattering at low energy |
   | 100 | ~2.92 | β⁻¹ dependence |
   | 150 | ~1.99 | Reference value |
   | 200 | ~1.52 | Lower scattering at high energy |

   **Note**: Previous document had incorrect values (150/200 both listed as ~1.5 mrad). These corrected values follow the Highland formula scaling.

3. **Step dependence**:
   - Verify √(x/X₀) dependence
   - Test with x = 0.5, 1.0, 2.0 mm

### 6.2 Integration Tests

1. **K2 coarse transport only**:
   - Set E_trigger = 0.05 MeV (force coarse-only)
   - Run 150 MeV pencil beam
   - Verify forward penetration and lateral spread

2. **Compare lateral spread**:
   - Measure σₓ at various depths
   - Compare with Fermi-Eyges theory

### 6.3 Regression Tests

| Metric | Before Fix (with 1/√2) | After Fix (no correction) | Expected |
|--------|------------------------|---------------------------|----------|
| Sigma at 150 MeV, 1mm | ~1.4 mrad | ~2.0 mrad | ~2.0 mrad (PDG) |
| Lateral σ at 80mm | Current | ~1.4× larger | Fermi-Eyges theory |
| Forward range | Current | Similar | ~158 mm (unchanged) |

---

## 7. Implementation Steps

### Step 1: Prepare Test Baseline
```bash
# Run current implementation
cd build
make run_simulation
./run_simulation
cp results/dose_2d.txt results/before_fix_dose.txt
```

### Step 2: Implement Changes
```bash
# Edit files
# src/cuda/device/device_physics.cuh
# src/include/physics/highland.hpp
```

### Step 3: Rebuild and Test
```bash
cd build
make -j$(nproc)
./run_simulation
cp results/dose_2d.txt results/after_fix_dose.txt
```

### Step 4: Compare Results
```bash
python3 << 'EOF'
import numpy as np

before = np.loadtxt('results/before_fix_dose.txt')
after = np.loadtxt('results/after_fix_dose.txt')

print(f"Before: Max dose = {before[:,3].max():.2f} Gy at z = {before[np.argmax(before[:,3]), 2]:.1f} mm")
print(f"After:  Max dose = {after[:,3].max():.2f} Gy at z = {after[np.argmax(after[:,3]), 2]:.1f} mm")

# Compare lateral profiles at mid-range
mid_depth = 80  # mm
before_lat = before[abs(before[:,2] - mid_depth) < 1]
after_lat = after[abs(after[:,2] - mid_depth) < 1]

if len(before_lat) > 0 and len(after_lat) > 0:
    sigma_before = np.sqrt(np.sum(before_lat[:,1]**2 * before_lat[:,3]) / np.sum(before_lat[:,3]))
    sigma_after = np.sqrt(np.sum(after_lat[:,1]**2 * after_lat[:,3]) / np.sum(after_lat[:,3]))
    print(f"Lateral σ at {mid_depth}mm: Before = {sigma_before:.2f} mm, After = {sigma_after:.2f} mm")
EOF
```

### Step 5: Validation
- Verify increased lateral spread (~1.4× larger)
- Verify Bragg peak position unchanged
- Verify total energy deposition conserved

---

## 8. Rollback Plan

If results are worse after the fix:

```bash
git checkout HEAD -- src/cuda/device/device_physics.cuh
git checkout HEAD -- src/include/physics/highland.hpp
cd build && make -j$(nproc)
```

Document findings and re-evaluate the physics interpretation.

---

## 9. Related Documents

- `docs/MCS_revision_plan_2D.md` - Overall MCS revision plan
- `dbg/H6_MCS_investigation_report.md` - Previous MCS investigation
- `docs/detailed/physics/01_physics_models.md` - Physics documentation

---

## 10. Decision Checklist

Before committing changes:

- [ ] Unit tests pass (Highland formula validation)
- [ ] Integration tests pass (coarse transport)
- [ ] Lateral spread increased by expected amount (~√2)
- [ ] Forward penetration maintained
- [ ] Energy conserved
- [ ] Documentation updated
- [ ] Code comments clarified

---

## 11. Theoretical Analysis: Lateral Spreading Methods

> **NOTE (2026-02-02)**: Previous version of this section contained significant errors in scaling analysis. This section has been corrected based on external physics review.

### 11.1 User's Hypothesis

The user proposed that the **weight distribution method** (current K2) may produce equivalent lateral spreading to the **angle accumulation method** through iterative Gaussian convolution:

> "각도를 누적하는 것이 물리적으로 옳은 이유는 최외곽에서 산란한 입자의 이전 산란 방향에 이번 iteration에서의 산란이 더해지기 때문이다. 만약 산란의 영향을 매 스텝마다 다시 계산한다면, 또는 현재 방식으로 산란에 의한 효과를 가중치 분산으로 매 번 계산한다면 동일한 결과를 줄 것으로 예상한다."

**Visualization of the hypothesis:**
```
Initial:    x[5]=0.01,  x[6]=0.09,  x[7]=0.41,  x[8]=0.90,  x[9]=1.00,  x[10]=0.90, ...
Step 1:     x[5]→(x[4],x[3]), x[6]→(x[5],x[4]), ...  (Gaussian spread)
Step 2:     Each position spreads again...
Result:     Small tails form across wide area
```

### 11.2 Corrected Mathematical Analysis

#### 11.2.1 Angle Accumulation Method (Particle-based, K3-style)

For a particle undergoing multiple Coulomb scattering:

```
Angle random walk: θ_n = Σ_{i=1}^n Δθ_i, where Δθ_i ~ N(0, σ_θ²)
Var(θ_n) = n × σ_θ²

Position is the integral of angle:
x_n = x_0 + Δz × Σ_{k=1}^n θ_k
    = x_0 + Δz × Σ_{k=1}^n Σ_{i=1}^k Δθ_i

This is a sum of correlated random variables. The variance is:
Var(x_n) = Δz² × σ_θ² × n(n+1)(2n+1)/6

For large n: Var(x_n) ≈ Δz² × σ_θ² × n³/3
Therefore: σ_x ∝ n^(3/2)
```

**Key correction**: Previous document incorrectly claimed σ_x ∝ n.
The correct scaling for angle accumulation is **O(n^(3/2))**.

#### 11.2.2 Weight Distribution Method (Distribution-based, K2-style)

For Gaussian convolution at each step:

```
Each step: w(x) = w(x) ⊗ G(0, σ_x²)
where G is Gaussian kernel with σ_x = sin(σ_θ) × Δz ≈ σ_θ × Δz

Gaussian ⊗ Gaussian = Gaussian with added variance:
σ_0² ⊗ σ_1² ⊗ ... ⊗ σ_n² = σ_0² + σ_1² + ... + σ_n²

For n identical steps: σ_x,total² = σ_0² + n × σ_x²
Therefore: σ_x ∝ √n
```

**Key correction**: Gaussian convolution **does add variance**, not cancel it.
The "redistribution cancels out" argument was incorrect.

### 11.3 Comparison of Scaling Laws

| Method | Scaling | Physical Meaning |
|--------|---------|------------------|
| Angle accumulation (K3) | O(n^(3/2)) | Particle trajectory accumulates angular deflections geometrically |
| Weight distribution (K2) | O(√n) | Probability distribution diffuses at each step |
| Fermi-Eyges theory | O(z^(3/2)) | Multiple scattering theory predicts ~z^(3/2) lateral spread |

**Conclusion**: Angle accumulation matches Fermi-Eyges scaling (O(n^(3/2))).
Weight distribution (O(√n)) will **underestimate** lateral spread for large n.

### 11.4 Why the Previous Simulation Was Misleading

The numerical simulation in the previous document showed minimal lateral spread growth,
which contradicted the theoretical O(√n) prediction. This discrepancy indicates
**implementation issues**, not physical validity:

**Possible causes:**
1. **Kernel truncation**: 10-cell spread radius may truncate significant probability mass
2. **Incorrect normalization**: If weights are renormalized after truncation, variance is lost
3. **Discrete bin effects**: Center-of-bin RMS calculation may not capture true spread
4. **Boundary effects**: Finite grid causes edge reflections/absorptions

**Verification needed**: The simulation code needs variance conservation check:
```python
# After each convolution, verify:
assert(abs(w.sum() - 1.0) < 1e-10)  # Weight conservation
assert(abs(sigma_squared_new - sigma_squared_old - sigma_x**2) < 1e-6)  # Variance accumulation
```

### 11.5 K2 Design Implications

Given the corrected analysis:

| Issue | Impact | Recommendation |
|-------|--------|----------------|
| **1/√2 correction** | Removes 29% of scattering | ✅ **Remove** - Highland θ₀ is already projected angle |
| **Weight distribution scaling** | O(√n) vs O(n^(3/2)) | ⚠️ **Acceptable for coarse** if n is small |
| **`sin(σ_θ)*step`** | Approximate for small angles | ✅ **OK with condition** σ_θ < 20 mrad |
| **Fermi-Eyges moments** | Would give correct scaling | ⭐ **Consider for future** |

### 11.6 Recommended Approach for K2

**Short term** (current plan):
1. Remove 1/√2 correction (HIGH priority)
2. Add condition: `if (sigma_theta > 0.02) return K3_THRESHOLD;`
3. Document that K2 uses approximate O(√n) scaling

**Long term** (more accurate):
1. Implement Fermi-Eyges moment evolution for K2:
   ```
   d⟨θ²⟩/dz = T (scattering power)
   d⟨xθ⟩/dz = ⟨θ²⟩
   d⟨x²⟩/dz = 2⟨xθ⟩
   ```
2. Use evolved moments to compute σ_x at cell exit
3. This preserves O(n^(3/2)) scaling without per-particle sampling

### 11.7 Implementation Detail Checks

1. **Highland energy dependence**:
   ```cpp
   // Use energy at step MIDDLE, not beginning
   float E_mid = E - dE_per_step / 2.0f;
   float sigma_theta = device_highland_sigma(E_mid, coarse_range_step);
   ```

2. **`device_emit_lateral_spread` variance conservation**:
   - CDF-based distribution must account for truncated tails
   - Test: total emitted weight = input weight AND variance is preserved

### 11.8 Summary of Corrections

| Previous (Incorrect) | Corrected |
|---------------------|-----------|
| Angle accumulation: σ_x ∝ n | σ_x ∝ n^(3/2) |
| Weight dist: "redistribution cancels" | Weight dist: variance adds (O(√n)) |
| "Simulation shows minimal growth" | Simulation has bugs (truncation/normalization) |
| "K2 is acceptable because √n" | K2 is acceptable BUT underestimates; Fermi-Eyges preferred |

---

**END OF PLAN_MCS_K2**
