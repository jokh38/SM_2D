# Phase 3: Physics Models Implementation

**Status**: Pending
**Duration**: 3-4 days
**Dependencies**: Phase 1 (LUT), Phase 2 (Data Structures)

---

## Objectives

1. Implement R-based step size control
2. Implement Highland MCS with variance accumulation
3. Implement nuclear attenuation with energy tracking
4. Implement 2-bin energy discretization
5. Implement angular quadrature (7-point)

---

## TDD Cycle 3.1: R-Based Step Size Control

### RED - Write Tests First

Create `tests/physics/test_step_control.cpp`:

```cpp
#include <gtest/gtest.h>
#include "physics/step_control.hpp"
#include "lut/r_lut.hpp"

TEST(StepControlTest, StepSizeAtHighEnergy) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    float ds = compute_max_step_physics(150.0f, lut);
    float R = lut.lookup_R(150.0f);

    // Max 2% of range
    EXPECT_LE(ds, 0.02f * R);
    EXPECT_GT(ds, 0);
}

TEST(StepControlTest, StepSizeAtLowEnergy) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    float ds = compute_max_step_physics(5.0f, lut);

    // Near Bragg: max 0.2 mm
    EXPECT_LE(ds, 0.2f);
}

TEST(StepControlTest, StepSizeAtMediumEnergy) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    float ds = compute_max_step_physics(50.0f, lut);

    // Medium energy: max 0.5 mm or 2% of range
    EXPECT_LE(ds, 0.5f);
}

TEST(StepControlTest, StepSizeDecreasesWithEnergy) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    float ds_high = compute_max_step_physics(150.0f, lut);
    float ds_low = compute_max_step_physics(10.0f, lut);

    // At lower energy, should have smaller step (due to Bragg refinement)
    EXPECT_LT(ds_low, ds_high);
}

TEST(StepControlTest, EnergyUpdateConsistency) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    float E0 = 150.0f;
    float ds = 1.0f;

    float R0 = lut.lookup_R(E0);
    float R1 = R0 - ds;
    float E1 = lut.lookup_E_inverse(R1);

    // Energy should decrease
    EXPECT_LT(E1, E0);
    EXPECT_GT(E1, 0);

    float dE = E0 - E1;
    EXPECT_GT(dE, 0);
    EXPECT_LT(dE, E0);
}

TEST(StepControlTest, FullRangeExhaustion) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    float E0 = 150.0f;
    float R0 = lut.lookup_R(E0);

    // Step beyond full range
    float ds = R0 * 1.1f;
    float R1 = R0 - ds;
    float E1 = lut.lookup_E_inverse(R1);

    // Should hit zero or very close
    EXPECT_LE(E1, E_cutoff);
}
```

### GREEN - Implementation

Create `include/physics/step_control.hpp`:

```cpp
#pragma once

#include "lut/r_lut.hpp"
#include <cmath>

// Cutoff energy
constexpr float E_cutoff = 0.1f;  // MeV

// Compute maximum step size based on R(E)
// Returns step size in mm
float compute_max_step_physics(float E, const RLUT& lut);

// Update energy using R-based control
// Returns (E_new, dE_deposited)
std::pair<float, float> update_energy_via_R(float E, float ds, const RLUT& lut);
```

Create `src/physics/step_control.cpp`:

```cpp
#include "physics/step_control.hpp"

float compute_max_step_physics(float E, const RLUT& lut) {
    float R = lut.lookup_R(E);

    // Option A: Fixed fraction of remaining range
    float delta_R_max = 0.02f * R;  // Max 2% range loss per substep

    // Energy-dependent refinement near Bragg
    if (E < 10.0f) {
        delta_R_max = fminf(delta_R_max, 0.2f);  // mm
    } else if (E < 50.0f) {
        delta_R_max = fminf(delta_R_max, 0.5f);
    }

    return delta_R_max;
}

std::pair<float, float> update_energy_via_R(float E, float ds, const RLUT& lut) {
    float R_current = lut.lookup_R(E);
    float R_new = R_current - ds;

    float E_new, dE;

    if (R_new <= 0) {
        E_new = 0;
        dE = E;
    } else {
        E_new = lut.lookup_E_inverse(R_new);
        dE = E - E_new;
    }

    return {E_new, dE};
}
```

---

## TDD Cycle 3.2: Highland Formula

### RED - Write Tests First

Create `tests/physics/test_highland.cpp`:

```cpp
#include <gtest/gtest.h>
#include "physics/highland.hpp"

TEST(HighlandTest, SigmaPositive) {
    float sigma = highland_sigma(150.0f, 1.0f);
    EXPECT_GT(sigma, 0);
    EXPECT_LT(sigma, 0.1f);  // Should be small for high energy
}

TEST(HighlandTest, EnergyDependence) {
    // More scattering at low energy
    float sigma_high_E = highland_sigma(150.0f, 1.0f);
    float sigma_low_E = highland_sigma(10.0f, 1.0f);

    EXPECT_GT(sigma_low_E, sigma_high_E);
}

TEST(HighlandTest, StepSizeDependence) {
    // Scales with sqrt(step_size)
    float sigma_1mm = highland_sigma(100.0f, 1.0f);
    float sigma_2mm = highland_sigma(100.0f, 2.0f);
    float sigma_4mm = highland_sigma(100.0f, 4.0f);

    EXPECT_GT(sigma_2mm, sigma_1mm);
    EXPECT_GT(sigma_4mm, sigma_2mm);

    // Check sqrt scaling
    float ratio_2 = sigma_2mm / sigma_1mm;
    float ratio_4 = sigma_4mm / sigma_1mm;
    EXPECT_NEAR(ratio_2, sqrtf(2.0f), 0.1f);
    EXPECT_NEAR(ratio_4, 2.0f, 0.1f);
}

TEST(HighlandTest, SmallStepReturnsZero) {
    float sigma = highland_sigma(100.0f, 1e-12f);
    EXPECT_NEAR(sigma, 0, 1e-6f);
}

TEST(HighlandTest, BracketNonNegative) {
    // For very small steps, ln term could be negative
    // But bracket = 1 + 0.038 * ln(t) should stay positive

    for (float ds : {1e-6f, 1e-4f, 0.001f, 0.01f, 0.1f, 1.0f, 10.0f}) {
        float sigma = highland_sigma(100.0f, ds);
        EXPECT_GE(sigma, 0);
    }
}

TEST(HighlandTest, ReasonableValues) {
    // Check that Highland gives reasonable values
    float sigma_150 = highland_sigma(150.0f, 1.0f);
    float sigma_70 = highland_sigma(70.0f, 1.0f);

    // 150 MeV, 1mm in water: expect ~0.002 rad
    EXPECT_GT(sigma_150, 0.001f);
    EXPECT_LT(sigma_150, 0.01f);

    // 70 MeV, 1mm in water: expect ~0.004 rad
    EXPECT_GT(sigma_70, 0.003f);
    EXPECT_LT(sigma_70, 0.02f);
}
```

### GREEN - Implementation

Create `include/physics/highland.hpp`:

```cpp
#pragma once

#include <cmath>

// Physical constants
constexpr float X0_water = 360.8f;  // Radiation length [mm]
constexpr float m_p = 938.272f;     // Proton mass [MeV/c²]

// Highland formula for MCS angle [rad]
// E: kinetic energy [MeV]
// ds: step length [mm]
float highland_sigma(float E_MeV, float ds);

// Returns -1 if step should be reduced (bracket near zero)
```

Create `src/physics/highland.cpp`:

```cpp
#include "physics/highland.hpp"

float highland_sigma(float E_MeV, float ds) {
    // Relativistic kinematics
    float beta = sqrtf(1.0f - powf(m_p / (E_MeV + m_p), 2));
    float p_MeV = sqrtf(powf(E_MeV + m_p, 2) - m_p * m_p);

    // Reduced thickness
    float t = ds / X0_water;

    if (t < 1e-10f) return 0.0f;

    // Highland formula
    float ln_term = logf(t);
    float bracket = 1.0f + 0.038f * ln_term;

    // Step reduction if bracket becomes unphysical
    if (bracket < 0.1f) {
        return -1.0f;  // Signal: reduce step size
    }

    float sigma_theta = (13.6f / (beta * p_MeV)) * sqrtf(t) * bracket;

    return sigma_theta;
}
```

---

## TDD Cycle 3.3: Variance Accumulation

### RED - Write Tests First

Create `tests/physics/test_variance_accum.cpp`:

```cpp
#include <gtest/gtest.h>
#include "physics/variance_accum.hpp"

TEST(VarianceTest, SingleStep) {
    VarianceAccumulator accum;

    float sigma = 0.05f;
    accum.add(sigma);

    EXPECT_NEAR(accum.rms(), sigma, 1e-6f);
}

TEST(VarianceTest, MultipleSteps) {
    VarianceAccumulator accum;

    float sigma = 0.03f;
    for (int i = 0; i < 4; ++i) {
        accum.add(sigma);
    }

    float rms = accum.rms();
    // After 4 steps: var = 4 * sigma^2, rms = 2 * sigma
    EXPECT_NEAR(rms, 2.0f * sigma, 1e-6f);
}

TEST(VarianceTest, DifferentSigmas) {
    VarianceAccumulator accum;

    accum.add(0.03f);
    accum.add(0.04f);

    float rms = accum.rms();
    float expected = sqrtf(0.03f*0.03f + 0.04f*0.04f);
    EXPECT_NEAR(rms, expected, 1e-6f);
}

TEST(VarianceTest, SplitCondition) {
    VarianceAccumulator accum;

    // Below threshold
    accum.add(0.02f);
    EXPECT_FALSE(accum.should_split(100.0f));

    // Above threshold
    accum.add(0.05f);
    EXPECT_TRUE(accum.should_split(100.0f));
}

TEST(VarianceTest, EnergyDependentThreshold) {
    VarianceAccumulator accum;

    // High energy: fixed threshold
    accum.add(0.06f);  // variance = 0.0036, rms = 0.06
    EXPECT_TRUE(accum.should_split(100.0f));  // threshold = 0.05

    accum.reset();

    // Low energy: relaxed threshold
    accum.add(0.06f);
    EXPECT_FALSE(accum.should_split(10.0f));  // threshold = 0.05 * sqrt(10/50) ≈ 0.022
}

TEST(VarianceTest, Reset) {
    VarianceAccumulator accum;

    accum.add(0.05f);
    EXPECT_GT(accum.rms(), 0);

    accum.reset();
    EXPECT_NEAR(accum.rms(), 0, 1e-10f);
}
```

### GREEN - Implementation

Create `include/physics/variance_accum.hpp`:

```cpp
#pragma once

#include <cmath>

// Variance-based accumulator for MCS angles
// CRITICAL: Accumulate variance, not sigma directly
class VarianceAccumulator {
public:
    VarianceAccumulator() : variance(0.0f) {}

    // Add a sigma contribution (accumulates variance)
    void add(float sigma) {
        variance += sigma * sigma;
    }

    // Get RMS of accumulated scattering
    float rms() const {
        return sqrtf(variance);
    }

    // Check if split should occur based on accumulated RMS
    bool should_split(float E) const {
        float rms_threshold = (E > 50.0f) ? 0.05f : 0.05f * sqrtf(E / 50.0f);
        return rms() > rms_threshold;
    }

    // Reset for new segment
    void reset() {
        variance = 0.0f;
    }

private:
    float variance;
};
```

---

## TDD Cycle 3.4: Nuclear Attenuation

### RED - Write Tests First

Create `tests/physics/test_nuclear.cpp`:

```cpp
#include <gtest/gtest.h>
#include "physics/nuclear.hpp"

TEST(NuclearTest, CrossSectionPositive) {
    EXPECT_GT(Sigma_total(10.0f), 0);
    EXPECT_GT(Sigma_total(50.0f), 0);
    EXPECT_GT(Sigma_total(100.0f), 0);
    EXPECT_GT(Sigma_total(200.0f), 0);
}

TEST(NuclearTest, CrossSectionDecreasesWithEnergy) {
    float sigma_10 = Sigma_total(10.0f);
    float sigma_50 = Sigma_total(50.0f);
    float sigma_100 = Sigma_total(100.0f);

    EXPECT_GE(sigma_10, sigma_50);  // More interaction at low E
    EXPECT_GE(sigma_50, sigma_100);
}

TEST(NuclearTest, SurvivalProbabilityValid) {
    float E = 150.0f;
    float ds = 1.0f;

    float surv = survival_probability(E, ds);

    EXPECT_GT(surv, 0.0f);
    EXPECT_LE(surv, 1.0f);
    EXPECT_LT(surv, 1.0f);  // Should decrease slightly
}

TEST(NuclearTest, WeightConservation) {
    float w_old = 1.0f;
    float E = 100.0f;
    float ds = 1.0f;

    auto result = apply_nuclear_attenuation(w_old, E, ds);
    float w_new = result.first;
    float w_removed = result.second;

    EXPECT_NEAR(w_new + w_removed, w_old, 1e-6f);
    EXPECT_GT(w_new, 0);
    EXPECT_LT(w_new, w_old);
    EXPECT_GT(w_removed, 0);
}

TEST(NuclearTest, EnergyTracking) {
    float w_removed = 0.01f;
    float E = 100.0f;

    float E_removed = w_removed * E;
    EXPECT_NEAR(E_removed, 1.0f, 1e-3f);
}

TEST(NuclearTest, SmallStepSmallEffect) {
    float ds = 0.001f;
    float E = 100.0f;

    float surv = survival_probability(E, ds);

    // For very small step, survival should be close to 1
    EXPECT_GT(surv, 0.999f);
}
```

### GREEN - Implementation

Create `include/physics/nuclear.hpp`:

```cpp
#pragma once

#include <cmath>

// Total nuclear cross-section [mm^-1]
float Sigma_total(float E_MeV);

// Survival probability after one step
float survival_probability(float E, float ds);

// Apply nuclear attenuation
// Returns (w_new, w_removed)
std::pair<float, float> apply_nuclear_attenuation(float w, float E, float ds);
```

Create `src/physics/nuclear.cpp`:

```cpp
#include "physics/nuclear.hpp"

float Sigma_total(float E_MeV) {
    // Energy-dependent cross-section
    if (E_MeV > 100.0f) return 0.0050f;  // mm^-1
    if (E_MeV > 50.0f)  return 0.0060f;
    return 0.0075f;
}

float survival_probability(float E, float ds) {
    return expf(-Sigma_total(E) * ds);
}

std::pair<float, float> apply_nuclear_attenuation(float w, float E, float ds) {
    float surv = survival_probability(E, ds);
    float w_new = w * surv;
    float w_removed = w - w_new;
    return {w_new, w_removed};
}
```

---

## TDD Cycle 3.5: Two-Bin Energy Discretization

### RED - Write Tests First

Create `tests/physics/test_2bin.cpp`:

```cpp
#include <gtest/gtest.h>
#include "physics/discretization.hpp"
#include "core/grids.hpp"

TEST(TwoBinTest, WeightConservation) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    float E = 150.0f, w = 1.0f;
    int bin_low, bin_high;
    float w_low, w_high;

    discretize_energy_2bin(grid, E, w, bin_low, bin_high, w_low, w_high);

    EXPECT_NEAR(w_low + w_high, w, 1e-6f);
}

TEST(TwoBinTest, BinsAdjacent) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    float E = 150.0f, w = 1.0f;
    int bin_low, bin_high;
    float w_low, w_high;

    discretize_energy_2bin(grid, E, w, bin_low, bin_high, w_low, w_high);

    EXPECT_GE(bin_low, 0);
    EXPECT_LT(bin_low, 256);
    EXPECT_GE(bin_high, 0);
    EXPECT_LT(bin_high, 256);

    // Bins should be adjacent or equal
    EXPECT_LE(abs(bin_high - bin_low), 1);
}

TEST(TwoBinTest, EnergyInBinRange) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    float E = 150.0f, w = 1.0f;
    int bin_low, bin_high;
    float w_low, w_high;

    discretize_energy_2bin(grid, E, w, bin_low, bin_high, w_low, w_high);

    // E should be between bin edges of low and high
    EXPECT_GE(E, grid.edges[bin_low]);
    EXPECT_LE(E, grid.edges[bin_high + 1]);
}

TEST(TwoBinTest, EdgeCases) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    // Below E_min
    int bin_low, bin_high;
    float w_low, w_high;

    discretize_energy_2bin(grid, 0.05f, 1.0f, bin_low, bin_high, w_low, w_high);
    EXPECT_EQ(bin_low, bin_high);  // Should clamp to same bin
    EXPECT_NEAR(w_low, 1.0f, 1e-6f);
    EXPECT_NEAR(w_high, 0.0f, 1e-6f);

    // Near E_max
    discretize_energy_2bin(grid, 249.9f, 1.0f, bin_low, bin_high, w_low, w_high);
    EXPECT_GE(bin_low, 0);
    EXPECT_LT(bin_low, 256);
}

TEST(TwoBinTest, MidpointSplit) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    // Find a bin midpoint
    int test_bin = 128;
    float E_test = sqrtf(grid.edges[test_bin] * grid.edges[test_bin + 1]);

    int bin_low, bin_high;
    float w_low, w_high;

    discretize_energy_2bin(grid, E_test, 1.0f, bin_low, bin_high, w_low, w_high);

    // At geometric mean, should split evenly
    EXPECT_NEAR(w_low, w_high, 0.01f);
}
```

### GREEN - Implementation

Create `include/physics/discretization.hpp`:

```cpp
#pragma once

#include "core/grids.hpp"
#include <cmath>

// Two-bin energy discretization
// Splits weight across adjacent energy bins
// Uses bin edges, not uniform dE assumption
void discretize_energy_2bin(
    const EnergyGrid& grid,
    float E, float w,
    int& bin_low, int& bin_high,
    float& w_low, float& w_high
);
```

Create `src/physics/discretization.cpp`:

```cpp
#include "physics/discretization.hpp"
#include <algorithm>

void discretize_energy_2bin(
    const EnergyGrid& grid,
    float E, float w,
    int& bin_low, int& bin_high,
    float& w_low, float& w_high
) {
    // Find interval: E_edges[i] <= E < E_edges[i+1]
    bin_low = grid.FindBin(E);
    bin_high = bin_low + 1;

    // Edge case: E near or above E_max
    if (bin_high >= grid.N_E) {
        bin_high = bin_low;
        w_low = w;
        w_high = 0.0f;
        return;
    }

    // Edge case: E below E_min
    if (bin_low < 0) {
        bin_low = 0;
        bin_high = 0;
        w_low = w;
        w_high = 0.0f;
        return;
    }

    // Linear interpolation factor
    float E_low = grid.edges[bin_low];
    float E_high = grid.edges[bin_high];
    float t = (E - E_low) / (E_high - E_low);
    t = fmaxf(0.0f, fminf(t, 1.0f));

    w_low = w * (1.0f - t);
    w_high = w * t;
}
```

---

## TDD Cycle 3.6: Angular Quadrature

### RED - Write Tests First

Create `tests/physics/test_quadrature.cpp`:

```cpp
#include <gtest/gtest.h>
#include "physics/quadrature.hpp"

TEST(QuadratureTest, WeightsSumToOne) {
    float sum = 0.0f;
    for (int i = 0; i < 7; ++i) {
        sum += g7_weights[i];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(QuadratureTest, OffsetsSymmetric) {
    // Offsets should be symmetric around 0
    EXPECT_NEAR(g7_offsets[0], -g7_offsets[6], 1e-6f);
    EXPECT_NEAR(g7_offsets[1], -g7_offsets[5], 1e-6f);
    EXPECT_NEAR(g7_offsets[2], -g7_offsets[4], 1e-6f);
    EXPECT_NEAR(g7_offsets[3], 0.0f, 1e-6f);
}

TEST(QuadratureTest, WeightsSymmetric) {
    EXPECT_NEAR(g7_weights[0], g7_weights[6], 1e-6f);
    EXPECT_NEAR(g7_weights[1], g7_weights[5], 1e-6f);
    EXPECT_NEAR(g7_weights[2], g7_weights[4], 1e-6f);
}

TEST(QuadratureTest, ApplyAngularSplit) {
    float theta0 = 0.0f;
    float sigma = 0.05f;
    float w = 1.0f;

    float theta_out[7];
    float w_out[7];

    apply_angular_quadrature(theta0, sigma, w, theta_out, w_out);

    float w_sum = 0.0f;
    for (int i = 0; i < 7; ++i) {
        w_sum += w_out[i];
    }

    EXPECT_NEAR(w_sum, w, 1e-6f);
}

TEST(QuadratureTest, AnglesClamped) {
    float theta0 = 1.5f;  // Near PI/2
    float sigma = 0.1f;
    float w = 1.0f;

    float theta_out[7];
    float w_out[7];

    apply_angular_quadrature(theta0, sigma, w, theta_out, w_out);

    for (int i = 0; i < 7; ++i) {
        EXPECT_GE(theta_out[i], -M_PI/2);
        EXPECT_LE(theta_out[i], M_PI/2);
    }
}
```

### GREEN - Implementation

Create `include/physics/quadrature.hpp`:

```cpp
#pragma once

#include <cmath>

// 7-point Gaussian quadrature for angular split
// Offsets in units of sigma
constexpr float g7_offsets[7] = {
    -3.0f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 3.0f
};

constexpr float g7_weights[7] = {
    0.05f, 0.10f, 0.20f, 0.30f, 0.20f, 0.10f, 0.05f
};

// Apply angular quadrature
// theta0: center angle [rad]
// sigma: scattering width [rad]
// w: input weight
// theta_out: output angles (size 7)
// w_out: output weights (size 7)
void apply_angular_quadrature(
    float theta0, float sigma, float w,
    float* theta_out, float* w_out
);
```

Create `src/physics/quadrature.cpp`:

```cpp
#include "physics/quadrature.hpp"

void apply_angular_quadrature(
    float theta0, float sigma, float w,
    float* theta_out, float* w_out
) {
    for (int i = 0; i < 7; ++i) {
        float theta_new = theta0 + g7_offsets[i] * sigma;
        // Clamp to valid range
        theta_new = fmaxf(-M_PI/2, fminf(M_PI/2, theta_new));

        theta_out[i] = theta_new;
        w_out[i] = w * g7_weights[i];
    }
}
```

---

## Exit Criteria Checklist

- [ ] All physics unit tests pass
- [ ] R-based steps are ≤2% of remaining range
- [ ] Highland sigma scales correctly with E and ds
- [ ] Variance accumulates as sum of squares (RMS-based)
- [ ] Two-bin discretization conserves weight
- [ ] Nuclear attenuation tracks both weight and energy
- [ ] Angular quadrature weights sum to 1.0

---

## Next Steps

After completing Phase 3, proceed to **Phase 4 (Transport Pipeline K1-K6)**.

```bash
# Test all physics
./bin/sm2d_tests --gtest_filter="*Physics*"
```
