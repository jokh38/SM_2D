# Phase 7: Physics Validation Against NIST

**Status**: Pending
**Duration**: 3-4 days
**Dependencies**: All previous phases

---

## Objectives

1. Run pencil beam simulations
2. Compare Bragg peak position to NIST
3. Compare lateral spread (sigma_x) to Fermi-Eyges
4. Validate distal falloff behavior
5. Verify determinism (run-to-run consistency)
6. Create validation report

---

## NIST Reference Values (Water)

| Energy | CSDA Range | Tolerance |
|--------|------------|-----------|
| 150 MeV | ~158 mm | ±2% |
| 70 MeV | ~40.8 mm | ±2% |
| 10 MeV | ~1.2 mm | ±5% |

---

## TDD Cycle 7.1: Pencil Beam Simulation

### RED - Write Tests First

Create `tests/validation/test_pencil_beam.cpp`:

```cpp
#include <gtest/gtest.h>
#include "validation/pencil_beam.hpp"

TEST(PencilBeam, SimulationRuns) {
    PencilBeamConfig config;
    config.E0 = 150.0f;  // MeV
    config.x0 = 0.0f;
    config.z0 = 0.0f;
    config.theta0 = 0.0f;
    config.Nx = 100;
    config.Nz = 200;
    config.dx = 1.0f;  // mm
    config.dz = 1.0f;  // mm
    config.max_steps = 100;

    auto result = run_pencil_beam(config);

    EXPECT_GT(result.edep.size(), 0);
    EXPECT_EQ(result.Nx, 100);
    EXPECT_EQ(result.Nz, 200);
}

TEST(PencilBeam, EnergyDeposited) {
    PencilBeamConfig config;
    config.E0 = 100.0f;
    config.Nx = 50;
    config.Nz = 150;

    auto result = run_pencil_beam(config);

    double total_edep = 0;
    for (const auto& row : result.edep) {
        for (double val : row) {
            total_edep += val;
        }
    }

    EXPECT_GT(total_edep, 0);
    EXPECT_LT(total_edep, config.E0 * 1.1);  // Should be close to E0
}

TEST(PencilBeam, BraggPeakExists) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 100;
    config.Nz = 200;

    auto result = run_pencil_beam(config);

    // Find depth of maximum dose
    int z_peak = find_bragg_peak_z(result);

    EXPECT_GT(z_peak, 100);  // Should be in second half
    EXPECT_LT(z_peak, 200);
}
```

### GREEN - Implementation

Create `include/validation/pencil_beam.hpp`:

```cpp
#pragma once

#include "core/grids.hpp"
#include "core/psi_storage.hpp"
#include <vector>

struct PencilBeamConfig {
    float E0 = 150.0f;      // Initial energy [MeV]
    float x0 = 0.0f;        // Initial x [mm]
    float z0 = 0.0f;        // Initial z [mm]
    float theta0 = 0.0f;    // Initial angle [rad]
    int Nx = 100;           // Number of x cells
    int Nz = 200;           // Number of z cells
    float dx = 1.0f;        // Cell size [mm]
    float dz = 1.0f;
    int max_steps = 100;    // Max transport steps
    float W_total = 1.0f;   // Total weight
};

struct SimulationResult {
    int Nx, Nz;
    float dx, dz;
    std::vector<std::vector<double>> edep;  // [Nx][Nz]
    std::vector<float> x_centers;
    std::vector<float> z_centers;
};

// Run pencil beam simulation
SimulationResult run_pencil_beam(const PencilBeamConfig& config);

// Find z-index of Bragg peak (maximum dose on central axis)
int find_bragg_peak_z(const SimulationResult& result);

// Get depth-dose curve on central axis
std::vector<double> get_depth_dose(const SimulationResult& result);
```

---

## TDD Cycle 7.2: Bragg Peak Position

### RED - Write Tests First

Create `tests/validation/test_bragg_peak.cpp`:

```cpp
#include <gtest/gtest.h>
#include "validation/bragg_peak.hpp"

TEST(BraggPeak, Position_150MeV) {
    // NIST reference: R(150) ≈ 158 mm in water
    float R_nist = 158.0f;

    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nz = 200;
    config.dz = 1.0f;

    auto result = run_pencil_beam(config);
    float z_peak_mm = find_bragg_peak_position_mm(result);

    float error = fabsf(z_peak_mm - R_nist) / R_nist;

    EXPECT_LT(error, 0.02f);  // ±2% tolerance
}

TEST(BraggPeak, Position_70MeV) {
    float R_nist = 40.8f;

    PencilBeamConfig config;
    config.E0 = 70.0f;
    config.Nz = 80;
    config.dz = 1.0f;

    auto result = run_pencil_beam(config);
    float z_peak_mm = find_bragg_peak_position_mm(result);

    float error = fabsf(z_peak_mm - R_nist) / R_nist;

    EXPECT_LT(error, 0.02f);  // ±2% tolerance
}

TEST(BraggPeak, Position_10MeV) {
    float R_nist = 1.2f;

    PencilBeamConfig config;
    config.E0 = 10.0f;
    config.Nz = 20;
    config.dz = 0.2f;

    auto result = run_pencil_beam(config);
    float z_peak_mm = find_bragg_peak_position_mm(result);

    float error = fabsf(z_peak_mm - R_nist) / R_nist;

    EXPECT_LT(error, 0.05f);  // ±5% tolerance (more at low E)
}

TEST(BraggPeak, PeakIsNarrow) {
    // Bragg peak should be narrow (FWHM small)
    PencilBeamConfig config;
    config.E0 = 150.0f;

    auto result = run_pencil_beam(config);

    float fwhm_mm = compute_bragg_peak_fwhm(result);

    // FWHM should be small (< 10 mm for 150 MeV)
    EXPECT_LT(fwhm_mm, 10.0f);
}
```

### GREEN - Implementation

Create `include/validation/bragg_peak.hpp`:

```cpp
#pragma once

#include "validation/pencil_beam.hpp"

// Find Bragg peak position in mm
float find_bragg_peak_position_mm(const SimulationResult& result);

// Compute FWHM of Bragg peak [mm]
float compute_bragg_peak_fwhm(const SimulationResult& result);

// Find R80 (depth at 80% distal to peak)
float find_R80(const SimulationResult& result);

// Find R20 (depth at 20% distal to peak)
float find_R20(const SimulationResult& result);

// Compute distal falloff (R80 - R20)
float compute_distal_falloff(const SimulationResult& result);
```

---

## TDD Cycle 7.3: Lateral Spread

### RED - Write Tests First

Create `tests/validation/test_lateral_spread.cpp`:

```cpp
#include <gtest/gtest.h>
#include "validation/lateral_spread.hpp"

TEST(LateralSpread, SigmaAtMidRange) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 100;
    config.Nz = 200;
    config.dx = 1.0f;

    auto result = run_pencil_beam(config);

    // Find mid-range depth
    float R_peak = find_bragg_peak_position_mm(result);
    float z_mid = R_peak / 2.0f;

    float sigma_sim = get_lateral_sigma_at_z(result, z_mid);

    // Compare to Fermi-Eyges
    float sigma_fe = compute_fermi_eyges_sigma(150.0f, z_mid);

    float error = fabsf(sigma_sim - sigma_fe) / sigma_fe;

    EXPECT_LT(error, 0.15f);  // ±15% tolerance
}

TEST(LateralSpread, SigmaIncreasesWithDepth) {
    PencilBeamConfig config;
    config.E0 = 150.0f;

    auto result = run_pencil_beam(config);

    float sigma_shallow = get_lateral_sigma_at_z(result, 20.0f);
    float sigma_deep = get_lateral_sigma_at_z(result, 100.0f);

    EXPECT_GT(sigma_deep, sigma_shallow);
}

TEST(LateralSpread, FermiEygesFormula) {
    // Test Fermi-Eyges calculation itself
    float sigma_50mm = compute_fermi_eyges_sigma(150.0f, 50.0f);
    float sigma_100mm = compute_fermi_eyges_sigma(150.0f, 100.0f);

    // Sigma should increase with depth (approximately sqrt(z))
    float ratio = sigma_100mm / sigma_50mm;
    float expected_ratio = sqrtf(100.0f / 50.0f);

    EXPECT_NEAR(ratio, expected_ratio, 0.1f);
}
```

### GREEN - Implementation

Create `include/validation/lateral_spread.hpp`:

```cpp
#pragma once

#include "validation/pencil_beam.hpp"
#include <vector>

// Compute lateral sigma (RMS spread) at given z depth
float get_lateral_sigma_at_z(const SimulationResult& result, float z_mm);

// Get lateral profile at given z
std::vector<double> get_lateral_profile(const SimulationResult& result, float z_mm);

// Compute Fermi-Eyges prediction for lateral sigma
float compute_fermi_eyges_sigma(float E0_MeV, float z_mm);

// Fit Gaussian to lateral profile, return sigma
float fit_gaussian_sigma(const std::vector<double>& profile, float dx);
```

Create `src/validation/lateral_spread.cpp`:

```cpp
#include "validation/lateral_spread.hpp"
#include "physics/highland.hpp"
#include <cmath>

float compute_fermi_eyges_sigma(float E0_MeV, float z_mm) {
    // Fermi-Eyges theory for proton lateral spread
    // σ_x²(z) ≈ ∫₀ᶻ (z-z')² × T(z') dz'
    // where T = (σ_θ/Δs)²

    // Simplified: use Highland formula integrated
    // This is an approximation - full Fermi-Eyges requires numerical integration

    const float X0 = X0_water;
    const float m_p = 938.272f;

    // Average scattering power
    float beta = sqrtf(1.0f - powf(m_p / (E0_MeV + m_p), 2));
    float p = sqrtf(powf(E0_MeV + m_p, 2) - m_p * m_p);

    // Highland-based scattering angle per mm
    float theta0 = (13.6f / (beta * p)) * sqrtf(1.0f / X0);

    // Fermi-Eyges: σ_x ≈ θ0 × z × sqrt(z / 3X0) for small angles
    float sigma_x = theta0 * z_mm * sqrtf(z_mm / (3.0f * X0));

    return sigma_x;
}

float get_lateral_sigma_at_z(const SimulationResult& result, float z_mm) {
    // Find z index
    int iz = static_cast<int>(z_mm / result.dz);
    if (iz < 0) iz = 0;
    if (iz >= result.Nz) iz = result.Nz - 1;

    // Get lateral profile at this z
    std::vector<double> profile(result.Nx);
    double total = 0;

    for (int ix = 0; ix < result.Nx; ++ix) {
        profile[ix] = result.edep[ix][iz];
        total += profile[ix];
    }

    if (total < 1e-10) return 0;

    // Compute centroid
    double x_centroid = 0;
    for (int ix = 0; ix < result.Nx; ++ix) {
        x_centroid += result.x_centers[ix] * profile[ix];
    }
    x_centroid /= total;

    // Compute sigma
    double sigma2 = 0;
    for (int ix = 0; ix < result.Nx; ++ix) {
        double dx = result.x_centers[ix] - x_centroid;
        sigma2 += profile[ix] * dx * dx;
    }
    sigma2 /= total;

    return sqrt(sigma2);
}
```

---

## TDD Cycle 7.4: Distal Falloff

### RED - Write Tests First

Create `tests/validation/test_distal_falloff.cpp`:

```cpp
#include <gtest/gtest.h>
#include "validation/distal_falloff.hpp"

TEST(DistalFalloff, R80_R20_Computable) {
    PencilBeamConfig config;
    config.E0 = 150.0f;

    auto result = run_pencil_beam(config);

    float R80 = find_R80(result);
    float R20 = find_R20(result);

    EXPECT_GT(R80, 140);
    EXPECT_LT(R80, 170);

    EXPECT_GT(R20, R80);
    EXPECT_LT(R20, 180);
}

TEST(DistalFalloff, FalloffWidthReasonable) {
    PencilBeamConfig config;
    config.E0 = 150.0f;

    auto result = run_pencil_beam(config);

    float falloff = compute_distal_falloff(result);

    // Expected ~4-6 mm for 150 MeV (without straggling)
    EXPECT_GT(falloff, 2.0f);
    EXPECT_LT(falloff, 10.0f);
}

TEST(DistalFalloff, EnergyDependence) {
    PencilBeamConfig config_high;
    config_high.E0 = 150.0f;
    auto result_high = run_pencil_beam(config_high);
    float falloff_high = compute_distal_falloff(result_high);

    PencilBeamConfig config_low;
    config_low.E0 = 70.0f;
    auto result_low = run_pencil_beam(config_low);
    float falloff_low = compute_distal_falloff(result_low);

    // Lower energy should have narrower falloff
    EXPECT_LT(falloff_low, falloff_high);
}
```

---

## TDD Cycle 7.5: Determinism

### RED - Write Tests First

Create `tests/validation/test_determinism.cpp`:

```cpp
#include <gtest/gtest.h>
#include "validation/determinism.hpp"

TEST(DeterminismTest, RunToRunConsistency) {
    PencilBeamConfig config;
    config.E0 = 150.0f;

    auto result1 = run_pencil_beam(config);
    auto result2 = run_pencil_beam(config);

    uint32_t checksum1 = compute_checksum(result1);
    uint32_t checksum2 = compute_checksum(result2);

    EXPECT_EQ(checksum1, checksum2);
}

TEST(DeterminismTest, ReproducibleAcrossSeeds) {
    // For stochastic components, verify seed control
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.random_seed = 42;

    auto result1 = run_pencil_beam(config);

    config.random_seed = 42;  // Same seed
    auto result2 = run_pencil_beam(config);

    uint32_t checksum1 = compute_checksum(result1);
    uint32_t checksum2 = compute_checksum(result2);

    EXPECT_EQ(checksum1, checksum2);
}
```

---

## TDD Cycle 7.6: Validation Report

### RED - Write Tests First

Create `tests/validation/test_report.cpp`:

```cpp
#include <gtest/gtest.h>
#include "validation/validation_report.hpp"

TEST(ValidationReport, GenerateReport) {
    ValidationResults results;

    // Bragg peak results
    results.bragg_150.error = 0.015f;  // 1.5%
    results.bragg_150.pass = true;

    results.bragg_70.error = 0.018f;
    results.bragg_70.pass = true;

    // Lateral spread
    results.lateral.error = 0.12f;  // 12%
    results.lateral.pass = true;

    // Conservation
    results.weight_error = 1e-7f;
    results.energy_error = 1e-6f;

    std::ostringstream oss;
    generate_validation_report(oss, results);

    std::string report = oss.str();

    EXPECT_NE(report.find("PASS"), std::string::npos);
    EXPECT_NE(report.find("Bragg Peak"), std::string::npos);
    EXPECT_NE(report.find("Lateral"), std::string::npos);
}
```

### GREEN - Implementation

Create `include/validation/validation_report.hpp`:

```cpp
#pragma once

#include <iosfwd>

struct BraggPeakResult {
    float energy_MeV;
    float R_sim;
    float R_nist;
    float error;  // relative
    bool pass;
};

struct LateralSpreadResult {
    float z_mm;
    float sigma_sim;
    float sigma_fe;
    float error;  // relative
    bool pass;
};

struct ConservationResult {
    float weight_error;
    float energy_error;
    bool pass;
};

struct ValidationResults {
    BraggPeakResult bragg_150;
    BraggPeakResult bragg_70;
    LateralSpreadResult lateral;
    ConservationResult conservation;
    bool overall_pass;
};

// Generate validation report
void generate_validation_report(std::ostream& os, const ValidationResults& results);

// Run all validation tests
ValidationResults run_full_validation();
```

---

## Exit Criteria Checklist

- [ ] Bragg peak position within ±2% of NIST (150 MeV, 70 MeV)
- [ ] Bragg peak position within ±5% of NIST (10 MeV)
- [ ] Lateral σ at mid-range within ±15% of Fermi-Eyges
- [ ] Distal falloff qualitatively correct (2-10 mm)
- [ ] Determinism verified (checksum match)
- [ ] Weight conservation < 1e-6
- [ ] Energy conservation < 1e-5
- [ ] Validation report generated

---

## Next Steps

After completing Phase 7, proceed to **Phase 8 (Performance Optimization)**.

```bash
# Run validation
./bin/sm2d_validation

# Generate report
./bin/sm2d_validation --report=validation_report.txt
```
