# Phase 1: LUT Generation and Validation

**Status**: Pending
**Duration**: 1-2 days
**Dependencies**: Phase 0

---

## Objectives

1. Download and process NIST PSTAR data for water
2. Generate R(E) and E(R) lookup tables
3. Validate against known reference points
4. Implement log-space interpolation
5. Create binary LUT storage with checksums

---

## TDD Cycle 1.1: NIST Data Download

### RED - Write Tests First

Create `tests/lut/test_nist_download.cpp`:

```cpp
#include <gtest/gtest.h>
#include "lut/nist_loader.hpp"

TEST(NistDataTest, DownloadFileExists) {
    std::string path = "data/nist/pstar_water.txt";
    EXPECT_TRUE(FileExists(path));
}

TEST(NistDataTest, FileFormatValid) {
    std::string path = "data/nist/pstar_water.txt";
    auto data = LoadNistData(path);

    // NIST PSTAR should have >100 data points
    EXPECT_GT(data.size(), 100);

    // Check first row has reasonable values
    EXPECT_GT(data[0].energy_MeV, 0);
    EXPECT_GT(data[0].csda_range_g_cm2, 0);
    EXPECT_GT(data[0].stopping_power, 0);
}

TEST(NistDataTest, DataMonotonic) {
    auto data = LoadNistData("data/nist/pstar_water.txt");

    // Energy should be monotonically increasing
    for (size_t i = 1; i < data.size(); ++i) {
        EXPECT_GT(data[i].energy_MeV, data[i-1].energy_MeV);
    }

    // Range should be monotonically increasing
    for (size_t i = 1; i < data.size(); ++i) {
        EXPECT_GT(data[i].csda_range_g_cm2, data[i-1].csda_range_g_cm2);
    }
}
```

### GREEN - Implementation

Create `scripts/lut_gen/download_nist.py`:

```python
#!/usr/bin/env python3
"""
Download NIST PSTAR data for water.

NIST PSTAR URL: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
"""

import requests
import re
from pathlib import Path

def download_pstar_water():
    """Download PSTAR data for water from NIST."""

    # For water, we can use the pre-formatted data
    # In production, scrape from NIST or use their downloadable files

    url = "https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html"

    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; SM_2D/1.0; +https://github.com/sm2d)'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Parse the table (simplified - actual implementation needs proper parsing)
    # For now, we'll use the downloadable PSTAR files

    # Alternative: Use NIST's downloadable PSTAR files
    data_url = "https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.txt"

    data_response = requests.get(data_url, headers=headers)

    output_path = Path("data/nist/pstar_water.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(data_response.text)

    print(f"Downloaded to {output_path}")

if __name__ == "__main__":
    download_pstar_water()
```

Create `include/lut/nist_loader.hpp`:

```cpp
#pragma once

#include <string>
#include <vector>

struct NistDataRow {
    float energy_MeV;           // Kinetic energy [MeV]
    float stopping_power;       // dE/dx [MeV cm²/g]
    float csda_range_g_cm2;     // CSDA range [g/cm²]
};

std::vector<NistDataRow> LoadNistData(const std::string& path);
bool FileExists(const std::string& path);
```

---

## TDD Cycle 1.2: Energy Grid Generation

### RED - Write Tests First

Create `tests/lut/test_energy_grid.cpp`:

```cpp
#include <gtest/gtest.h>
#include "lut/energy_grid.hpp"

TEST(EnergyGridTest, BinEdgesCorrect) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    EXPECT_EQ(grid.edges.size(), 257);  // N_E + 1
    EXPECT_NEAR(grid.edges[0], 0.1f, 1e-6f);
    EXPECT_NEAR(grid.edges[256], 250.0f, 1e-4f);
}

TEST(EnergyGridTest, LogSpacing) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    // Check that ratio between consecutive bins is approximately constant
    float ratio = grid.edges[1] / grid.edges[0];

    for (size_t i = 2; i < 10; ++i) {  // Check first 10
        float current_ratio = grid.edges[i] / grid.edges[i-1];
        EXPECT_NEAR(current_ratio, ratio, 0.01f);
    }
}

TEST(EnergyGridTest, RepresentativeEnergyGeometricMean) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    // E_rep[i] should be sqrt(E_edges[i] * E_edges[i+1])
    for (int i = 0; i < 256; ++i) {
        float expected = sqrtf(grid.edges[i] * grid.edges[i+1]);
        EXPECT_NEAR(grid.rep[i], expected, 1e-5f * expected);
    }
}
```

### GREEN - Implementation

Create `include/lut/energy_grid.hpp`:

```cpp
#pragma once

#include <vector>
#include <cmath>

struct EnergyGrid {
    const int N_E;
    const float E_min;
    const float E_max;
    std::vector<float> edges;   // N_E + 1 values
    std::vector<float> rep;     // N_E representative values

    EnergyGrid(float E_min, float E_max, int N_E);

    // Find bin index for given energy
    int FindBin(float E) const;

    // Get representative energy for bin
    float GetRepEnergy(int bin) const;
};
```

Create `src/lut/energy_grid.cpp`:

```cpp
#include "lut/energy_grid.hpp"
#include <algorithm>

EnergyGrid::EnergyGrid(float E_min, float E_max, int N_E)
    : N_E(N_E), E_min(E_min), E_max(E_max)
{
    edges.resize(N_E + 1);
    rep.resize(N_E);

    // Generate log-spaced bin edges
    float log_E_min = logf(E_min);
    float log_E_max = logf(E_max);
    float delta_log = (log_E_max - log_E_min) / N_E;

    for (int i = 0; i <= N_E; ++i) {
        edges[i] = expf(log_E_min + i * delta_log);
    }

    // Representative energy = geometric mean
    for (int i = 0; i < N_E; ++i) {
        rep[i] = sqrtf(edges[i] * edges[i+1]);
    }
}

int EnergyGrid::FindBin(float E) const {
    if (E < edges[0]) return 0;
    if (E >= edges[N_E]) return N_E - 1;

    // Binary search
    int lo = 0, hi = N_E;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (edges[mid + 1] <= E) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

float EnergyGrid::GetRepEnergy(int bin) const {
    if (bin < 0) bin = 0;
    if (bin >= N_E) bin = N_E - 1;
    return rep[bin];
}
```

---

## TDD Cycle 1.3: R(E) Lookup Table

### RED - Write Tests First

Create `tests/lut/test_r_lookup.cpp`:

```cpp
#include <gtest/gtest.h>
#include "lut/r_lut.hpp"

TEST(RLUTTest, R_EnergyMonotonic) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    for (size_t i = 1; i < lut.R.size(); ++i) {
        EXPECT_GT(lut.R[i], lut.R[i-1]);
    }
}

TEST(RLUTTest, R_Positive) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    for (size_t i = 0; i < lut.R.size(); ++i) {
        EXPECT_GT(lut.R[i], 0);
    }
}
```

### GREEN - Implementation

Create `include/lut/r_lut.hpp`:

```cpp
#pragma once

#include "lut/energy_grid.hpp"
#include <vector>

struct RLUT {
    EnergyGrid grid;
    std::vector<float> R;         // CSDA range [mm]
    std::vector<float> log_E;     // Pre-computed log(E)
    std::vector<float> log_R;     // Pre-computed log(R)

    // Lookup R(E) using log-log interpolation
    float lookup_R(float E) const;

    // Inverse lookup: E from R
    float lookup_E_inverse(float R) const;
};

// Generate LUT from NIST data
RLUT GenerateRLUT(float E_min, float E_max, int N_E);
```

---

## TDD Cycle 1.4: Reference Validation

### RED - Write Tests First

Create `tests/lut/test_reference_validation.cpp`:

```cpp
#include <gtest/gtest.h>
#include "lut/r_lut.hpp"

TEST(LUTValidation, R_150MeV_WithinTolerance) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    float R = lut.lookup_R(150.0f);
    // NIST: ~158 mm for 150 MeV in water
    float R_expected = 158.0f;

    float error = fabsf(R - R_expected) / R_expected;
    EXPECT_LT(error, 0.013f);  // ±1.3%
}

TEST(LUTValidation, R_70MeV_WithinTolerance) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    float R = lut.lookup_R(70.0f);
    // NIST: ~40.8 mm for 70 MeV in water
    float R_expected = 40.8f;

    float error = fabsf(R - R_expected) / R_expected;
    EXPECT_LT(error, 0.012f);  // ±1.2%
}

TEST(LUTValidation, R_10MeV_WithinTolerance) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    float R = lut.lookup_R(10.0f);
    // NIST: ~1.2 mm for 10 MeV in water
    float R_expected = 1.2f;

    float error = fabsf(R - R_expected) / R_expected;
    EXPECT_LT(error, 0.05f);  // ±5% (more tolerance at low E)
}

TEST(LUTValidation, InverseConsistency) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);

    std::vector<float> test_energies = {1.0f, 10.0f, 50.0f, 100.0f, 150.0f, 200.0f};

    for (float E_test : test_energies) {
        float R = lut.lookup_R(E_test);
        float E_recovered = lut.lookup_E_inverse(R);

        float error = fabsf(E_recovered - E_test) / E_test;
        EXPECT_LT(error, 0.001f) << "Failed at E=" << E_test;
    }
}
```

### GREEN - Implementation

Create `src/lut/r_lut.cpp`:

```cpp
#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include <cmath>
#include <algorithm>

RLUT GenerateRLUT(float E_min, float E_max, int N_E) {
    RLUT lut;
    lut.grid = EnergyGrid(E_min, E_max, N_E);
    lut.R.resize(N_E);
    lut.log_E.resize(N_E);
    lut.log_R.resize(N_E);

    // Load NIST data
    auto nist_data = LoadNistData("data/nist/pstar_water.txt");

    // Convert NIST range from g/cm² to mm (water density = 1.0 g/cm³)
    const float rho = 1.0f;  // g/cm³
    const float g_cm2_to_mm = 10.0f / rho;  // 1 g/cm² = 10 mm

    // Generate LUT by interpolating NIST data
    for (int i = 0; i < N_E; ++i) {
        float E = lut.grid.rep[i];

        // Find surrounding NIST data points
        auto it = std::lower_bound(
            nist_data.begin(), nist_data.end(),
            E,
            [](const NistDataRow& row, float val) {
                return row.energy_MeV < val;
            }
        );

        if (it == nist_data.begin()) {
            lut.R[i] = nist_data[0].csda_range_g_cm2 * g_cm2_to_mm;
        } else if (it == nist_data.end()) {
            lut.R[i] = nist_data.back().csda_range_g_cm2 * g_cm2_to_mm;
        } else {
            // Linear interpolation in log-log space
            float E0 = (it - 1)->energy_MeV;
            float E1 = it->energy_MeV;
            float R0 = (it - 1)->csda_range_g_cm2 * g_cm2_to_mm;
            float R1 = it->csda_range_g_cm2 * g_cm2_to_mm;

            float log_R = logf(R0) + (logf(R1) - logf(R0)) *
                         (logf(E) - logf(E0)) / (logf(E1) - logf(E0));
            lut.R[i] = expf(log_R);
        }

        lut.log_E[i] = logf(lut.grid.rep[i]);
        lut.log_R[i] = logf(lut.R[i]);
    }

    return lut;
}

float RLUT::lookup_R(float E) const {
    // Clamp to grid bounds
    float E_clamped = fmaxf(E_min, fminf(E, E_max));

    // Find bin
    int bin = grid.FindBin(E_clamped);

    // Log-log interpolation
    float log_E = logf(E_clamped);
    float log_E0 = log_E[bin];
    float log_E1 = log_E[std::min(bin + 1, grid.N_E - 1)];
    float log_R0 = log_R[bin];
    float log_R1 = log_R[std::min(bin + 1, grid.N_E - 1)];

    float log_R = log_R0 + (log_R1 - log_R0) * (log_E - log_E0) / (log_E1 - log_E0);

    return expf(log_R);
}

float RLUT::lookup_E_inverse(float R) const {
    // Find bin by searching in R space
    auto it = std::lower_bound(R.begin(), R.end(), R);
    int bin = std::max(0, std::min(static_cast<int>(it - R.begin()), grid.N_E - 2));

    // Log-log interpolation
    float log_R = logf(R);
    float log_R0 = log_R[bin];
    float log_R1 = log_R[bin + 1];
    float log_E0 = log_E[bin];
    float log_E1 = log_E[bin + 1];

    float log_E = log_E0 + (log_E1 - log_E0) * (log_R - log_R0) / (log_R1 - log_R0);

    return expf(log_E);
}
```

---

## TDD Cycle 1.5: Binary Storage

### RED - Write Tests First

Create `tests/lut/test_lut_storage.cpp`:

```cpp
#include <gtest/gtest.h>
#include "lut/r_lut.hpp"

TEST(LUTStorage, SaveAndLoad) {
    RLUT lut_orig = GenerateRLUT(0.1f, 250.0f, 256);

    SaveLUT("data/lut/water_R_lut.bin", lut_orig);

    RLUT lut_loaded = LoadLUT("data/lut/water_R_lut.bin");

    // Check values match
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(lut_loaded.R[i], lut_orig.R[i], 1e-5f);
    }
}

TEST(LUTStorage, ChecksumValidation) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);
    SaveLUT("data/lut/water_R_lut.bin", lut);

    // Corrupt the file
    FILE* f = fopen("data/lut/water_R_lut.bin", "r+");
    if (f) {
        fseek(f, 100, SEEK_SET);
        uint8_t corrupt = 0xFF;
        fwrite(&corrupt, 1, 1, f);
        fclose(f);
    }

    // Should detect corruption
    EXPECT_THROW(LoadLUT("data/lut/water_R_lut.bin"), std::runtime_error);
}
```

### GREEN - Implementation

Create `include/lut/lut_storage.hpp`:

```cpp
#pragma once

#include "lut/r_lut.hpp"
#include <string>

void SaveLUT(const std::string& path, const RLUT& lut);
RLUT LoadLUT(const std::string& path);
```

---

## Exit Criteria Checklist

- [ ] All NIST reference tests pass (R(150), R(70), R(10))
- [ ] Inverse lookup consistency: |E - E'|/E < 0.1%
- [ ] LUT file saves and loads correctly
- [ ] Checksum validation detects corruption
- [ ] Unit test coverage > 95% for LUT module
- [ ] Generated LUT file in `data/lut/water_R_lut.bin`

---

## Next Steps

After completing Phase 1, proceed to **Phase 3 (Physics Models)**. Phase 2 (Data Structures) can be developed in parallel.

```bash
# Validate LUT
./bin/sm2d_tests --gtest_filter="*LUT*"

# Verify reference values
./bin/sm2d_tests --gtest_filter="*Validation*"
```
