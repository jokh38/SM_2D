# Phase 5: Source and Boundary Conditions

**Status**: Pending
**Duration**: 2-3 days
**Dependencies**: Phase 0, Phase 2, Phase 4

---

## Objectives

1. Implement pencil beam source
2. Implement Gaussian beam source
3. Implement boundary conditions (absorb, reflect)
4. Implement boundary loss tracking
5. Implement neighbor lookup with boundary checks

---

## TDD Cycle 5.1: Pencil Beam Source

### RED - Write Tests First

Create `tests/source/test_pencil.cpp`:

```cpp
#include <gtest/gtest.h>
#include "source/pencil_source.hpp"
#include "core/grids.hpp"

TEST(PencilSourceTest, DefaultValues) {
    PencilSource src;

    EXPECT_FLOAT_EQ(src.x0, 0.0f);
    EXPECT_FLOAT_EQ(src.z0, 0.0f);
    EXPECT_FLOAT_EQ(src.theta0, 0.0f);
    EXPECT_FLOAT_EQ(src.E0, 150.0f);
    EXPECT_FLOAT_EQ(src.W_total, 1.0f);
}

TEST(PencilSourceTest, InjectionCorrectCell) {
    PencilSource src;
    src.x0 = 0.5f;  // Middle of cell 0
    src.z0 = 0.5f;
    src.theta0 = 0.0f;
    src.E0 = 150.0f;
    src.W_total = 1.0f;

    EnergyGrid e_grid(0.1f, 250.0f, 256);
    AngularGrid a_grid(-M_PI/2, M_PI/2, 512);

    PsiC psi(4, 4, 32);

    inject_source(psi, src, e_grid, a_grid);

    // Cell 0 should have weight
    EXPECT_GT(sum_psi(psi, 0), 0);
}

TEST(PencilSourceTest, InjectionEnergyBin) {
    PencilSource src;
    src.E0 = 150.0f;
    src.W_total = 1.0f;

    EnergyGrid e_grid(0.1f, 250.0f, 256);
    AngularGrid a_grid(-M_PI/2, M_PI/2, 512);

    PsiC psi(4, 4, 32);

    inject_source(psi, src, e_grid, a_grid);

    // Find the energy bin
    int E_bin = e_grid.FindBin(150.0f);

    // Check that weight is in correct block
    // Block ID encodes (b_theta, b_E)
    uint32_t expected_b_E = E_bin / N_E_local;

    bool found = false;
    for (int slot = 0; slot < 32; ++slot) {
        uint32_t bid = psi.block_id[0][slot];
        if (bid == EMPTY_BLOCK_ID) continue;

        uint32_t b_theta = bid & 0xFFF;
        uint32_t b_E = (bid >> 12) & 0xFFF;

        if (b_E == expected_b_E) {
            found = true;
            break;
        }
    }

    EXPECT_TRUE(found);
}

TEST(PencilSourceTest, InjectionWeightConserved) {
    PencilSource src;
    src.W_total = 1.0f;

    EnergyGrid e_grid(0.1f, 250.0f, 256);
    AngularGrid a_grid(-M_PI/2, M_PI/2, 512);

    PsiC psi(4, 4, 32);

    inject_source(psi, src, e_grid, a_grid);

    float total = 0;
    for (int cell = 0; cell < 16; ++cell) {
        total += sum_psi(psi, cell);
    }

    EXPECT_NEAR(total, src.W_total, 1e-6f);
}

TEST(PencilSourceTest, InjectionThetaBin) {
    PencilSource src;
    src.theta0 = 0.0f;

    EnergyGrid e_grid(0.1f, 250.0f, 256);
    AngularGrid a_grid(-M_PI/2, M_PI/2, 512);

    PsiC psi(4, 4, 32);

    inject_source(psi, src, e_grid, a_grid);

    // Theta = 0 should be near middle of angular grid
    int theta_bin = a_grid.FindBin(0.0f);
    uint32_t expected_b_theta = theta_bin / N_theta_local;

    bool found = false;
    for (int slot = 0; slot < 32; ++slot) {
        uint32_t bid = psi.block_id[0][slot];
        if (bid == EMPTY_BLOCK_ID) continue;

        uint32_t b_theta = bid & 0xFFF;
        uint32_t b_E = (bid >> 12) & 0xFFF;

        if (b_theta == expected_b_theta) {
            found = true;
            break;
        }
    }

    EXPECT_TRUE(found);
}
```

### GREEN - Implementation

Create `include/source/pencil_source.hpp`:

```cpp
#pragma once

#include "core/grids.hpp"
#include "core/psi_storage.hpp"

struct PencilSource {
    float x0 = 0.0f;      // Entry x position [mm]
    float z0 = 0.0f;      // Entry z position [mm]
    float theta0 = 0.0f;  // Entry angle [rad]
    float E0 = 150.0f;    // Initial energy [MeV]
    float W_total = 1.0f; // Total weight
};

// Inject pencil beam into PsiC
void inject_source(
    PsiC& psi,
    const PencilSource& src,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid
);
```

Create `src/source/pencil_source.cpp`:

```cpp
#include "source/pencil_source.hpp"
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"

void inject_source(
    PsiC& psi,
    const PencilSource& src,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid
) {
    // Find cell for (x0, z0)
    // Assuming cell_size = 1.0 mm for simplicity
    float dx = 1.0f, dz = 1.0f;
    int ix = static_cast<int>(src.x0 / dx);
    int iz = static_cast<int>(src.z0 / dz);
    int cell = ix + iz * psi.Nx;

    if (cell < 0 || cell >= psi.Nx * psi.Nz) return;

    // Find bins
    int theta_bin = a_grid.FindBin(src.theta0);
    int E_bin = e_grid.FindBin(src.E0);

    // Encode block ID
    uint32_t bid = encode_block(
        theta_bin / N_theta_local,
        E_bin / N_E_local
    );

    // Get local indices
    int theta_local = theta_bin % N_theta_local;
    int E_local = E_bin % N_E_local;
    uint16_t lidx = encode_local_idx(theta_local, E_local);

    // Find or allocate slot
    int slot = psi.find_or_allocate_slot(cell, bid);
    if (slot < 0) return;  // Full

    // Add weight
    float current_w = psi.get_weight(cell, slot, lidx);
    psi.set_weight(cell, slot, lidx, current_w + src.W_total);
}
```

---

## TDD Cycle 5.2: Gaussian Beam Source

### RED - Write Tests First

Create `tests/source/test_gaussian.cpp`:

```cpp
#include <gtest/gtest.h>
#include "source/gaussian_source.hpp"

TEST(GaussianSourceTest, DefaultValues) {
    GaussianSource src;

    EXPECT_FLOAT_EQ(src.x0, 0.0f);
    EXPECT_FLOAT_EQ(src.z0, 0.0f);
    EXPECT_FLOAT_EQ(src.sigma_x, 5.0f);
    EXPECT_FLOAT_EQ(src.sigma_theta, 0.01f);
    EXPECT_FLOAT_EQ(src.E0, 150.0f);
    EXPECT_FLOAT_EQ(src.sigma_E, 1.0f);
    EXPECT_FLOAT_EQ(src.W_total, 1.0f);
}

TEST(GaussianSourceTest, WeightSumConserved) {
    GaussianSource src;
    src.W_total = 1.0f;
    src.x0 = 5.0f;  // Center of domain
    src.z0 = 5.0f;
    src.sigma_x = 2.0f;
    src.sigma_theta = 0.01f;

    EnergyGrid e_grid(0.1f, 250.0f, 256);
    AngularGrid a_grid(-M_PI/2, M_PI/2, 512);

    PsiC psi(10, 10, 32);

    inject_source(psi, src, e_grid, a_grid);

    float total = 0;
    for (int cell = 0; cell < 100; ++cell) {
        total += sum_psi(psi, cell);
    }

    EXPECT_NEAR(total, src.W_total, 1e-3f);  // Allow small discretization error
}

TEST(GaussianSourceTest, SpatialDistribution) {
    GaussianSource src;
    src.x0 = 5.0f;
    src.z0 = 5.0f;
    src.sigma_x = 1.0f;
    src.W_total = 1.0f;

    EnergyGrid e_grid(0.1f, 250.0f, 256);
    AngularGrid a_grid(-M_PI/2, M_PI/2, 512);

    PsiC psi(10, 10, 32);

    inject_source(psi, src, e_grid, a_grid);

    // Center cell should have more weight than edge cells
    float center_weight = sum_psi(psi, 55);  // Near (5, 5)
    float edge_weight = sum_psi(psi, 0);    // Corner

    EXPECT_GT(center_weight, edge_weight);
}

TEST(GaussianSourceTest, AngularDistribution) {
    GaussianSource src;
    src.theta0 = 0.0f;
    src.sigma_theta = 0.01f;  // Small spread
    src.W_total = 1.0f;

    EnergyGrid e_grid(0.1f, 250.0f, 256);
    AngularGrid a_grid(-M_PI/2, M_PI/2, 512);

    PsiC psi(10, 10, 32);

    inject_source(psi, src, e_grid, a_grid);

    // Most weight should be in theta bins near 0
    float central_weight = 0;
    float total_weight = 0;

    for (int cell = 0; cell < 100; ++cell) {
        for (int slot = 0; slot < 32; ++slot) {
            uint32_t bid = psi.block_id[cell][slot];
            if (bid == EMPTY_BLOCK_ID) continue;

            uint32_t b_theta = bid & 0xFFF;
            // Check if b_theta corresponds to central angles
            if (b_theta >= 30 && b_theta <= 34) {  // Near center
                for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                    central_weight += psi.get_weight(cell, slot, lidx);
                    total_weight += psi.get_weight(cell, slot, lidx);
                }
            }
        }
    }

    EXPECT_GT(central_weight / total_weight, 0.8f);  // Most weight near center
}
```

### GREEN - Implementation

Create `include/source/gaussian_source.hpp`:

```cpp
#pragma once

#include "core/grids.hpp"
#include "core/psi_storage.hpp"
#include <random>

struct GaussianSource {
    float x0 = 0.0f;       // Mean x position [mm]
    float z0 = 0.0f;       // Mean z position [mm]
    float sigma_x = 5.0f;  // Spatial spread [mm]
    float sigma_theta = 0.01f;  // Angular spread [rad]
    float E0 = 150.0f;     // Mean energy [MeV]
    float sigma_E = 1.0f;  // Energy spread [MeV]
    float W_total = 1.0f;  // Total weight
    int n_samples = 1000;  // Number of samples for discretization
};

// Inject Gaussian beam into PsiC
void inject_source(
    PsiC& psi,
    const GaussianSource& src,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid
);
```

Create `src/source/gaussian_source.cpp`:

```cpp
#include "source/gaussian_source.hpp"
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"
#include <cmath>
#include <random>

void inject_source(
    PsiC& psi,
    const GaussianSource& src,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid
) {
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> x_dist(src.x0, src.sigma_x);
    std::normal_distribution<float> theta_dist(src.theta0, src.sigma_theta);
    std::normal_distribution<float> E_dist(src.E0, src.sigma_E);

    float w_per_sample = src.W_total / src.n_samples;
    float dx = 1.0f, dz = 1.0f;

    for (int i = 0; i < src.n_samples; ++i) {
        float x = x_dist(rng);
        float theta = theta_dist(rng);
        float E = E_dist(rng);

        // Clamp to valid ranges
        E = fmaxf(e_grid.E_min, fminf(E, e_grid.E_max));
        theta = fmaxf(a_grid.theta_min, fminf(theta, a_grid.theta_max));

        // Find cell
        int ix = static_cast<int>(x / dx);
        int iz = 0;  // Start at z = 0
        if (ix < 0 || ix >= psi.Nx) continue;

        int cell = ix;

        // Find bins
        int theta_bin = a_grid.FindBin(theta);
        int E_bin = e_grid.FindBin(E);

        // Encode block
        uint32_t bid = encode_block(
            theta_bin / N_theta_local,
            E_bin / N_E_local
        );

        int theta_local = theta_bin % N_theta_local;
        int E_local = E_bin % N_E_local;
        uint16_t lidx = encode_local_idx(theta_local, E_local);

        // Add weight
        int slot = psi.find_or_allocate_slot(cell, bid);
        if (slot < 0) continue;

        float current_w = psi.get_weight(cell, slot, lidx);
        psi.set_weight(cell, slot, lidx, current_w + w_per_sample);
    }
}
```

---

## TDD Cycle 5.3: Boundary Conditions

### RED - Write Tests First

Create `tests/boundary/test_boundaries.cpp`:

```cpp
#include <gtest/gtest.h>
#include "boundary/boundaries.hpp"

TEST(BoundaryTest, AbsorbAtZmax) {
    BoundaryConfig config;
    config.z_max = ABSORB;

    // At z boundary
    int cell = 31;  // Last cell in z (assuming Nz = 8)
    int face = 0;   // +z face

    int neighbor = get_neighbor(cell, face, config, 8, 8);

    EXPECT_EQ(neighbor, -1);  // No neighbor = boundary
}

TEST(BoundaryTest, AbsorbAtXmax) {
    BoundaryConfig config;
    config.x_max = ABSORB;

    int cell = 7;   // Last cell in x (assuming Nx = 8)
    int face = 2;   // +x face

    int neighbor = get_neighbor(cell, face, config, 8, 8);

    EXPECT_EQ(neighbor, -1);
}

TEST(BoundaryTest, NormalNeighbor) {
    BoundaryConfig config;

    int cell = 10;  // Interior cell
    int face = 0;   // +z face

    int neighbor = get_neighbor(cell, face, config, 8, 8);

    EXPECT_EQ(neighbor, 18);  // cell + Nx
}

TEST(BoundaryTest, AllFaces) {
    BoundaryConfig config;

    int cell = 10;  // Interior cell
    int Nx = 8;

    // +z
    EXPECT_EQ(get_neighbor(cell, 0, config, Nx, 8), cell + Nx);
    // -z
    EXPECT_EQ(get_neighbor(cell, 1, config, Nx, 8), cell - Nx);
    // +x
    EXPECT_EQ(get_neighbor(cell, 2, config, Nx, 8), cell + 1);
    // -x
    EXPECT_EQ(get_neighbor(cell, 3, config, Nx, 8), cell - 1);
}
```

### GREEN - Implementation

Create `include/boundary/boundaries.hpp`:

```cpp
#pragma once

#include <cstdint>

enum class BoundaryType : int {
    ABSORB = 0,
    REFLECT = 1,
    PERIODIC = 2
};

struct BoundaryConfig {
    BoundaryType z_min = BoundaryType::ABSORB;
    BoundaryType z_max = BoundaryType::ABSORB;
    BoundaryType x_min = BoundaryType::ABSORB;
    BoundaryType x_max = BoundaryType::ABSORB;
};

// Get neighbor cell index
// Returns -1 if boundary is hit
int get_neighbor(int cell, int face, const BoundaryConfig& config, int Nx, int Nz);
```

Create `src/boundary/boundaries.cpp`:

```cpp
#include "boundary/boundaries.hpp"

int get_neighbor(int cell, int face, const BoundaryConfig& config, int Nx, int Nz) {
    int ix = cell % Nx;
    int iz = cell / Nx;

    int neighbor = -1;

    switch (face) {
        case 0:  // +z
            if (iz + 1 >= Nz) {
                neighbor = (config.z_max == BoundaryType::ABSORB) ? -1 : cell;
            } else {
                neighbor = cell + Nx;
            }
            break;
        case 1:  // -z
            if (iz <= 0) {
                neighbor = (config.z_min == BoundaryType::ABSORB) ? -1 : cell;
            } else {
                neighbor = cell - Nx;
            }
            break;
        case 2:  // +x
            if (ix + 1 >= Nx) {
                neighbor = (config.x_max == BoundaryType::ABSORB) ? -1 : cell;
            } else {
                neighbor = cell + 1;
            }
            break;
        case 3:  // -x
            if (ix <= 0) {
                neighbor = (config.x_min == BoundaryType::ABSORB) ? -1 : cell;
            } else {
                neighbor = cell - 1;
            }
            break;
    }

    return neighbor;
}
```

---

## TDD Cycle 5.4: Boundary Loss Tracking

### RED - Write Tests First

Create `tests/boundary/test_loss_tracking.cpp`:

```cpp
#include <gtest/gtest.h>
#include "boundary/loss_tracking.hpp"

TEST(BoundaryLossTest, InitializeZero) {
    BoundaryLoss loss;

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(loss.weight[i], 0.0f);
        EXPECT_DOUBLE_EQ(loss.energy[i], 0.0);
    }
}

TEST(BoundaryLossTest, RecordLoss) {
    BoundaryLoss loss;

    record_boundary_loss(loss, 0, 1.0f, 100.0f);

    EXPECT_FLOAT_EQ(loss.weight[0], 1.0f);
    EXPECT_DOUBLE_EQ(loss.energy[0], 100.0);
}

TEST(BoundaryLossTest, MultipleLosses) {
    BoundaryLoss loss;

    record_boundary_loss(loss, 0, 1.0f, 100.0f);
    record_boundary_loss(loss, 0, 0.5f, 50.0f);
    record_boundary_loss(loss, 1, 0.3f, 30.0f);

    EXPECT_FLOAT_EQ(loss.weight[0], 1.5f);
    EXPECT_DOUBLE_EQ(loss.energy[0], 150.0);
    EXPECT_FLOAT_EQ(loss.weight[1], 0.3f);
    EXPECT_DOUBLE_EQ(loss.energy[1], 30.0);
}

TEST(BoundaryLossTest, TotalLoss) {
    BoundaryLoss loss;

    record_boundary_loss(loss, 0, 0.5f, 50.0f);
    record_boundary_loss(loss, 1, 0.3f, 30.0f);
    record_boundary_loss(loss, 2, 0.1f, 10.0f);
    record_boundary_loss(loss, 3, 0.1f, 10.0f);

    float total_w = total_boundary_weight_loss(loss);
    EXPECT_NEAR(total_w, 1.0f, 1e-6f);
}
```

### GREEN - Implementation

Create `include/boundary/loss_tracking.hpp`:

```cpp
#pragma once

#include <array>

struct BoundaryLoss {
    std::array<float, 4> weight;  // [z_min, z_max, x_min, x_max]
    std::array<double, 4> energy;

    BoundaryLoss() {
        weight.fill(0.0f);
        energy.fill(0.0);
    }
};

// Record loss at boundary face
void record_boundary_loss(BoundaryLoss& loss, int face, float w, double E);

// Get total weight lost at boundaries
float total_boundary_weight_loss(const BoundaryLoss& loss);

// Get total energy lost at boundaries
double total_boundary_energy_loss(const BoundaryLoss& loss);
```

Create `src/boundary/loss_tracking.cpp`:

```cpp
#include "boundary/loss_tracking.hpp"

void record_boundary_loss(BoundaryLoss& loss, int face, float w, double E) {
    if (face >= 0 && face < 4) {
        loss.weight[face] += w;
        loss.energy[face] += E;
    }
}

float total_boundary_weight_loss(const BoundaryLoss& loss) {
    float total = 0;
    for (int i = 0; i < 4; ++i) {
        total += loss.weight[i];
    }
    return total;
}

double total_boundary_energy_loss(const BoundaryLoss& loss) {
    double total = 0;
    for (int i = 0; i < 4; ++i) {
        total += loss.energy[i];
    }
    return total;
}
```

---

## Exit Criteria Checklist

- [ ] Pencil source injects to correct cell/bins
- [ ] Gaussian source conserves total weight (±1e-3)
- [ ] Gaussian distribution spatially correct
- [ ] Gaussian distribution angularly correct
- [ ] Boundary conditions correctly identify neighbors
- [ ] Absorbing boundary returns -1 for neighbor
- [ ] Boundary loss tracking accumulates correctly
- [ ] Total boundary loss can be retrieved

---

## Next Steps

After completing Phase 5, proceed to **Phase 6 (Conservation Audit)** which integrates with the full pipeline.

```bash
# Test sources and boundaries
./bin/sm2d_tests --gtest_filter="*Source*:*Boundary*"
```
