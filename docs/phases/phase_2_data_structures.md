# Phase 2: Core Data Structures and Memory Layout

**Status**: Pending
**Duration**: 2-3 days
**Dependencies**: Phase 0

---

## Objectives

1. Implement energy and angular grid definitions
2. Implement block-sparse storage (PsiC)
3. Implement bucket structures for outflow
4. Implement local bin encoding/decoding
5. Implement block ID encoding

---

## TDD Cycle 2.1: Energy Grid

### RED - Write Tests First

Create `tests/core/test_energy_grid.cpp`:

```cpp
#include <gtest/gtest.h>
#include "core/grids.hpp"

TEST(EnergyGridTest, BinEdgesCorrect) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    EXPECT_EQ(grid.edges.size(), 257);  // N_E + 1
    EXPECT_NEAR(grid.edges[0], 0.1f, 1e-6f);
    EXPECT_NEAR(grid.edges[256], 250.0f, 1e-4f);
}

TEST(EnergyGridTest, FindEnergyBin) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    // Test at representative energies
    int bin = grid.FindBin(150.0f);
    EXPECT_GE(bin, 0);
    EXPECT_LT(bin, 256);
    EXPECT_GE(150.0f, grid.edges[bin]);
    EXPECT_LT(150.0f, grid.edges[bin+1]);
}

TEST(EnergyGridTest, EdgeCases) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    // Below minimum
    int bin_min = grid.FindBin(0.01f);
    EXPECT_EQ(bin_min, 0);

    // Above maximum
    int bin_max = grid.FindBin(300.0f);
    EXPECT_EQ(bin_max, 255);
}

TEST(EnergyGridTest, RepresentativeEnergyWithinBin) {
    EnergyGrid grid(0.1f, 250.0f, 256);

    for (int i = 0; i < 256; ++i) {
        float E_rep = grid.GetRepEnergy(i);
        EXPECT_GE(E_rep, grid.edges[i]);
        EXPECT_LT(E_rep, grid.edges[i+1]);
    }
}
```

### GREEN - Implementation

Create `include/core/grids.hpp`:

```cpp
#pragma once

#include <vector>
#include <cmath>

struct EnergyGrid {
    const int N_E;
    const float E_min;
    const float E_max;
    std::vector<float> edges;   // N_E + 1 bin edges
    std::vector<float> rep;     // N_E representative (geometric mean)

    EnergyGrid(float E_min, float E_max, int N_E);
    int FindBin(float E) const;
    float GetRepEnergy(int bin) const;
};
```

---

## TDD Cycle 2.2: Angular Grid

### RED - Write Tests First

Create `tests/core/test_angular_grid.cpp`:

```cpp
#include <gtest/gtest.h>
#include "core/grids.hpp"

TEST(AngularGridTest, BinEdgesCorrect) {
    AngularGrid grid(-M_PI/2, M_PI/2, 512);

    EXPECT_EQ(grid.edges.size(), 513);  // N_theta + 1
    EXPECT_NEAR(grid.edges[0], -M_PI/2, 1e-6f);
    EXPECT_NEAR(grid.edges[512], M_PI/2, 1e-6f);
}

TEST(AngularGridTest, UniformSpacing) {
    AngularGrid grid(-M_PI/2, M_PI/2, 512);

    float delta = grid.edges[1] - grid.edges[0];

    for (size_t i = 2; i < 100; ++i) {
        EXPECT_NEAR(grid.edges[i] - grid.edges[i-1], delta, 1e-6f);
    }
}

TEST(AngularGridTest, FindThetaBin) {
    AngularGrid grid(-M_PI/2, M_PI/2, 512);

    // Zero angle should be near the middle
    int bin_zero = grid.FindBin(0.0f);
    EXPECT_GT(bin_zero, 200);
    EXPECT_LT(bin_zero, 312);
}
```

### GREEN - Implementation

Extend `include/core/grids.hpp`:

```cpp
struct AngularGrid {
    const int N_theta;
    const float theta_min;
    const float theta_max;
    std::vector<float> edges;   // N_theta + 1 bin edges
    std::vector<float> rep;     // N_theta representative (midpoint)

    AngularGrid(float theta_min, float theta_max, int N_theta);
    int FindBin(float theta) const;
    float GetRepTheta(int bin) const;
};
```

---

## TDD Cycle 2.3: Local Bin Encoding

### RED - Write Tests First

Create `tests/core/test_local_bins.cpp`:

```cpp
#include <gtest/gtest.h>
#include "core/local_bins.hpp"

TEST(LocalBinsTest, LocalBinsEquals32) {
    // Compile-time check
    EXPECT_EQ(LOCAL_BINS, 32);
    EXPECT_EQ(N_theta_local, 8);
    EXPECT_EQ(N_E_local, 4);
    EXPECT_EQ(LOCAL_BINS, N_theta_local * N_E_local);
}

TEST(LocalBinsTest, EncodeDecodeRoundTrip) {
    for (int t = 0; t < N_theta_local; ++t) {
        for (int e = 0; e < N_E_local; ++e) {
            uint16_t encoded = encode_local_idx(t, e);
            int t_decoded, e_decoded;
            decode_local_idx(encoded, t_decoded, e_decoded);

            EXPECT_EQ(t_decoded, t);
            EXPECT_EQ(e_decoded, e);
        }
    }
}

TEST(LocalBinsTest, EncodeRange) {
    // Maximum encoded value should fit in uint16_t
    uint16_t max_encoded = encode_local_idx(N_theta_local - 1, N_E_local - 1);
    EXPECT_LT(max_encoded, LOCAL_BINS);
}

TEST(LocalBinsTest, DecodeAllValues) {
    for (uint16_t i = 0; i < LOCAL_BINS; ++i) {
        int t, e;
        decode_local_idx(i, t, e);

        EXPECT_GE(t, 0);
        EXPECT_LT(t, N_theta_local);
        EXPECT_GE(e, 0);
        EXPECT_LT(e, N_E_local);
    }
}
```

### GREEN - Implementation

Create `include/core/local_bins.hpp`:

```cpp
#pragma once

#include <cstdint>

// Local bin decomposition parameters
constexpr int N_theta_local = 8;   // Angular sub-bins per block
constexpr int N_E_local = 4;       // Energy sub-bins per block
constexpr int LOCAL_BINS = N_theta_local * N_E_local;  // = 32

// Encode local index from (theta_local, E_local)
inline uint16_t encode_local_idx(int theta_local, int E_local) {
    return static_cast<uint16_t>(theta_local * N_E_local + E_local);
}

// Decode local index to (theta_local, E_local)
inline void decode_local_idx(uint16_t lidx, int& theta_local, int& E_local) {
    theta_local = static_cast<int>(lidx) / N_E_local;
    E_local = static_cast<int>(lidx) % N_E_local;
}

// Static assertion to ensure LOCAL_BINS fits in uint16_t
static_assert(LOCAL_BINS <= 65536, "LOCAL_BINS too large for uint16_t");
```

---

## TDD Cycle 2.4: Block ID Encoding

### RED - Write Tests First

Create `tests/core/test_block_encoding.cpp`:

```cpp
#include <gtest/gtest.h>
#include "core/block_encoding.hpp"

TEST(BlockEncodingTest, EncodeDecodeRoundTrip) {
    for (uint32_t b_theta = 0; b_theta < 1000; ++b_theta) {
        for (uint32_t b_E = 0; b_E < 1000; ++b_E) {
            uint32_t block_id = encode_block(b_theta, b_E);

            uint32_t theta_decoded, E_decoded;
            decode_block(block_id, theta_decoded, E_decoded);

            EXPECT_EQ(theta_decoded, b_theta);
            EXPECT_EQ(E_decoded, b_E);
        }
    }
}

TEST(BlockEncodingTest, BitLayout) {
    uint32_t b_theta = 0xABC;
    uint32_t b_E = 0xDEF;

    uint32_t block_id = encode_block(b_theta, b_E);

    // Lower 12 bits = b_theta
    EXPECT_EQ(block_id & 0xFFF, b_theta);

    // Upper 12 bits = b_E
    EXPECT_EQ((block_id >> 12) & 0xFFF, b_E);
}

TEST(BlockEncodingTest, MaxValues) {
    // Maximum block indices that fit in 12 bits
    uint32_t max_b_theta = 0xFFF;  // 4095
    uint32_t max_b_E = 0xFFF;

    uint32_t block_id = encode_block(max_b_theta, max_b_E);
    EXPECT_NE(block_id, 0);

    uint32_t theta_decoded, E_decoded;
    decode_block(block_id, theta_decoded, E_decoded);

    EXPECT_EQ(theta_decoded, max_b_theta);
    EXPECT_EQ(E_decoded, max_b_E);
}
```

### GREEN - Implementation

Create `include/core/block_encoding.hpp`:

```cpp
#pragma once

#include <cstdint>

// Block ID: 24-bit encoding
// Bits 0-11:   b_theta (12 bits)
// Bits 12-23:  b_E (12 bits)

inline uint32_t encode_block(uint32_t b_theta, uint32_t b_E) {
    return (b_theta & 0xFFF) | ((b_E & 0xFFF) << 12);
}

inline void decode_block(uint32_t block_id, uint32_t& b_theta, uint32_t& b_E) {
    b_theta = block_id & 0xFFF;
    b_E = (block_id >> 12) & 0xFFF;
}
```

---

## TDD Cycle 2.5: Sparse Storage (PsiC)

### RED - Write Tests First

Create `tests/core/test_psi_storage.cpp`:

```cpp
#include <gtest/gtest.h>
#include "core/psi_storage.hpp"

TEST(PsiStorageTest, InitializeEmpty) {
    PsiC psi(32, 32, 32);  // Nx=32, Nz=32, Kb=32

    for (int cell = 0; cell < 32 * 32; ++cell) {
        for (int slot = 0; slot < 32; ++slot) {
            EXPECT_EQ(psi.block_id[cell][slot], EMPTY_BLOCK_ID);
        }
    }
}

TEST(PsiStorageTest, AllocateAndAccess) {
    PsiC psi(32, 32, 32);

    uint32_t bid = encode_block(10, 20);
    uint16_t lidx = encode_local_idx(3, 1);
    float w = 1.5f;

    int slot = psi.find_or_allocate_slot(0, bid);
    ASSERT_GE(slot, 0);
    EXPECT_LT(slot, 32);

    psi.set_weight(0, slot, lidx, w);
    EXPECT_NEAR(psi.get_weight(0, slot, lidx), w, 1e-6f);
}

TEST(PsiStorageTest, SlotReuse) {
    PsiC psi(32, 32, 32);

    uint32_t bid = encode_block(5, 10);

    int slot1 = psi.find_or_allocate_slot(0, bid);
    int slot2 = psi.find_or_allocate_slot(0, bid);

    EXPECT_EQ(slot1, slot2);  // Should reuse same slot
}

TEST(PsiStorageTest, DifferentBlocksDifferentSlots) {
    PsiC psi(32, 32, 32);

    uint32_t bid1 = encode_block(1, 2);
    uint32_t bid2 = encode_block(3, 4);

    int slot1 = psi.find_or_allocate_slot(0, bid1);
    int slot2 = psi.find_or_allocate_slot(0, bid2);

    EXPECT_NE(slot1, slot2);  // Different blocks, different slots
}

TEST(PsiStorageTest, DifferentCellsIndependent) {
    PsiC psi(32, 32, 32);

    uint32_t bid = encode_block(5, 5);

    int slot0 = psi.find_or_allocate_slot(0, bid);
    int slot1 = psi.find_or_allocate_slot(1, bid);

    EXPECT_EQ(slot0, slot1);  // Same block ID, can reuse slot number

    // But values should be independent
    psi.set_weight(0, slot0, 0, 1.0f);
    psi.set_weight(1, slot1, 0, 2.0f);

    EXPECT_NEAR(psi.get_weight(0, slot0, 0), 1.0f, 1e-6f);
    EXPECT_NEAR(psi.get_weight(1, slot1, 0), 2.0f, 1e-6f);
}
```

### GREEN - Implementation

Create `include/core/psi_storage.hpp`:

```cpp
#pragma once

#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"
#include <vector>
#include <array>
#include <cstdint>

constexpr uint32_t EMPTY_BLOCK_ID = 0xFFFFFFFF;

struct PsiC {
    const int Nx;
    const int Nz;
    const int Kb;

    // block_id[cell][slot] = block ID, or EMPTY if slot unused
    std::vector<std::array<uint32_t, 32>> block_id;

    // value[cell][slot][local_idx] = weight
    std::vector<std::array<std::array<float, LOCAL_BINS>, 32>> value;

    PsiC(int Nx, int Nz, int Kb);

    // Find existing slot for block, or allocate new one
    // Returns slot index, or -1 if full
    int find_or_allocate_slot(int cell, uint32_t bid);

    // Access weight by (cell, slot, local_idx)
    float get_weight(int cell, int slot, uint16_t lidx) const;
    void set_weight(int cell, int slot, uint16_t lidx, float w);

    // Clear all data
    void clear();

private:
    int N_cells;
};
```

Create `src/core/psi_storage.cpp`:

```cpp
#include "core/psi_storage.hpp"
#include <cstring>

PsiC::PsiC(int Nx, int Nz, int Kb)
    : Nx(Nx), Nz(Nz), Kb(Kb), N_cells(Nx * Nz)
{
    block_id.resize(N_cells);
    value.resize(N_cells);

    for (int cell = 0; cell < N_cells; ++cell) {
        block_id[cell].fill(EMPTY_BLOCK_ID);
        for (int slot = 0; slot < Kb; ++slot) {
            value[cell][slot].fill(0.0f);
        }
    }
}

int PsiC::find_or_allocate_slot(int cell, uint32_t bid) {
    // Search for existing slot
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == bid) {
            return slot;
        }
    }

    // Allocate new slot
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == EMPTY_BLOCK_ID) {
            block_id[cell][slot] = bid;
            return slot;
        }
    }

    return -1;  // Full
}

float PsiC::get_weight(int cell, int slot, uint16_t lidx) const {
    if (cell < 0 || cell >= N_cells) return 0.0f;
    if (slot < 0 || slot >= Kb) return 0.0f;
    if (lidx >= LOCAL_BINS) return 0.0f;
    return value[cell][slot][lidx];
}

void PsiC::set_weight(int cell, int slot, uint16_t lidx, float w) {
    if (cell >= 0 && cell < N_cells && slot >= 0 && slot < Kb && lidx < LOCAL_BINS) {
        value[cell][slot][lidx] = w;
    }
}

void PsiC::clear() {
    for (int cell = 0; cell < N_cells; ++cell) {
        block_id[cell].fill(EMPTY_BLOCK_ID);
        for (int slot = 0; slot < Kb; ++slot) {
            value[cell][slot].fill(0.0f);
        }
    }
}
```

---

## TDD Cycle 2.6: Bucket Structures

### RED - Write Tests First

Create `tests/core/test_buckets.cpp`:

```cpp
#include <gtest/gtest.h>
#include "core/buckets.hpp"

TEST(BucketTest, InitializeEmpty) {
    OutflowBucket bucket;

    for (int i = 0; i < Kb_out; ++i) {
        EXPECT_EQ(bucket.block_id[i], EMPTY_BLOCK_ID);
        EXPECT_EQ(bucket.local_count[i], 0);
        for (int j = 0; j < LOCAL_BINS; ++j) {
            EXPECT_FLOAT_EQ(bucket.value[i][j], 0.0f);
        }
    }
}

TEST(BucketTest, EmitToBucket) {
    OutflowBucket bucket;
    uint32_t bid = encode_block(10, 20);
    int theta_local = 3;
    int E_local = 1;
    uint16_t lidx = encode_local_idx(theta_local, E_local);
    float w = 1.0f;

    int slot = bucket.find_or_allocate_slot(bid);
    ASSERT_GE(slot, 0);

    atomic_add(bucket.value[slot][lidx], w);
    bucket.local_count[slot]++;

    EXPECT_FLOAT_EQ(bucket.value[slot][lidx], w);
}

TEST(BucketTest, WeightConservation) {
    OutflowBucket bucket;

    // Simulate emission with 2-bin energy discretization
    float w_total = 1.0f;
    uint16_t lidx_low = 0;
    uint16_t lidx_high = 1;
    float w_low = 0.7f;
    float w_high = 0.3f;

    uint32_t bid = encode_block(5, 10);
    int slot = bucket.find_or_allocate_slot(bid);

    atomic_add(bucket.value[slot][lidx_low], w_low);
    atomic_add(bucket.value[slot][lidx_high], w_high);

    // Sum all weights
    float total = 0.0f;
    for (int i = 0; i < Kb_out; ++i) {
        for (int j = 0; j < LOCAL_BINS; ++j) {
            total += bucket.value[i][j];
        }
    }

    EXPECT_NEAR(total, w_total, 1e-5f);
}

TEST(BucketTest, ClearBucket) {
    OutflowBucket bucket;

    // Add some data
    uint32_t bid = encode_block(1, 2);
    int slot = bucket.find_or_allocate_slot(bid);
    bucket.value[slot][0] = 1.0f;

    bucket.clear();

    // Should be empty again
    for (int i = 0; i < Kb_out; ++i) {
        EXPECT_EQ(bucket.block_id[i], EMPTY_BLOCK_ID);
    }
}
```

### GREEN - Implementation

Create `include/core/buckets.hpp`:

```cpp
#pragma once

#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"
#include <array>
#include <cstdint>

constexpr int Kb_out = 64;

struct OutflowBucket {
    std::array<uint32_t, Kb_out> block_id;
    std::array<uint16_t, Kb_out> local_count;
    std::array<std::array<float, LOCAL_BINS>, Kb_out> value;

    OutflowBucket();

    // Find slot for block, or allocate new one
    int find_or_allocate_slot(uint32_t bid);

    // Clear all data
    void clear();

    // Atomic add for weight (CPU version - GPU uses atomicAdd)
    void atomic_add_slot(int slot, uint16_t lidx, float w);
};

// Helper function
inline void atomic_add(float& target, float value) {
    // CPU version - just add
    target += value;
}
```

---

## Exit Criteria Checklist

- [ ] All grid tests pass (bin finding, representative values)
- [ ] Encoding/decoding round-trips correctly for all values
- [ ] Sparse storage allocates and reuses slots correctly
- [ ] Bucket emission conserves weight (w_in = w_out)
- [ ] Local bins constant: LOCAL_BINS = 32
- [ ] Memory usage documented and < 100MB for data structures

---

## Next Steps

After completing Phase 2, proceed to **Phase 3 (Physics Models)** which depends on both Phase 1 (LUT) and Phase 2 (data structures).

```bash
# Test data structures
./bin/sm2d_tests --gtest_filter="*Grid*:*LocalBin*:*Block*:*PsiStorage*:*Bucket*"
```
