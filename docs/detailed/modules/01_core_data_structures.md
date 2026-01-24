# Core Data Structures Module

## Overview

The core data structures module provides the foundational storage and encoding mechanisms for the phase-space particle representation. This module implements a hierarchical, block-sparse storage system optimized for GPU computation.

---

## 1. Energy Grid (`grids.hpp/cpp`)

### Purpose
Logarithmic energy grid for representing proton kinetic energy from 0.1 to 250 MeV.

### Structure

```cpp
struct EnergyGrid {
    const int N_E;                     // Number of energy bins (256)
    const float E_min;                 // Minimum energy (0.1 MeV)
    const float E_max;                 // Maximum energy (250.0 MeV)
    std::vector<float> edges;          // N_E + 1 bin edges (log-spaced)
    std::vector<float> rep;            // N_E representative energies
};
```

### Key Methods

| Method | Description | Complexity |
|--------|-------------|------------|
| `EnergyGrid(E_min, E_max, N_E)` | Constructor - creates log-spaced grid | O(N) |
| `FindBin(float E)` | Binary search for energy bin | O(log N) |
| `GetRepEnergy(int bin)` | Get representative (geometric mean) energy | O(1) |

### Representative Energy Calculation
```cpp
rep[i] = sqrt(edges[i] * edges[i+1])  // Geometric mean
```

### Usage Example
```cpp
EnergyGrid grid(0.1f, 250.0f, 256);
int bin = grid.FindBin(150.0f);        // Find bin for 150 MeV
float E_rep = grid.GetRepEnergy(bin);  // Get representative energy
```

---

## 2. Angular Grid (`grids.hpp/cpp`)

### Purpose
Uniform angular grid for particle direction in the X-Z plane.

### Structure

```cpp
struct AngularGrid {
    const int N_theta;                 // Number of theta bins (512)
    const float theta_min;             // Minimum angle (-90°)
    const float theta_max;             // Maximum angle (+90°)
    std::vector<float> edges;          // N_theta + 1 bin edges (uniform)
    std::vector<float> rep;            // N_theta representative angles
};
```

### Key Methods

| Method | Description | Complexity |
|--------|-------------|------------|
| `AngularGrid(theta_min, theta_max, N_theta)` | Constructor | O(N) |
| `FindBin(float theta)` | Find angular bin (clamped) | O(1) |
| `GetRepTheta(int bin)` | Get representative (midpoint) angle | O(1) |

### Representative Angle Calculation
```cpp
rep[i] = 0.5 * (edges[i] + edges[i+1])  // Arithmetic mean
```

---

## 3. Block Encoding (`block_encoding.hpp`)

### Purpose
Compact 24-bit encoding of (θ, E) phase-space coordinates for efficient GPU storage.

### Encoding Scheme

```
┌─────────────────────────┬──────────────────────────┐
│     b_E (12 bits)       │    b_theta (12 bits)     │
│    Bits 12-23           │     Bits 0-11            │
│    Range: 0-4095        │     Range: 0-4095        │
└─────────────────────────┴──────────────────────────┘
                    24-bit Block ID
```

### Functions

```cpp
// Encode: (b_theta, b_E) → block_id
__host__ __device__ inline uint32_t encode_block(uint32_t b_theta, uint32_t b_E) {
    return (b_theta & 0xFFF) | ((b_E & 0xFFF) << 12);
}

// Decode: block_id → (b_theta, b_E)
__host__ __device__ inline void decode_block(uint32_t block_id,
                                               uint32_t& b_theta,
                                               uint32_t& b_E) {
    b_theta = block_id & 0xFFF;
    b_E = (block_id >> 12) & 0xFFF;
}
```

### Special Values
```cpp
constexpr uint32_t EMPTY_BLOCK_ID = 0xFFFFFFFF;  // Marks unused slots
```

### Why 12 bits each?
- Energy bins: 256 needed → 12 bits allows up to 4096
- Angular bins: 512 needed → 12 bits allows up to 4096
- Fits in single 32-bit integer for GPU efficiency

---

## 4. Local Bins (`local_bins.hpp`)

### Purpose
4D sub-cell partitioning for variance-preserving intra-cell particle tracking.

### Structure

| Dimension | Bins | Description |
|-----------|------|-------------|
| θ_local | 8 | Local angular subdivision |
| E_local | 4 | Local energy subdivision |
| x_sub | 4 | Transverse position within cell |
| z_sub | 4 | Depth position within cell |

**Total**: `8 × 4 × 4 × 4 = 512` local bins per block

### Index Encoding

```cpp
constexpr int N_theta_local = 8;
constexpr int N_E_local = 4;
constexpr int N_x_sub = 4;
constexpr int N_z_sub = 4;
constexpr int LOCAL_BINS = 512;

// Encode 4D coordinates to 16-bit index
__host__ __device__ inline uint16_t encode_local_idx_4d(
    int theta_local, int E_local, int x_sub, int z_sub
) {
    int inner = E_local + N_E_local * (x_sub + N_x_sub * z_sub);
    return static_cast<uint16_t>(theta_local + N_theta_local * inner);
}
```

### Position Conversion

```cpp
// X offset from sub-bin center (range: -0.375*dx to +0.375*dx)
__host__ __device__ inline float get_x_offset_from_bin(int x_sub, float dx) {
    return dx * (-0.375f + 0.25f * x_sub);
}

// Z offset from sub-bin center
__host__ __device__ inline float get_z_offset_from_bin(int z_sub, float dz) {
    return dz * (-0.375f + 0.25f * z_sub);
}
```

### Sub-bin Centers

| x_sub | Offset (×dx) |
|-------|--------------|
| 0 | -0.375 |
| 1 | -0.125 |
| 2 | +0.125 |
| 3 | +0.375 |

---

## 5. Phase-Space Storage (`psi_storage.hpp/cpp`)

### Purpose
Hierarchical cell-based storage for particle weights in 4D phase-space.

### Data Structure

```cpp
struct PsiC {
    const int Nx;                     // Grid X dimension
    const int Nz;                     // Grid Z dimension
    const int Kb;                     // Max blocks per cell (32)

    // Storage layout: [cell][slot][local_bin]
    std::vector<std::array<uint32_t, 32>> block_id;  // Block IDs
    std::vector<std::array<std::array<float, LOCAL_BINS>, 32>> value;  // Weights

private:
    int N_cells;  // Nx × Nz
};
```

### Memory Layout

```
PsiC[cell = 0]
├── slot[0]: block_id=0x000123 → value[0..511]
├── slot[1]: block_id=0x000456 → value[0..511]
├── ...
└── slot[31]: block_id=EMPTY → value[unused]

PsiC[cell = 1]
├── ...
```

### Key Methods

```cpp
// Find existing block or allocate new slot
int find_or_allocate_slot(int cell, uint32_t bid);

// Get/set weight in specific local bin
float get_weight(int cell, int slot, uint16_t lidx) const;
void set_weight(int cell, int slot, uint16_t lidx, float w);

// Clear all data
void clear();
```

### Slot Allocation Algorithm

```cpp
int PsiC::find_or_allocate_slot(int cell, uint32_t bid) {
    // First pass: check if block already exists
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == bid) {
            return slot;  // Found existing slot
        }
    }
    // Second pass: find empty slot
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == EMPTY_BLOCK_ID) {
            block_id[cell][slot] = bid;  // Allocate new slot
            return slot;
        }
    }
    return -1;  // No space available
}
```

### Capacity Analysis

For a 200×640 grid with 32 slots per cell:
- Total cells: 128,000
- Total slots: 4,096,000
- Memory per slot: 512 bins × 4 bytes = 2KB
- Total storage: ~8GB (fully dense)
- Actual usage: ~1.1GB (sparse)

---

## 6. Bucket Emission (`buckets.hpp/cpp`)

### Purpose
Efficient inter-cell particle transfer through boundary bucket emission.

### Structure

```cpp
struct OutflowBucket {
    static constexpr int Kb_out = 64;  // Max emission buckets

    std::array<uint32_t, Kb_out> block_id;        // Block IDs
    std::array<uint16_t, Kb_out> local_count;     // Particle counts
    std::array<std::array<float, LOCAL_BINS>, Kb_out> value;  // Weights
};
```

### Face Indexing

| Face | Direction | Index |
|------|-----------|-------|
| +z | Forward | 0 |
| -z | Backward | 1 |
| +x | Right | 2 |
| -x | Left | 3 |

### Key Methods

```cpp
int find_or_allocate_slot(uint32_t bid);  // Find/allocate emission slot
void clear();                             // Clear all emission data
```

### Transfer Flow

```
Cell (i,j)
├── emits to bucket[+z] → received by Cell (i,j+1)
├── emits to bucket[-z] → received by Cell (i,j-1)
├── emits to bucket[+x] → received by Cell (i+1,j)
└── emits to bucket[-x] → received by Cell (i-1,j)
```

---

## Memory Summary

| Component | Size per Instance | Total Size |
|-----------|-------------------|------------|
| EnergyGrid | ~4KB | 4KB |
| AngularGrid | ~8KB | 8KB |
| PsiC | ~1.1GB | 1.1GB |
| OutflowBuckets | ~32KB per cell | ~4GB (all cells) |
| **Active Working Set** | | **~2.2GB** |

---

## Design Rationale

1. **Why block-sparse?**
   - Phase space is mostly empty (particles occupy limited (θ,E) regions)
   - Saves >70% memory vs dense storage

2. **Why 24-bit encoding?**
   - Single integer lookup instead of pair hashing
   - GPU-friendly integer operations
   - Sufficient capacity for therapeutic energy/angle ranges

3. **Why 512 local bins?**
   - Balances variance preservation vs memory
   - Powers of 2 for efficient bit manipulation
   - Empirically sufficient for clinical accuracy

4. **Why fixed slots per cell?**
   - Predictable memory layout for GPU coalescing
   - O(Kb) search is fast (Kb=32 is small)
   - Avoids dynamic allocation on GPU
