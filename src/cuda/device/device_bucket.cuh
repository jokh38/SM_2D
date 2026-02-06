#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"

// Forward declarations for physics functions (defined in device_physics.cuh)
// Avoid circular dependency by using forward declarations where possible
extern __device__ float device_gaussian_cdf(float x, float mu, float sigma);
extern __device__ void device_gaussian_spread_weights(float* weights, float x_mean, float sigma_x, float dx, int N);
extern __device__ void device_gaussian_spread_weights_subcell(float* weights, float x_mean, float sigma_x, float dx, int N_x_sub);

// ============================================================================
// Device Bucket Emission Functions for K3 FineTransport
// ============================================================================
// P3 FIX: Device-accessible bucket emission for GPU kernel
//
// These functions handle the transfer of particle weights from a cell
// to its neighboring cells through boundary crossing.
//
// Flow: K3 detects boundary crossing → emits to bucket → K4 transfers to neighbor
// ============================================================================

constexpr uint32_t DEVICE_EMPTY_BLOCK_ID = 0xFFFFFFFF;
constexpr int DEVICE_Kb_out = 8;  // Reduced from 64 to match DEVICE_Kb and save memory
constexpr int DEVICE_LOCAL_BINS = LOCAL_BINS;  // Now 128 for 3D (theta, E, x_sub)

// Face definitions (must match K4 get_neighbor)
// Moved to top so these can be used by other functions
enum DeviceFace {
    FACE_Z_PLUS = 0,   // +z direction
    FACE_Z_MINUS = 1,  // -z direction
    FACE_X_PLUS = 2,   // +x direction
    FACE_X_MINUS = 3   // -x direction
};

// Device-accessible OutflowBucket structure
// Matches CPU OutflowBucket layout for direct memory transfer
//
// PLAN_MCS Phase B-1: Added Fermi-Eyges moment tracking for O(z^(3/2)) lateral spreading
// These moments track the evolution of angular and position variances:
//   moment_A = ⟨θ²⟩ : Angular variance (scattering angle squared)
//   moment_B = ⟨xθ⟩ : Position-angle covariance
//   moment_C = ⟨x²⟩ : Position variance (lateral spread squared)
//
// Reference: Fermi-Eyges theory of multiple Coulomb scattering
struct DeviceOutflowBucket {
    uint32_t block_id[DEVICE_Kb_out];
    uint16_t local_count[DEVICE_Kb_out];
    float value[DEVICE_Kb_out][DEVICE_LOCAL_BINS];

    // Fermi-Eyges moments for lateral spreading (Phase B)
    float moment_A;  // ⟨θ²⟩ - Angular variance [rad²]
    float moment_B;  // ⟨xθ⟩ - Position-angle covariance [mm·rad]
    float moment_C;  // ⟨x²⟩ - Position variance [mm²]
};

// Find or allocate a slot in the bucket for a given block ID
// Returns slot index or -1 if bucket is full
__device__ inline int device_bucket_find_or_allocate_slot(
    DeviceOutflowBucket& bucket,
    uint32_t bid
) {
    // First, try to find existing slot
    for (int slot = 0; slot < DEVICE_Kb_out; ++slot) {
        if (bucket.block_id[slot] == bid) {
            return slot;
        }
    }

    // Not found, try to allocate new slot
    for (int slot = 0; slot < DEVICE_Kb_out; ++slot) {
        if (bucket.block_id[slot] == DEVICE_EMPTY_BLOCK_ID) {
            // Initialize new slot
            bucket.block_id[slot] = bid;
            bucket.local_count[slot] = 0;
            for (int i = 0; i < DEVICE_LOCAL_BINS; ++i) {
                bucket.value[slot][i] = 0.0f;
            }
            return slot;
        }
    }

    // Bucket is full
    return -1;
}

// Emit a particle weight to a bucket
// P3 FIX: This function was missing from original implementation
__device__ inline void device_emit_to_bucket(
    DeviceOutflowBucket& bucket,
    uint32_t bid,           // Coarse block ID
    uint16_t lidx,          // Local bin index (0-31)
    float weight            // Weight to add
) {
    if (weight <= 0.0f) return;

    int slot = device_bucket_find_or_allocate_slot(bucket, bid);
    if (slot >= 0) {
        // Atomic add for thread safety
        atomicAdd(&bucket.value[slot][lidx], weight);
        bucket.local_count[slot]++;
    }
    // Note: If bucket is full, weight is lost (should be rare with Kb_out=64)
}

// Emit a component to a bucket with full phase-space encoding (2D version for compatibility)
// This is the main emission function called by K3 FineTransport
__device__ inline void device_emit_component_to_bucket(
    DeviceOutflowBucket& bucket,
    float theta,            // Polar angle [rad]
    float E,                // Energy [MeV]
    float weight,           // Statistical weight
    const float* __restrict__ theta_edges,  // Angular grid edges (for bin lookup)
    const float* __restrict__ E_edges,      // Energy grid edges (for bin lookup)
    int N_theta,            // Number of angular bins
    int N_E,                // Number of energy bins
    int N_theta_local,      // Local angular bins per block (8)
    int N_E_local           // Local energy bins per block (4)
) {
    if (weight <= 0.0f) return;

    // Find global bins
    int theta_bin = 0;
    int E_bin = 0;

    // Simple bin finding (can be optimized with binary search)
    // For uniform theta grid
    float theta_min = theta_edges[0];
    float theta_max = theta_edges[N_theta];
    float dtheta = (theta_max - theta_min) / N_theta;
    theta_bin = (int)((theta - theta_min) / dtheta);
    theta_bin = max(0, min(theta_bin, N_theta - 1));

    // Option D2: Use binary search for energy bin (works for both log-spaced and piecewise-uniform)
    // Binary search in E_edges to find the bin containing energy E
    float E_clamped = max(E_edges[0], min(E, E_edges[N_E]));
    if (E_clamped <= E_edges[0]) {
        E_bin = 0;
    } else if (E_clamped >= E_edges[N_E]) {
        E_bin = N_E - 1;
    } else {
        int lo = 0, hi = N_E;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (E_edges[mid + 1] <= E_clamped) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        E_bin = lo;
    }
    E_bin = max(0, min(E_bin, N_E - 1));

    // Encode to coarse block and local index
    uint32_t b_theta = theta_bin / N_theta_local;
    uint32_t b_E = E_bin / N_E_local;
    uint32_t bid = encode_block(b_theta, b_E);

    int theta_local = theta_bin % N_theta_local;
    int E_local = E_bin % N_E_local;
    uint16_t lidx = encode_local_idx(theta_local, E_local);

    // Emit to bucket
    device_emit_to_bucket(bucket, bid, lidx, weight);
}

// ============================================================================
// 3D Phase-Space Emission with Linear Interpolation (theta, E, x_sub)
// ============================================================================

__device__ inline void device_emit_component_to_bucket_3d_interp(
    DeviceOutflowBucket& bucket,
    float theta,            // Polar angle [rad]
    float E,                // Energy [MeV]
    float weight,           // Statistical weight
    int x_sub,              // Sub-cell x bin (0-3)
    const float* __restrict__ theta_edges,  // Angular grid edges
    const float* __restrict__ E_edges,      // Energy grid edges
    int N_theta,            // Number of angular bins
    int N_E,                // Number of energy bins
    int N_theta_local,      // Local angular bins per block (8)
    int N_E_local           // Local energy bins per block (4)
) {
    if (weight <= 0.0f) return;

    // Get grid bounds
    float theta_min = theta_edges[0];
    float theta_max = theta_edges[N_theta];
    float E_min = E_edges[0];
    float E_max = E_edges[N_E];

    // Clamp values to grid bounds
    theta = fmaxf(theta_min, fminf(theta, theta_max));
    E = fmaxf(E_min, fminf(E, E_max));

    // Calculate continuous bin positions
    float dtheta = (theta_max - theta_min) / N_theta;
    float theta_cont = (theta - theta_min) / dtheta;
    int theta_bin = (int)theta_cont;
    float frac_theta = theta_cont - theta_bin;

    // H6 FIX: Use piecewise-uniform energy grid (Option D2) instead of log-spaced
    // Binary search in E_edges to find the bin containing energy E
    int E_bin = 0;
    float frac_E = 0.0f;
    if (E <= E_edges[0]) {
        E_bin = 0;
        frac_E = 0.0f;
    } else if (E >= E_edges[N_E]) {
        E_bin = N_E - 1;
        frac_E = 0.0f;
    } else {
        int lo = 0, hi = N_E;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (E_edges[mid + 1] <= E) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        E_bin = lo;
        // Calculate fractional position within bin for interpolation
        float E_bin_lower = E_edges[E_bin];
        float E_bin_upper = E_edges[E_bin + 1];
        float bin_width = E_bin_upper - E_bin_lower;
        if (bin_width > 1e-10f) {
            frac_E = (E - E_bin_lower) / bin_width;
        }
    }

    // Handle boundary conditions
    if (theta_bin >= N_theta - 1) {
        theta_bin = N_theta - 1;
        frac_theta = 0.0f;
    }
    if (E_bin >= N_E - 1) {
        E_bin = N_E - 1;
        frac_E = 0.0f;
    }

    // Bilinear interpolation weights
    float w00 = (1.0f - frac_theta) * (1.0f - frac_E);
    float w10 = frac_theta * (1.0f - frac_E);
    float w01 = (1.0f - frac_theta) * frac_E;
    float w11 = frac_theta * frac_E;

    for (int ti = 0; ti < 2; ++ti) {
        for (int ei = 0; ei < 2; ++ei) {
            int t_bin = theta_bin + ti;
            int e_bin = E_bin + ei;

            if (t_bin >= N_theta || e_bin >= N_E) continue;

            float w = (ti == 0 && ei == 0) ? w00 :
                      (ti == 1 && ei == 0) ? w10 :
                      (ti == 0 && ei == 1) ? w01 : w11;

            if (w <= 0.0f) continue;

            uint32_t b_theta = t_bin / N_theta_local;
            uint32_t b_E = e_bin / N_E_local;
            uint32_t bid = encode_block(b_theta, b_E);

            int theta_local = t_bin % N_theta_local;
            int E_local = e_bin % N_E_local;

            uint16_t lidx = encode_local_idx_3d(theta_local, E_local, x_sub);
            device_emit_to_bucket(bucket, bid, lidx, weight * w);
        }
    }
}

// ============================================================================
// 2D Phase-Space Emission with Linear Interpolation (theta, E)
// ============================================================================

__device__ inline void device_emit_component_to_bucket_interp(
    DeviceOutflowBucket& bucket,
    float theta,            // Polar angle [rad]
    float E,                // Energy [MeV]
    float weight,           // Statistical weight
    const float* __restrict__ theta_edges,  // Angular grid edges
    const float* __restrict__ E_edges,      // Energy grid edges
    int N_theta,            // Number of angular bins
    int N_E,                // Number of energy bins
    int N_theta_local,      // Local angular bins per block (8)
    int N_E_local           // Local energy bins per block (4)
) {
    if (weight <= 0.0f) return;

    // Get grid bounds
    float theta_min = theta_edges[0];
    float theta_max = theta_edges[N_theta];
    float E_min = E_edges[0];
    float E_max = E_edges[N_E];

    // Clamp values to grid bounds
    theta = fmaxf(theta_min, fminf(theta, theta_max));
    E = fmaxf(E_min, fminf(E, E_max));

    // Calculate continuous bin positions
    float dtheta = (theta_max - theta_min) / N_theta;
    float theta_cont = (theta - theta_min) / dtheta;
    int theta_bin = (int)theta_cont;
    float frac_theta = theta_cont - theta_bin;

    // H6 FIX: Use piecewise-uniform energy grid (Option D2) instead of log-spaced
    // Binary search in E_edges to find the bin containing energy E
    int E_bin = 0;
    float frac_E = 0.0f;
    if (E <= E_edges[0]) {
        E_bin = 0;
        frac_E = 0.0f;
    } else if (E >= E_edges[N_E]) {
        E_bin = N_E - 1;
        frac_E = 0.0f;
    } else {
        int lo = 0, hi = N_E;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (E_edges[mid + 1] <= E) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        E_bin = lo;
        // Calculate fractional position within bin for interpolation
        float E_bin_lower = E_edges[E_bin];
        float E_bin_upper = E_edges[E_bin + 1];
        float bin_width = E_bin_upper - E_bin_lower;
        if (bin_width > 1e-10f) {
            frac_E = (E - E_bin_lower) / bin_width;
        }
    }

    // Handle boundary conditions
    if (theta_bin >= N_theta - 1) {
        theta_bin = N_theta - 1;
        frac_theta = 0.0f;
    }
    if (E_bin >= N_E - 1) {
        E_bin = N_E - 1;
        frac_E = 0.0f;
    }

    // Bilinear interpolation weights
    float w00 = (1.0f - frac_theta) * (1.0f - frac_E);
    float w10 = frac_theta * (1.0f - frac_E);
    float w01 = (1.0f - frac_theta) * frac_E;
    float w11 = frac_theta * frac_E;

    for (int ti = 0; ti < 2; ++ti) {
        for (int ei = 0; ei < 2; ++ei) {
            int t_bin = theta_bin + ti;
            int e_bin = E_bin + ei;

            if (t_bin >= N_theta || e_bin >= N_E) continue;

            float w = (ti == 0 && ei == 0) ? w00 :
                      (ti == 1 && ei == 0) ? w10 :
                      (ti == 0 && ei == 1) ? w01 : w11;

            if (w <= 0.0f) continue;

            uint32_t b_theta = t_bin / N_theta_local;
            uint32_t b_E = e_bin / N_E_local;
            uint32_t bid = encode_block(b_theta, b_E);

            int theta_local = t_bin % N_theta_local;
            int E_local = e_bin % N_E_local;

            uint16_t lidx = encode_local_idx(theta_local, E_local);
            device_emit_to_bucket(bucket, bid, lidx, weight * w);
        }
    }
}

// ============================================================================
// 4D Phase-Space Emission (theta, E, x_sub, z_sub)
// FIX Problem 1: Added z_sub for complete particle position tracking
// ============================================================================

__device__ inline void device_emit_component_to_bucket_4d(
    DeviceOutflowBucket& bucket,
    float theta,            // Polar angle [rad]
    float E,                // Energy [MeV]
    float weight,           // Statistical weight
    int x_sub,              // Sub-cell x bin (0-3)
    int z_sub,              // Sub-cell z bin (0-3)
    const float* __restrict__ theta_edges,  // Angular grid edges
    const float* __restrict__ E_edges,      // Energy grid edges
    int N_theta,            // Number of angular bins
    int N_E,                // Number of energy bins
    int N_theta_local,      // Local angular bins per block (8)
    int N_E_local           // Local energy bins per block (4)
) {
    if (weight <= 0.0f) return;

    // Find global bins
    int theta_bin = 0;
    int E_bin = 0;

    // For uniform theta grid
    float theta_min = theta_edges[0];
    float theta_max = theta_edges[N_theta];
    float dtheta = (theta_max - theta_min) / N_theta;
    theta_bin = (int)((theta - theta_min) / dtheta);
    theta_bin = max(0, min(theta_bin, N_theta - 1));

    // Option D2: Use binary search for energy bin (works for both log-spaced and piecewise-uniform)
    // Binary search in E_edges to find the bin containing energy E
    float E_clamped = max(E_edges[0], min(E, E_edges[N_E]));
    if (E_clamped <= E_edges[0]) {
        E_bin = 0;
    } else if (E_clamped >= E_edges[N_E]) {
        E_bin = N_E - 1;
    } else {
        int lo = 0, hi = N_E;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (E_edges[mid + 1] <= E_clamped) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        E_bin = lo;
    }
    E_bin = max(0, min(E_bin, N_E - 1));

    // Encode to coarse block and local index
    uint32_t b_theta = theta_bin / N_theta_local;
    uint32_t b_E = E_bin / N_E_local;
    uint32_t bid = encode_block(b_theta, b_E);

    int theta_local = theta_bin % N_theta_local;
    int E_local = E_bin % N_E_local;

    // 4D local index encoding with x_sub and z_sub
    uint16_t lidx = encode_local_idx_4d(theta_local, E_local, x_sub, z_sub);

    // Emit to bucket
    device_emit_to_bucket(bucket, bid, lidx, weight);
}

// ============================================================================
// 4D Phase-Space Emission with Linear Interpolation (theta, E, x_sub, z_sub)
// ============================================================================
// Uses bilinear interpolation in theta-E space for improved accuracy
// Distributes particle weight across 4 neighboring phase-space bins
//
// Interpolation diagram (theta-E plane):
//
//         E_bin+1
//            ↑
//            │  (0,1)        (1,1)
//            │    o------------o
//            │    │            │
//            │    │  ● value   │  frac_E
//            │    │            │
//            │    o------------o
//         E_bin│  (0,0)        (1,1)
//            │
//            └────────────────────→
//          theta_bin        theta_bin+1
//                frac_theta
//
// Weight distribution:
//   w00 = (1-frac_theta) * (1-frac_E)  → bin (theta_bin, E_bin)
//   w10 = frac_theta     * (1-frac_E)  → bin (theta_bin+1, E_bin)
//   w01 = (1-frac_theta) * frac_E      → bin (theta_bin, E_bin+1)
//   w11 = frac_theta     * frac_E      → bin (theta_bin+1, E_bin+1)
// ============================================================================

__device__ inline void device_emit_component_to_bucket_4d_interp(
    DeviceOutflowBucket& bucket,
    float theta,            // Polar angle [rad]
    float E,                // Energy [MeV]
    float weight,           // Statistical weight
    int x_sub,              // Sub-cell x bin (0-3)
    int z_sub,              // Sub-cell z bin (0-3)
    const float* __restrict__ theta_edges,  // Angular grid edges
    const float* __restrict__ E_edges,      // Energy grid edges
    int N_theta,            // Number of angular bins
    int N_E,                // Number of energy bins
    int N_theta_local,      // Local angular bins per block (8)
    int N_E_local           // Local energy bins per block (4)
) {
    if (weight <= 0.0f) return;

    // Get grid bounds
    float theta_min = theta_edges[0];
    float theta_max = theta_edges[N_theta];
    float E_min = E_edges[0];
    float E_max = E_edges[N_E];

    // Clamp values to grid bounds
    theta = fmaxf(theta_min, fminf(theta, theta_max));
    E = fmaxf(E_min, fminf(E, E_max));

    // Calculate continuous bin positions
    float dtheta = (theta_max - theta_min) / N_theta;
    float theta_cont = (theta - theta_min) / dtheta;  // Continuous position in [0, N_theta]
    int theta_bin = (int)theta_cont;                  // Lower bin index
    float frac_theta = theta_cont - theta_bin;         // Fraction toward upper bin

    // H6 FIX: Use piecewise-uniform energy grid (Option D2) instead of log-spaced
    // Binary search in E_edges to find the bin containing energy E
    int E_bin = 0;
    float frac_E = 0.0f;
    if (E <= E_edges[0]) {
        E_bin = 0;
        frac_E = 0.0f;
    } else if (E >= E_edges[N_E]) {
        E_bin = N_E - 1;
        frac_E = 0.0f;
    } else {
        int lo = 0, hi = N_E;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (E_edges[mid + 1] <= E) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        E_bin = lo;
        // Calculate fractional position within bin for interpolation
        float E_bin_lower = E_edges[E_bin];
        float E_bin_upper = E_edges[E_bin + 1];
        float bin_width = E_bin_upper - E_bin_lower;
        if (bin_width > 1e-10f) {
            frac_E = (E - E_bin_lower) / bin_width;
        }
    }

    // Handle boundary conditions
    // If at upper boundary, don't interpolate beyond grid
    if (theta_bin >= N_theta - 1) {
        theta_bin = N_theta - 1;
        frac_theta = 0.0f;
    }
    if (E_bin >= N_E - 1) {
        E_bin = N_E - 1;
        frac_E = 0.0f;
    }

    // Calculate interpolation weights (bilinear)
    float w00 = (1.0f - frac_theta) * (1.0f - frac_E);  // (theta_bin, E_bin)
    float w10 = frac_theta * (1.0f - frac_E);            // (theta_bin+1, E_bin)
    float w01 = (1.0f - frac_theta) * frac_E;            // (theta_bin, E_bin+1)
    float w11 = frac_theta * frac_E;                     // (theta_bin+1, E_bin+1)

    // Emit to 4 neighboring bins (or fewer if at boundary)
    int theta_offsets[2] = {0, 1};
    int E_offsets[2] = {0, 1};
    float weights[4] = {w00, w10, w01, w11};

    for (int ti = 0; ti < 2; ++ti) {
        for (int ei = 0; ei < 2; ++ei) {
            int t_bin = theta_bin + theta_offsets[ti];
            int e_bin = E_bin + E_offsets[ei];

            // Skip if out of bounds (shouldn't happen with clamping above)
            if (t_bin >= N_theta || e_bin >= N_E) continue;

            float w = weights[ti * 2 + ei];
            if (w <= 0.0f) continue;

            // Encode to coarse block and local index
            uint32_t b_theta = t_bin / N_theta_local;
            uint32_t b_E = e_bin / N_E_local;
            uint32_t bid = encode_block(b_theta, b_E);

            int theta_local = t_bin % N_theta_local;
            int E_local = e_bin % N_E_local;

            // 4D local index encoding with x_sub and z_sub
            uint16_t lidx = encode_local_idx_4d(theta_local, E_local, x_sub, z_sub);

            // Emit interpolated weight to bucket
            device_emit_to_bucket(bucket, bid, lidx, weight * w);
        }
    }
}

// Transform x_sub when crossing to neighbor cell
// Face mapping:
//   +z (0): x_offset unchanged
//   -z (1): x_offset unchanged
//   +x (2): source cell right edge -> neighbor left edge (x_sub -> 0)
//   -x (3): source cell left edge -> neighbor right edge (x_sub -> N_x_sub-1)
__device__ inline int device_transform_x_sub_for_neighbor(int x_sub, int face) {
    switch (face) {
        case FACE_X_PLUS:   // Moving to +x neighbor, reset to leftmost bin
            return 0;
        case FACE_X_MINUS:  // Moving to -x neighbor, reset to rightmost bin
            return N_x_sub - 1;
        default:            // Z direction: preserve x_sub
            return x_sub;
    }
}

// Transform z_sub when crossing to neighbor cell
// FIX Problem 1: Added z tracking for particle position preservation
// Face mapping:
//   +z (0): source cell top edge -> neighbor bottom edge (z_sub -> 0)
//   -z (1): source cell bottom edge -> neighbor top edge (z_sub -> N_z_sub-1)
//   +x (2): z_offset unchanged
//   -x (3): z_offset unchanged
__device__ inline int device_transform_z_sub_for_neighbor(int z_sub, int face) {
    switch (face) {
        case FACE_Z_PLUS:   // Moving to +z neighbor, reset to bottommost bin
            return 0;
        case FACE_Z_MINUS:  // Moving to -z neighbor, reset to topmost bin
            return N_z_sub - 1;
        default:            // X direction: preserve z_sub
            return z_sub;
    }
}

// Calculate x offset in neighbor cell after crossing face
// Returns the offset from neighbor cell center
__device__ inline float device_get_neighbor_x_offset(
    float x_exit,       // Exit position in current cell coordinates
    int face,           // Exit face
    float dx            // Cell size
) {
    switch (face) {
        case FACE_X_PLUS:   // Exiting +x face, entering neighbor from left
            return -dx * 0.5f;  // Neighbor cell's left edge (relative to center)
        case FACE_X_MINUS:  // Exiting -x face, entering neighbor from right
            return dx * 0.5f;   // Neighbor cell's right edge (relative to center)
        default:            // Z direction: preserve x offset
            return x_exit - dx * 0.5f;
    }
}

// Clear a bucket (set all values to empty/zero)
// PLAN_MCS Phase B-1: Also initialize Fermi-Eyges moments to zero
__device__ inline void device_bucket_clear(DeviceOutflowBucket& bucket) {
    int idx = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int stride = blockDim.x * blockDim.y * blockDim.z;

    for (int i = idx; i < DEVICE_Kb_out; i += stride) {
        bucket.block_id[i] = DEVICE_EMPTY_BLOCK_ID;
        bucket.local_count[i] = 0;
        for (int j = 0; j < DEVICE_LOCAL_BINS; ++j) {
            bucket.value[i][j] = 0.0f;
        }
    }

    // Initialize Fermi-Eyges moments (single thread)
    if (idx == 0) {
        bucket.moment_A = 0.0f;
        bucket.moment_B = 0.0f;
        bucket.moment_C = 0.0f;
    }
}

// ============================================================================
// Bucket Index Calculation
// ============================================================================

// Calculate the bucket index for a given cell and face
// Buckets are stored per cell, per face (4 faces: +z, -z, +x, -x)
__device__ inline int device_bucket_index(int cell, int face, int Nx, int Nz) {
    return cell * 4 + face;
}

// Determine which face a particle will exit through
// Returns -1 if particle remains in cell
// NOTE: Assumes centered coordinate system: x ∈ [-dx/2, +dx/2], z ∈ [-dz/2, +dz/2]
__device__ inline int device_determine_exit_face(
    float x_old, float z_old,  // Old position [mm]
    float x_new, float z_new,  // New position [mm]
    float dx, float dz         // Cell dimensions [mm]
) {
    // Check boundaries (order matters for corner cases)
    // Using centered coordinate system: [-dx/2, +dx/2] x [-dz/2, +dz/2]
    //
    // CRITICAL FIX: The step is limited to 99.9% of boundary distance, so particles
    // never actually reach the boundary. We need epsilon LARGER than the step
    // safety margin to detect particles that TRY to cross.
    //
    // For half_dz = 0.25 mm, step_limited = 0.25 * 0.999 = 0.24975 mm
    // We need epsilon > (0.25 - 0.24975) = 0.00025 mm
    constexpr float BOUNDARY_EPSILON = 0.001f;  // 0.001 mm (1 μm) tolerance
    float half_dz = dz * 0.5f;
    float half_dx = dx * 0.5f;

    if (z_new >= half_dz - BOUNDARY_EPSILON) return FACE_Z_PLUS;
    if (z_new <= -half_dz + BOUNDARY_EPSILON) return FACE_Z_MINUS;
    if (x_new >= half_dx - BOUNDARY_EPSILON) return FACE_X_PLUS;
    if (x_new <= -half_dx + BOUNDARY_EPSILON) return FACE_X_MINUS;

    return -1;  // Remains in cell
}

// Get neighbor cell index for a given face
__device__ inline int device_get_neighbor(int cell, int face, int Nx, int Nz) {
    int ix = cell % Nx;
    int iz = cell / Nx;

    switch (face) {
        case FACE_Z_PLUS:
            if (iz + 1 >= Nz) return -1;
            return cell + Nx;
        case FACE_Z_MINUS:
            if (iz <= 0) return -1;
            return cell - Nx;
        case FACE_X_PLUS:
            if (ix + 1 >= Nx) return -1;
            return cell + 1;
        case FACE_X_MINUS:
            if (ix <= 0) return -1;
            return cell - 1;
    }
    return -1;
}

// ============================================================================
// Multi-Cell Lateral Spreading Emission (PLAN_fix_scattering)
// ============================================================================
// Emits particle weight to multiple neighboring cells using Gaussian
// lateral spreading instead of theta accumulation.
//
// This implements stochastic interpolation for K2 coarse transport:
// - Weight is distributed across N cells using Gaussian distribution
// - Each cell receives weight proportional to Gaussian CDF
// - Thread-safe using atomic operations
// ============================================================================

// Emit weight with lateral spreading to multiple cells
// Parameters:
//   OutflowBuckets: Array of all buckets (accessed via bucket_index)
//   source_cell:    Current cell index
//   target_z:       Target z row index (typically source_cell/Nx + 1 for +z direction)
//   theta:          Scattering angle (NOT accumulated)
//   E:              Energy
//   weight:         Total weight to distribute
//   x_offset:       Lateral position within source cell (centered: [-dx/2, +dx/2])
//   sigma_x:        Lateral spread standard deviation (mm)
//   dx:             Cell size (mm)
//   Nx, Nz:         Grid dimensions
//   N_spread:       Number of cells to spread across (must be even)
//   theta_edges, E_edges, N_theta, N_E, N_theta_local, N_E_local: Grid info
__device__ inline void device_emit_lateral_spread(
    DeviceOutflowBucket* __restrict__ OutflowBuckets,
    int source_cell,
    int target_z,
    float theta,
    float E,
    float weight,
    float x_offset,            // Lateral position in cell (centered)
    float sigma_x,             // Lateral spread (mm)
    float dx,
    int Nx, int Nz,
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,
    int N_spread = 10          // Number of cells to spread across
) {
    if (weight <= 0.0f || N_spread <= 0) return;

    // Clamp sigma_x to avoid numerical issues
    sigma_x = fmaxf(sigma_x, 1e-6f);

    // If sigma_x is very small compared to cell size, just emit to single cell
    if (sigma_x < dx * 0.1f) {
        // Single-cell emission (center)
        int ix_center = source_cell % Nx;
        int iz_source = source_cell / Nx;

        if (target_z >= 0 && target_z < Nz) {
            int emit_cell = ix_center + iz_source * Nx;
            int bucket_idx = device_bucket_index(emit_cell, FACE_Z_PLUS, Nx, Nz);
            DeviceOutflowBucket& bucket = OutflowBuckets[bucket_idx];

            // Use center sub-cell bins
            int x_sub_center = N_x_sub / 2;  // Center x bin
            int z_sub_neighbor = 0;  // Bottom of neighbor cell (entering from -z)

            device_emit_component_to_bucket_4d(
                bucket, theta, E, weight, x_sub_center, z_sub_neighbor,
                theta_edges, E_edges, N_theta, N_E,
                N_theta_local, N_E_local
            );
        }
        return;
    }

    // Calculate Gaussian weight distribution
    // Fixed-size array for device code
    constexpr int MAX_SPREAD_CELLS = 10;
    int N_actual = fminf(N_spread, MAX_SPREAD_CELLS);

    float weights[MAX_SPREAD_CELLS];

    // x_mean is the mean lateral position (x_offset = position within cell)
    device_gaussian_spread_weights(weights, x_offset, sigma_x, dx, N_actual);

    // Get source cell indices
    int ix_source = source_cell % Nx;
    int iz_source = source_cell / Nx;

    // Emit to each target cell with distributed weight
    int x_offset_start = -(N_actual / 2);  // -5 for N=10

    for (int i = 0; i < N_actual; i++) {
        float w_frac = weights[i];
        if (w_frac < 1e-6f) continue;  // Skip negligible weights

        int x_offset = x_offset_start + i;
        int ix_target = ix_source + x_offset;

        // Boundary check
        if (ix_target < 0 || ix_target >= Nx) continue;
        if (target_z < 0 || target_z >= Nz) continue;

        // KEY: To send weight to cell (ix_target, target_z), we emit to the +z bucket
        // of cell (ix_target, iz_source). K4 will then transfer it to (ix_target, target_z).
        int emit_cell = ix_target + iz_source * Nx;
        int bucket_idx = device_bucket_index(emit_cell, FACE_Z_PLUS, Nx, Nz);
        DeviceOutflowBucket& bucket = OutflowBuckets[bucket_idx];

        // Sub-cell bins for entry into neighbor cell
        int x_sub_target;
        if (x_offset < 0) {
            x_sub_target = N_x_sub - 1;  // Rightmost bin (entering from left)
        } else if (x_offset > 0) {
            x_sub_target = 0;  // Leftmost bin (entering from right)
        } else {
            x_sub_target = N_x_sub / 2;  // Center
        }

        int z_sub_neighbor = 0;  // Bottom of neighbor cell

        // Emit with distributed weight
        device_emit_component_to_bucket_4d(
            bucket, theta, E, weight * w_frac, x_sub_target, z_sub_neighbor,
            theta_edges, E_edges, N_theta, N_E,
            N_theta_local, N_E_local
        );
    }
}
