#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"

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
constexpr int DEVICE_Kb_out = 64;
constexpr int DEVICE_LOCAL_BINS = 32;

// Device-accessible OutflowBucket structure
// Matches CPU OutflowBucket layout for direct memory transfer
struct DeviceOutflowBucket {
    uint32_t block_id[DEVICE_Kb_out];
    uint16_t local_count[DEVICE_Kb_out];
    float value[DEVICE_Kb_out][DEVICE_LOCAL_BINS];
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

// Emit a component to a bucket with full phase-space encoding
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

    // For log-spaced E grid
    float E_min = E_edges[0];
    float E_max = E_edges[N_E];
    float log_E = logf(E);
    float log_E_min = logf(E_min);
    float log_E_max = logf(E_max);
    float dlog = (log_E_max - log_E_min) / N_E;
    E_bin = (int)((log_E - log_E_min) / dlog);
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

// Clear a bucket (set all values to empty/zero)
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
}

// ============================================================================
// Bucket Index Calculation
// ============================================================================

// Calculate the bucket index for a given cell and face
// Buckets are stored per cell, per face (4 faces: +z, -z, +x, -x)
__device__ inline int device_bucket_index(int cell, int face, int Nx, int Nz) {
    return cell * 4 + face;
}

// Face definitions (must match K4 get_neighbor)
enum DeviceFace {
    FACE_Z_PLUS = 0,   // +z direction
    FACE_Z_MINUS = 1,  // -z direction
    FACE_X_PLUS = 2,   // +x direction
    FACE_X_MINUS = 3   // -x direction
};

// Determine which face a particle will exit through
// Returns -1 if particle remains in cell
__device__ inline int device_determine_exit_face(
    float x_old, float z_old,  // Old position [mm]
    float x_new, float z_new,  // New position [mm]
    float dx, float dz         // Cell dimensions [mm]
) {
    // Check boundaries (order matters for corner cases)
    if (z_new >= dz) return FACE_Z_PLUS;
    if (z_new < 0) return FACE_Z_MINUS;
    if (x_new >= dx) return FACE_X_PLUS;
    if (x_new < 0) return FACE_X_MINUS;

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
