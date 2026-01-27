#pragma once
#include <cstdint>

#ifdef SM2D_HAS_CUDA
#include <cuda_runtime.h>
#endif

// For CPU builds, define __host__ and __device__ as empty
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

// NOTE: SPEC v0.8 requires N_theta_local=8, N_E_local=4, LOCAL_BINS=32 (without x_sub/z_sub)
// This code uses extended 4D encoding with x_sub, z_sub for sub-cell position tracking
// To fit in 8GB VRAM, using reduced values (memory optimization)
constexpr int N_theta_local = 4;   // Angular sub-bins per block
constexpr int N_E_local = 2;       // Energy sub-bins per block
constexpr int N_x_sub = 4;        // Sub-cell x position bins per cell
constexpr int N_z_sub = 4;        // Sub-cell z position bins per cell
constexpr int LOCAL_BINS = N_theta_local * N_E_local * N_x_sub * N_z_sub;  // = 4*2*4*4 = 128

// ============================================================================
// 4D Local Index Encoding (theta_local, E_local, x_sub, z_sub)
// Layout: lidx = theta_local + N_theta_local * (E_local + N_E_local * (x_sub + N_x_sub * z_sub))
// ============================================================================
// FIX Problem 1: Added z_sub to preserve z-position information across steps
// This prevents all particles from starting at z_cell = dz/2 each step

__host__ __device__ inline uint16_t encode_local_idx_4d(
    int theta_local, int E_local, int x_sub, int z_sub
) {
    int inner = E_local + N_E_local * (x_sub + N_x_sub * z_sub);
    return static_cast<uint16_t>(theta_local + N_theta_local * inner);
}

__host__ __device__ inline void decode_local_idx_4d(
    uint16_t lidx, int& theta_local, int& E_local, int& x_sub, int& z_sub
) {
    theta_local = static_cast<int>(lidx) % N_theta_local;
    int remainder = static_cast<int>(lidx) / N_theta_local;
    E_local = remainder % N_E_local;
    remainder /= N_E_local;
    x_sub = remainder % N_x_sub;
    z_sub = remainder / N_x_sub;
}

// ============================================================================
// Legacy 3D encoding (deprecated, kept for compatibility during transition)
// ============================================================================

__host__ __device__ inline uint16_t encode_local_idx_3d(
    int theta_local, int E_local, int x_sub
) {
    return static_cast<uint16_t>(
        theta_local + N_theta_local * (E_local + N_E_local * x_sub)
    );
}

__host__ __device__ inline void decode_local_idx_3d(
    uint16_t lidx, int& theta_local, int& E_local, int& x_sub
) {
    theta_local = static_cast<int>(lidx) % N_theta_local;
    int remainder = static_cast<int>(lidx) / N_theta_local;
    E_local = remainder % N_E_local;
    x_sub = remainder / N_E_local;
}

// ============================================================================
// x_offset <-> x_sub conversion
// Sub-cell x range: [-dx/2, +dx/2] divided into N_x_sub equal bins
// bin 0: [-dx/2, -dx/4), bin 1: [-dx/4, 0), bin 2: [0, +dx/4), bin 3: [+dx/4, +dx/2)
// Bin centers: -3/8*dx, -1/8*dx, +1/8*dx, +3/8*dx
// ============================================================================

__host__ __device__ inline float get_x_offset_from_bin(int x_sub, float dx) {
    // Return bin center offset from cell center
    // bin 0 -> -0.375*dx, bin 1 -> -0.125*dx, bin 2 -> +0.125*dx, bin 3 -> +0.375*dx
    return dx * (-0.375f + 0.25f * x_sub);
}

__host__ __device__ inline int get_x_sub_bin(float x_offset, float dx) {
    // x_offset ∈ [-dx/2, +dx/2] → bin ∈ [0, N_x_sub-1]
    float normalized = (x_offset / dx) + 0.5f;  // Map to [0, 1]
    int bin = static_cast<int>(normalized * N_x_sub);
    if (bin < 0) bin = 0;
    if (bin >= N_x_sub) bin = N_x_sub - 1;
    return bin;
}

// ============================================================================
// z_offset <-> z_sub conversion
// Sub-cell z range: [-dz/2, +dz/2] divided into N_z_sub equal bins
// bin 0: [-dz/2, -dz/4), bin 1: [-dz/4, 0), bin 2: [0, +dz/4), bin 3: [+dz/4, +dz/2)
// Bin centers: -3/8*dz, -1/8*dz, +1/8*dz, +3/8*dz
// FIX Problem 1: Added z tracking to preserve particle position across steps
// ============================================================================

__host__ __device__ inline float get_z_offset_from_bin(int z_sub, float dz) {
    // Return bin center offset from cell center
    // bin 0 -> -0.375*dz, bin 1 -> -0.125*dz, bin 2 -> +0.125*dz, bin 3 -> +0.375*dz
    return dz * (-0.375f + 0.25f * z_sub);
}

__host__ __device__ inline int get_z_sub_bin(float z_offset, float dz) {
    // z_offset ∈ [-dz/2, +dz/2] → bin ∈ [0, N_z_sub-1]
    float normalized = (z_offset / dz) + 0.5f;  // Map to [0, 1]
    int bin = static_cast<int>(normalized * N_z_sub);
    if (bin < 0) bin = 0;
    if (bin >= N_z_sub) bin = N_z_sub - 1;
    return bin;
}

// ============================================================================
// Legacy 2D encoding (deprecated, kept for compatibility during transition)
// ============================================================================
__host__ __device__ inline uint16_t encode_local_idx(int theta_local, int E_local) {
    return encode_local_idx_3d(theta_local, E_local, 0);  // Default to x_sub = 0
}

__host__ __device__ inline void decode_local_idx(uint16_t lidx, int& theta_local, int& E_local) {
    int x_sub_dummy;
    decode_local_idx_3d(lidx, theta_local, E_local, x_sub_dummy);
}

static_assert(LOCAL_BINS <= 65536, "LOCAL_BINS too large for uint16_t");
