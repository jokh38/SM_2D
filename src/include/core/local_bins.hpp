#pragma once
#include <cstdint>
#include <cuda_runtime.h>

constexpr int N_theta_local = 8;
constexpr int N_E_local = 4;
constexpr int N_x_sub = 4;                    // Sub-cell x position bins per cell
constexpr int LOCAL_BINS = N_theta_local * N_E_local * N_x_sub;  // = 128

// ============================================================================
// 3D Local Index Encoding (theta_local, E_local, x_sub)
// Layout: lidx = theta_local + N_theta_local * (E_local + N_E_local * x_sub)
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
