#pragma once
#include <cstdint>
#include <cuda_runtime.h>

constexpr int N_theta_local = 8;
constexpr int N_E_local = 4;
constexpr int LOCAL_BINS = N_theta_local * N_E_local;  // = 32

// P3 FIX: Make these functions available on both host and device
__host__ __device__ inline uint16_t encode_local_idx(int theta_local, int E_local) {
    return static_cast<uint16_t>(theta_local * N_E_local + E_local);
}

__host__ __device__ inline void decode_local_idx(uint16_t lidx, int& theta_local, int& E_local) {
    theta_local = static_cast<int>(lidx) / N_E_local;
    E_local = static_cast<int>(lidx) % N_E_local;
}

static_assert(LOCAL_BINS <= 65536, "LOCAL_BINS too large for uint16_t");
