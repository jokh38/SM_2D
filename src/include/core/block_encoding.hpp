#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Block ID: 24-bit encoding
// Bits 0-11:   b_theta (12 bits)
// Bits 12-23:  b_E (12 bits)

// P3 FIX: Make these functions available on both host and device
__host__ __device__ inline uint32_t encode_block(uint32_t b_theta, uint32_t b_E) {
    return (b_theta & 0xFFF) | ((b_E & 0xFFF) << 12);
}

__host__ __device__ inline void decode_block(uint32_t block_id, uint32_t& b_theta, uint32_t& b_E) {
    b_theta = block_id & 0xFFF;
    b_E = (block_id >> 12) & 0xFFF;
}

constexpr uint32_t EMPTY_BLOCK_ID = 0xFFFFFFFF;
