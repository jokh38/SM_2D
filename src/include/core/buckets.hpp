#pragma once
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"
#include <array>
#include <cstdint>

constexpr int Kb_out = 64;

// PLAN_MCS Phase B-1: CPU-side OutflowBucket structure
// Must match GPU DeviceOutflowBucket layout for memory transfer
//
// Fermi-Eyges moment tracking for O(z^(3/2)) lateral spreading:
//   moment_A = ⟨θ²⟩ : Angular variance [rad²]
//   moment_B = ⟨xθ⟩ : Position-angle covariance [mm·rad]
//   moment_C = ⟨x²⟩ : Position variance [mm²]
struct OutflowBucket {
    std::array<uint32_t, Kb_out> block_id;
    std::array<uint16_t, Kb_out> local_count;
    std::array<std::array<float, LOCAL_BINS>, Kb_out> value;

    // Fermi-Eyges moments for lateral spreading (Phase B-1)
    float moment_A;  // ⟨θ²⟩ - Angular variance [rad²]
    float moment_B;  // ⟨xθ⟩ - Position-angle covariance [mm·rad]
    float moment_C;  // ⟨x²⟩ - Position variance [mm²]

    OutflowBucket();

    int find_or_allocate_slot(uint32_t bid);
    void clear();
};

inline void atomic_add(float& target, float value) {
    target += value;
}
