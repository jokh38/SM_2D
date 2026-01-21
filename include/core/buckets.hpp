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

    int find_or_allocate_slot(uint32_t bid);
    void clear();
};

inline void atomic_add(float& target, float value) {
    target += value;
}
