#pragma once

#include <cstddef>
#include <string>

#include "perf/fine_batch_planner.hpp"

namespace sm_2d {

struct MemoryPreflightInput {
    int Nx = 0;
    int Nz = 0;
    int N_theta = 0;
    int N_E = 0;
    int fine_batch_requested_cells = 0;
    float preflight_vram_margin = 0.85f;

    // Optional override for testing.
    std::size_t free_vram_bytes = 0;
    std::size_t total_vram_bytes = 0;
};

struct DenseMemoryEstimate {
    std::size_t psi_buffers_bytes = 0;
    std::size_t outflow_buckets_bytes = 0;
    std::size_t pipeline_state_bytes = 0;
    std::size_t lut_bytes = 0;
    std::size_t runtime_overhead_bytes = 0;
    std::size_t total_required_bytes = 0;
    std::size_t bytes_per_dense_cell = 0;
};

struct MemoryPreflightResult {
    bool ok = false;
    std::string message;
    std::size_t free_vram_bytes = 0;
    std::size_t total_vram_bytes = 0;
    std::size_t usable_vram_bytes = 0;
    DenseMemoryEstimate estimate;
    FineBatchPlan fine_batch_plan;
};

DenseMemoryEstimate estimate_dense_k1k6_memory(
    int Nx,
    int Nz,
    int N_theta,
    int N_E
);

MemoryPreflightResult run_memory_preflight(const MemoryPreflightInput& input);

std::string format_binary_bytes(std::size_t bytes);

} // namespace sm_2d
