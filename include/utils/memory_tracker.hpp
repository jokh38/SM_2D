#pragma once
#ifdef MOCK_CUDA
#include "cuda_runtime_mock.hpp"
#else
#include <cuda_runtime.h>
#endif
#include <cstddef>

class MemoryTracker {
public:
    MemoryTracker();
    size_t get_current_usage() const;
    size_t get_total_memory() const;
    void set_warning_threshold(size_t bytes);
    bool check_warning() const;
    void simulate_allocation(size_t bytes);
    size_t get_peak_usage() const;
private:
    size_t warning_threshold_;
    size_t simulated_bytes_;
    mutable size_t peak_usage_;
};