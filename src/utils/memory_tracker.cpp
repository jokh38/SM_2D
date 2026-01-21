#include "utils/memory_tracker.hpp"
#include <iostream>

MemoryTracker::MemoryTracker()
    : warning_threshold_(6ULL * 1024 * 1024 * 1024)
    , simulated_bytes_(0)
    , peak_usage_(0)
{}

size_t MemoryTracker::get_current_usage() const {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    size_t used = total - free + simulated_bytes_;
    peak_usage_ = std::max(peak_usage_, used);
    return used;
}

size_t MemoryTracker::get_total_memory() const {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    return total;
}

void MemoryTracker::set_warning_threshold(size_t bytes) {
    warning_threshold_ = bytes;
}

bool MemoryTracker::check_warning() const {
    return get_current_usage() > warning_threshold_;
}

void MemoryTracker::simulate_allocation(size_t bytes) {
    simulated_bytes_ += bytes;
}

size_t MemoryTracker::get_peak_usage() const {
    return peak_usage_;
}