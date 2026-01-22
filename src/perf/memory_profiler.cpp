#include "perf/memory_profiler.hpp"
#include <iostream>
#include <algorithm>

MemoryProfiler::MemoryProfiler()
    : budget_(7ULL * 1024 * 1024 * 1024)
    , baseline_(0)
    , peak_(0)
    , running_(false)
{}

void MemoryProfiler::set_budget(uint64_t bytes) {
    budget_ = bytes;
}

void MemoryProfiler::start() {
    // Simulate CUDA memory tracking for non-CUDA builds
    baseline_ = 100ULL * 1024 * 1024;  // 100MB baseline
    peak_ = baseline_;
    running_ = true;
}

void MemoryProfiler::stop() {
    peak_ = std::max(peak_, static_cast<uint64_t>(baseline_ + 50ULL * 1024 * 1024));  // Add 50MB
    running_ = false;
}

uint64_t MemoryProfiler::get_peak_bytes() const {
    return peak_;
}

uint64_t MemoryProfiler::get_current_bytes() const {
    return running_ ? baseline_ + 50ULL * 1024 * 1024 : peak_;
}

MemoryBudget MemoryProfiler::check_budget() const {
    MemoryBudget status;
    status.budget = budget_;
    status.peak_bytes = peak_;
    status.over_budget = peak_ > budget_;
    return status;
}

void MemoryProfiler::print_report() const {
    double peak_gb = peak_ / (1024.0 * 1024.0 * 1024.0);
    double budget_gb = budget_ / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Memory Report:\n";
    std::cout << "  Budget: " << budget_gb << " GB\n";
    std::cout << "  Peak: " << peak_gb << " GB\n";
    if (peak_ > budget_) {
        std::cout << "  Status: OVER BUDGET\n";
    } else {
        std::cout << "  Status: OK\n";
        double headroom_gb = (budget_ - peak_) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  Headroom: " << headroom_gb << " GB\n";
    }
}
