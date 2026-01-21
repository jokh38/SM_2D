#include "perf/kernel_profiler.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

void KernelProfiler::start(const std::string& kernel_name) {
    starts_[kernel_name] = std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

void KernelProfiler::stop(const std::string& kernel_name) {
    auto it = starts_.find(kernel_name);
    if (it == starts_.end()) return;

    auto end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    long long duration_us = (end - it->second) / 1000;  // Convert to microseconds
    float ms = duration_us / 1000.0f;

    times_ms_[kernel_name] += ms;
    total_time_ms_ += ms;
}

float KernelProfiler::get_time_ms(const std::string& kernel_name) const {
    auto it = times_ms_.find(kernel_name);
    return (it != times_ms_.end()) ? it->second : 0;
}

float KernelProfiler::get_total_time_ms() const {
    return total_time_ms_;
}

void KernelProfiler::print_report() const {
    std::cout << "\n=== Kernel Performance Report ===\n";
    for (const auto& kv : times_ms_) {
        std::cout << std::setw(20) << kv.first << ": "
                  << std::fixed << std::setprecision(2)
                  << kv.second << " ms";
        if (total_time_ms_ > 0) {
            float pct = 100.0f * kv.second / total_time_ms_;
            std::cout << " (" << pct << "%)";
        }
        std::cout << "\n";
    }
    std::cout << "-----------------------------------\n";
    std::cout << "Total: " << total_time_ms_ << " ms\n";
}
