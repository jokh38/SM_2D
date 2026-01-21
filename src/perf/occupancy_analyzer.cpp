#include "perf/occupancy_analyzer.hpp"

void OccupancyAnalyzer::analyze_kernel(const std::string& kernel_name) {
    // Stub: Real analysis requires Nsight Compute
    // Set reasonable defaults for RTX 2080
    occupancy_ = 50.0f;
    limiting_factor_ = "registers";
}

float OccupancyAnalyzer::get_occupancy_percent() const {
    return occupancy_;
}

std::string OccupancyAnalyzer::get_limiting_factor() const {
    return limiting_factor_;
}
