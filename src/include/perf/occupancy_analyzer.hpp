#pragma once
#include <string>

class OccupancyAnalyzer {
public:
    void analyze_kernel(const std::string& kernel_name);
    float get_occupancy_percent() const;
    std::string get_limiting_factor() const;
private:
    float occupancy_ = 50.0f;  // Estimated
    std::string limiting_factor_ = "registers";
};
