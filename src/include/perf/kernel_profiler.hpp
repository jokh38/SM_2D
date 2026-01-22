#pragma once
#include <string>
#include <unordered_map>

class KernelProfiler {
public:
    void start(const std::string& kernel_name);
    void stop(const std::string& kernel_name);
    float get_time_ms(const std::string& kernel_name) const;
    float get_total_time_ms() const;
    void print_report() const;
private:
    std::unordered_map<std::string, float> times_ms_;
    std::unordered_map<std::string, long long> starts_;
    float total_time_ms_ = 0;
};
