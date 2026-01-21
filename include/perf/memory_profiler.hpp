#pragma once
#include <cstdint>

struct MemoryBudget {
    uint64_t budget;
    uint64_t peak_bytes;
    bool over_budget;
};

class MemoryProfiler {
public:
    MemoryProfiler();
    void set_budget(uint64_t bytes);
    void start();
    void stop();
    uint64_t get_peak_bytes() const;
    uint64_t get_current_bytes() const;
    MemoryBudget check_budget() const;
    void print_report() const;
private:
    uint64_t budget_;
    uint64_t baseline_;
    uint64_t peak_;
    bool running_;
};
