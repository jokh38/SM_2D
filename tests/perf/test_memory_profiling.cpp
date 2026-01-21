#include <gtest/gtest.h>
#include "perf/memory_profiler.hpp"

TEST(MemoryProfilerTest, InitialState) {
    MemoryProfiler profiler;

    EXPECT_EQ(profiler.get_peak_bytes(), 0ULL);
    EXPECT_EQ(profiler.get_current_bytes(), 0ULL);
}

TEST(MemoryProfilerTest, BudgetSetting) {
    MemoryProfiler profiler;
    uint64_t budget = 5ULL * 1024 * 1024 * 1024;  // 5GB

    profiler.set_budget(budget);

    auto status = profiler.check_budget();
    EXPECT_EQ(status.budget, budget);
    EXPECT_FALSE(status.over_budget);
}

TEST(MemoryProfilerTest, StartStopTracking) {
    MemoryProfiler profiler;

    profiler.start();
    EXPECT_GT(profiler.get_current_bytes(), 0ULL);

    profiler.stop();
    uint64_t peak = profiler.get_peak_bytes();
    EXPECT_GT(peak, 0ULL);
    EXPECT_EQ(profiler.get_current_bytes(), peak);
}

TEST(MemoryProfilerTest, BudgetCheck) {
    MemoryProfiler profiler;
    profiler.set_budget(1ULL * 1024 * 1024);  // 1MB budget

    profiler.start();
    profiler.stop();

    auto status = profiler.check_budget();
    EXPECT_TRUE(status.over_budget);  // Should exceed 1MB
}

TEST(MemoryProfilerTest, ReportGeneration) {
    MemoryProfiler profiler;
    profiler.set_budget(7ULL * 1024 * 1024 * 1024);  // 7GB

    profiler.start();
    profiler.stop();

    // Just ensure it doesn't crash
    profiler.print_report();

    EXPECT_TRUE(true);
}
