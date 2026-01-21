#include <gtest/gtest.h>
#include "perf/kernel_profiler.hpp"
#include <thread>
#include <chrono>

TEST(KernelProfilerTest, InitialState) {
    KernelProfiler profiler;

    EXPECT_FLOAT_EQ(profiler.get_time_ms("test_kernel"), 0.0f);
    EXPECT_FLOAT_EQ(profiler.get_total_time_ms(), 0.0f);
}

TEST(KernelProfilerTest, SingleKernelTiming) {
    KernelProfiler profiler;

    profiler.start("test_kernel");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.stop("test_kernel");

    float time_ms = profiler.get_time_ms("test_kernel");
    EXPECT_GT(time_ms, 8.0f);  // At least 8ms (allowing for timing variance)
    EXPECT_LT(time_ms, 50.0f); // But less than 50ms
}

TEST(KernelProfilerTest, MultipleKernelTimings) {
    KernelProfiler profiler;

    profiler.start("kernel_a");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    profiler.stop("kernel_a");

    profiler.start("kernel_b");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.stop("kernel_b");

    float time_a = profiler.get_time_ms("kernel_a");
    float time_b = profiler.get_time_ms("kernel_b");

    EXPECT_GT(time_a, 3.0f);
    EXPECT_GT(time_b, 8.0f);
    EXPECT_GT(time_b, time_a);  // kernel_b should take longer

    float total = profiler.get_total_time_ms();
    EXPECT_GT(total, time_a + time_b - 1.0f);  // Allow 1ms measurement error
}

TEST(KernelProfilerTest, AccumulatedTiming) {
    KernelProfiler profiler;

    // Run kernel multiple times
    profiler.start("repeated_kernel");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    profiler.stop("repeated_kernel");

    profiler.start("repeated_kernel");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    profiler.stop("repeated_kernel");

    float time_ms = profiler.get_time_ms("repeated_kernel");
    EXPECT_GT(time_ms, 8.0f);  // Should accumulate
}

TEST(KernelProfilerTest, ReportGeneration) {
    KernelProfiler profiler;

    profiler.start("kernel1");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    profiler.stop("kernel1");

    // Just ensure it doesn't crash
    profiler.print_report();

    EXPECT_TRUE(true);
}

TEST(KernelProfilerTest, StopWithoutStart) {
    KernelProfiler profiler;

    // Stopping without starting should not crash
    profiler.stop("nonexistent_kernel");

    EXPECT_FLOAT_EQ(profiler.get_time_ms("nonexistent_kernel"), 0.0f);
}
