#include <gtest/gtest.h>
#include "utils/memory_tracker.hpp"

TEST(MemoryBudgetTest, InitialUsageLow) {
    MemoryTracker tracker;
    size_t initial = tracker.get_current_usage();
    EXPECT_LT(initial, 500ULL * 1024 * 1024);
}

TEST(MemoryBudgetTest, AllocationTracked) {
    MemoryTracker tracker;
    size_t before = tracker.get_current_usage();
    float* d_ptr = nullptr;
    size_t alloc_size = 100ULL * 1024 * 1024;
    cudaMalloc(&d_ptr, alloc_size);
    size_t after = tracker.get_current_usage();
    EXPECT_GE(after - before, alloc_size * 0.9);
    cudaFree(d_ptr);
}