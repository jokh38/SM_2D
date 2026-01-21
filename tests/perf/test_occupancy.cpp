#include <gtest/gtest.h>
#include "perf/occupancy_analyzer.hpp"

TEST(OccupancyAnalyzerTest, InitialState) {
    OccupancyAnalyzer analyzer;

    EXPECT_FLOAT_EQ(analyzer.get_occupancy_percent(), 50.0f);
    EXPECT_EQ(analyzer.get_limiting_factor(), "registers");
}

TEST(OccupancyAnalyzerTest, AnalyzeKernel) {
    OccupancyAnalyzer analyzer;

    analyzer.analyze_kernel("test_kernel");

    // Stub implementation - just ensure it doesn't crash
    EXPECT_FLOAT_EQ(analyzer.get_occupancy_percent(), 50.0f);
    EXPECT_EQ(analyzer.get_limiting_factor(), "registers");
}

TEST(OccupancyAnalyzerTest, MultipleKernels) {
    OccupancyAnalyzer analyzer;

    analyzer.analyze_kernel("kernel_a");
    float occ_a = analyzer.get_occupancy_percent();

    analyzer.analyze_kernel("kernel_b");
    float occ_b = analyzer.get_occupancy_percent();

    // Stub returns same values for all kernels
    EXPECT_FLOAT_EQ(occ_a, 50.0f);
    EXPECT_FLOAT_EQ(occ_b, 50.0f);
}

TEST(OccupancyAnalyzerTest, LimitingFactorRetrieval) {
    OccupancyAnalyzer analyzer;

    analyzer.analyze_kernel("some_kernel");
    std::string factor = analyzer.get_limiting_factor();

    EXPECT_FALSE(factor.empty());
    EXPECT_EQ(factor, "registers");
}
