#include <gtest/gtest.h>

#include "perf/fine_batch_planner.hpp"
#include "perf/memory_preflight.hpp"

TEST(FineBatchPlannerTest, AutoModeUsesBudgetCap) {
    sm_2d::FineBatchPlan plan = sm_2d::plan_fine_batch_cells(
        0,      // auto
        1000,   // candidate cells
        64000,  // budget bytes
        128     // bytes per cell
    );

    EXPECT_EQ(plan.max_cells, 500);
    EXPECT_EQ(plan.planned_cells, 500);
    EXPECT_FALSE(plan.clamped);
}

TEST(FineBatchPlannerTest, RequestedModeClampsToBudgetCap) {
    sm_2d::FineBatchPlan plan = sm_2d::plan_fine_batch_cells(
        700,    // requested
        1000,   // candidate cells
        64000,  // budget bytes
        128     // bytes per cell
    );

    EXPECT_EQ(plan.max_cells, 500);
    EXPECT_EQ(plan.planned_cells, 500);
    EXPECT_TRUE(plan.clamped);
}

TEST(MemoryPreflightTest, PassesForValidationSizedGridWith8GiBBudget) {
    sm_2d::MemoryPreflightInput input;
    input.Nx = 100;
    input.Nz = 320;
    input.N_theta = 36;
    input.N_E = 991;
    input.preflight_vram_margin = 0.85f;
    input.fine_batch_requested_cells = 0;
    input.free_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    input.total_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

    const sm_2d::MemoryPreflightResult result = sm_2d::run_memory_preflight(input);
    EXPECT_TRUE(result.ok);
    EXPECT_GT(result.estimate.total_required_bytes, 0u);
    EXPECT_GT(result.fine_batch_plan.planned_cells, 0);
}

TEST(MemoryPreflightTest, FailsForLargeGridAtSameBudget) {
    sm_2d::MemoryPreflightInput input;
    input.Nx = 200;
    input.Nz = 640;
    input.N_theta = 36;
    input.N_E = 991;
    input.preflight_vram_margin = 0.85f;
    input.fine_batch_requested_cells = 0;
    input.free_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    input.total_vram_bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;

    const sm_2d::MemoryPreflightResult result = sm_2d::run_memory_preflight(input);
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.message.find("exceeds usable"), std::string::npos);
}
