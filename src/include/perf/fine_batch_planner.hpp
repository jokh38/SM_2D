#pragma once

#include <cstddef>

namespace sm_2d {

struct FineBatchPlan {
    int requested_cells = 0;
    int max_cells = 0;
    int planned_cells = 0;
    std::size_t bytes_per_cell = 0;
    std::size_t budget_bytes = 0;
    bool clamped = false;
};

FineBatchPlan plan_fine_batch_cells(
    int requested_cells,
    int candidate_cells,
    std::size_t budget_bytes,
    std::size_t bytes_per_cell
);

} // namespace sm_2d
