#include "perf/fine_batch_planner.hpp"

#include <algorithm>

namespace sm_2d {

FineBatchPlan plan_fine_batch_cells(
    int requested_cells,
    int candidate_cells,
    std::size_t budget_bytes,
    std::size_t bytes_per_cell
) {
    FineBatchPlan plan;
    plan.requested_cells = requested_cells;
    plan.bytes_per_cell = bytes_per_cell;
    plan.budget_bytes = budget_bytes;

    if (candidate_cells <= 0 || bytes_per_cell == 0) {
        return plan;
    }

    const std::size_t max_by_budget = budget_bytes / bytes_per_cell;
    const int budget_cap = (max_by_budget > static_cast<std::size_t>(candidate_cells))
        ? candidate_cells
        : static_cast<int>(max_by_budget);

    plan.max_cells = std::max(0, budget_cap);
    if (requested_cells <= 0) {
        plan.planned_cells = plan.max_cells;
        return plan;
    }

    plan.planned_cells = std::min(requested_cells, plan.max_cells);
    plan.clamped = requested_cells > plan.max_cells;
    return plan;
}

} // namespace sm_2d
