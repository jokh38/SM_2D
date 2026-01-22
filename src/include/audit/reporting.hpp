#pragma once
#include "audit/conservation.hpp"
#include "audit/global_budget.hpp"
#include <iosfwd>
#include <vector>

void print_cell_weight_report(std::ostream& os, const CellWeightAudit& audit, int cell);
void print_global_report(std::ostream& os, const GlobalAudit& audit);
void print_failed_cells(std::ostream& os, const std::vector<int>& failed_cells);
void print_summary(std::ostream& os, int n_cells, int n_passed, int n_failed, const GlobalAudit& global);
