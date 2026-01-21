#include "audit/reporting.hpp"
#include <iostream>
#include <iomanip>

void print_cell_weight_report(std::ostream& os, const CellWeightAudit& audit, int cell) {
    os << "Cell " << cell << ": "
       << "W_in=" << audit.W_in << " "
       << "W_out=" << audit.W_out << " "
       << "W_error=" << audit.W_error << " "
       << (audit.W_error < 1e-6f ? "PASS" : "FAIL") << "\n";
}

void print_global_report(std::ostream& os, const GlobalAudit& audit) {
    os << "=== Global Audit Report ===\n";
    os << "Weight: in=" << audit.W_total_in
       << " out=" << audit.W_total_out
       << " cutoff=" << audit.W_total_cutoff
       << " nuclear=" << audit.W_total_nuclear
       << " boundary=" << audit.W_boundary
       << " error=" << audit.W_error << "\n";
    os << "Energy: in=" << audit.E_total_in
       << " dep=" << audit.E_total_dep
       << " nuclear=" << audit.E_total_nuclear
       << " boundary=" << audit.E_boundary
       << " error=" << audit.E_error << "\n";
}

void print_failed_cells(std::ostream& os, const std::vector<int>& failed_cells) {
    os << "Failed cells: ";
    for (int cell : failed_cells) {
        os << cell << " ";
    }
    os << "\n";
}

void print_summary(std::ostream& os, int n_cells, int n_passed, int n_failed, const GlobalAudit& global) {
    os << "=== Summary ===\n";
    os << "Cells: " << n_cells << " (passed=" << n_passed << ", failed=" << n_failed << ")\n";
    os << "Weight conservation: " << (global.W_error < 1e-6f ? "PASS" : "FAIL") << "\n";
    os << "Energy conservation: " << (global.E_error < 1e-5f ? "PASS" : "FAIL") << "\n";
}
