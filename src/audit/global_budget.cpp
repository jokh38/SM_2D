#include "audit/global_budget.hpp"
#include <cmath>

GlobalAudit aggregate_cell_audits(
    const std::vector<CellWeightAudit>& weight_audits,
    const std::vector<CellEnergyAudit>& energy_audits
) {
    GlobalAudit audit;
    for (const auto& cell : weight_audits) {
        audit.W_total_in += cell.W_in;
        audit.W_total_out += cell.W_out;
        audit.W_total_cutoff += cell.W_cutoff;
        audit.W_total_nuclear += cell.W_nuclear;
    }
    for (const auto& cell : energy_audits) {
        audit.E_total_in += cell.E_in;
        audit.E_total_dep += cell.E_dep;
        audit.E_total_nuclear += cell.E_nuclear;
    }
    return audit;
}

bool check_global_weight_conservation(GlobalAudit& audit) {
    float W_expected = audit.W_total_out + audit.W_total_cutoff +
                      audit.W_total_nuclear + audit.W_boundary;
    float W_diff = fabsf(audit.W_total_in - W_expected);
    audit.W_error = W_diff / fmaxf(audit.W_total_in, 1e-20f);
    return audit.W_error < 1e-6f;
}

bool check_global_energy_conservation(GlobalAudit& audit) {
    double E_expected = audit.E_total_dep + audit.E_total_nuclear + audit.E_boundary;
    double E_diff = fabs(audit.E_total_in - E_expected);
    audit.E_error = E_diff / fmax(audit.E_total_in, 1e-20);
    return audit.E_error < 1e-5;
}
