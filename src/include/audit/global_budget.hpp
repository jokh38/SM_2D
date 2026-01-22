#pragma once
#include "audit/conservation.hpp"
#include <vector>

struct GlobalAudit {
    float W_total_in = 0;
    float W_total_out = 0;
    float W_total_cutoff = 0;
    float W_total_nuclear = 0;
    float W_boundary = 0;
    float W_error = 0;
    double E_total_in = 0;
    double E_total_dep = 0;
    double E_total_nuclear = 0;
    double E_boundary = 0;
    double E_error = 0;
};

GlobalAudit aggregate_cell_audits(
    const std::vector<CellWeightAudit>& weight_audits,
    const std::vector<CellEnergyAudit>& energy_audits
);

bool check_global_weight_conservation(GlobalAudit& audit);
bool check_global_energy_conservation(GlobalAudit& audit);
