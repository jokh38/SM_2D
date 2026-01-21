#pragma once
#include <cstdint>

struct CellWeightAudit {
    float W_in = 0;
    float W_out = 0;
    float W_cutoff = 0;
    float W_nuclear = 0;
    float W_error = 0;
};

struct CellEnergyAudit {
    double E_in = 0;
    double E_out = 0;
    double E_dep = 0;
    double E_nuclear = 0;
    double E_error = 0;
};

bool check_weight_conservation(CellWeightAudit& audit);
bool check_energy_conservation(CellEnergyAudit& audit);
float compute_weight_error(const CellWeightAudit& audit);
double compute_energy_error(const CellEnergyAudit& audit);
