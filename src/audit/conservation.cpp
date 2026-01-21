#include "audit/conservation.hpp"
#include <cmath>

bool check_weight_conservation(CellWeightAudit& audit) {
    float W_expected = audit.W_out + audit.W_cutoff + audit.W_nuclear;
    float W_diff = fabsf(audit.W_in - W_expected);
    float W_rel = W_diff / fmaxf(audit.W_in, 1e-20f);
    audit.W_error = W_rel;
    return W_rel < 1e-6f;
}

float compute_weight_error(const CellWeightAudit& audit) {
    float W_expected = audit.W_out + audit.W_cutoff + audit.W_nuclear;
    float W_diff = fabsf(audit.W_in - W_expected);
    return W_diff / fmaxf(audit.W_in, 1e-20f);
}

bool check_energy_conservation(CellEnergyAudit& audit) {
    double E_expected = audit.E_out + audit.E_dep + audit.E_nuclear;
    double E_diff = fabs(audit.E_in - E_expected);
    double E_rel = E_diff / fmax(audit.E_in, 1e-20);
    audit.E_error = E_rel;
    return E_rel < 1e-5;
}

double compute_energy_error(const CellEnergyAudit& audit) {
    double E_expected = audit.E_out + audit.E_dep + audit.E_nuclear;
    double E_diff = fabs(audit.E_in - E_expected);
    return E_diff / fmax(audit.E_in, 1e-20);
}
