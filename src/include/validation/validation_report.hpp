#pragma once
#include <iosfwd>

struct BraggPeakResult {
    float energy_MeV;
    float R_sim;
    float R_nist;
    float error;
    bool pass;
};

struct LateralSpreadResult {
    float z_mm;
    float sigma_sim;
    float sigma_fe;
    float error;
    bool pass;
};

struct ConservationResult {
    float weight_error;
    float energy_error;
    bool pass;
};

struct ValidationResults {
    BraggPeakResult bragg_150;
    BraggPeakResult bragg_70;
    LateralSpreadResult lateral;
    ConservationResult conservation;
    bool overall_pass;
};

void generate_validation_report(std::ostream& os, const ValidationResults& results);
ValidationResults run_full_validation();
