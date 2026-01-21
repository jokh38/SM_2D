#pragma once
#include "core/grids.hpp"
#include <vector>

struct RLUT {
    EnergyGrid grid;
    std::vector<float> R;         // CSDA range [mm]
    std::vector<float> log_E;     // Pre-computed log(E)
    std::vector<float> log_R;     // Pre-computed log(R)

    // Lookup R(E) using log-log interpolation
    float lookup_R(float E) const;

    // Inverse lookup: E from R
    float lookup_E_inverse(float R) const;
};

// Generate LUT from NIST data
RLUT GenerateRLUT(float E_min, float E_max, int N_E);
