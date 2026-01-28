#pragma once
#include "core/grids.hpp"
#include <vector>

struct RLUT {
    EnergyGrid grid;
    std::vector<float> R;         // CSDA range [mm]
    std::vector<float> S;         // Stopping power [MeV cm²/g]
    std::vector<float> log_E;     // Pre-computed log(E)
    std::vector<float> log_R;     // Pre-computed log(R)
    std::vector<float> log_S;     // Pre-computed log(S)

    // Lookup R(E) using log-log interpolation
    float lookup_R(float E) const;

    // Lookup S(E) using log-log interpolation
    float lookup_S(float E) const;

    // Inverse lookup: E from R
    float lookup_E_inverse(float R) const;
};

// Generate LUT from NIST data (log-spaced grid)
RLUT GenerateRLUT(float E_min, float E_max, int N_E);

// Generate LUT from NIST data (using existing EnergyGrid, e.g., piecewise-uniform)
RLUT GenerateRLUT(const EnergyGrid& grid);
