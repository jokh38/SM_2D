#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include <cmath>
#include <algorithm>

RLUT GenerateRLUT(float E_min, float E_max, int N_E) {
    RLUT lut{EnergyGrid(E_min, E_max, N_E), {}, {}, {}};
    lut.R.resize(N_E);
    lut.log_E.resize(N_E);
    lut.log_R.resize(N_E);

    // Load NIST data
    auto nist_data = LoadNistData("data/nist/pstar_water.txt");

    // Convert NIST range from g/cm² to mm (water density = 1.0 g/cm³)
    const float rho = 1.0f;
    const float g_cm2_to_mm = 10.0f / rho;

    // Generate LUT by interpolating NIST data
    for (int i = 0; i < N_E; ++i) {
        float E = lut.grid.rep[i];

        // Find surrounding NIST data points
        auto it = std::lower_bound(
            nist_data.begin(), nist_data.end(),
            E,
            [](const NistDataRow& row, float val) {
                return row.energy_MeV < val;
            }
        );

        if (it == nist_data.begin()) {
            lut.R[i] = nist_data[0].csda_range_g_cm2 * g_cm2_to_mm;
        } else if (it == nist_data.end()) {
            lut.R[i] = nist_data.back().csda_range_g_cm2 * g_cm2_to_mm;
        } else {
            // Linear interpolation in log-log space
            float E0 = (it - 1)->energy_MeV;
            float E1 = it->energy_MeV;
            float R0 = (it - 1)->csda_range_g_cm2 * g_cm2_to_mm;
            float R1 = it->csda_range_g_cm2 * g_cm2_to_mm;

            float log_R = logf(R0) + (logf(R1) - logf(R0)) *
                         (logf(E) - logf(E0)) / (logf(E1) - logf(E0));
            lut.R[i] = expf(log_R);
        }

        lut.log_E[i] = logf(lut.grid.rep[i]);
        lut.log_R[i] = logf(lut.R[i]);
    }

    return lut;
}

float RLUT::lookup_R(float E) const {
    float E_clamped = fmaxf(grid.E_min, fminf(E, grid.E_max));
    int bin = grid.FindBin(E_clamped);

    float log_E_val = logf(E_clamped);
    float log_E0 = log_E[bin];
    float log_E1 = log_E[std::min(bin + 1, grid.N_E - 1)];
    float log_R0 = log_R[bin];
    float log_R1 = log_R[std::min(bin + 1, grid.N_E - 1)];

    float log_R_val = log_R0 + (log_R1 - log_R0) * (log_E_val - log_E0) / (log_E1 - log_E0);
    return expf(log_R_val);
}

float RLUT::lookup_E_inverse(float R_input) const {
    auto it = std::lower_bound(R.begin(), R.end(), R_input);
    int bin = std::max(0, std::min(static_cast<int>(it - R.begin()), grid.N_E - 2));

    float log_R_val = logf(R_input);
    float log_R0 = log_R[bin];
    float log_R1 = log_R[bin + 1];
    float log_E0 = log_E[bin];
    float log_E1 = log_E[bin + 1];

    float log_E_val = log_E0 + (log_E1 - log_E0) * (log_R_val - log_R0) / (log_R1 - log_R0);
    return expf(log_E_val);
}
