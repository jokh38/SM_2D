#include "validation/bragg_peak.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

float find_bragg_peak_position_mm(const SimulationResult& result) {
    float max_dose = 0.0f;
    float z_peak = 0.0f;

    for (int j = 0; j < result.Nz; ++j) {
        float dose_z = 0.0f;
        for (int i = 0; i < result.Nx; ++i) {
            dose_z += static_cast<float>(result.edep[j][i]);
        }

        if (dose_z > max_dose) {
            max_dose = dose_z;
            z_peak = result.z_centers[j];
        }
    }

    return z_peak;
}

float compute_bragg_peak_fwhm(const SimulationResult& result) {
    // Find peak position and dose
    float z_peak = find_bragg_peak_position_mm(result);

    int peak_idx = static_cast<int>(z_peak / result.dz);
    float peak_dose = 0.0f;
    for (int i = 0; i < result.Nx; ++i) {
        peak_dose += static_cast<float>(result.edep[peak_idx][i]);
    }

    float half_max = peak_dose * 0.5f;

    // Find left crossing
    float z_left = z_peak;
    for (int j = peak_idx; j >= 0; --j) {
        float dose_z = 0.0f;
        for (int i = 0; i < result.Nx; ++i) {
            dose_z += static_cast<float>(result.edep[j][i]);
        }

        if (dose_z < half_max) {
            z_left = result.z_centers[j];
            break;
        }
    }

    // Find right crossing
    float z_right = z_peak;
    for (int j = peak_idx; j < result.Nz; ++j) {
        float dose_z = 0.0f;
        for (int i = 0; i < result.Nx; ++i) {
            dose_z += static_cast<float>(result.edep[j][i]);
        }

        if (dose_z < half_max) {
            z_right = result.z_centers[j];
            break;
        }
    }

    return z_right - z_left;
}

float find_R80(const SimulationResult& result) {
    float z_peak = find_bragg_peak_position_mm(result);
    int peak_idx = static_cast<int>(z_peak / result.dz);

    float peak_dose = 0.0f;
    for (int i = 0; i < result.Nx; ++i) {
        peak_dose += static_cast<float>(result.edep[peak_idx][i]);
    }

    float threshold = peak_dose * 0.8f;

    for (int j = peak_idx; j < result.Nz; ++j) {
        float dose_z = 0.0f;
        for (int i = 0; i < result.Nx; ++i) {
            dose_z += static_cast<float>(result.edep[j][i]);
        }

        if (dose_z < threshold) {
            return result.z_centers[j];
        }
    }

    return result.z_centers[result.Nz - 1];
}

float find_R20(const SimulationResult& result) {
    float z_peak = find_bragg_peak_position_mm(result);
    int peak_idx = static_cast<int>(z_peak / result.dz);

    float peak_dose = 0.0f;
    for (int i = 0; i < result.Nx; ++i) {
        peak_dose += static_cast<float>(result.edep[peak_idx][i]);
    }

    float threshold = peak_dose * 0.2f;

    for (int j = peak_idx; j < result.Nz; ++j) {
        float dose_z = 0.0f;
        for (int i = 0; i < result.Nx; ++i) {
            dose_z += static_cast<float>(result.edep[j][i]);
        }

        if (dose_z < threshold) {
            return result.z_centers[j];
        }
    }

    return result.z_centers[result.Nz - 1];
}

float compute_distal_falloff(const SimulationResult& result) {
    // Distal falloff = R80 - R20 (distance from 80% to 20% dose)
    float r80 = find_R80(result);
    float r20 = find_R20(result);

    return r20 - r80;
}
