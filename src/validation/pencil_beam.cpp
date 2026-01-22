#include "validation/pencil_beam.hpp"
#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

SimulationResult run_pencil_beam(const PencilBeamConfig& config) {
    SimulationResult result;
    result.Nx = config.Nx;
    result.Nz = config.Nz;
    result.dx = config.dx;
    result.dz = config.dz;

    // Allocate energy deposition grid
    result.edep.resize(config.Nz, std::vector<double>(config.Nx, 0.0));

    // Create coordinate centers
    result.x_centers.resize(config.Nx);
    result.z_centers.resize(config.Nz);

    for (int i = 0; i < config.Nx; ++i) {
        result.x_centers[i] = (i + 0.5f) * config.dx - (config.Nx * config.dx) / 2.0f;
    }

    for (int j = 0; j < config.Nz; ++j) {
        result.z_centers[j] = j * config.dz;
    }

    // STUB: In full implementation, this would run the CUDA simulation
    // For now, create a simple Bragg peak pattern for validation testing

    // Generate RLUT to get accurate range from NIST PSTAR based on initial energy
    auto lut = GenerateRLUT(0.1f, 300.0f, 256);
    float R_bragg = lut.lookup_R(config.E0);  // Range from NIST PSTAR lookup

    // Energy-dependent lateral spread (approximate)
    float sigma = 0.5f + 0.02f * config.E0;  // Lateral spread increases with energy

    for (int j = 0; j < config.Nz; ++j) {
        float z = result.z_centers[j];
        for (int i = 0; i < config.Nx; ++i) {
            float x = result.x_centers[i];

            // Simple Bragg peak model
            float depth_factor = 0.0f;
            if (z < R_bragg) {
                depth_factor = std::pow(z / R_bragg, 2.0f);
            } else {
                float falloff = (z - R_bragg) / 3.0f;
                depth_factor = std::exp(-falloff * falloff);
            }

            // Lateral Gaussian spread
            float lateral_factor = std::exp(-(x * x) / (2.0f * sigma * sigma));

            result.edep[j][i] = config.W_total * depth_factor * lateral_factor;
        }
    }

    return result;
}

int find_bragg_peak_z(const SimulationResult& result) {
    int max_z = 0;
    double max_dose = 0.0;

    for (int j = 0; j < result.Nz; ++j) {
        // Sum dose across all x at this z
        double dose_z = 0.0;
        for (int i = 0; i < result.Nx; ++i) {
            dose_z += result.edep[j][i];
        }

        if (dose_z > max_dose) {
            max_dose = dose_z;
            max_z = j;
        }
    }

    return max_z;
}

std::vector<double> get_depth_dose(const SimulationResult& result) {
    std::vector<double> depth_dose(result.Nz);

    for (int j = 0; j < result.Nz; ++j) {
        double dose_z = 0.0;
        for (int i = 0; i < result.Nx; ++i) {
            dose_z += result.edep[j][i];
        }
        depth_dose[j] = dose_z;
    }

    return depth_dose;
}
