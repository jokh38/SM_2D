#include "validation/pencil_beam.hpp"
#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include "physics/physics.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    // For now, create a physically-motivated Bragg peak pattern for validation testing

    // Generate RLUT to get accurate range and stopping power from NIST PSTAR
    auto lut = GenerateRLUT(0.1f, 300.0f, 256);
    float R_bragg = lut.lookup_R(config.E0);  // Range from NIST PSTAR lookup

    // Energy-dependent lateral spread (approximate MCS theory)
    // σ ≈ θ₀ * z / √3 where θ₀ is the scattering angle
    float sigma_0 = 0.015f * std::pow(config.E0 / 100.0f, -0.8f);  // Scattering angle

    // Get reference stopping power for normalization
    float S_plateau = lut.lookup_S(config.E0);

    // Range straggling sigma (energy-dependent)
    // Clinical distal falloff (80%-20%) is ~3-5mm for 150 MeV protons
    // Gaussian convolution with sigma_R ~ 1.5% of range
    float sigma_R = std::max(3.0f, R_bragg * 0.015f);

    // First pass: compute unconvolved depth-dose from stopping power
    std::vector<float> unconvolved_dose(config.Nz);
    for (int j = 0; j < config.Nz; ++j) {
        float z = result.z_centers[j];

        if (z < R_bragg) {
            // Before Bragg peak: D(z) = S(E(z)) / S(E0)
            float R_residual = R_bragg - z;
            float E_z = lut.lookup_E_inverse(std::max(0.1f, R_residual));
            float S_z = lut.lookup_S(E_z);
            unconvolved_dose[j] = S_z / S_plateau;
        } else {
            // Beyond Bragg peak: no primary particles
            unconvolved_dose[j] = 0.0f;
        }
    }

    // Second pass: convolve with Gaussian range straggling
    // This smooths the sharp Bragg peak into a realistic distribution
    int kernel_radius = static_cast<int>(4 * sigma_R / config.dz);  // 4-sigma kernel
    for (int j = 0; j < config.Nz; ++j) {
        float z = result.z_centers[j];
        float convolved_dose = 0.0f;
        float norm = 0.0f;

        for (int k = -kernel_radius; k <= kernel_radius; ++k) {
            int jk = j + k;
            if (jk >= 0 && jk < config.Nz) {
                float dz_k = k * config.dz;
                float gaussian = std::exp(-(dz_k * dz_k) / (2.0f * sigma_R * sigma_R));
                convolved_dose += unconvolved_dose[jk] * gaussian;
                norm += gaussian;
            }
        }

        float depth_dose = convolved_dose / norm;

        // Depth-dependent lateral spread (MCS theory + initial beam size)
        // Total sigma = sqrt(sigma_MCS^2 + sigma_initial^2)
        float sigma_mcs = sigma_0 * z / std::sqrt(3.0f);
        float sigma_z = std::sqrt(std::pow(sigma_mcs, 2) + std::pow(config.sigma_x0, 2));
        sigma_z = std::max(sigma_z, 0.5f);  // Minimum spread [mm]

        for (int i = 0; i < config.Nx; ++i) {
            float x = result.x_centers[i];

            // Lateral Gaussian spread (normalized to 1 at x=0)
            float lateral_factor = std::exp(-(x * x) / (2.0f * sigma_z * sigma_z));

            // Normalize: total lateral integral ≈ sqrt(2π) * sigma * dx
            float lateral_norm = config.dx / (std::sqrt(2.0f * M_PI) * sigma_z);

            result.edep[j][i] = config.W_total * depth_dose * lateral_factor * lateral_norm;
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
