#include "validation/deterministic_beam.hpp"
#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include "physics/physics.hpp"
#include "physics/highland.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward declaration for Highland formula (defined in physics/highland.hpp)
extern float highland_sigma(float E_MeV, float ds, float X0);

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

    /**
     * DETERMINISTIC S-MATRIX TRANSPORT - ANALYTICAL VALIDATION MODE
     * ============================================================
     * This implements an analytical solution for validation against NIST PSTAR.
     * For production simulations, the K1-K6 CUDA kernels implement the full
     * deterministic transport with angular quadrature for MCS.
     *
     * LATERAL SPREAD CALCULATION (Fixed):
     * ===================================
     * Uses Fermi-Eyges theory with proper Highland angular diffusion integration.
     *
     * Total lateral variance at depth z:
     *   σ²_total(z) = σ²_initial + σ²_geometric(z) + σ²_diffusion(z)
     *
     * where the diffusion term is computed by integrating Highland scattering
     * along the path, accounting for energy degradation.
     *
     * σ_diffusion²(z) = ∫[0 to z] σ_θ²(E(z')) × (z-z')² / 3 dz'
     *
     * This is the CORRECT implementation of lateral spreading from MCS.
     */
    auto lut = GenerateRLUT(0.1f, 300.0f, 256);
    float R_bragg = lut.lookup_R(config.E0);  // Range from NIST PSTAR

    // Get reference stopping power for normalization
    float S_plateau = lut.lookup_S(config.E0);

    // Range straggling sigma (energy-dependent)
    float sigma_R = std::max(3.0f, R_bragg * 0.015f);

    // First pass: compute unconvolved depth-dose from stopping power
    std::vector<float> unconvolved_dose(config.Nz);
    for (int j = 0; j < config.Nz; ++j) {
        float z = result.z_centers[j];

        if (z < R_bragg) {
            float R_residual = R_bragg - z;
            float E_z = lut.lookup_E_inverse(std::max(0.1f, R_residual));
            float S_z = lut.lookup_S(E_z);
            unconvolved_dose[j] = S_z / S_plateau;
        } else {
            unconvolved_dose[j] = 0.0f;
        }
    }

    // Second pass: convolve with range straggling and compute lateral spread
    int kernel_radius = static_cast<int>(4 * sigma_R / config.dz);
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

        /**
         * FIXED: Proper lateral spread calculation using Highland theory
         *
         * The lateral spread σ(z) at depth z has three components:
         * 1. σ_initial: Initial beam width (config.sigma_x0)
         * 2. σ_geometric: Geometric spread from initial angular divergence
         * 3. σ_diffusion: Highland angular diffusion accumulated along path
         *
         * The diffusion term is computed by integrating Highland scattering
         * along the path, with each scattering event contributing to lateral
         * displacement according to the lever arm (z - z').
         */
        float sigma_diffusion_sq = 0.0f;

        // Integrate scattering contribution along path using Fermi-Eyges:
        // σ_x^2(z) = ∫ (z - z')^2 * T(z') dz', where T = d<θ^2>/dz.
        // We approximate T dz by the Highland σ_θ^2 over each dz step.
        int n_steps = static_cast<int>(std::ceil(z / config.dz));
        for (int step = 0; step < n_steps; ++step) {
            float step_start = step * config.dz;
            if (step_start >= z) {
                break;
            }
            float step_length = std::min(config.dz, z - step_start);
            float z_step = step_start + 0.5f * step_length;
            float z_remaining = z - z_step;

            // Energy at this depth (residual range method)
            float R_residual = R_bragg - z_step;
            float E_z = R_residual > 0 ? lut.lookup_E_inverse(std::max(0.1f, R_residual)) : 0.1f;

            // Highland scattering angle for this step (2D projection included)
            float sigma_theta = highland_sigma(E_z, step_length, X0_water);

            // Accumulate diffusion contribution with lever arm to depth z
            sigma_diffusion_sq += sigma_theta * sigma_theta * z_remaining * z_remaining;
        }

        float sigma_diffusion = std::sqrt(sigma_diffusion_sq);

        // Geometric spread from initial angular divergence (small-angle approximation)
        float sigma_geometric = std::abs(config.sigma_theta0) * z;

        // Beam center shifts with mean angle (tilt)
        float x_mean = std::tan(config.theta0) * z;

        // Total lateral spread (quadrature sum of all components)
        float sigma_z = std::sqrt(
            config.sigma_x0 * config.sigma_x0 +           // Initial beam size
            sigma_geometric * sigma_geometric +            // Geometric spread
            sigma_diffusion * sigma_diffusion             // MCS diffusion (FIXED)
        );

        // Ensure minimum spread for numerical stability
        sigma_z = std::max(sigma_z, config.dx);

        // Apply lateral Gaussian profile
        for (int i = 0; i < config.Nx; ++i) {
            float x = result.x_centers[i];
            float x_shift = x - x_mean;
            float lateral_factor = std::exp(-(x_shift * x_shift) / (2.0f * sigma_z * sigma_z));
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
