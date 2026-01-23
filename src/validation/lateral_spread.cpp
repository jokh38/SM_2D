#include "validation/lateral_spread.hpp"
#include "core/local_bins.hpp"
#include <cmath>
#include <algorithm>

float get_lateral_sigma_at_z(const SimulationResult& result, float z_mm) {
    // Find closest z index
    int z_idx = static_cast<int>(std::round(z_mm / result.dz));
    z_idx = std::max(0, std::min(result.Nz - 1, z_idx));

    // Get lateral profile at this z
    std::vector<double> profile(result.Nx);
    for (int i = 0; i < result.Nx; ++i) {
        profile[i] = result.edep[z_idx][i];
    }

    return fit_gaussian_sigma(profile, result.dx);
}

std::vector<double> get_lateral_profile(const SimulationResult& result, float z_mm) {
    // Find closest z index
    int z_idx = static_cast<int>(std::round(z_mm / result.dz));
    z_idx = std::max(0, std::min(result.Nz - 1, z_idx));

    std::vector<double> profile(result.Nx);
    for (int i = 0; i < result.Nx; ++i) {
        profile[i] = result.edep[z_idx][i];
    }

    return profile;
}

float compute_fermi_eyges_sigma(float E0_MeV, float z_mm) {
    // Fermi-Eyges theory for lateral spread
    // Approximation for water: sigma^2 = (z/3) * (1/Emax)
    // where Emax is in MeV and z is in cm

    float z_cm = z_mm / 10.0f;  // Convert mm to cm

    // Highland formula approximation for proton scattering
    // sigma_theta ≈ 13.6 MeV / (beta * c * p) * sqrt(x/X0)
    // For protons in water, simplified:
    float sigma_theta = 13.6f / (E0_MeV) * std::sqrt(z_cm / 36.08f);  // X0 ≈ 36.08 g/cm² for water

    // Lateral spread = theta * z / sqrt(3)
    float sigma = sigma_theta * z_cm / std::sqrt(3.0f);

    return sigma * 10.0f;  // Convert back to mm
}

float fit_gaussian_sigma(const std::vector<double>& profile, float dx) {
    // Simple Gaussian fit using second moment
    // Find center of mass
    double sum = 0.0;
    double sum_x = 0.0;
    double sum_x2 = 0.0;

    for (size_t i = 0; i < profile.size(); ++i) {
        double x = i * dx;
        sum += profile[i];
        sum_x += x * profile[i];
        sum_x2 += x * x * profile[i];
    }

    if (sum == 0.0) {
        return 0.0f;
    }

    double mean = sum_x / sum;
    double mean_x2 = sum_x2 / sum;
    double variance = mean_x2 - mean * mean;

    return static_cast<float>(std::sqrt(std::max(0.0, variance)));
}

// ============================================================================
// Sub-cell resolution profile extraction (for 3D phase-space analysis)
// ============================================================================

std::vector<double> get_lateral_profile_subcell(const SimulationResult& result, float z_mm, const PsiC& psi) {
    // Find closest z index
    int z_idx = static_cast<int>(std::round(z_mm / result.dz));
    z_idx = std::max(0, std::min(result.Nz - 1, z_idx));

    // High-resolution profile with sub-cell bins
    int total_x_bins = result.Nx * N_x_sub;
    std::vector<double> profile(total_x_bins, 0.0);

    for (int ix = 0; ix < result.Nx; ++ix) {
        for (int x_sub = 0; x_sub < N_x_sub; ++x_sub) {
            int profile_idx = ix * N_x_sub + x_sub;
            int cell = ix + z_idx * result.Nx;

            // Sum over all (theta, E) bins for this sub-cell
            double cell_sum = 0.0;
            for (int slot = 0; slot < psi.Kb; ++slot) {
                for (int theta_local = 0; theta_local < N_theta_local; ++theta_local) {
                    for (int E_local = 0; E_local < N_E_local; ++E_local) {
                        uint16_t lidx = encode_local_idx_3d(theta_local, E_local, x_sub);
                        if (cell < static_cast<int>(psi.value.size()) &&
                            slot < static_cast<int>(psi.value[cell].size())) {
                            cell_sum += psi.value[cell][slot][lidx];
                        }
                    }
                }
            }
            profile[profile_idx] = cell_sum;
        }
    }

    return profile;
}
