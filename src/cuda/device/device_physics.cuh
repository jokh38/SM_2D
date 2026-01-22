#pragma once
#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// Device Physics Functions for K3 FineTransport GPU Kernel
// ============================================================================
// P2 FIX: Device-accessible physics functions for GPU execution
//
// These are GPU-compatible versions of the CPU physics functions:
// - Multiple Coulomb Scattering (Highland formula)
// - Energy straggling (Vavilov model)
// - Nuclear interactions (ICRU 63 cross-sections)
// ============================================================================

// ============================================================================
// P2 FIX: 2D MCS Projection Correction Applied
// ============================================================================
// This simulation uses a 2D geometry (x-z plane).
// The Highland formula gives the 3D scattering angle sigma_3D.
//
// When projecting 3D scattering onto 2D (x-z plane):
//   - The azimuthal angle φ is uniformly distributed in [0, 2π]
//   - The 2D scattering angle is: θ_2D = θ_3D * cos(φ)
//   - The expected value is: E[|cos(φ)|] = 2/π ≈ 0.637
//
// FIXED: 2D projection correction factor (2/π) is now applied to convert
// sigma_3D to sigma_2D for accurate 2D simulation.
// ============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr float DEVICE_MCS_2D_CORRECTION = 2.0f / (float)M_PI;  // ≈ 0.637

// Physics constants
constexpr float DEVICE_m_p_MeV = 938.272f;      // Proton rest mass [MeV/c²]
constexpr float DEVICE_m_ec2_MeV = 0.511f;      // Electron rest energy [MeV]
constexpr float DEVICE_X0_water = 360.8f;       // Radiation length of water [mm]
constexpr float DEVICE_K = 0.307f;              // MeV cm²/g (Vavilov constant)
constexpr float DEVICE_Z_A_water = 0.555f;      // Z/A for water

// ============================================================================
// Multiple Coulomb Scattering (Highland Formula - PDG 2024)
// ============================================================================

// Highland formula for MCS scattering angle sigma [radians]
// P2 FIX: Now applies 2D projection correction (2/π)
__device__ inline float device_highland_sigma(float E_MeV, float ds, float X0 = DEVICE_X0_water) {
    constexpr float z = 1.0f;  // Proton charge

    // Relativistic kinematics
    float gamma = (E_MeV + DEVICE_m_p_MeV) / DEVICE_m_p_MeV;
    float beta2 = 1.0f - 1.0f / (gamma * gamma);
    float beta = sqrtf(fmaxf(beta2, 0.0f));
    float p_MeV = sqrtf(fmaxf((E_MeV + DEVICE_m_p_MeV) * (E_MeV + DEVICE_m_p_MeV) -
                             DEVICE_m_p_MeV * DEVICE_m_p_MeV, 0.0f));

    float t = ds / X0;
    if (t < 1e-6f) return 0.0f;

    // Highland correction factor (PDG 2024: bracket >= 0.25)
    float ln_t = logf(t);
    float bracket = 1.0f + 0.038f * ln_t;
    bracket = fmaxf(bracket, 0.25f);

    // P2 FIX: Apply 2D projection correction
    float sigma_3d = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
    return sigma_3d * DEVICE_MCS_2D_CORRECTION;
}

// Sample Gaussian scattering angle using Box-Muller transform
__device__ inline float device_sample_gaussian(unsigned& seed) {
    // Simple linear congruential generator for GPU
    seed = 1664525u * seed + 1013904223u;

    float u1 = (seed >> 16) / 65536.0f;  // Use upper bits for better quality
    seed = 1664525u * seed + 1013904223u;
    float u2 = (seed >> 16) / 65536.0f;

    u1 = fmaxf(u1, 1e-6f);

    // Box-Muller transform
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return z;
}

// Sample MCS scattering angle
__device__ inline float device_sample_mcs_angle(float sigma_theta, unsigned& seed) {
    if (sigma_theta <= 0.0f) return 0.0f;
    return device_sample_gaussian(seed) * sigma_theta;
}

// Update direction cosines after MCS
__device__ inline void device_update_direction_mcs(
    float theta_old,
    float theta_scatter,
    float& mu_out,
    float& eta_out
) {
    float theta_new = theta_old + theta_scatter;

    mu_out = cosf(theta_new);
    eta_out = sinf(theta_new);

    // Normalize
    float norm = sqrtf(mu_out * mu_out + eta_out * eta_out);
    if (norm > 1e-6f) {
        mu_out /= norm;
        eta_out /= norm;
    }
}

// ============================================================================
// Energy Straggling (Vavilov Model)
// ============================================================================

// Vavilov characteristic energy ξ [MeV]
__device__ inline float device_vavilov_xi(float beta, float rho_g_cm3, float ds_cm) {
    constexpr float z = 1.0f;  // Proton charge
    return (DEVICE_K / 2.0f) * DEVICE_Z_A_water * (z * z / (beta * beta)) * rho_g_cm3 * ds_cm;
}

// Maximum energy transfer to electron
__device__ inline float device_vavilov_T_max(float beta, float gamma) {
    float m_ratio = DEVICE_m_ec2_MeV / DEVICE_m_p_MeV;
    float num = 2.0f * DEVICE_m_ec2_MeV * beta * beta * gamma * gamma;
    float denom = 1.0f + 2.0f * gamma * m_ratio + m_ratio * m_ratio;
    return num / fmaxf(denom, 1e-10f);
}

// Vavilov parameter κ = ξ / T_max
__device__ inline float device_vavilov_kappa(float E_MeV, float ds_mm, float rho = 1.0f) {
    float gamma = (E_MeV + DEVICE_m_p_MeV) / DEVICE_m_p_MeV;
    float beta2 = 1.0f - 1.0f / (gamma * gamma);
    float beta = sqrtf(fmaxf(beta2, 0.0f));

    float ds_cm = ds_mm / 10.0f;
    float xi = device_vavilov_xi(beta, rho, ds_cm);
    float T_max = device_vavilov_T_max(beta, gamma);

    float kappa = xi / fmaxf(T_max, 1e-10f);
    return fmaxf(kappa, 1e-6f);
}

// Bohr straggling (Gaussian regime, κ >> 1)
__device__ inline float device_bohr_straggling_sigma(float ds_mm, float rho = 1.0f) {
    constexpr float kappa_water = 0.156f;  // MeV/√mm for water
    return kappa_water * sqrtf(rho * ds_mm);
}

// Energy straggling sigma with full Vavilov regime handling
__device__ inline float device_energy_straggling_sigma(float E_MeV, float ds_mm, float rho = 1.0f) {
    float kappa = device_vavilov_kappa(E_MeV, ds_mm, rho);

    if (kappa > 10.0f) {
        // Bohr (Gaussian) regime
        return device_bohr_straggling_sigma(ds_mm, rho);
    } else if (kappa < 0.01f) {
        // Landau regime - use effective width
        float gamma = (E_MeV + DEVICE_m_p_MeV) / DEVICE_m_p_MeV;
        float beta = sqrtf(fmaxf(1.0f - 1.0f / (gamma * gamma), 0.0f));
        float ds_cm = ds_mm / 10.0f;
        float xi = device_vavilov_xi(beta, rho, ds_cm);
        return 4.0f * xi / 2.355f;  // FWHM / 2.355
    } else {
        // Vavilov regime - interpolate
        float sigma_bohr = device_bohr_straggling_sigma(ds_mm, rho);
        float gamma = (E_MeV + DEVICE_m_p_MeV) / DEVICE_m_p_MeV;
        float beta = sqrtf(fmaxf(1.0f - 1.0f / (gamma * gamma), 0.0f));
        float ds_cm = ds_mm / 10.0f;
        float xi = device_vavilov_xi(beta, rho, ds_cm);
        float sigma_landau = 4.0f * xi / 2.355f;

        float w = 1.0f / (1.0f + kappa);
        return w * sigma_landau + (1.0f - w) * sigma_bohr;
    }
}

// Sample energy loss with Gaussian straggling
__device__ inline float device_sample_energy_loss(float mean_E_loss, float sigma_E, unsigned& seed) {
    if (sigma_E <= 0.0f) return mean_E_loss;

    float z = device_sample_gaussian(seed);
    float E_loss = mean_E_loss + z * sigma_E;
    return fmaxf(E_loss, 0.0f);
}

// ============================================================================
// Nuclear Interactions (ICRU 63)
// ============================================================================

// Nuclear cross-section [mm⁻¹]
__device__ inline float device_nuclear_cross_section(float E_MeV) {
    if (E_MeV < 5.0f) return 0.0f;

    constexpr float sigma_100 = 0.0012f;  // mm⁻¹ at 100 MeV (ICRU 63)
    constexpr float E_ref = 100.0f;

    if (E_MeV >= 20.0f) {
        float log_factor = 1.0f - 0.15f * logf(E_MeV / E_ref);
        return sigma_100 * fmaxf(log_factor, 0.4f);
    } else {
        constexpr float sigma_20 = 0.0016f;
        float frac = (E_MeV - 5.0f) / 15.0f;
        return sigma_20 * frac;
    }
}

// Apply nuclear attenuation
__device__ inline float device_apply_nuclear_attenuation(
    float w_old,
    float E,
    float step_length,
    float& w_removed_out,
    float& E_removed_out
) {
    float sigma = device_nuclear_cross_section(E);
    float survival = expf(-sigma * step_length);
    float w_new = w_old * survival;
    float w_removed = w_old - w_new;

    w_removed_out = w_removed;
    E_removed_out = w_removed * E;

    return w_new;
}

// ============================================================================
// Combined Physics Step
// ============================================================================

// Perform complete physics update for one component (device version)
// This consolidates all physics functions for efficient GPU execution
__device__ inline void device_physics_step(
    const DeviceRLUT& lut,      // Range lookup table
    float E_in,                 // Input energy [MeV]
    float w_in,                 // Input weight
    float theta_in,             // Input angle [rad]
    float dx, float dz,         // Cell dimensions [mm]
    float x_in, float z_in,     // Input position [mm]
    unsigned seed,              // Random seed
    float& E_out,               // Output energy
    float& w_out,               // Output weight
    float& theta_out,           // Output angle
    float& x_out, float& z_out, // Output position
    float& Edep,                // Energy deposited [MeV]
    float& w_nuclear,           // Nuclear weight removed
    float& E_nuclear,           // Nuclear energy removed
    int& boundary_face          // Boundary face: -1=none, 0=+z, 1=-z, 2=+x, 3=-x
) {
    E_out = E_in;
    w_out = w_in;
    theta_out = theta_in;
    x_out = x_in;
    z_out = z_in;
    Edep = 0;
    w_nuclear = 0;
    E_nuclear = 0;
    boundary_face = -1;

    // Cutoff check
    if (E_in <= 0.1f) {
        Edep = E_in * w_in;
        E_out = 0;
        return;
    }

    // Compute step size
    float step = device_compute_max_step(lut, E_in);

    // Energy loss with straggling
    float mean_dE = device_compute_energy_deposition(lut, E_in, step);
    float sigma_dE = device_energy_straggling_sigma(E_in, step, 1.0f);
    float dE = device_sample_energy_loss(mean_dE, sigma_dE, seed);
    dE = fminf(dE, E_in);

    E_out = E_in - dE;
    Edep = dE * w_in;

    // Nuclear attenuation
    float w_rem, E_rem;
    w_out = device_apply_nuclear_attenuation(w_in, E_in, step, w_rem, E_rem);
    w_nuclear = w_rem;
    E_nuclear = E_rem;
    Edep += E_rem;

    // MCS direction update
    float sigma_mcs = device_highland_sigma(E_in, step, DEVICE_X0_water);
    float theta_scatter = device_sample_mcs_angle(sigma_mcs, seed);
    float mu, eta;
    device_update_direction_mcs(theta_in, theta_scatter, mu, eta);
    theta_out = theta_in + theta_scatter;

    // Position update
    float x_new = x_in + eta * step;
    float z_new = z_in + mu * step;

    // Check boundary crossing
    // Assuming cell origin at (0, 0) with size (dx, dz)
    if (z_new >= dz) {
        boundary_face = 0;  // +z
    } else if (z_new < 0) {
        boundary_face = 1;  // -z
    } else if (x_new >= dx) {
        boundary_face = 2;  // +x
    } else if (x_new < 0) {
        boundary_face = 3;  // -x
    }

    x_out = fmaxf(0.0f, fminf(x_new, dx));
    z_out = fmaxf(0.0f, fminf(z_new, dz));

    // Terminate if energy depleted
    if (E_out <= 0.1f) {
        Edep += E_out * w_out;
        E_out = 0;
    }
}
