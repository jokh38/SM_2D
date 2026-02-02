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
// MCS Highland Formula (PDG 2024)
// ============================================================================
// This simulation uses a 2D geometry (x-z plane).
//
// PHYSICS CORRECTION (2026-02):
// The Highland formula θ₀ IS defined as the RMS "projected" scattering
// angle for one plane (PDG 2024). No additional 2D correction is needed
// for x-z plane simulation.
//
// REMOVED: DEVICE_MCS_2D_CORRECTION was incorrectly applied. The Highland
// formula already returns the plane-projected angle, not 3D angle.
// ============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// DEPRECATED: 1/√2 correction removed - Highland theta_0 IS the projected RMS
// constexpr float DEVICE_MCS_2D_CORRECTION = 0.70710678f;  // DEPRECATED (2026-02)

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
// Returns the plane-projected RMS angle (PDG 2024 definition)
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

    // Highland sigma already IS the projected angle RMS (PDG 2024)
    float sigma = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
    return sigma;
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
    // Note: cos²θ + sin²θ = 1, so normalization is unnecessary
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
// FIX: Added 1/β energy dependence to match CPU implementation
// Bohr straggling: σ²_E = 4π N_e r_e² (m_e c²)² z² L ρ ds ∝ 1/β²
// Therefore: σ_E ∝ 1/β
__device__ inline float device_bohr_straggling_sigma(float E_MeV, float ds_mm, float rho = 1.0f) {
    constexpr float kappa_water = 0.156f;  // MeV/√mm for water at β≈1

    // Relativistic β calculation for energy dependence
    float gamma = (E_MeV + DEVICE_m_p_MeV) / DEVICE_m_p_MeV;
    float beta2 = 1.0f - 1.0f / (gamma * gamma);
    float beta = sqrtf(fmaxf(beta2, 0.01f));  // Avoid division by zero

    return kappa_water * sqrtf(rho * ds_mm) / beta;  // 1/β correction added
}

// Energy straggling sigma with full Vavilov regime handling
__device__ inline float device_energy_straggling_sigma(float E_MeV, float ds_mm, float rho = 1.0f) {
    float kappa = device_vavilov_kappa(E_MeV, ds_mm, rho);

    if (kappa > 10.0f) {
        // Bohr (Gaussian) regime
        return device_bohr_straggling_sigma(E_MeV, ds_mm, rho);
    } else if (kappa < 0.01f) {
        // Landau regime - use effective width
        // PHYSICS NOTE: Landau is asymmetric, FWHM/2.355 approximation used.
        // Therapeutic protons typically in Vavilov regime (0.01 < κ < 10).
        float gamma = (E_MeV + DEVICE_m_p_MeV) / DEVICE_m_p_MeV;
        float beta = sqrtf(fmaxf(1.0f - 1.0f / (gamma * gamma), 0.0f));
        float ds_cm = ds_mm / 10.0f;
        float xi = device_vavilov_xi(beta, rho, ds_cm);
        return 4.0f * xi / 2.355f;  // FWHM / 2.355
    } else {
        // Vavilov regime - interpolate
        float sigma_bohr = device_bohr_straggling_sigma(E_MeV, ds_mm, rho);
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
// PHYSICS NOTE: This simplified model treats all nuclear-removed energy as
// locally deposited. In reality, secondary particles transport ~70-80% of
// this energy away from the primary track, causing ~1-2% local dose overestimate.
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
// Stochastic Interpolation for Lateral Scattering (K2 Coarse Transport)
// ============================================================================
// PLAN_fix_scattering: Implement Gaussian-based lateral spreading
// instead of theta accumulation for proper beam widening
// ============================================================================

// Approximate Gaussian CDF using error function approximation
// Returns: P(X <= x) for X ~ N(mu, sigma^2)
__device__ inline float device_gaussian_cdf(float x, float mu, float sigma) {
    if (sigma < 1e-10f) return (x >= mu) ? 1.0f : 0.0f;

    float t = (x - mu) / (sigma * 0.70710678f);  // / sqrt(2)
    float abs_t = fabsf(t);
    float r;

    // Abramowitz and Stegun approximation for erf
    if (abs_t < 1.0f) {
        r = 0.5f * t * (1.0f + abs_t * (0.16666667f + abs_t * (0.04166667f +
            abs_t * (0.00833333f + abs_t * 0.00142857f))));
    } else {
        r = 1.0f - 0.5f * expf(-abs_t * (1.0f + abs_t * (0.5f + abs_t * (0.33333333f +
            abs_t * 0.25f)))) / sqrtf(2.0f * (float)M_PI);
    }

    return (t > 0.0f) ? 0.5f + r : 0.5f - r;
}

// Calculate Gaussian weight distribution for lateral scattering
// Distributes particle weight across N cells using Gaussian CDF
//
// Parameters:
//   weights[out]:     Array of N weights (sum = 1.0)
//   x_mean:           Mean lateral position (cell-centered)
//   sigma_x:          Lateral spread standard deviation (mm)
//   dx:               Cell size (mm)
//   N:                Number of cells to spread across (must be even)
//
// The distribution covers ±(N/2)*dx around x_mean
__device__ inline void device_gaussian_spread_weights(
    float* weights,
    float x_mean,
    float sigma_x,
    float dx,
    int N
) {
    // Clamp sigma_x to avoid division issues
    sigma_x = fmaxf(sigma_x, 1e-6f);

    // Calculate cell boundaries (N cells cover range from -N/2*dx to +N/2*dx)
    float x_min = x_mean - (N / 2) * dx;

    // Calculate weights using Gaussian CDF
    float cdf_prev = device_gaussian_cdf(x_min, x_mean, sigma_x);

    for (int i = 0; i < N; i++) {
        float x_boundary = x_min + (i + 1) * dx;
        float cdf_curr = device_gaussian_cdf(x_boundary, x_mean, sigma_x);
        weights[i] = cdf_curr - cdf_prev;
        cdf_prev = cdf_curr;
    }

    // Normalize to ensure sum = 1.0
    float w_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        w_sum += weights[i];
    }

    if (w_sum > 1e-10f) {
        for (int i = 0; i < N; i++) {
            weights[i] /= w_sum;
        }
    } else {
        // If sigma_x is very small, put all weight in center cell
        for (int i = 0; i < N; i++) {
            weights[i] = (i == N / 2 - 1) ? 1.0f : 0.0f;
        }
    }
}

// Calculate lateral spread sigma_x from scattering angle
// Applies sqrt(3) correction for continuous scattering distributed within step
//
// For continuous scattering over a step of length ds:
// - If scattering occurs at the beginning: sigma_x = sigma_theta * ds
// - If scattering is uniformly distributed: sigma_x = sigma_theta * ds / sqrt(3)
//
// The /sqrt(3) factor accounts for the variance of uniform distribution
__device__ inline float device_lateral_spread_sigma(float sigma_theta, float step) {
    float sin_theta = sinf(fminf(sigma_theta, 1.57f));  // sin(theta), theta < pi/2
    return sin_theta * step / 1.7320508f;  // Divide by sqrt(3) for continuous scattering
}
