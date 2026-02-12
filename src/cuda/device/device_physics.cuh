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
    constexpr float INV_SQRT_2 = 0.70710678f;
    float arg = (x - mu) / sigma * INV_SQRT_2;
    float cdf = 0.5f * (1.0f + erff(arg));
    return fminf(1.0f, fmaxf(0.0f, cdf));
}

// Calculate Gaussian weight distribution for lateral scattering
// Distributes particle weight across N cells using Gaussian CDF
//
// Parameters:
//   weights[out]:     Array of N weights (sum = 1.0)
//   x_mean:           Mean lateral position (cell-centered)
//   sigma_x:          Lateral spread standard deviation (mm)
//   dx:               Cell size (mm)
//   N:                Number of cells to spread across (odd N preferred)
//
// The distribution covers N cells centered around x_mean.
__device__ inline void device_gaussian_spread_weights(
    float* weights,
    float x_mean,
    float sigma_x,
    float dx,
    int N
) {
    // Clamp sigma_x to avoid division issues
    sigma_x = fmaxf(sigma_x, 1e-6f);

    // Center the N-cell window around x_mean.
    // Using 0.5*N keeps odd-N windows truly symmetric around the source cell.
    float x_min = x_mean - 0.5f * static_cast<float>(N) * dx;

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
            weights[i] = (i == N / 2) ? 1.0f : 0.0f;
        }
    }
}

// ============================================================================
// FIX B: Sub-cell Gaussian spreading (for lateral spreading within a cell)
// ============================================================================
// This function correctly handles spreading across sub-cell bins (N_x_sub = 8)
// where each bin has spacing dx/N_x_sub instead of dx.
//
// Parameters:
//   weights:          Output array of N weights (sum = 1.0)
//   x_mean:           Mean lateral position within cell [-dx/2, +dx/2]
//   sigma_x:          Lateral spread standard deviation (mm)
//   dx:               Cell size (mm)
//   N_x_sub:          Number of sub-cell bins (default 8)
//
// The distribution covers all N_x_sub sub-bins within the cell
// ============================================================================
__device__ inline void device_gaussian_spread_weights_subcell(
    float* weights,
    float x_mean,
    float sigma_x,
    float dx,
    int N_x_sub = 8
) {
    // Clamp sigma_x to avoid division issues
    sigma_x = fmaxf(sigma_x, 1e-6f);

    // Sub-cell spacing
    float dx_sub = dx / N_x_sub;

    // Calculate sub-bin boundaries within cell
    // Cell spans from -dx/2 to +dx/2
    float x_min = -dx * 0.5f;

    // Calculate weights using Gaussian CDF
    float cdf_prev = device_gaussian_cdf(x_min, x_mean, sigma_x);

    for (int i = 0; i < N_x_sub; i++) {
        float x_boundary = x_min + (i + 1) * dx_sub;
        float cdf_curr = device_gaussian_cdf(x_boundary, x_mean, sigma_x);
        weights[i] = cdf_curr - cdf_prev;
        cdf_prev = cdf_curr;
    }

    // Normalize to ensure sum = 1.0
    float w_sum = 0.0f;
    for (int i = 0; i < N_x_sub; i++) {
        w_sum += weights[i];
    }

    if (w_sum > 1e-10f) {
        for (int i = 0; i < N_x_sub; i++) {
            weights[i] /= w_sum;
        }
    } else {
        // If sigma_x is very small, put all weight in center bin
        for (int i = 0; i < N_x_sub; i++) {
            weights[i] = (i == N_x_sub / 2) ? 1.0f : 0.0f;
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

// ============================================================================
// Fermi-Eyges Moment Tracking (Phase B - PLAN_MCS)
// ============================================================================
// Implements moment-based lateral spreading for correct O(z^(3/2)) scaling
//
// Moments tracked:
//   A = ⟨θ²⟩  : Angular variance
//   B = ⟨xθ⟩  : Position-angle covariance
//   C = ⟨x²⟩  : Position variance (lateral spread squared)
//
// Reference: Fermi-Eyges theory of multiple Coulomb scattering
// ============================================================================

// Calculate scattering power T = θ₀²/ds [rad²/mm]
// This is the rate of increase of angular variance per unit path length
__device__ inline float device_scattering_power_T(float E_MeV, float ds, float X0 = DEVICE_X0_water) {
    // Highland theta0 at step energy
    float theta0 = device_highland_sigma(E_MeV, ds, X0);

    // Guard against ds too small
    if (ds < 1e-6f) return 0.0f;

    // Scattering power: T = θ₀² / ds
    float T = theta0 * theta0 / ds;
    return T;
}

// Fermi-Eyges moment evolution for one step
// Updates A, B, C moments based on scattering power T and step size ds
//
// Input:  A, B, C (old moments), T (scattering power), ds (step size)
// Output: A, B, C (updated moments)
//
// Equations (from Fermi-Eyges theory):
//   d⟨θ²⟩/dz = T           → A_new = A_old + T*ds
//   d⟨xθ⟩/dz = ⟨θ²⟩        → B_new = B_old + A_old*ds + 0.5*T*ds²
//   d⟨x²⟩/dz = 2⟨xθ⟩       → C_new = C_old + 2*B_old*ds + A_old*ds² + (1/3)*T*ds³
__device__ inline void device_fermi_eyges_step(
    float& A, float& B, float& C,  // Moments (in/out)
    float T, float ds               // Scattering power, step size
) {
    // Store old values for sequential update
    float A_old = A;
    float B_old = B;

    // Update moments (Fermi-Eyges evolution equations)
    A = A_old + T * ds;                                              // ⟨θ²⟩ increases by T*ds
    B = B_old + A_old * ds + 0.5f * T * ds * ds;                    // ⟨xθ⟩ increases
    C = C + 2.0f * B_old * ds + A_old * ds * ds + (1.0f / 3.0f) * T * ds * ds * ds;  // ⟨x²⟩ increases

    // Ensure moments remain non-negative
    A = fmaxf(A, 0.0f);
    C = fmaxf(C, 0.0f);
}

// Calculate accumulated lateral spread sigma_x from Fermi-Eyges C moment
// Returns: sigma_x = sqrt(⟨x²⟩) = sqrt(C)
__device__ inline float device_accumulated_sigma_x(float moment_C) {
    return sqrtf(fmaxf(moment_C, 0.0f));
}

// Calculate accumulated angular spread sigma_theta from Fermi-Eyges A moment
// Returns: sigma_theta = sqrt(⟨θ²⟩) = sqrt(A)
__device__ inline float device_accumulated_sigma_theta(float moment_A) {
    return sqrtf(fmaxf(moment_A, 0.0f));
}

// ============================================================================
// TOTAL ACCUMULATED LATERAL SPREAD (from source to current depth)
// ============================================================================
// Computes total lateral spread sigma_x for a particle that has traveled from
// the beam source (z=0) to depth z at energy E.
//
// Uses Fermi-Eyges theory for total accumulated scattering:
//   σ_x²(z) = σ_x,initial² + (σ_θ,total² × z²) / 3
//
// where σ_θ,total is the total RMS scattering angle accumulated over path z,
// computed using Highland formula for the full path length at current energy.
//
// This gives the correct O(z^(3/2)) scaling for lateral spread:
//   σ_θ ∝ √z  →  σ_x² ∝ z × z² / 3 = z³/3  →  σ_x ∝ z^(3/2)
//
// Parameters:
//   path_mm: total path length from source to current depth [mm]
//   sigma_x_initial: initial beam width (at source) [mm]
//   E_MeV: current particle energy [MeV]
//   X0: radiation length [mm]
// Returns: total lateral spread sigma_x [mm]
// ============================================================================
__device__ inline float device_total_lateral_spread(
    float path_mm,
    float sigma_x_initial,
    float E_MeV,
    float X0 = DEVICE_X0_water
) {
    // Highland formula for total RMS scattering angle over full path
    // σ_θ_total = Highland_sigma(E, path_mm)
    float sigma_theta_total = device_highland_sigma(E_MeV, path_mm, X0);

    // Fermi-Eyges: lateral variance from accumulated angular variance
    // For uniform scattering over path z, σ_x² = σ_θ² × z² / 3
    float sigma_theta_sq = sigma_theta_total * sigma_theta_total;
    float lateral_variance = sigma_theta_sq * path_mm * path_mm / 3.0f;

    // Add initial beam width (in quadrature)
    float total_variance = sigma_x_initial * sigma_x_initial + lateral_variance;

    return sqrtf(fmaxf(total_variance, 0.0f));
}

// ============================================================================
// Hybrid Moment Tracking (Phase B - PLAN_MCS2)
// ============================================================================
// Combines analytic Fermi-Eyges theory with accumulated moments for
// robust O(z^(3/2)) lateral spreading at all depths.
//
// Key insight: At shallow depths, accumulated tracking is noisy.
// At deep depths, pure theory diverges from reality.
// Solution: Use max(accumulated, analytic) with energy correction.
// ============================================================================

// Read accumulated C moment for a given (z_cell, E_bin)
__device__ inline float device_read_moment_C(int iz, int E_bin, int N_E, const float* __restrict__ d_C_array) {
    int idx = iz * N_E + E_bin;
    return d_C_array[idx];
}

// Update accumulated C moment after a transport step
__device__ inline void device_update_moment_C(
    int iz, int E_bin, int N_E, float* __restrict__ d_C_array, float C_add
) {
    int idx = iz * N_E + E_bin;
    atomicAdd(&d_C_array[idx], C_add);
}

// Calculate C_add contribution from one transport step
// C_add = T * ds³ / 3  (from Fermi-Eyges theory)
__device__ inline float device_compute_C_add(float T, float ds) {
    return T * ds * ds * ds / 3.0f;
}

// Get sigma_x from hybrid moment tracking
// Combines accumulated moments with analytic theory
__device__ inline float device_hybrid_sigma_x(
    float z_depth_mm,
    float E_MeV,
    int iz,
    int E_bin,
    int N_E,
    const float* __restrict__ d_C_array
) {
    // Read accumulated moment C
    float C = device_read_moment_C(iz, E_bin, N_E, d_C_array);

    // Calculate analytic C from depth using Fermi-Eyges
    // For thin target: sigma_x ≈ theta_0 * z / sqrt(3)
    float theta_0 = device_highland_sigma(E_MeV, z_depth_mm);
    float sigma_x_analytic = theta_0 * z_depth_mm / 1.7320508f;  // / sqrt(3)
    float C_analytic = sigma_x_analytic * sigma_x_analytic;

    // Energy correction factor (accounts for increased scattering at low energy)
    // As energy decreases, scattering increases: correction > 1
    float z_cm = z_depth_mm / 10.0f;
    float energy_correction = 1.0f + 0.1f * z_cm / fmaxf(E_MeV, 1.0f);
    C_analytic *= energy_correction * energy_correction;

    // Use maximum of accumulated and analytic for robust spreading
    // - Accumulated: accurate at shallow depths (low noise)
    // - Analytic: accurate at deep depths (prevents underestimation)
    float C_effective = fmaxf(C, C_analytic);

    return sqrtf(fmaxf(C_effective, 0.0f));
}

// Alternative hybrid sigma_x using analytic as fallback
// Uses accumulated moment when available, falls back to analytic formula
__device__ inline float device_hybrid_sigma_x_fallback(
    float z_depth_mm,
    float E_MeV,
    float C_accumulated,
    float min_threshold_C2
) {
    // Analytic sigma_x from Fermi-Eyges theory
    float theta_0 = device_highland_sigma(E_MeV, z_depth_mm);
    float sigma_x_analytic = theta_0 * z_depth_mm / 1.7320508f;

    // Accumulated sigma_x
    float sigma_x_accum = sqrtf(fmaxf(C_accumulated, 0.0f));

    // Use analytic if accumulated is below threshold (too noisy)
    if (C_accumulated < min_threshold_C2) {
        return sigma_x_analytic;
    }

    // Otherwise use accumulated
    return sigma_x_accum;
}
