#pragma once
#include <cmath>

// Define CUDA macros for non-CUDA builds
#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

// Forward declare X0_water (defined in physics.hpp which includes this header)
// Use default value if not yet included
#ifndef X0_water
constexpr float X0_water = 360.8f;  // Radiation length of water [mm]
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Proton rest mass [MeV/c²]
namespace { const float m_p_MeV = 938.272f; }

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
// REMOVED: MCS_2D_CORRECTION was incorrectly applied. The Highland
// formula already returns the plane-projected angle, not 3D angle.
// ============================================================================

// DEPRECATED: 1/√2 correction removed - Highland theta_0 IS the projected RMS
// constexpr float MCS_2D_CORRECTION = 0.70710678f;  // DEPRECATED (2026-02)

// Highland formula for multiple Coulomb scattering (PDG 2024)
// Returns sigma_theta [radians] for 2D simulation (x-z plane)
//
// σ_θ = (13.6 MeV / βcp) * z * sqrt(x/X_0) * [1 + 0.038 * ln(x/X_0)]
//
// where:
//   βcp = momentum * velocity [MeV/c]
//   z = projectile charge (1 for protons)
//   x = step length [mm]
//   X_0 = radiation length [mm] (360.8 mm for water)
//
// Valid for: 1e-5 < t < 100 where t = x/X_0
//
// NOTE: The Highland formula already returns the plane-projected RMS angle,
// not the 3D angle. No additional 2D correction is needed.
__host__ __device__ inline float highland_sigma(float E_MeV, float ds, float X0 = 360.8f) {
    constexpr float z = 1.0f;  // Proton charge

    // Relativistic kinematics
    float gamma = (E_MeV + m_p_MeV) / m_p_MeV;
    float beta = sqrtf(fmaxf(1.0f - 1.0f / (gamma * gamma), 0.0f));
    float p_MeV = sqrtf(fmaxf((E_MeV + m_p_MeV) * (E_MeV + m_p_MeV) - m_p_MeV * m_p_MeV, 0.0f));

    float t = ds / X0;
    if (t < 1e-6f) return 0.0f;

    // Highland correction factor
    // Valid for 1e-5 < t < 100; clamp to physical minimum
    float ln_t = logf(t);
    float bracket = 1.0f + 0.038f * ln_t;
    // PDG 2024 recommends bracket ≥ 0.25 (was 0.5)
    bracket = fmaxf(bracket, 0.25f);  // PDG 2024 recommendation

    // Highland sigma already IS the projected angle RMS (PDG 2024)
    float sigma = (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
    return sigma;
}

// 7-point Gaussian quadrature weights for angular splitting
constexpr int N_QUADRATURE = 7;
constexpr float QUADRATURE_WEIGHTS[N_QUADRATURE] = {
    0.05f, 0.10f, 0.20f, 0.30f, 0.20f, 0.10f, 0.05f
};
constexpr float QUADRATURE_DELTAS[N_QUADRATURE] = {
    -3.0f, -1.5f, -0.5f, 0.0f, +0.5f, +1.5f, +3.0f
};

// ============================================================================
// MCS Direction Update (IC-6: Missing from original implementation)
// ============================================================================

// Sample Gaussian scattering angle for MCS
// Returns theta_scatter [radians] using Box-Muller transform
__host__ __device__ inline float sample_mcs_angle(float sigma_theta, unsigned& seed) {
    if (sigma_theta <= 0.0f) {
        return 0.0f;
    }

    // Box-Muller transform for Gaussian random sampling
    float u1 = rand_r(&seed) / (float)RAND_MAX;
    float u2 = rand_r(&seed) / (float)RAND_MAX;

    // Avoid log(0)
    u1 = fmaxf(u1, 1e-10f);

    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return z * sigma_theta;
}

// Update direction cosines (mu, eta) after MCS scattering
// theta_new = theta_old + theta_scatter (small angle approximation)
inline void update_direction_after_mcs(
    float theta_old,
    float theta_scatter,
    float& mu_out,
    float& eta_out
) {
    // New polar angle (small angle: theta_new ≈ theta_old + theta_scatter)
    float theta_new = theta_old + theta_scatter;

    // Update direction cosines
    // Note: cos²θ + sin²θ = 1 exactly, no normalization needed
    mu_out = cosf(theta_new);
    eta_out = sinf(theta_new);
}

// Complete MCS step: sample and update direction
inline void apply_mcs_step(
    float E_MeV,
    float ds,
    float theta_old,
    float& mu_out,
    float& eta_out,
    unsigned& seed
) {
    float sigma = highland_sigma(E_MeV, ds, X0_water);
    float theta_scatter = sample_mcs_angle(sigma, seed);
    update_direction_after_mcs(theta_old, theta_scatter, mu_out, eta_out);
}
