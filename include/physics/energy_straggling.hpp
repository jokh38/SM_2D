#pragma once
#include <cmath>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Bohr energy straggling for protons in water
// Returns standard deviation of energy loss [MeV]
inline float bohr_energy_straggling_sigma(float E_MeV, float ds_mm, float rho = 1.0f) {
    // Bohr straggling theory: σ²_E = 4π N_e r_e² (m_e c²)² z² L ρ ds
    // where:
    //   N_e = electron density [electrons/cm³]
    //   r_e = classical electron radius [cm]
    //   m_e c² = electron rest energy [MeV]
    //   z = projectile charge (1 for proton)
    //   L = stopping number (≈ 1 for biological materials)
    //   ρ = density [g/cm³]
    //   ds = path length [cm]

    // Simplified for protons in water:
    // σ_E ≈ κ * sqrt(ρ * ds) where κ ≈ 0.156 MeV/√mm for water
    // This approximation is valid for the Bohr regime (κ >> 1)

    constexpr float kappa = 0.156f;  // Bohr constant for water [MeV/√mm]
    return kappa * sqrtf(rho * ds_mm);
}

// Sample energy loss with Gaussian straggling (Box-Muller transform)
inline float sample_energy_loss_with_straggling(float mean_E_loss, float sigma_E, unsigned& seed) {
    if (sigma_E <= 0.0f) {
        return mean_E_loss;
    }

    // Box-Muller transform for Gaussian random sampling
    float u1 = rand_r(&seed) / (float)RAND_MAX;
    float u2 = rand_r(&seed) / (float)RAND_MAX;

    // Avoid log(0)
    u1 = fmaxf(u1, 1e-10f);

    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);

    // Ensure energy loss is non-negative
    float E_loss = mean_E_loss + z * sigma_E;
    return fmaxf(E_loss, 0.0f);
}

// Vavilov distribution parameter kappa (for future implementation)
// Returns κ = ξ / (T_max) where ξ is the characteristic energy
inline float vavilov_kappa(float E_MeV, float ds_mm, float rho = 1.0f) {
    // For protons in water, κ is typically < 0.01 (Landau regime)
    // This means Landau approximation is more appropriate than Gaussian
    constexpr float kappa_approx = 0.01f;  // Approximate for therapeutic protons
    return kappa_approx;
}

// Most probable energy loss (Landau approximation)
inline float most_probable_energy_loss(float E_MeV, float ds_mm, float rho = 1.0f) {
    // For Landau regime: Δp = ξ [ln(κ) + ln(ξ/T_max) + 0.2 - β²]
    // Simplified: use mean loss with small correction
    return bohr_energy_straggling_sigma(E_MeV, ds_mm, rho) * 2.0f;
}
