#pragma once
#include <cmath>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Physics constants
// Proton mass m_p = 938.272 MeV/c²
// Electron rest energy m_ec² = 0.511 MeV
constexpr float m_ec2_MeV = 0.511f;

// ============================================================================
// Bohr Energy Straggling
// ============================================================================
// Bohr straggling theory: σ²_E = 4π N_e r_e² (m_e c²)² z² L ρ ds
// Returns standard deviation of energy loss [MeV]
// Valid for thick absorbers (κ >> 1, Gaussian regime)
inline float bohr_energy_straggling_sigma(float E_MeV, float ds_mm, float rho = 1.0f) {
    // Bohr straggling: σ²_E = 4π N_e r_e² (m_e c²)² z² L ρ ds
    // The energy dependence enters through the β factor in the log term L
    // Simplified for protons in water with β correction:
    // σ_E ≈ κ * sqrt(ρ * ds) / β

    // Relativistic kinematics for β correction
    float gamma = (E_MeV + 938.272f) / 938.272f;
    float beta2 = 1.0f - 1.0f / (gamma * gamma);
    float beta = sqrtf(fmaxf(beta2, 0.01f));  // Avoid division by zero

    constexpr float kappa_0 = 0.156f;  // Bohr constant for water [MeV/√mm] at β≈1
    return kappa_0 * sqrtf(rho * ds_mm) / beta;
}

// ============================================================================
// Vavilov Parameter (κ = ξ / T_max)
// ============================================================================
// Characteristic energy ξ [MeV]
// ξ = (K/2) * (Z/A) * (z²/β²) * ρ * x
// where K = 4π N_A r_e² m_e c² = 0.307 MeV cm²/g for protons
inline float vavilov_xi(float beta, float Z_A, float rho_g_cm3, float ds_cm) {
    constexpr float K = 0.307f;  // MeV cm²/g
    constexpr float z = 1.0f;    // Proton charge

    return (K / 2.0f) * Z_A * (z * z / (beta * beta)) * rho_g_cm3 * ds_cm;
}

// Maximum energy transfer to electron
// T_max = (2 m_e c² β² γ²) / (1 + 2γ m_e/m_p + (m_e/m_p)²)
inline float vavilov_T_max(float beta, float gamma) {
    float m_ratio = m_ec2_MeV / 938.272f;
    float num = 2.0f * m_ec2_MeV * beta * beta * gamma * gamma;
    float denom = 1.0f + 2.0f * gamma * m_ratio + m_ratio * m_ratio;
    return num / fmaxf(denom, 1e-10f);
}

// Vavilov distribution parameter κ = ξ / T_max
// κ >> 1: Bohr (Gaussian) regime
// κ << 1: Landau regime (asymmetric distribution)
// 0.01 < κ < 10: Vavilov regime
inline float vavilov_kappa(float E_MeV, float ds_mm, float rho = 1.0f) {
    // Water: Z/A = 10/18 ≈ 0.555
    constexpr float Z_A = 0.555f;

    float gamma = (E_MeV + 938.272f) / 938.272f;
    float beta = sqrtf(fmaxf(1.0f - 1.0f / (gamma * gamma), 0.0f));

    float ds_cm = ds_mm / 10.0f;
    float xi = vavilov_xi(beta, Z_A, rho, ds_cm);
    float T_max = vavilov_T_max(beta, gamma);

    float kappa = xi / fmaxf(T_max, 1e-10f);
    return fmaxf(kappa, 1e-6f);  // Avoid division by zero
}

// ============================================================================
// Regime-Dependent Energy Straggling
// ============================================================================
inline float energy_straggling_sigma(float E_MeV, float ds_mm, float rho = 1.0f) {
    float kappa = vavilov_kappa(E_MeV, ds_mm, rho);

    if (kappa > 10.0f) {
        // Bohr (Gaussian) regime - thick absorber
        return bohr_energy_straggling_sigma(E_MeV, ds_mm, rho);
    } else if (kappa < 0.01f) {
        // Landau regime - thin absorber (therapeutic protons typically here)
        //
        // PHYSICS NOTE: Landau distribution is asymmetric with no well-defined sigma.
        // The approximation FWHM ≈ 4ξ and sigma_eff = FWHM/2.355 is commonly used
        // but introduces quantitative error. For therapeutic protons (50-250 MeV),
        // kappa is typically in the Vavilov regime (0.01 < κ < 10), so this
        // approximation has limited impact.
        //
        // Use effective sigma = FWHM / 2.355 for comparison with Gaussian
        constexpr float Z_A = 0.555f;
        float gamma = (E_MeV + 938.272f) / 938.272f;
        float beta = sqrtf(fmaxf(1.0f - 1.0f / (gamma * gamma), 0.0f));
        float ds_cm = ds_mm / 10.0f;
        float xi = vavilov_xi(beta, Z_A, rho, ds_cm);
        return 4.0f * xi / 2.355f;  // Effective sigma from Landau width
    } else {
        // Vavilov regime - intermediate
        // Interpolate between Bohr and Landau
        float sigma_bohr = bohr_energy_straggling_sigma(E_MeV, ds_mm, rho);
        constexpr float Z_A = 0.555f;
        float gamma = (E_MeV + 938.272f) / 938.272f;
        float beta = sqrtf(fmaxf(1.0f - 1.0f / (gamma * gamma), 0.0f));
        float ds_cm = ds_mm / 10.0f;
        float xi = vavilov_xi(beta, Z_A, rho, ds_cm);
        float sigma_landau = 4.0f * xi / 2.355f;

        // Smooth interpolation based on kappa
        float w = 1.0f / (1.0f + kappa);  // w → 1 for Landau, w → 0 for Bohr
        return w * sigma_landau + (1.0f - w) * sigma_bohr;
    }
}

// ============================================================================
// Most Probable Energy Loss (Landau)
// ============================================================================
// Δp = ξ [ln(ξ/T_max) + ln(1 + β²γ²) + 0.2 - β² - δ/2]
// Simplified for water
inline float most_probable_energy_loss(float E_MeV, float ds_mm, float rho = 1.0f) {
    constexpr float Z_A = 0.555f;

    float gamma = (E_MeV + 938.272f) / 938.272f;
    float beta = sqrtf(fmaxf(1.0f - 1.0f / (gamma * gamma), 0.0f));

    float ds_cm = ds_mm / 10.0f;
    float xi = vavilov_xi(beta, Z_A, rho, ds_cm);
    float T_max = vavilov_T_max(beta, gamma);

    // Landau most probable value (simplified, neglecting density effect)
    float ln_term = logf(xi / fmaxf(T_max, 1e-10f));
    float delta_p = xi * (ln_term + 0.2f - beta * beta);

    return fmaxf(delta_p, 0.0f);
}

// ============================================================================
// Sampling
// ============================================================================
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
