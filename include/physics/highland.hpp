#pragma once
#include <cmath>

// Proton rest mass [MeV/c²]
namespace { const float m_p_MeV = 938.272f; }

// Highland formula for multiple Coulomb scattering (PDG 2024)
// Returns sigma_theta [radians]
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
inline float highland_sigma(float E_MeV, float ds, float X0 = 360.8f) {
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
    bracket = fmaxf(bracket, 0.5f);  // Physical minimum (PDG recommendation)

    return (13.6f * z / (beta * p_MeV)) * sqrtf(t) * bracket;
}

// 7-point Gaussian quadrature weights for angular splitting
constexpr int N_QUADRATURE = 7;
constexpr float QUADRATURE_WEIGHTS[N_QUADRATURE] = {
    0.05f, 0.10f, 0.20f, 0.30f, 0.20f, 0.10f, 0.05f
};
constexpr float QUADRATURE_DELTAS[N_QUADRATURE] = {
    -3.0f, -1.5f, -0.5f, 0.0f, +0.5f, +1.5f, +3.0f
};
