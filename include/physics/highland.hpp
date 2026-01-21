#pragma once
#include <cmath>

// Highland formula for multiple Coulomb scattering
// Returns sigma_theta [radians]
inline float highland_sigma(float E_MeV, float ds, float X0 = 360.8f) {
    // X0 = 360.8 mm for water
    float beta = sqrtf(1.0f - powf(938.272f / (E_MeV + 938.272f), 2));
    float p_MeV = sqrtf(powf(E_MeV + 938.272f, 2) - 938.272f * 938.272f);
    float t = ds / X0;

    if (t < 1e-10f) return 0.0f;

    float ln_term = logf(t);
    float bracket = 1.0f + 0.038f * ln_term;

    // IC-8: Reduce step, don't clamp bracket
    if (bracket < 0.1f) {
        return -1.0f;  // Signal: reduce step size
    }

    return (13.6f / (beta * p_MeV)) * sqrtf(t) * bracket;
}

// 7-point Gaussian quadrature weights for angular splitting
constexpr int N_QUADRATURE = 7;
constexpr float QUADRATURE_WEIGHTS[N_QUADRATURE] = {
    0.05f, 0.10f, 0.20f, 0.30f, 0.20f, 0.10f, 0.05f
};
constexpr float QUADRATURE_DELTAS[N_QUADRATURE] = {
    -3.0f, -1.5f, -0.5f, 0.0f, +0.5f, +1.5f, +3.0f
};
