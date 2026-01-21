#pragma once
#include <cmath>

// Nuclear cross-section (energy-dependent)
inline float Sigma_total(float E_MeV) {
    if (E_MeV > 100.0f) return 0.0050f;  // mm⁻¹
    if (E_MeV > 50.0f)  return 0.0060f;
    return 0.0075f;
}

// Nuclear attenuation with energy budget tracking (IC-5)
inline float apply_nuclear_attenuation(
    float w_old,
    float E,
    float step_length,
    float& w_removed_out,
    float& E_removed_out
) {
    float survival = expf(-Sigma_total(E) * step_length);
    float w_new = w_old * survival;
    float w_removed = w_old - w_new;

    // Track removed weight and energy for conservation audit
    w_removed_out = w_removed;
    E_removed_out = w_removed * E;

    return w_new;
}
