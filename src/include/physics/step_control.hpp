#pragma once
#include "lut/r_lut.hpp"
#include <cmath>

// ============================================================================
// R-based step size control with adaptive refinement near Bragg peak
// ============================================================================
// IC-1: Uses R(E) LUT, not S(E) directly
//
// Step size is limited by:
// 1. Fraction of remaining range (2% default)
// 2. Energy-dependent refinement near Bragg peak
// 3. Maximum absolute step size (1 mm)
//
// Near Bragg peak (E < 20 MeV), stopping power varies rapidly,
// requiring smaller steps for accuracy.
inline float compute_max_step_physics(const RLUT& lut, float E) {
    float R = lut.lookup_R(E);

    // Base: 2% of remaining range
    float delta_R_max = 0.02f * R;

    // Energy-dependent refinement factor
    // dS/dE increases near Bragg peak, requiring smaller steps
    float dS_factor = 1.0f;

    if (E < 5.0f) {
        // Very near end of range: extreme refinement
        dS_factor = 0.2f;
        delta_R_max = fminf(delta_R_max, 0.1f);  // Max 0.1 mm
    } else if (E < 10.0f) {
        // Near Bragg peak: high refinement
        dS_factor = 0.3f;
        delta_R_max = fminf(delta_R_max, 0.2f);  // Max 0.2 mm
    } else if (E < 20.0f) {
        // Bragg peak region: moderate refinement
        dS_factor = 0.5f;
        delta_R_max = fminf(delta_R_max, 0.5f);  // Max 0.5 mm
    } else if (E < 50.0f) {
        // Pre-Bragg: light refinement
        dS_factor = 0.7f;
        delta_R_max = fminf(delta_R_max, 0.7f);  // Max 0.7 mm
    }

    // Apply refinement factor
    delta_R_max = delta_R_max * dS_factor;

    // Hard upper limit (prevents overly large steps at high energy)
    delta_R_max = fminf(delta_R_max, 1.0f);  // Max 1 mm

    // Hard lower limit (prevents excessive subdivision)
    delta_R_max = fmaxf(delta_R_max, 0.05f);  // Min 0.05 mm

    return delta_R_max;  // dR/ds ≈ 1 in CSDA approximation
}

// Energy update using R-based control
inline float compute_energy_after_step(const RLUT& lut, float E, float step_length) {
    float R_current = lut.lookup_R(E);
    float R_new = R_current - step_length;

    if (R_new <= 0) {
        return 0.0f;
    }

    return lut.lookup_E_inverse(R_new);
}

// Energy deposited in this step
inline float compute_energy_deposition(const RLUT& lut, float E, float step_length) {
    float E_new = compute_energy_after_step(lut, E, step_length);
    return E - E_new;
}

// Energy deposition using stopping power (more physically accurate)
// dE = S(E) * ds * rho, where S is in [MeV cm²/g], ds in [mm], rho in [g/cm³]
inline float compute_energy_deposition_stopping_power(const RLUT& lut, float E, float step_length, float rho = 1.0f) {
    float S = lut.lookup_S(E);  // Stopping power [MeV cm²/g]
    // Convert: S [MeV cm²/g] * rho [g/cm³] * ds [mm] / 10 [mm/cm] = [MeV]
    return S * rho * step_length / 10.0f;
}
