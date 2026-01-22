#pragma once
#include "lut/r_lut.hpp"
#include <cmath>

// R-based step size control (IC-1: NO S(E) USAGE)
inline float compute_max_step_physics(const RLUT& lut, float E) {
    float R = lut.lookup_R(E);

    // Option A: Fixed fraction of remaining range
    float delta_R_max = 0.02f * R;  // Max 2% range loss per substep

    // Energy-dependent refinement near Bragg
    if (E < 10.0f) {
        delta_R_max = fminf(delta_R_max, 0.2f);
    } else if (E < 50.0f) {
        delta_R_max = fminf(delta_R_max, 0.5f);
    }

    return delta_R_max;  // This IS the step size (dR/ds ≈ 1 in CSDA)
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
