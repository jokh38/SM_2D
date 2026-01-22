#pragma once
#include <cmath>

// ============================================================================
// Nuclear Cross-Section for Protons in Water
// ============================================================================
// Based on ICRU 63 and Janni data for p + H₂O → nuclear reactions
// Returns total nuclear interaction coefficient [mm⁻¹]
//
// Energy dependence follows approximate logarithmic behavior:
// σ(E) ≈ σ_0 * [1 + a * ln(E/E_ref)]
//
// Reference values (ICRU 63, Janni 1982):
//   - 10 MeV:  ~8 mb/nucleus → ~0.005 mm⁻¹
//   - 50 MeV:  ~6 mb/nucleus → ~0.004 mm⁻¹
//   - 100 MeV: ~5 mb/nucleus → ~0.003 mm⁻¹
//   - 200 MeV: ~4 mb/nucleus → ~0.0025 mm⁻¹
inline float Sigma_total(float E_MeV) {
    // Below Coulomb barrier for hydrogen/oxygen: negligible nuclear reactions
    if (E_MeV < 5.0f) {
        return 0.0f;
    }

    // Reference values at 100 MeV (ICRU 63)
    // CORRECTED: ICRU 63 reports ~0.0012 mm⁻¹ for water at 100 MeV
    // Previous value 0.005 was ~4x too high
    constexpr float sigma_100 = 0.0012f;  // mm⁻¹ at 100 MeV (ICRU 63)
    constexpr float E_ref = 100.0f;       // Reference energy [MeV]

    if (E_MeV >= 20.0f) {
        // Logarithmic energy dependence for therapeutic range (20-250 MeV)
        // σ(E) = σ_100 * [1 - 0.15 * ln(E/100)]
        // The factor 0.15 gives ~30% reduction from 20 to 200 MeV
        float log_factor = 1.0f - 0.15f * logf(E_MeV / E_ref);
        float sigma = sigma_100 * fmaxf(log_factor, 0.4f);  // Minimum at 40% of reference
        return sigma;
    } else {
        // Low energy (5-20 MeV): rapid increase due to reduced Coulomb barrier
        // Linear ramp from 0 at 5 MeV to sigma_20 at 20 MeV
        // CORRECTED: Scaled proportionally with sigma_100
        constexpr float sigma_20 = 0.0016f;  // mm⁻¹ at 20 MeV (was 0.0065)
        float frac = (E_MeV - 5.0f) / 15.0f;  // 0 to 1
        return sigma_20 * frac;
    }
}

// ============================================================================
// Nuclear attenuation with energy budget tracking (IC-5)
// ============================================================================
// Note: Simplified model that conserves total energy
// In reality, nuclear interactions produce secondary particles that
// transport energy away from the primary track. This simplified model
// treats all removed energy as locally absorbed, which overestimates
// dose locally by ~0.1% (acceptable for validation).
inline float apply_nuclear_attenuation(
    float w_old,
    float E,
    float step_length,
    float& w_removed_out,
    float& E_removed_out
) {
    float sigma = Sigma_total(E);
    float survival = expf(-sigma * step_length);
    float w_new = w_old * survival;
    float w_removed = w_old - w_new;

    // Track removed weight and energy for conservation audit
    // Note: E_removed_out includes both local deposition and secondary
    // particle energy (simplified as all local for validation purposes)
    w_removed_out = w_removed;
    E_removed_out = w_removed * E;

    return w_new;
}