#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "physics/highland.hpp"

// ============================================================================
// FIX Problem 5: Fermi-Eyges Moment Tracking
// ============================================================================
// Purpose: Track lateral spreading moments using Fermi-Eyges theory
//
// The Fermi-Eyges theory describes the lateral spread of a charged particle
// beam due to multiple Coulomb scattering. The key quantities are:
//
// Scattering power: T(z) = dσ_θ²/dz [rad²/mm]
//   Rate of angular variance increase per unit path length
//
// Moments:
//   A₀(z) = ∫₀ᶻ T(z') dz'        : Total angular variance
//   A₁(z) = ∫₀ᶻ z' × T(z') dz'    : First spatial moment
//   A₂(z) = ∫₀ᶻ z'² × T(z') dz'   : Second spatial moment
//
// Lateral variance at depth z:
//   σ²_x(z) = A₀×z² - 2×A₁×z + A₂
//
// Reference: Fermi, Eyges; "On the Multiple Scattering of Charged Particles"
// ============================================================================

// Fermi-Eyges moment accumulator (per component)
struct FermiEygesMoments {
    double A0;  // ∫ T dz : Total angular variance [rad²]
    double A1;  // ∫ z×T dz : First moment [rad²·mm]
    double A2;  // ∫ z²×T dz : Second moment [rad²·mm²]

    __host__ __device__ FermiEygesMoments() : A0(0.0), A1(0.0), A2(0.0) {}

    __host__ __device__ FermiEygesMoments(double a0, double a1, double a2)
        : A0(a0), A1(a1), A2(a2) {}

    // Compute lateral variance at depth z
    __host__ __device__ double lateral_variance(double z) const {
        return A0 * z * z - 2.0 * A1 * z + A2;
    }

    __host__ __device__ double lateral_sigma(double z) const {
        double var = lateral_variance(z);
        return var > 0.0 ? sqrt(var) : 0.0;
    }

    // Accumulate contribution from a step
    __host__ __device__ void accumulate_step(double T, double z_step, double ds) {
        A0 += T * ds;
        A1 += T * z_step * ds;
        A2 += T * z_step * z_step * ds;
    }
};

// Compute scattering power T = dσ_θ²/dz from Highland formula
// Returns T in units of [rad²/mm]
__host__ __device__ inline float fermi_eyges_scattering_power(
    float E_MeV,     // Energy [MeV]
    float ds,        // Step size [mm]
    float X0 = 360.8f  // Radiation length [mm]
) {
    // Highland sigma for this step
    float sigma_theta = highland_sigma(E_MeV, ds, X0);

    // Scattering power: T ≈ σ_θ² / ds [rad²/mm]
    if (ds > 1e-6f) {
        return sigma_theta * sigma_theta / ds;
    }
    return 0.0f;
}

// Device function to update Fermi-Eyges moments during transport
__device__ inline void device_update_fermi_eyges_moments(
    FermiEygesMoments& moments,
    float E_MeV,
    float z_step,    // Current depth position [mm]
    float ds,        // Step size [mm]
    float X0 = 360.8f
) {
    float T = fermi_eyges_scattering_power(E_MeV, ds, X0);
    moments.A0 += T * ds;
    moments.A1 += T * z_step * ds;
    moments.A2 += T * z_step * z_step * ds;
}

// Compute lateral spread correction based on Fermi-Eyges theory
// This can be used to adjust particle positions after transport
__device__ inline void device_apply_fermi_eyges_spread(
    float& x,
    float z,
    const FermiEygesMoments& moments,
    unsigned& seed
) {
    // Compute expected lateral variance at this depth
    double sigma_sq = moments.lateral_variance(z);
    if (sigma_sq <= 0.0) return;

    // Sample from Gaussian with this variance
    float sigma = sqrtf(static_cast<float>(sigma_sq));

    // Box-Muller transform
    float u1 = (seed & 0xFFFF) / 65536.0f;
    float u2 = ((seed >> 16) & 0xFFFF) / 65536.0f;
    u1 = fmaxf(u1, 1e-10f);

    float z_gauss = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);

    // Apply lateral displacement
    x += z_gauss * sigma;
}
