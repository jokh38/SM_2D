#pragma once
#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// Device-side Range Lookup Table (LUT) for GPU kernels
// ============================================================================
// P2 FIX: Device-accessible LUT structure for K3 FineTransport GPU kernel
//
// This structure mirrors the CPU RLUT but is designed for GPU memory access:
// - Fixed-size arrays for efficient coalesced reads
// - Pointers to device memory for R, S, log_E, log_R, log_S
// - Inline __device__ functions for fast interpolation
// ============================================================================

struct DeviceRLUT {
    int N_E;                  // Number of energy bins
    float E_min;              // Minimum energy [MeV]
    float E_max;              // Maximum energy [MeV]
    const float* __restrict__ R;       // CSDA range [mm] (device pointer)
    const float* __restrict__ S;       // Stopping power [MeV cm²/g] (device pointer)
    const float* __restrict__ log_E;   // Pre-computed log(E) (device pointer)
    const float* __restrict__ log_R;   // Pre-computed log(R) (device pointer)
    const float* __restrict__ log_S;   // Pre-computed log(S) (device pointer)
};

// ============================================================================
// Device Functions for LUT Lookup
// ============================================================================

// Find energy bin using binary search (device version)
// Note: For log-spaced grid, can also compute directly: floor(log(E/E_min) / dlog)
__device__ inline int device_find_bin(float E, int N_E, float E_min, float E_max) {
    if (E <= E_min) return 0;
    if (E >= E_max) return N_E - 1;

    // For log-spaced grid, use direct computation (faster than binary search)
    float log_E = logf(E);
    float log_E_min = logf(E_min);
    float log_E_max = logf(E_max);
    float dlog = (log_E_max - log_E_min) / N_E;

    int bin = static_cast<int>((log_E - log_E_min) / dlog);
    return max(0, min(bin, N_E - 1));
}

// Lookup R(E) using log-log interpolation on device
__device__ inline float device_lookup_R(const DeviceRLUT& lut, float E) {
    float E_clamped = fmaxf(lut.E_min, fminf(E, lut.E_max));
    int bin = device_find_bin(E_clamped, lut.N_E, lut.E_min, lut.E_max);

    float log_E_val = logf(E_clamped);
    float log_E0 = lut.log_E[bin];
    float log_E1 = lut.log_E[min(bin + 1, lut.N_E - 1)];
    float log_R0 = lut.log_R[bin];
    float log_R1 = lut.log_R[min(bin + 1, lut.N_E - 1)];

    float d_log_E = log_E1 - log_E0;
    if (fabsf(d_log_E) < 1e-10f) return expf(log_R0);

    float log_R_val = log_R0 + (log_R1 - log_R0) * (log_E_val - log_E0) / d_log_E;
    return expf(log_R_val);
}

// Lookup S(E) using log-log interpolation on device
__device__ inline float device_lookup_S(const DeviceRLUT& lut, float E) {
    float E_clamped = fmaxf(lut.E_min, fminf(E, lut.E_max));
    int bin = device_find_bin(E_clamped, lut.N_E, lut.E_min, lut.E_max);

    float log_E_val = logf(E_clamped);
    float log_E0 = lut.log_E[bin];
    float log_E1 = lut.log_E[min(bin + 1, lut.N_E - 1)];
    float log_S0 = lut.log_S[bin];
    float log_S1 = lut.log_S[min(bin + 1, lut.N_E - 1)];

    float d_log_E = log_E1 - log_E0;
    if (fabsf(d_log_E) < 1e-10f) return expf(log_S0);

    float log_S_val = log_S0 + (log_S1 - log_S0) * (log_E_val - log_E0) / d_log_E;
    return expf(log_S_val);
}

// Inverse lookup: E from R (device version)
__device__ inline float device_lookup_E_inverse(const DeviceRLUT& lut, float R_input) {
    // Handle boundary cases
    if (R_input <= 0.0f) return lut.E_min;

    // Binary search for R bin (R is monotonically increasing)
    int lo = 0, hi = lut.N_E;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (lut.R[mid] < R_input) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int bin = max(0, min(lo, lut.N_E - 2));

    // Log-log interpolation
    float log_R_val = logf(R_input);
    float log_R0 = lut.log_R[bin];
    float log_R1 = lut.log_R[bin + 1];
    float log_E0 = lut.log_E[bin];
    float log_E1 = lut.log_E[bin + 1];

    float d_log_R = log_R1 - log_R0;
    if (fabsf(d_log_R) < 1e-10f) return expf(log_E0);

    float log_E_val = log_E0 + (log_E1 - log_E0) * (log_R_val - log_R0) / d_log_R;
    return expf(log_E_val);
}

// ============================================================================
// Device Functions for Energy Update
// ============================================================================

// Compute energy after step using R-based control (device version)
__device__ inline float device_compute_energy_after_step(const DeviceRLUT& lut, float E, float step_length) {
    float R_current = device_lookup_R(lut, E);
    float R_new = R_current - step_length;

    if (R_new <= 0) return 0.0f;
    return device_lookup_E_inverse(lut, R_new);
}

// Compute energy deposition in this step (device version)
__device__ inline float device_compute_energy_deposition(const DeviceRLUT& lut, float E, float step_length) {
    float E_new = device_compute_energy_after_step(lut, E, step_length);
    return E - E_new;
}

// ============================================================================
// Device Functions for Step Size Control
// ============================================================================

// Compute maximum step size based on physics (device version)
__device__ inline float device_compute_max_step(const DeviceRLUT& lut, float E) {
    float R = device_lookup_R(lut, E);

    // Base: 2% of remaining range
    float delta_R_max = 0.02f * R;

    // Energy-dependent refinement factor
    float dS_factor = 1.0f;

    if (E < 5.0f) {
        dS_factor = 0.2f;
        delta_R_max = fminf(delta_R_max, 0.1f);
    } else if (E < 10.0f) {
        dS_factor = 0.3f;
        delta_R_max = fminf(delta_R_max, 0.2f);
    } else if (E < 20.0f) {
        dS_factor = 0.5f;
        delta_R_max = fminf(delta_R_max, 0.5f);
    } else if (E < 50.0f) {
        dS_factor = 0.7f;
        delta_R_max = fminf(delta_R_max, 0.7f);
    }

    delta_R_max = delta_R_max * dS_factor;
    delta_R_max = fminf(delta_R_max, 1.0f);
    delta_R_max = fmaxf(delta_R_max, 0.05f);

    return delta_R_max;
}
