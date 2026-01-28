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

// FIX: Compute maximum step size based on physics AND cell geometry
// Step size is designed to:
// 1. Resolve sub-cell features (sub-cell size = dx / N_x_sub = 0.5 / 4 = 0.125mm)
// 2. Allow boundary crossing (cell half-width = 0.25mm, need ~2-3 steps minimum)
// 3. Adapt to energy-dependent stopping power variations
__device__ inline float device_compute_max_step(const DeviceRLUT& lut, float E, float dx = 1.0f, float dz = 1.0f) {
    float R = device_lookup_R(lut, E);

    // Base: 2% of remaining range
    float delta_R_max = 0.02f * R;

    // Energy-dependent refinement factor
    // Adjusted for E_trigger = 10 MeV threshold between coarse and fine transport
    float dS_factor = 1.0f;

    if (E < 5.0f) {
        // Very low energy (near end of range) - use sub-cell resolution
        // Sub-cell size = 0.125mm, use 0.1mm to resolve features while allowing progress
        dS_factor = 0.6f;
        delta_R_max = fminf(delta_R_max, 0.1f);
    } else if (E < 10.0f) {
        // Low energy (Bragg peak region) - intermediate step
        // Need enough resolution but must allow boundary crossing
        dS_factor = 0.7f;
        delta_R_max = fminf(delta_R_max, 0.15f);
    } else if (E < 20.0f) {
        // Below coarse transport threshold - larger steps
        dS_factor = 0.8f;
        delta_R_max = fminf(delta_R_max, 0.3f);
    } else if (E < 50.0f) {
        // Just above threshold - even larger steps
        dS_factor = 0.9f;
        delta_R_max = fminf(delta_R_max, 0.5f);
    }

    delta_R_max = delta_R_max * dS_factor;
    // H5 FIX: Removed 1.0mm cap per SPEC.md:203 which requires delta_R_max = 0.02 * R
    // At 150 MeV: delta_R_max = 0.02 * 157.7mm = 3.15mm (was limited to 1.0mm)
    // This 3.15x smaller step size was causing particles to stop at 42mm instead of 158mm

    // CRITICAL FIX: Minimum step must allow boundary crossing in 2-3 iterations
    // Cell half-width = 0.25mm, so minimum step should be at least 0.1mm
    // This prevents particles from getting trapped in cells
    delta_R_max = fmaxf(delta_R_max, 0.1f);

    // REMOVED: Artificial cell_limit was causing step size to be limited to 0.125mm (0.25 * 0.5mm)
    // This prevented 150MeV protons from traveling their full ~158mm range
    // Boundary crossing detection already handles cell-size limiting properly
    // float cell_limit = 0.25f * fminf(dx, dz);
    // delta_R_max = fminf(delta_R_max, cell_limit);

    return delta_R_max;
}
