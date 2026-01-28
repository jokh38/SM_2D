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
    const float* __restrict__ E_edges; // Energy bin edges (N_E+1 values) - Option D2
};

// ============================================================================
// Device Functions for LUT Lookup
// ============================================================================

// Find energy bin using binary search (device version)
// Works for both log-spaced and piecewise-uniform grids
__device__ inline int device_find_bin_edges(const float* __restrict__ edges, int N_E, float E) {
    if (E <= edges[0]) return 0;
    if (E >= edges[N_E]) return N_E - 1;

    // Binary search for piecewise-uniform grid
    int lo = 0, hi = N_E;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (edges[mid + 1] <= E) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Legacy: Find bin using log-spaced formula (kept for compatibility)
__device__ inline int device_find_bin_log(float E, int N_E, float E_min, float E_max) {
    if (E <= E_min) return 0;
    if (E >= E_max) return N_E - 1;

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
    // Option D2: Use binary search with E_edges for piecewise-uniform grid
    int bin = device_find_bin_edges(lut.E_edges, lut.N_E, E_clamped);

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
    // Option D2: Use binary search with E_edges for piecewise-uniform grid
    int bin = device_find_bin_edges(lut.E_edges, lut.N_E, E_clamped);

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
// 3. Adapt to energy-dependent stopping power variations (Option D2 adaptive)
__device__ inline float device_compute_max_step(const DeviceRLUT& lut, float E, float dx = 1.0f, float dz = 1.0f) {
    float R = device_lookup_R(lut, E);

    // Base: 2% of remaining range
    float delta_R_max = 0.02f * R;

    // Option D2: Adaptive step size per energy group
    // Each group has a minimum step to ensure particles cross energy bins
    float step_min_group;

    if (E < 2.0f) {
        // [0.1-2 MeV]: 0.1 MeV/bin, use 0.2mm step to cross bins
        step_min_group = 0.2f;
    } else if (E < 20.0f) {
        // [2-20 MeV]: 0.25 MeV/bin, use 0.5mm step
        step_min_group = 0.5f;
    } else if (E < 100.0f) {
        // [20-100 MeV]: 0.5 MeV/bin, use 1.0mm step
        step_min_group = 1.0f;
    } else {
        // [100-250 MeV]: 1 MeV/bin, use 2.0mm step
        step_min_group = 2.0f;
    }

    // Use the larger of: 2% of range OR group minimum step
    delta_R_max = fmaxf(delta_R_max, step_min_group);

    // Additional refinement near Bragg peak for accuracy
    if (E < 0.5f) {
        // End of range - use very small steps
        delta_R_max = fminf(delta_R_max, 0.1f);
    }

    // REMOVED: Artificial cell_limit was causing step size to be limited to 0.125mm (0.25 * 0.5mm)
    // This prevented 150MeV protons from traveling their full ~158mm range
    // Boundary crossing detection already handles cell-size limiting properly
    // float cell_limit = 0.25f * fminf(dx, dz);
    // delta_R_max = fminf(delta_R_max, cell_limit);

    return delta_R_max;
}
