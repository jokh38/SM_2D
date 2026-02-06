#pragma once
#include "physics/physics.hpp"
#include "lut/r_lut.hpp"
#include "device/device_lut.cuh"
#include "device/device_bucket.cuh"
#include <cstdint>

// Component state
struct Component {
    float theta, E, w, x, z, mu, eta;
};

// Result from single component transport
struct K3Result {
    float Edep = 0;
    float E_new = 0;           // Updated energy after transport (IC-2 fix)
    float nuclear_weight_removed = 0;
    float nuclear_energy_removed = 0;
    int bucket_emissions = 0;
    bool remained_in_cell = true;
    bool terminated = false;
    int split_count = 0;

    // IC-6: MCS direction update (was missing)
    float mu_new = 1.0f;       // Updated direction cosine
    float eta_new = 0.0f;      // Updated direction sine
    float theta_scatter = 0.0f; // Scattering angle for diagnostics
};

// P1 FIX: Updated GPU kernel signature with device LUT and bucket support
// This replaces the stub implementation with full physics transport
// Physics flags added for selective process enabling (testing/validation)
//
// NOTE: This is NOT a Monte Carlo code!
// Lateral spreading is implemented deterministically using Gaussian weight
// distribution across cells, not random sampling of scattering angles.
__global__ void K3_FineTransport(
    // Inputs
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint32_t* __restrict__ ActiveList,
    // Grid
    int Nx, int Nz, float dx, float dz,
    int n_active,
    // Device LUT (P2 FIX)
    const DeviceRLUT dlut,
    // Grid edges for bin finding
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,
    // Physics configuration flags (for testing/validation)
    bool enable_straggling,   // Enable energy straggling (Vavilov)
    bool enable_nuclear,      // Enable nuclear interactions
    // FIX C: Initial beam width for lateral spreading (from input config)
    float sigma_x_initial,
    // Lateral spreading is ALWAYS enabled (deterministic, not Monte Carlo)
    // Outputs
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,
    // Outflow buckets for boundary crossing (P3 FIX)
    DeviceOutflowBucket* __restrict__ OutflowBuckets,
    // CRITICAL FIX: Output phase space for particles remaining in cell
    uint32_t* __restrict__ block_ids_out,
    float* __restrict__ values_out
);

// Debug counters for slot allocation failures in K3 output PsiC.
void k3_reset_debug_counters();
void k3_get_debug_counters(
    unsigned long long& slot_drop_count,
    double& slot_drop_weight,
    double& slot_drop_energy,
    unsigned long long& bucket_drop_count,
    double& bucket_drop_weight,
    double& bucket_drop_energy,
    unsigned long long& pruned_weight_count,
    double& pruned_weight_sum,
    double& pruned_energy_sum
);

// CPU test stubs (unchanged)
K3Result run_K3_single_component(const Component& c);
K3Result run_K3_with_forced_split(const Component& c);

// ============================================================================
// Helper: Create DeviceRLUT from CPU RLUT
// ============================================================================
// This function copies CPU LUT data to GPU memory for kernel use
// P2 FIX: Enables device LUT access in K3 kernel
// Option D2: Added E_edges for piecewise-uniform grid support
struct DeviceLUTWrapper {
    DeviceRLUT dlut;
    float* d_R;
    float* d_S;
    float* d_log_E;
    float* d_log_R;
    float* d_log_S;
    float* d_E_edges;  // For piecewise-uniform grid binary search

    DeviceLUTWrapper() : d_R(nullptr), d_S(nullptr), d_log_E(nullptr),
                         d_log_R(nullptr), d_log_S(nullptr), d_E_edges(nullptr) {}

    // Initialize from CPU RLUT (allocates and copies to GPU)
    bool init(const RLUT& cpu_lut) {
        int N_E = cpu_lut.grid.N_E;

        // Allocate device memory
        size_t data_size = N_E * sizeof(float);
        size_t edges_size = (N_E + 1) * sizeof(float);

        cudaMalloc(&d_R, data_size);
        cudaMalloc(&d_S, data_size);
        cudaMalloc(&d_log_E, data_size);
        cudaMalloc(&d_log_R, data_size);
        cudaMalloc(&d_log_S, data_size);
        cudaMalloc(&d_E_edges, edges_size);  // Option D2: Edges for binary search

        // Copy data to device
        cudaMemcpy(d_R, cpu_lut.R.data(), data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_S, cpu_lut.S.data(), data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_log_E, cpu_lut.log_E.data(), data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_log_R, cpu_lut.log_R.data(), data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_log_S, cpu_lut.log_S.data(), data_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_E_edges, cpu_lut.grid.edges.data(), edges_size, cudaMemcpyHostToDevice);

        // Fill device LUT structure
        dlut.N_E = N_E;
        dlut.E_min = cpu_lut.grid.E_min;
        dlut.E_max = cpu_lut.grid.E_max;
        dlut.R = d_R;
        dlut.S = d_S;
        dlut.log_E = d_log_E;
        dlut.log_R = d_log_R;
        dlut.log_S = d_log_S;
        dlut.E_edges = d_E_edges;  // Option D2: Edges for binary search

        return true;
    }

    // Free GPU memory
    void cleanup() {
        if (d_R) cudaFree(d_R);
        if (d_S) cudaFree(d_S);
        if (d_log_E) cudaFree(d_log_E);
        if (d_log_R) cudaFree(d_log_R);
        if (d_log_S) cudaFree(d_log_S);
        if (d_E_edges) cudaFree(d_E_edges);  // Option D2
        d_R = d_S = d_log_E = d_log_R = d_log_S = d_E_edges = nullptr;
    }

    ~DeviceLUTWrapper() {
        cleanup();
    }
};
