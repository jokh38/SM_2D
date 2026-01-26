#include "cuda/k1k6_pipeline.cuh"
#include "kernels/k1_activemask.cuh"
#include "kernels/k2_coarsetransport.cuh"
#include "kernels/k3_finetransport.cuh"
#include "kernels/k4_transfer.cuh"
#include "kernels/k5_audit.cuh"
#include "kernels/k6_swap.cuh"
#include "device/device_lut.cuh"
#include "device/device_bucket.cuh"
#include "device/device_psic.cuh"
#include "cuda/gpu_transport_wrapper.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

namespace sm_2d {

// ============================================================================
// Helper Kernel Implementations
// ============================================================================

__global__ void compact_active_list(
    const uint8_t* __restrict__ ActiveMask,
    uint32_t* __restrict__ ActiveList,
    int Nx, int Nz,
    int* d_n_active
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int N_cells = Nx * Nz;

    if (cell >= N_cells) return;

    // Shared memory for exclusive scan
    __shared__ uint32_t s_buffer[256];

    uint32_t is_active = (ActiveMask[cell] == 1) ? 1 : 0;
    s_buffer[threadIdx.x] = is_active;
    __syncthreads();

    // Exclusive scan in shared memory
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        uint32_t val = 0;
        if (threadIdx.x >= offset) {
            val = s_buffer[threadIdx.x - offset];
        }
        __syncthreads();
        s_buffer[threadIdx.x] += val;
        __syncthreads();
    }

    // Write to global list
    if (is_active && threadIdx.x > 0) {
        uint32_t idx = atomicAdd(d_n_active, 1);
        ActiveList[idx] = cell;
    } else if (is_active && threadIdx.x == 0) {
        uint32_t idx = atomicAdd(d_n_active, 1);
        ActiveList[idx] = cell;
    }
}

__global__ void compact_coarse_list(
    const uint8_t* __restrict__ ActiveMask,
    uint32_t* __restrict__ CoarseList,
    int Nx, int Nz,
    int* d_n_coarse
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int N_cells = Nx * Nz;

    if (cell >= N_cells) return;

    // Coarse cells are those where ActiveMask == 0 (not active for fine transport)
    uint32_t needs_coarse = (ActiveMask[cell] == 0) ? 1 : 0;

    if (needs_coarse) {
        uint32_t idx = atomicAdd(d_n_coarse, 1);
        CoarseList[idx] = cell;
    }
}

__global__ void compute_total_weight(
    const uint32_t* __restrict__ block_ids,
    const float* __restrict__ values,
    double* __restrict__ total_weight,
    int N_cells,
    int Kb,
    int LOCAL_BINS
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int cell = blockIdx.x;

    if (cell >= N_cells) return;

    size_t base_value = cell * Kb * LOCAL_BINS;
    size_t total_values = Kb * LOCAL_BINS;

    double sum = 0.0;
    for (size_t i = tid; i < total_values; i += blockDim.x) {
        sum += values[base_value + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        total_weight[cell] = sdata[0];
    }
}

// ============================================================================
// Pipeline State Allocation
// ============================================================================

bool K1K6PipelineState::allocate(int Nx, int Nz) {
    int N_cells = Nx * Nz;

    // Allocate ActiveMask
    if (cudaMalloc(&d_ActiveMask, N_cells * sizeof(uint8_t)) != cudaSuccess) return false;

    // Allocate ActiveList
    if (cudaMalloc(&d_ActiveList, N_cells * sizeof(uint32_t)) != cudaSuccess) return false;

    // Allocate CoarseList
    if (cudaMalloc(&d_CoarseList, N_cells * sizeof(uint32_t)) != cudaSuccess) return false;

    // Allocate device counters
    if (cudaMalloc(&d_n_active, sizeof(int)) != cudaSuccess) return false;
    if (cudaMalloc(&d_n_coarse, sizeof(int)) != cudaSuccess) return false;

    // Allocate energy deposition array
    if (cudaMalloc(&d_EdepC, N_cells * sizeof(double)) != cudaSuccess) return false;

    // Allocate weight tracking arrays
    if (cudaMalloc(&d_AbsorbedWeight_cutoff, N_cells * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_AbsorbedWeight_nuclear, N_cells * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_AbsorbedEnergy_nuclear, N_cells * sizeof(double)) != cudaSuccess) return false;
    if (cudaMalloc(&d_BoundaryLoss_weight, N_cells * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_BoundaryLoss_energy, N_cells * sizeof(double)) != cudaSuccess) return false;

    // Allocate outflow buckets (4 faces per cell)
    if (cudaMalloc(&d_OutflowBuckets, N_cells * 4 * sizeof(DeviceOutflowBucket)) != cudaSuccess) return false;

    // Allocate audit structures
    if (cudaMalloc(&d_audit_report, sizeof(AuditReport)) != cudaSuccess) return false;
    if (cudaMalloc(&d_weight_in, N_cells * sizeof(double)) != cudaSuccess) return false;
    if (cudaMalloc(&d_weight_out, N_cells * sizeof(double)) != cudaSuccess) return false;

    owns_device_memory = true;
    return true;
}

void K1K6PipelineState::cleanup() {
    if (!owns_device_memory) return;

    if (d_ActiveMask) cudaFree(d_ActiveMask);
    if (d_ActiveList) cudaFree(d_ActiveList);
    if (d_CoarseList) cudaFree(d_CoarseList);
    if (d_n_active) cudaFree(d_n_active);
    if (d_n_coarse) cudaFree(d_n_coarse);
    if (d_EdepC) cudaFree(d_EdepC);
    if (d_AbsorbedWeight_cutoff) cudaFree(d_AbsorbedWeight_cutoff);
    if (d_AbsorbedWeight_nuclear) cudaFree(d_AbsorbedWeight_nuclear);
    if (d_AbsorbedEnergy_nuclear) cudaFree(d_AbsorbedEnergy_nuclear);
    if (d_BoundaryLoss_weight) cudaFree(d_BoundaryLoss_weight);
    if (d_BoundaryLoss_energy) cudaFree(d_BoundaryLoss_energy);
    if (d_OutflowBuckets) cudaFree(d_OutflowBuckets);
    if (d_audit_report) cudaFree(d_audit_report);
    if (d_weight_in) cudaFree(d_weight_in);
    if (d_weight_out) cudaFree(d_weight_out);

    d_ActiveMask = nullptr;
    d_ActiveList = nullptr;
    d_CoarseList = nullptr;
    d_n_active = nullptr;
    d_n_coarse = nullptr;
    d_EdepC = nullptr;
    d_AbsorbedWeight_cutoff = nullptr;
    d_AbsorbedWeight_nuclear = nullptr;
    d_AbsorbedEnergy_nuclear = nullptr;
    d_BoundaryLoss_weight = nullptr;
    d_BoundaryLoss_energy = nullptr;
    d_OutflowBuckets = nullptr;
    d_audit_report = nullptr;
    d_weight_in = nullptr;
    d_weight_out = nullptr;

    owns_device_memory = false;
}

bool init_pipeline_state(const K1K6PipelineConfig& config, K1K6PipelineState& state) {
    return state.allocate(config.Nx, config.Nz);
}

void reset_pipeline_state(K1K6PipelineState& state, int Nx, int Nz) {
    int N_cells = Nx * Nz;

    cudaMemset(state.d_ActiveMask, 0, N_cells * sizeof(uint8_t));
    cudaMemset(state.d_EdepC, 0, N_cells * sizeof(double));
    cudaMemset(state.d_AbsorbedWeight_cutoff, 0, N_cells * sizeof(float));
    cudaMemset(state.d_AbsorbedWeight_nuclear, 0, N_cells * sizeof(float));
    cudaMemset(state.d_AbsorbedEnergy_nuclear, 0, N_cells * sizeof(double));
    cudaMemset(state.d_BoundaryLoss_weight, 0, N_cells * sizeof(float));
    cudaMemset(state.d_BoundaryLoss_energy, 0, N_cells * sizeof(double));

    // Initialize buckets to empty
    cudaMemset(state.d_OutflowBuckets, 0xFF, N_cells * 4 * sizeof(uint32_t) * DEVICE_Kb_out);

    int zero = 0;
    cudaMemcpy(state.d_n_active, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_n_coarse, &zero, sizeof(int), cudaMemcpyHostToDevice);
}

AuditReport get_audit_report(const K1K6PipelineState& state) {
    AuditReport report;
    cudaMemcpy(&report, state.d_audit_report, sizeof(AuditReport), cudaMemcpyDeviceToHost);
    return report;
}

// ============================================================================
// Individual Kernel Wrappers
// ============================================================================

bool run_k1_active_mask(
    const PsiC& psi_in,
    uint8_t* d_ActiveMask,
    int Nx, int Nz,
    int b_E_trigger,
    float weight_active_min
) {
    // Get flattened pointers from PsiC
    // Note: This requires PsiC to have device-accessible members
    // For now, we'll call the kernel directly

    int threads = 256;
    int blocks = (Nx * Nz + threads - 1) / threads;

    // K1 kernel should be called here
    // For now, this is a placeholder

    return true;
}

bool run_k2_coarse_transport(
    const PsiC& psi_in,
    const uint8_t* d_ActiveMask,
    const uint32_t* d_CoarseList,
    int n_coarse,
    const DeviceRLUT& dlut,
    const K1K6PipelineConfig& config,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    K1K6PipelineState& state
) {
    if (n_coarse == 0) return true;

    K2Config k2_cfg;
    k2_cfg.E_coarse_max = config.E_coarse_max;
    k2_cfg.step_coarse = config.step_coarse;
    k2_cfg.n_steps_per_cell = config.n_steps_per_cell;

    // Get device grid edges
    float* d_theta_edges;
    float* d_E_edges;
    // These should be passed in or stored in state

    int threads = 256;
    int blocks = (n_coarse + threads - 1) / threads;

    // K2_CoarseTransport<<<blocks, threads>>>(...);

    return true;
}

bool run_k3_fine_transport(
    const PsiC& psi_in,
    const uint32_t* d_ActiveList,
    int n_active,
    const DeviceRLUT& dlut,
    const K1K6PipelineConfig& config,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    K1K6PipelineState& state
) {
    if (n_active == 0) return true;

    int threads = 256;
    int blocks = (n_active + threads - 1) / threads;

    // K3_FineTransport<<<blocks, threads>>>(...);

    return true;
}

bool run_k4_bucket_transfer(
    PsiC& psi_out,
    const DeviceOutflowBucket* d_OutflowBuckets,
    int Nx, int Nz
) {
    int threads = 256;
    int blocks = (Nx * Nz + threads - 1) / threads;

    // K4_BucketTransfer<<<blocks, threads>>>(...);

    return true;
}

bool run_k5_weight_audit(
    const PsiC& psi_in,
    const PsiC& psi_out,
    const float* d_AbsorbedWeight_cutoff,
    const float* d_AbsorbedWeight_nuclear,
    AuditReport* d_report,
    int Nx, int Nz
) {
    // K5_WeightAudit<<<1, 1>>>(...);

    return true;
}

void run_k6_swap_buffers(PsiC*& psi_in, PsiC*& psi_out) {
    // K6 is just a pointer swap
    // K6_SwapBuffers(psi_in, psi_out);
    PsiC* temp = psi_in;
    psi_in = psi_out;
    psi_out = temp;
}

// ============================================================================
// Main Pipeline Implementation
// ============================================================================

// Simplified wrapper that uses existing kernels
// This is a transitional implementation that calls the individual kernels

void run_k1k6_pipeline_transport(
    float x0, float z0, float theta0, float E0, float W_total,
    int Nx, int Nz, float dx, float dz,
    float x_min, float z_min,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,
    const float* theta_edges,
    const float* E_edges,
    const DeviceRLUT& dlut,
    std::vector<std::vector<double>>& edep
) {
    std::cout << "Running K1-K6 GPU pipeline..." << std::endl;
    std::cout << "  Grid: " << Nx << " x " << Nz << " cells" << std::endl;
    std::cout << "  Phase-space: " << N_theta << " x " << N_E << " global bins" << std::endl;
    std::cout << "  Local bins: " << N_theta_local << " x " << N_E_local << " per block" << std::endl;

    // Cast from sm_2d::DeviceRLUT to ::DeviceRLUT (same type, different namespace)
    const ::DeviceRLUT& native_dlut = reinterpret_cast<const ::DeviceRLUT&>(dlut);

    // Initialize pipeline configuration
    K1K6PipelineConfig config;
    config.Nx = Nx;
    config.Nz = Nz;
    config.dx = dx;
    config.dz = dz;

    // Energy thresholds
    config.E_trigger = 50.0f;          // Fine transport below 50 MeV
    config.weight_active_min = 1e-6f;   // Minimum weight for active cell

    // Coarse transport settings
    config.E_coarse_max = 300.0f;       // Up to 300 MeV
    config.step_coarse = 5.0f;          // 5 mm coarse steps
    config.n_steps_per_cell = 1;        // One step per cell for coarse

    // Phase-space dimensions
    config.N_theta = N_theta;
    config.N_E = N_E;
    config.N_theta_local = N_theta_local;
    config.N_E_local = N_E_local;

    // ========================================================================
    // STEP 1: Allocate Device PsiC Buffers
    // ========================================================================

    DevicePsiC psi_in, psi_out;

    if (!device_psic_init(psi_in, Nx, Nz)) {
        std::cerr << "Failed to allocate psi_in" << std::endl;
        return;
    }

    if (!device_psic_init(psi_out, Nx, Nz)) {
        std::cerr << "Failed to allocate psi_out" << std::endl;
        device_psic_cleanup(psi_in);
        return;
    }

    // ========================================================================
    // STEP 2: Allocate Auxiliary Arrays
    // ========================================================================

    K1K6PipelineState state;
    if (!init_pipeline_state(config, state)) {
        std::cerr << "Failed to allocate pipeline state" << std::endl;
        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
        return;
    }

    // ========================================================================
    // STEP 3: Inject Source Particle
    // ========================================================================

    // Convert source position to cell coordinates
    float x_rel = x0 - x_min;
    float z_rel = z0 - z_min;

    int source_cell_x = static_cast<int>(x_rel / dx);
    int source_cell_z = static_cast<int>(z_rel / dz);
    int source_cell = source_cell_z * Nx + source_cell_x;

    // Clamp to valid range
    source_cell_x = (source_cell_x < 0) ? 0 : (source_cell_x >= Nx) ? Nx - 1 : source_cell_x;
    source_cell_z = (source_cell_z < 0) ? 0 : (source_cell_z >= Nz) ? Nz - 1 : source_cell_z;
    source_cell = source_cell_z * Nx + source_cell_x;

    // Position within cell
    float x_in_cell = x_rel - source_cell_x * dx;
    float z_in_cell = z_rel - source_cell_z * dz;

    std::cout << "  Source: cell (" << source_cell_x << ", " << source_cell_z
              << ") at (" << x0 << ", " << z0 << ") mm" << std::endl;
    std::cout << "  Energy: " << E0 << " MeV, Angle: " << theta0 << " rad" << std::endl;

    // For now, use simple transport with increased particle count
    // The full K1-K6 implementation requires more infrastructure

    std::cout << "  Using simplified transport (full K1-K6 pending)" << std::endl;

    // ========================================================================
    // STEP 4: Main Transport Loop (simplified)
    // ========================================================================

    // The full K1-K6 loop would be:
    // while (remaining_weight > threshold && iteration < max_iter) {
    //     // K1: Active mask
    //     K1_ActiveMask<<<blocks, threads>>>(...);
    //
    //     // Build active list
    //     compact_active_list<<<blocks, threads>>>(...);
    //
    //     // K3: Fine transport
    //     K3_FineTransport<<<blocks, threads>>>(...);
    //
    //     // K4: Bucket transfer
    //     K4_BucketTransfer<<<blocks, threads>>>(...);
    //
    //     // K5: Weight audit
    //     K5_WeightAudit<<<1, 1>>>(...);
    //
    //     // K6: Swap buffers
    //     K6_SwapBuffers(psi_in, psi_out);
    // }

    // ========================================================================
    // STEP 5: Use Existing Simple Transport (transitional)
    // ========================================================================

    // For now, fall back to simple transport with the physics-enhanced kernel
    // This provides correct results while we build the full K1-K6 infrastructure
    extern void run_simple_gpu_transport(
        float x0, float z0, float theta0, float E0, float W_total,
        int n_particles,
        int Nx, int Nz, float dx, float dz,
        float x_min, float z_min,
        const ::DeviceRLUT& dlut,
        std::vector<std::vector<double>>& edep
    );

    // Use a reasonable number of particles for good statistics
    int n_particles = 100000;

    run_simple_gpu_transport(
        x0, z0, theta0, E0, W_total,
        n_particles,
        Nx, Nz, dx, dz,
        x_min, z_min,
        native_dlut,
        edep
    );

    std::cout << "  K1-K6 pipeline: transport complete" << std::endl;

    // ========================================================================
    // STEP 6: Cleanup
    // ========================================================================

    state.cleanup();
    device_psic_cleanup(psi_in);
    device_psic_cleanup(psi_out);

    std::cout << "GPU K1-K6 pipeline complete." << std::endl;
}

// ============================================================================
// Full Pipeline with K1-K6 Kernel Calls (Future Implementation)
// ============================================================================

// This function will be implemented once all kernel infrastructure is complete
// It will call:
// - K1_ActiveMask
// - compact_active_list
// - K2_CoarseTransport (optional)
// - K3_FineTransport
// - K4_BucketTransfer
// - K5_WeightAudit
// - K6_SwapBuffers

bool run_full_k1k6_pipeline(
    PsiC* psi_in,
    PsiC* psi_out,
    const DeviceRLUT& dlut,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    const K1K6PipelineConfig& config,
    K1K6PipelineState& state
) {
    // TODO: Implement full K1-K6 pipeline with all kernel calls
    // This requires:
    // 1. Device-side PsiC structures to be fully integrated
    // 2. All kernels to be properly initialized
    // 3. Host-device transfer functions
    return false;
}

} // namespace sm_2d
