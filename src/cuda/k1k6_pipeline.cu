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

// Source particle injection kernel
__global__ void inject_source_kernel(
    DevicePsiC psi,
    int source_cell,
    float theta0, float E0, float W_total,
    float x_in_cell, float z_in_cell,
    float dx, float dz,
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local
) {
    // Only one thread needed for source injection
    device_psic_inject_source(
        psi,
        source_cell,
        theta0, E0, W_total,
        x_in_cell, z_in_cell,
        dx, dz,
        theta_edges, E_edges,
        N_theta, N_E,
        N_theta_local, N_E_local
    );
}

// Clear all outflow buckets
__global__ void clear_buckets_kernel(
    DeviceOutflowBucket* buckets,
    int num_buckets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_buckets) return;

    device_bucket_clear(buckets[idx]);
}

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
    const DevicePsiC& psi_in,
    uint8_t* d_ActiveMask,
    int Nx, int Nz,
    int b_E_trigger,
    float weight_active_min
) {
    int threads = 256;
    int blocks = (Nx * Nz + threads - 1) / threads;

    // Call K1_ActiveMask kernel
    K1_ActiveMask<<<blocks, threads>>>(
        psi_in.block_id,      // Block IDs from DevicePsiC
        psi_in.value,         // Values from DevicePsiC
        d_ActiveMask,
        Nx, Nz,
        b_E_trigger,
        weight_active_min
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "K1 kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    cudaDeviceSynchronize();
    return true;
}

bool run_k2_coarse_transport(
    const DevicePsiC& psi_in,
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

    // Allocate device grid edges
    float* d_theta_edges;
    float* d_E_edges;
    cudaMalloc(&d_theta_edges, (config.N_theta + 1) * sizeof(float));
    cudaMalloc(&d_E_edges, (config.N_E + 1) * sizeof(float));

    // Copy grid edges to device
    cudaMemcpy(d_theta_edges, a_grid.edges.data(), (config.N_theta + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E_edges, e_grid.edges.data(), (config.N_E + 1) * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n_coarse + threads - 1) / threads;

    // Call K2_CoarseTransport kernel
    K2_CoarseTransport<<<blocks, threads>>>(
        psi_in.block_id,
        psi_in.value,
        d_ActiveMask,
        config.Nx, config.Nz, config.dx, config.dz,
        n_coarse,
        dlut,
        d_theta_edges,
        d_E_edges,
        config.N_theta, config.N_E,
        config.N_theta_local, config.N_E_local,
        k2_cfg,
        state.d_EdepC,
        state.d_AbsorbedWeight_cutoff,
        state.d_AbsorbedWeight_nuclear,
        state.d_AbsorbedEnergy_nuclear,
        state.d_BoundaryLoss_weight,
        state.d_BoundaryLoss_energy,
        state.d_OutflowBuckets
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "K2 kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_theta_edges);
        cudaFree(d_E_edges);
        return false;
    }

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_theta_edges);
    cudaFree(d_E_edges);

    return true;
}

bool run_k3_fine_transport(
    const DevicePsiC& psi_in,
    const uint32_t* d_ActiveList,
    int n_active,
    const DeviceRLUT& dlut,
    const K1K6PipelineConfig& config,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    K1K6PipelineState& state
) {
    if (n_active == 0) return true;

    // Allocate device grid edges
    float* d_theta_edges;
    float* d_E_edges;
    cudaMalloc(&d_theta_edges, (config.N_theta + 1) * sizeof(float));
    cudaMalloc(&d_E_edges, (config.N_E + 1) * sizeof(float));

    // Copy grid edges to device
    cudaMemcpy(d_theta_edges, a_grid.edges.data(), (config.N_theta + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E_edges, e_grid.edges.data(), (config.N_E + 1) * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n_active + threads - 1) / threads;

    // Call K3_FineTransport kernel
    K3_FineTransport<<<blocks, threads>>>(
        psi_in.block_id,
        psi_in.value,
        d_ActiveList,
        config.Nx, config.Nz, config.dx, config.dz,
        n_active,
        dlut,
        d_theta_edges,
        d_E_edges,
        config.N_theta, config.N_E,
        config.N_theta_local, config.N_E_local,
        state.d_EdepC,
        state.d_AbsorbedWeight_cutoff,
        state.d_AbsorbedWeight_nuclear,
        state.d_AbsorbedEnergy_nuclear,
        state.d_BoundaryLoss_weight,
        state.d_BoundaryLoss_energy,
        state.d_OutflowBuckets
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "K3 kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_theta_edges);
        cudaFree(d_E_edges);
        return false;
    }

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_theta_edges);
    cudaFree(d_E_edges);

    return true;
}

bool run_k4_bucket_transfer(
    DevicePsiC& psi_out,
    const DeviceOutflowBucket* d_OutflowBuckets,
    int Nx, int Nz
) {
    int threads = 256;
    int blocks = (Nx * Nz + threads - 1) / threads;

    // Call K4_BucketTransfer kernel
    K4_BucketTransfer<<<blocks, threads>>>(
        d_OutflowBuckets,
        psi_out.value,
        psi_out.block_id,
        Nx, Nz
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "K4 kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    cudaDeviceSynchronize();
    return true;
}

bool run_k5_weight_audit(
    const DevicePsiC& psi_in,
    const DevicePsiC& psi_out,
    const float* d_AbsorbedWeight_cutoff,
    const float* d_AbsorbedWeight_nuclear,
    AuditReport* d_report,
    int Nx, int Nz
) {
    int N_cells = Nx * Nz;

    // Call K5_WeightAudit kernel
    K5_WeightAudit<<<1, 1>>>(
        psi_in.block_id,
        psi_in.value,
        psi_out.block_id,
        psi_out.value,
        d_AbsorbedWeight_cutoff,
        d_AbsorbedWeight_nuclear,
        d_report,
        N_cells
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "K5 kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    cudaDeviceSynchronize();
    return true;
}

void run_k6_swap_buffers(DevicePsiC*& psi_in, DevicePsiC*& psi_out) {
    // K6 is just a pointer swap
    DevicePsiC* temp = psi_in;
    psi_in = psi_out;
    psi_out = temp;
}

// ============================================================================
// Main Pipeline Implementation
// ============================================================================

bool run_k1k6_pipeline_transport(
    DevicePsiC* psi_in,
    DevicePsiC* psi_out,
    const DeviceRLUT& dlut,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    const K1K6PipelineConfig& config,
    K1K6PipelineState& state
) {
    std::cout << "Running K1-K6 GPU pipeline..." << std::endl;
    std::cout << "  Grid: " << config.Nx << " x " << config.Nz << " cells" << std::endl;
    std::cout << "  Phase-space: " << config.N_theta << " x " << config.N_E << " global bins" << std::endl;

    // Compute b_E_trigger from E_trigger
    int b_E_trigger = compute_b_E_trigger(config.E_trigger, e_grid, config.N_E_local);

    std::cout << "  Energy threshold: " << config.E_trigger << " MeV (b_E_trigger=" << b_E_trigger << ")" << std::endl;
    std::cout << "  Min weight: " << config.weight_active_min << std::endl;

    // Reset pipeline state
    reset_pipeline_state(state, config.Nx, config.Nz);

    // Clear output buffer
    device_psic_clear(*psi_out);

    // Maximum iterations
    const int max_iter = 100;
    const double weight_threshold = 1e-6;

    int iter = 0;
    double total_weight_remaining = 0.0;

    std::cout << "Starting main transport loop..." << std::endl;

    // ========================================================================
    // Main Transport Loop
    // ========================================================================
    while (iter < max_iter) {
        iter++;

        // --------------------------------------------------------------------
        // K1: Active Mask Identification
        // --------------------------------------------------------------------
        if (!run_k1_active_mask(*psi_in, state.d_ActiveMask, config.Nx, config.Nz,
                                 b_E_trigger, config.weight_active_min)) {
            std::cerr << "K1 failed at iteration " << iter << std::endl;
            return false;
        }

        // --------------------------------------------------------------------
        // Compact Active List and Coarse List
        // --------------------------------------------------------------------
        // Reset counters
        int zero = 0;
        cudaMemcpy(state.d_n_active, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(state.d_n_coarse, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // Launch compaction kernels
        int threads = 256;
        int blocks = (config.Nx * config.Nz + threads - 1) / threads;

        compact_active_list<<<blocks, threads>>>(
            state.d_ActiveMask,
            state.d_ActiveList,
            config.Nx, config.Nz,
            state.d_n_active
        );

        compact_coarse_list<<<blocks, threads>>>(
            state.d_ActiveMask,
            state.d_CoarseList,
            config.Nx, config.Nz,
            state.d_n_coarse
        );

        cudaDeviceSynchronize();

        // Get counts
        cudaMemcpy(&state.n_active, state.d_n_active, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&state.n_coarse, state.d_n_coarse, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "  Iteration " << iter << ": "
                  << state.n_active << " active, "
                  << state.n_coarse << " coarse cells" << std::endl;

        // Check if we're done
        if (state.n_active == 0 && state.n_coarse == 0) {
            std::cout << "  No active cells remaining, transport complete" << std::endl;
            break;
        }

        // --------------------------------------------------------------------
        // K2: Coarse Transport (high energy cells)
        // --------------------------------------------------------------------
        if (state.n_coarse > 0) {
            if (!run_k2_coarse_transport(*psi_in, state.d_ActiveMask, state.d_CoarseList,
                                          state.n_coarse, dlut, config, e_grid, a_grid, state)) {
                std::cerr << "K2 failed at iteration " << iter << std::endl;
                return false;
            }
        }

        // --------------------------------------------------------------------
        // K3: Fine Transport (low energy cells)
        // --------------------------------------------------------------------
        if (state.n_active > 0) {
            if (!run_k3_fine_transport(*psi_in, state.d_ActiveList, state.n_active,
                                       dlut, config, e_grid, a_grid, state)) {
                std::cerr << "K3 failed at iteration " << iter << std::endl;
                return false;
            }
        }

        // --------------------------------------------------------------------
        // K4: Bucket Transfer (boundary crossings)
        // --------------------------------------------------------------------
        if (!run_k4_bucket_transfer(*psi_out, state.d_OutflowBuckets, config.Nx, config.Nz)) {
            std::cerr << "K4 failed at iteration " << iter << std::endl;
            return false;
        }

        // Clear outflow buckets for next iteration
        size_t num_buckets = config.Nx * config.Nz * 4;
        int b_threads = 256;
        int b_blocks = (num_buckets + b_threads - 1) / b_threads;
        clear_buckets_kernel<<<b_blocks, b_threads>>>(state.d_OutflowBuckets, num_buckets);
        cudaDeviceSynchronize();

        // --------------------------------------------------------------------
        // K5: Weight Audit (conservation check)
        // --------------------------------------------------------------------
        if (!run_k5_weight_audit(*psi_in, *psi_out,
                                 state.d_AbsorbedWeight_cutoff,
                                 state.d_AbsorbedWeight_nuclear,
                                 state.d_audit_report,
                                 config.Nx, config.Nz)) {
            std::cerr << "K5 failed at iteration " << iter << std::endl;
            return false;
        }

        // Get audit report
        AuditReport report = get_audit_report(state);
        std::cout << "  Weight audit: error=" << report.W_error
                  << " pass=" << (report.W_pass ? "yes" : "no") << std::endl;

        // --------------------------------------------------------------------
        // K6: Swap Buffers
        // --------------------------------------------------------------------
        run_k6_swap_buffers(psi_in, psi_out);

        // Clear new input buffer for next iteration
        device_psic_clear(*psi_in);
    }

    std::cout << "K1-K6 pipeline: completed " << iter << " iterations" << std::endl;
    std::cout << "GPU K1-K6 pipeline complete." << std::endl;

    return true;
}

} // namespace sm_2d
