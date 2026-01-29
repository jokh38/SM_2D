#include "k1k6_pipeline.cuh"
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

    // Clear the bucket directly
    DeviceOutflowBucket& bucket = buckets[idx];
    for (int i = 0; i < DEVICE_Kb_out; ++i) {
        bucket.block_id[i] = DEVICE_EMPTY_BLOCK_ID;
        bucket.local_count[i] = 0;
        for (int j = 0; j < DEVICE_LOCAL_BINS; ++j) {
            bucket.value[i][j] = 0.0f;
        }
    }
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
        // DEBUG: Track atomicAdd
        if (cell < 2000) {
            printf("compact_active write: cell=%d, idx=%u, threadIdx.x=%d\n", cell, idx, threadIdx.x);
        }
    } else if (is_active && threadIdx.x == 0) {
        uint32_t idx = atomicAdd(d_n_active, 1);
        ActiveList[idx] = cell;
        // DEBUG: Track atomicAdd
        if (cell < 2000) {
            printf("compact_active write (thread 0): cell=%d, idx=%u\n", cell, idx);
        }
    }
}

__global__ void compact_coarse_list(
    const uint8_t* __restrict__ ActiveMask,
    const uint32_t* __restrict__ block_ids,  // CRITICAL FIX: Check for actual weights
    const float* __restrict__ values,         // CRITICAL FIX: Check for actual weights
    uint32_t* __restrict__ CoarseList,
    int Nx, int Nz,
    int* d_n_coarse
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int N_cells = Nx * Nz;

    if (cell >= N_cells) return;

    // CRITICAL FIX: Only collect cells that BOTH:
    // 1. Have ActiveMask == 0 (need coarse transport)
    // 2. Have actual weight (non-empty cells)
    if (ActiveMask[cell] != 0) return;  // Skip active cells

    // Check if cell has any non-zero weight
    constexpr int Kb = DEVICE_Kb;
    constexpr int LOCAL_BINS_val = DEVICE_LOCAL_BINS;
    bool has_weight = false;

    for (int slot = 0; slot < Kb && !has_weight; ++slot) {
        uint32_t bid = block_ids[cell * Kb + slot];
        if (bid == DEVICE_EMPTY_BLOCK_ID) continue;

        for (int lidx = 0; lidx < LOCAL_BINS_val && !has_weight; ++lidx) {
            float w = values[(cell * Kb + slot) * LOCAL_BINS_val + lidx];
            if (w > 1e-12f) {
                has_weight = true;
                break;
            }
        }
    }

    if (has_weight) {
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
    cudaError_t e;

    std::cout << "Allocating pipeline state for " << Nx << "x" << Nz << " (" << N_cells << " cells)" << std::endl;

    // Allocate ActiveMask
    e = cudaMalloc(&d_ActiveMask, N_cells * sizeof(uint8_t));
    if (e != cudaSuccess) { std::cerr << "Failed d_ActiveMask: " << N_cells << " bytes - " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate ActiveList
    e = cudaMalloc(&d_ActiveList, N_cells * sizeof(uint32_t));
    if (e != cudaSuccess) { std::cerr << "Failed d_ActiveList: " << N_cells * 4 << " bytes - " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate CoarseList
    e = cudaMalloc(&d_CoarseList, N_cells * sizeof(uint32_t));
    if (e != cudaSuccess) { std::cerr << "Failed d_CoarseList: " << N_cells * 4 << " bytes - " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate device counters
    e = cudaMalloc(&d_n_active, sizeof(int));
    if (e != cudaSuccess) { std::cerr << "Failed d_n_active: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_n_coarse, sizeof(int));
    if (e != cudaSuccess) { std::cerr << "Failed d_n_coarse: " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate energy deposition array
    e = cudaMalloc(&d_EdepC, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_EdepC: " << N_cells * 8 << " bytes - " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate weight tracking arrays
    e = cudaMalloc(&d_AbsorbedWeight_cutoff, N_cells * sizeof(float));
    if (e != cudaSuccess) { std::cerr << "Failed d_AbsorbedWeight_cutoff: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_AbsorbedWeight_nuclear, N_cells * sizeof(float));
    if (e != cudaSuccess) { std::cerr << "Failed d_AbsorbedWeight_nuclear: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_AbsorbedEnergy_nuclear, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_AbsorbedEnergy_nuclear: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_BoundaryLoss_weight, N_cells * sizeof(float));
    if (e != cudaSuccess) { std::cerr << "Failed d_BoundaryLoss_weight: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_BoundaryLoss_energy, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_BoundaryLoss_energy: " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate outflow buckets (4 faces per cell)
    size_t bucket_bytes = N_cells * 4 * sizeof(DeviceOutflowBucket);
    e = cudaMalloc(&d_OutflowBuckets, bucket_bytes);
    if (e != cudaSuccess) { std::cerr << "Failed d_OutflowBuckets: " << bucket_bytes << " bytes - " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate audit structures
    e = cudaMalloc(&d_audit_report, sizeof(AuditReport));
    if (e != cudaSuccess) { std::cerr << "Failed d_audit_report: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_weight_in, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_weight_in: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_weight_out, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_weight_out: " << cudaGetErrorString(e) << std::endl; return false; }

    std::cout << "Pipeline state allocation successful" << std::endl;
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
    DevicePsiC& psi_out,  // CRITICAL FIX: Output phase space
    const uint8_t* d_ActiveMask,
    const uint32_t* d_CoarseList,
    int n_coarse,
    const ::DeviceRLUT& dlut,
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
        d_CoarseList,  // CRITICAL FIX: Pass CoarseList (was missing)
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
        state.d_OutflowBuckets,
        psi_out.block_id,  // CRITICAL FIX: Output phase space
        psi_out.value
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
    DevicePsiC& psi_out,  // CRITICAL FIX: Output phase space
    const uint32_t* d_ActiveList,
    int n_active,
    const ::DeviceRLUT& dlut,
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
    // Physics flags: enable all physics for normal pipeline operation
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
        true,   // enable_straggling (full physics)
        true,   // enable_nuclear (full physics)
        true,   // enable_mcs (full physics)
        state.d_EdepC,
        state.d_AbsorbedWeight_cutoff,
        state.d_AbsorbedWeight_nuclear,
        state.d_AbsorbedEnergy_nuclear,
        state.d_BoundaryLoss_weight,
        state.d_BoundaryLoss_energy,
        state.d_OutflowBuckets,
        psi_out.block_id,  // CRITICAL FIX: Output phase space
        psi_out.value
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
    const ::DeviceRLUT& dlut,
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
    // FIX: Need enough iterations for 150 MeV protons to travel ~158mm
    // With coarse_step = 0.3mm, need at least 158/0.3 ≈ 527 iterations
    const int max_iter = 600;
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
            psi_in->block_id,  // CRITICAL FIX: Check for actual weights
            psi_in->value,     // CRITICAL FIX: Check for actual weights
            state.d_CoarseList,
            config.Nx, config.Nz,
            state.d_n_coarse
        );

        cudaDeviceSynchronize();

        // Get counts
        cudaMemcpy(&state.n_active, state.d_n_active, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&state.n_coarse, state.d_n_coarse, sizeof(int), cudaMemcpyDeviceToHost);

        // DEBUG: Print n_active directly
        if (iter <= 5 || state.n_active > 0) {
            std::cout << "  DEBUG n_active=" << state.n_active << ", n_coarse=" << state.n_coarse << std::endl;
        }

        // EXTRA DEBUG: Read d_n_active directly to verify
        int n_active_verify = 0;
        cudaMemcpy(&n_active_verify, state.d_n_active, sizeof(int), cudaMemcpyDeviceToHost);
        if (n_active_verify != state.n_active) {
            std::cout << "  WARNING: n_active mismatch! state=" << state.n_active << ", verify=" << n_active_verify << std::endl;
        }

        std::cout << "  Iteration " << iter << ": "
                  << state.n_active << " active, "
                  << state.n_coarse << " coarse cells" << std::endl;

        // DEBUG: Check weight in psi_in before processing
        if (iter <= 5) {
            std::vector<float> h_psi_in_check(psi_in->N_cells * psi_in->Kb * DEVICE_LOCAL_BINS);
            cudaMemcpy(h_psi_in_check.data(), psi_in->value, psi_in->N_cells * psi_in->Kb * DEVICE_LOCAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);
            double total_weight_in = 0.0;
            int nonzero_count = 0;
            for (size_t i = 0; i < h_psi_in_check.size(); ++i) {
                total_weight_in += h_psi_in_check[i];
                if (h_psi_in_check[i] > 1e-12f) nonzero_count++;
            }
            std::cout << "    psi_in: weight=" << total_weight_in << ", nonzero_bins=" << nonzero_count << std::endl;
        }

        // DEBUG: Check energy accumulation every 10 iterations
        if (iter % 10 == 0) {
            int total_cells = config.Nx * config.Nz;
            std::vector<double> h_EdepC_check(total_cells);
            cudaMemcpy(h_EdepC_check.data(), state.d_EdepC, total_cells * sizeof(double), cudaMemcpyDeviceToHost);
            double total_edep_check = 0.0;
            for (int i = 0; i < total_cells; ++i) {
                total_edep_check += h_EdepC_check[i];
            }
            std::cout << "    Energy so far: " << total_edep_check << " MeV" << std::endl;
        }

        // Check if we're done
        if (state.n_active == 0 && state.n_coarse == 0) {
            std::cout << "  No active cells remaining, transport complete" << std::endl;
            break;
        }

        // CRITICAL FIX: Clear psi_out BEFORE K2/K3 write to it
        // This prevents accumulation of particles from previous iterations
        device_psic_clear(*psi_out);

        // CRITICAL FIX: Clear buckets BEFORE K2/K3 run
        // This ensures buckets only contain transfers from CURRENT iteration
        size_t num_buckets = config.Nx * config.Nz * 4;
        int b_threads = 256;
        int b_blocks = (num_buckets + b_threads - 1) / b_threads;
        clear_buckets_kernel<<<b_blocks, b_threads>>>(state.d_OutflowBuckets, num_buckets);
        cudaDeviceSynchronize();

        // --------------------------------------------------------------------
        // K2: Coarse Transport (high energy cells)
        // --------------------------------------------------------------------
        if (state.n_coarse > 0) {
            std::cout << "    Calling K2 with n_coarse=" << state.n_coarse << std::endl;
            if (!run_k2_coarse_transport(*psi_in, *psi_out, state.d_ActiveMask, state.d_CoarseList,
                                          state.n_coarse, dlut, config, e_grid, a_grid, state)) {
                std::cerr << "K2 failed at iteration " << iter << std::endl;
                return false;
            }
            std::cout << "    K2 complete" << std::endl;

            // DEBUG: Check psi_out after K2
            if (iter <= 3) {
                std::vector<float> h_psi_out_after_k2(psi_out->N_cells * psi_out->Kb * DEVICE_LOCAL_BINS);
                cudaMemcpy(h_psi_out_after_k2.data(), psi_out->value, psi_out->N_cells * psi_out->Kb * DEVICE_LOCAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);
                double total_weight_out_k2 = 0.0;
                int nonzero_count_out_k2 = 0;
                for (size_t i = 0; i < h_psi_out_after_k2.size(); ++i) {
                    total_weight_out_k2 += h_psi_out_after_k2[i];
                    if (h_psi_out_after_k2[i] > 1e-12f) nonzero_count_out_k2++;
                }
                std::cout << "    psi_out AFTER K2 (before K4): weight=" << total_weight_out_k2 << ", nonzero_bins=" << nonzero_count_out_k2 << std::endl;
            }
        }

        // --------------------------------------------------------------------
        // K3: Fine Transport (low energy cells)
        // --------------------------------------------------------------------
        if (state.n_active > 0) {
            if (!run_k3_fine_transport(*psi_in, *psi_out, state.d_ActiveList, state.n_active,
                                       dlut, config, e_grid, a_grid, state)) {
                std::cerr << "K3 failed at iteration " << iter << std::endl;
                return false;
            }
        }

        // --------------------------------------------------------------------
        // K4: Bucket Transfer (boundary crossings)
        // --------------------------------------------------------------------

        // DEBUG: Check psi_out before K4
        if (iter <= 5) {
            std::vector<float> h_psi_out_check(psi_out->N_cells * psi_out->Kb * DEVICE_LOCAL_BINS);
            cudaMemcpy(h_psi_out_check.data(), psi_out->value, psi_out->N_cells * psi_out->Kb * DEVICE_LOCAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);
            double total_weight_out = 0.0;
            int nonzero_count_out = 0;
            for (size_t i = 0; i < h_psi_out_check.size(); ++i) {
                total_weight_out += h_psi_out_check[i];
                if (h_psi_out_check[i] > 1e-12f) nonzero_count_out++;
            }
            std::cout << "    psi_out before K4: weight=" << total_weight_out << ", nonzero_bins=" << nonzero_count_out << std::endl;
        }

        if (!run_k4_bucket_transfer(*psi_out, state.d_OutflowBuckets, config.Nx, config.Nz)) {
            std::cerr << "K4 failed at iteration " << iter << std::endl;
            return false;
        }

        // DEBUG: Check psi_out after K4
        if (iter <= 5) {
            std::vector<float> h_psi_out_check(psi_out->N_cells * psi_out->Kb * DEVICE_LOCAL_BINS);
            cudaMemcpy(h_psi_out_check.data(), psi_out->value, psi_out->N_cells * psi_out->Kb * DEVICE_LOCAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);
            double total_weight_out = 0.0;
            int nonzero_count_out = 0;
            for (size_t i = 0; i < h_psi_out_check.size(); ++i) {
                total_weight_out += h_psi_out_check[i];
                if (h_psi_out_check[i] > 1e-12f) nonzero_count_out++;
            }
            std::cout << "    psi_out AFTER K4: weight=" << total_weight_out << ", nonzero_bins=" << nonzero_count_out << std::endl;
        }

        // Note: Buckets are cleared at START of next iteration (before K2/K3)
        // This is correct because K4 needs to read the buckets AFTER K2/K3 write to them

        // --------------------------------------------------------------------
        // K5: Weight Audit (conservation check)
        // --------------------------------------------------------------------
        // Note: psi_in now contains processed particles (moved from original)
        //       psi_out contains K2+K3+K4 results (cleared at start of iteration)
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

        // Clear psi_out (which will receive fresh bucket transfers in next iteration)
        // After swap, psi_in contains transferred particles to process next iteration
        // psi_out contains the old input which is now done
        device_psic_clear(*psi_out);
    }

    std::cout << "K1-K6 pipeline: completed " << iter << " iterations" << std::endl;
    std::cout << "GPU K1-K6 pipeline complete." << std::endl;

    // ========================================================================
    // Energy Conservation Report
    // ========================================================================
    int total_cells = config.Nx * config.Nz;

    // Energy deposition
    std::vector<double> h_EdepC(total_cells);
    cudaMemcpy(h_EdepC.data(), state.d_EdepC, total_cells * sizeof(double), cudaMemcpyDeviceToHost);
    double total_edep = 0.0;
    for (int i = 0; i < total_cells; ++i) {
        total_edep += h_EdepC[i];
    }

    // Boundary loss energy
    std::vector<double> h_BoundaryLoss_energy(total_cells);
    cudaMemcpy(h_BoundaryLoss_energy.data(), state.d_BoundaryLoss_energy, total_cells * sizeof(double), cudaMemcpyDeviceToHost);
    double total_boundary_loss_energy = 0.0;
    for (int i = 0; i < total_cells; ++i) {
        total_boundary_loss_energy += h_BoundaryLoss_energy[i];
    }

    // Cutoff weight (particles below energy threshold)
    std::vector<float> h_AbsorbedWeight_cutoff(total_cells);
    cudaMemcpy(h_AbsorbedWeight_cutoff.data(), state.d_AbsorbedWeight_cutoff, total_cells * sizeof(float), cudaMemcpyDeviceToHost);
    double total_cutoff_weight = 0.0;
    for (int i = 0; i < total_cells; ++i) {
        total_cutoff_weight += h_AbsorbedWeight_cutoff[i];
    }

    double total_accounted = total_edep + total_boundary_loss_energy;

    std::cout << "\n=== Energy Conservation Report ===" << std::endl;
    std::cout << "  Energy Deposited: " << total_edep << " MeV" << std::endl;
    std::cout << "  Boundary Loss Energy: " << total_boundary_loss_energy << " MeV" << std::endl;
    std::cout << "  Cutoff Weight: " << total_cutoff_weight << " (particles below E < 0.1 MeV)" << std::endl;
    std::cout << "  Total Accounted Energy: " << total_accounted << " MeV" << std::endl;
    std::cout << "=====================================" << std::endl;

    return true;
}

} // namespace sm_2d
