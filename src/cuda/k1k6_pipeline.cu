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
#include "core/local_bins.hpp"  // For decode_local_idx_4d
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <iomanip>

namespace sm_2d {

// ============================================================================
// Optional debug dump: non-zero cell information to CSV
// ============================================================================
#if defined(SM2D_ENABLE_DEBUG_DUMPS)
void dump_nonzero_cells_to_csv(
    const DevicePsiC& psi,
    int Nx, int Nz, float dx, float dz,
    int iteration,
    const char* stage_name,  // "after_K2K3", "after_K4", etc.
    const float* theta_edges,  // Add theta edges for angle calculation
    const float* E_edges,      // Add E edges for energy calculation
    int N_theta, int N_E,
    int N_theta_local, int N_E_local
) {
    int N_cells = Nx * Nz;
    size_t total_values = N_cells * psi.Kb * LOCAL_BINS;

    // Copy all values from device
    std::vector<float> h_all_values(total_values);
    std::vector<uint32_t> h_all_block_ids(N_cells * psi.Kb);

    cudaMemcpy(h_all_values.data(), psi.value, total_values * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_block_ids.data(), psi.block_id, N_cells * psi.Kb * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Collect non-zero cells with detailed energy/angle info
    struct CellInfo {
        int cell;
        int ix;
        int iz;
        float x_center;
        float z_center;
        float total_weight;
        int num_slots;
        float mean_E;        // Mean energy weighted by weight
        float mean_theta;    // Mean theta weighted by weight
        float min_E;         // Minimum energy in cell
        float max_E;         // Maximum energy in cell
        float min_theta;     // Minimum theta in cell
        float max_theta;     // Maximum theta in cell
    };
    std::vector<CellInfo> nonzero_cells;

    // Pre-compute theta and E bin centers for lookup
    std::vector<float> theta_centers(N_theta);
    std::vector<float> E_centers(N_E);
    float theta_min = theta_edges[0];
    float theta_max = theta_edges[N_theta];
    float dtheta = (theta_max - theta_min) / N_theta;
    for (int i = 0; i < N_theta; ++i) {
        theta_centers[i] = theta_min + (i + 0.5f) * dtheta;
    }
    for (int i = 0; i < N_E; ++i) {
        // Use lower edge + 50% of half-width (same as K2/K3)
        float E_lower = E_edges[i];
        float E_upper = E_edges[i + 1];
        float E_half_width = (E_upper - E_lower) * 0.5f;
        E_centers[i] = E_lower + 0.50f * E_half_width;
    }

    for (int cell = 0; cell < N_cells; ++cell) {
        float cell_weight = 0.0f;
        int num_slots = 0;

        // Weighted sums for mean energy and theta
        float sum_E_weighted = 0.0f;
        float sum_theta_weighted = 0.0f;
        float min_E_cell = 1e30f;
        float max_E_cell = -1e30f;
        float min_theta_cell = 1e30f;
        float max_theta_cell = -1e30f;

        for (int slot = 0; slot < psi.Kb; ++slot) {
            uint32_t bid = h_all_block_ids[cell * psi.Kb + slot];
            if (bid == 0xFFFFFFFF) continue;  // EMPTY_BLOCK_ID

            // Decode block ID
            uint32_t b_theta = bid & 0xFFF;
            uint32_t b_E = (bid >> 12) & 0xFFF;

            for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                size_t idx = (cell * psi.Kb + slot) * LOCAL_BINS + lidx;
                float w = h_all_values[idx];
                if (w > 1e-12f) {
                    cell_weight += w;
                    num_slots++;

                    // Decode local index to get theta_local, E_local
                    int theta_local, E_local, x_sub, z_sub;
                    decode_local_idx_4d(lidx, theta_local, E_local, x_sub, z_sub);

                    // Get global bin indices
                    int theta_bin = b_theta * N_theta_local + theta_local;
                    int E_bin = b_E * N_E_local + E_local;

                    // Get theta and E values
                    float theta = theta_centers[theta_bin];
                    float E = E_centers[E_bin];

                    // Accumulate weighted sums
                    sum_E_weighted += E * w;
                    sum_theta_weighted += theta * w;

                    // Track min/max
                    if (E < min_E_cell) min_E_cell = E;
                    if (E > max_E_cell) max_E_cell = E;
                    if (theta < min_theta_cell) min_theta_cell = theta;
                    if (theta > max_theta_cell) max_theta_cell = theta;
                }
            }
        }

        if (cell_weight > 1e-12f) {
            int ix = cell % Nx;
            int iz = cell / Nx;
            float x_center = (ix + 0.5f) * dx;
            float z_center = (iz + 0.5f) * dz;

            float mean_E = sum_E_weighted / cell_weight;
            float mean_theta = sum_theta_weighted / cell_weight;

            nonzero_cells.push_back({
                cell, ix, iz, x_center, z_center, cell_weight, num_slots,
                mean_E, mean_theta, min_E_cell, max_E_cell, min_theta_cell, max_theta_cell
            });
        }
    }

    // Create filename: debug_cells_iter_<N>_<stage>.csv
    char filename[256];
    snprintf(filename, sizeof(filename), "results/debug_cells_iter_%02d_%s.csv", iteration, stage_name);

    // Write CSV
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }

    // Header with new columns
    ofs << "cell,ix,iz,x_mm,z_mm,total_weight,num_slots,mean_E_MeV,mean_theta_rad,min_E_MeV,max_E_MeV,min_theta_rad,max_theta_rad\n";
    ofs << std::fixed << std::setprecision(6);

    for (const auto& info : nonzero_cells) {
        ofs << info.cell << ","
            << info.ix << ","
            << info.iz << ","
            << info.x_center << ","
            << info.z_center << ","
            << info.total_weight << ","
            << info.num_slots << ","
            << info.mean_E << ","
            << info.mean_theta << ","
            << info.min_E << ","
            << info.max_E << ","
            << info.min_theta << ","
            << info.max_theta << "\n";
    }

    ofs.close();

    std::cout << "  DEBUG: Wrote " << nonzero_cells.size() << " non-zero cells to "
              << filename << " (z range: ";
    if (!nonzero_cells.empty()) {
        float z_min = nonzero_cells.front().z_center;
        float z_max = nonzero_cells.back().z_center;
        std::cout << z_min << " - " << z_max << " mm)";
    }
    std::cout << std::endl;
}
#endif

// ============================================================================
// Helper Kernel Implementations
// ============================================================================

// Source particle injection kernel with angular and spatial distribution
// Distributes source across ALL cells within the beam width using proper Gaussian distribution
__global__ void inject_source_kernel(
    DevicePsiC psi,
    int Nx, int Nz, float dx, float dz, float x_min, float z_min,
    float x0, float z0,
    float theta0, float sigma_theta,
    float E0, float W_total,
    float sigma_x,
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local
) {
    // Number of samples for angular distribution (Gauss-Hermite still used for angular)
    const int N_theta_samples = 7;

    // Gauss-Hermite quadrature for angular spread only
    __constant__ static float gh_abscissas[7] = {-2.65f, -1.67f, -0.82f, 0.0f, 0.82f, 1.67f, 2.65f};
    // Normalized GH weights (sum = 1.0 for proper probability distribution)
    __constant__ static float gh_weights_norm[7] = {0.0105f, 0.0769f, 0.2448f, 0.3357f, 0.2448f, 0.0769f, 0.0105f};

    // Determine if we need distributed sampling
    bool use_angular_spread = (sigma_theta > 0.0001f);
    bool use_spatial_spread = (sigma_x > 0.01f);

    // Pencil beam case: single cell injection
    if (!use_angular_spread && !use_spatial_spread) {
        float x_rel = x0 - x_min;
        float z_rel = z0 - z_min;
        int source_cell_x = static_cast<int>(x_rel / dx);
        int source_cell_z = static_cast<int>(z_rel / dz);
        source_cell_x = (source_cell_x < 0) ? 0 : (source_cell_x >= Nx) ? Nx - 1 : source_cell_x;
        source_cell_z = (source_cell_z < 0) ? 0 : (source_cell_z >= Nz) ? Nz - 1 : source_cell_z;
        int source_cell = source_cell_z * Nx + source_cell_x;
        float x_in_cell = x_rel - source_cell_x * dx;
        float z_in_cell = z_rel - source_cell_z * dz;

        device_psic_inject_source(
            psi, source_cell, theta0, E0, W_total,
            x_in_cell, z_in_cell, dx, dz,
            theta_edges, E_edges, N_theta, N_E,
            N_theta_local, N_E_local
        );
        return;
    }

    // =========================================================================
    // CONTINUOUS GAUSSIAN DISTRIBUTION FOR SPATIAL SPREAD
    // =========================================================================
    // Distribute source weight across ALL cells within ±4 sigma_x of beam center
    // Each cell receives weight proportional to Gaussian PDF at cell center

    // Calculate cell range that contains significant Gaussian weight (±4 sigma)
    const float N_sigma = 4.0f;  // Cover 99.99% of Gaussian distribution
    float x_min_beam = x0 - N_sigma * sigma_x;
    float x_max_beam = x0 + N_sigma * sigma_x;

    // Convert to cell indices
    int ix_start = static_cast<int>((x_min_beam - x_min) / dx);
    int ix_end = static_cast<int>((x_max_beam - x_min) / dx) + 1;

    // Clamp to valid grid range
    ix_start = (ix_start < 0) ? 0 : ix_start;
    ix_end = (ix_end > Nx) ? Nx : ix_end;

    // Angular spread configuration
    int theta_start = (use_angular_spread) ? 0 : 0;
    int theta_end = (use_angular_spread) ? N_theta_samples : 1;

    // First pass: compute total spatial weight for normalization
    // (This ensures weight sums to exactly 1.0 regardless of grid coverage)
    float total_spatial_weight = 0.0f;
    for (int ix = ix_start; ix < ix_end; ++ix) {
        float x_center = x_min + (ix + 0.5f) * dx;  // Cell center position
        float dx_from_beam = x_center - x0;
        float gaussian = expf(-0.5f * dx_from_beam * dx_from_beam / (sigma_x * sigma_x));
        total_spatial_weight += gaussian * dx;  // Integral approximation
    }

    // Avoid division by zero
    if (total_spatial_weight <= 0.0f) total_spatial_weight = 1.0f;

    // Angular distribution loop (GH quadrature for angular)
    for (int it = theta_start; it < theta_end; ++it) {
        float theta = theta0;
        float w_theta = 1.0f;
        if (use_angular_spread) {
            theta = theta0 + gh_abscissas[it] * sigma_theta;
            w_theta = gh_weights_norm[it];  // Use normalized weights
        }

        // Spatial distribution: loop over ALL cells in beam width
        for (int ix = ix_start; ix < ix_end; ++ix) {
            // Cell center in global coordinates
            float x_center = x_min + (ix + 0.5f) * dx;
            float dx_from_beam = x_center - x0;

            // Gaussian weight for this cell
            float gaussian = expf(-0.5f * dx_from_beam * dx_from_beam / (sigma_x * sigma_x));

            // Normalize weight so sum over all cells = 1.0
            float w_x = (gaussian * dx) / total_spatial_weight;

            // Z coordinate (always at z0 for source plane)
            float z_rel = z0 - z_min;
            int cell_z = static_cast<int>(z_rel / dz);
            cell_z = (cell_z < 0) ? 0 : (cell_z >= Nz) ? Nz - 1 : cell_z;

            int cell = cell_z * Nx + ix;

            // Position within cell (use cell center for uniform distribution)
            float x_in_cell = dx * 0.5f;  // Center of cell in x
            float z_in_cell = z_rel - cell_z * dz;
            z_in_cell = fmaxf(0.0f, fminf(z_in_cell, dz));

            // Combined weight
            float w_sample = W_total * w_theta * w_x;

            // Inject this sample
            device_psic_inject_source(
                psi, cell, theta, E0, w_sample,
                x_in_cell, z_in_cell, dx, dz,
                theta_edges, E_edges, N_theta, N_E,
                N_theta_local, N_E_local
            );
        }
    }
}

// ============================================================================
// Gaussian Beam Source Injection
// ============================================================================

// Box-Muller transform for Gaussian sampling (matches device_sample_gaussian)
__device__ inline float gaussian_sample(unsigned& seed) {
    // Simple linear congruential generator
    seed = 1664525u * seed + 1013904223u;

    float u1 = (seed >> 16) / 65536.0f;  // [0, 1)
    seed = 1664525u * seed + 1013904223u;

    float u2 = (seed >> 16) / 65536.0f;  // [0, 1)

    // Box-Muller transform
    float r = sqrtf(-2.0f * logf(fmaxf(u1, 1e-10f)));
    float theta = 2.0f * M_PI * u2;

    return r * cosf(theta);  // Return one of two independent normals
}

// Inject Gaussian beam source into PsiC
// Each thread processes one sample from the Gaussian distribution
__global__ void inject_gaussian_source_kernel(
    DevicePsiC psi,
    float x0, float z0,              // Central beam position [mm]
    float theta0, float E0, float W_total,  // Central angle, energy, total weight
    float sigma_x, float sigma_theta, float sigma_E,  // Gaussian spreads
    int n_samples,                    // Number of samples to draw
    unsigned int random_seed,         // Seed for RNG
    float x_min, float z_min,         // Grid origin
    float dx, float dz,               // Cell size
    int Nx, int Nz,                   // Grid dimensions
    float* d_injected_weight,
    float* d_out_of_grid_weight,
    float* d_slot_dropped_weight,
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local
) {
    // Each thread processes one sample
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= n_samples) return;

    // Initialize RNG with unique seed per thread
    unsigned seed = random_seed + sample_idx;

    // Sample from Gaussian distributions using Box-Muller
    float x_sample = x0 + sigma_x * gaussian_sample(seed);
    float theta_sample = theta0 + sigma_theta * gaussian_sample(seed);
    float E_sample = E0 + sigma_E * gaussian_sample(seed);

    // Clamp energy to valid range
    float E_min_grid = E_edges[0];
    float E_max_grid = E_edges[N_E];
    E_sample = fmaxf(E_min_grid, fminf(E_sample, E_max_grid));

    // Clamp angle to valid range
    float theta_min_grid = theta_edges[0];
    float theta_max_grid = theta_edges[N_theta];
    theta_sample = fmaxf(theta_min_grid, fminf(theta_sample, theta_max_grid));

    // Calculate grid cell for this sample
    float x_rel = x_sample - x_min;
    float z_rel = z0 - z_min;  // All samples at z=0 (incident plane)

    int ix = static_cast<int>(x_rel / dx);
    int iz = static_cast<int>(z_rel / dz);

    float w_sample = W_total / n_samples;

    // Skip if outside grid bounds
    if (ix < 0 || ix >= Nx || iz < 0 || iz >= Nz) {
        if (d_out_of_grid_weight != nullptr) {
            atomicAdd(d_out_of_grid_weight, w_sample);
        }
        return;
    }

    int cell = iz * Nx + ix;

    // Calculate position within cell
    float x_in_cell = x_rel - ix * dx;          // [0, dx)
    float z_in_cell = z_rel - iz * dz;          // [0, dz)

    // Convert to centered coordinates for device_psic_inject_source
    // device_psic_inject_source expects: x, z in [0, dx] and [0, dz]
    // and internally converts to offsets from center

    // Inject this sample into the phase space
    bool injected = device_psic_inject_source(
        psi,
        cell,
        theta_sample, E_sample, w_sample,
        x_in_cell, z_in_cell,
        dx, dz,
        theta_edges, E_edges,
        N_theta, N_E,
        N_theta_local, N_E_local
    );

    if (injected) {
        if (d_injected_weight != nullptr) {
            atomicAdd(d_injected_weight, w_sample);
        }
    } else if (d_slot_dropped_weight != nullptr) {
        atomicAdd(d_slot_dropped_weight, w_sample);
    }
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
    if (is_active) {
        uint32_t idx = atomicAdd(d_n_active, 1);
        ActiveList[idx] = cell;
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
    e = cudaMalloc(&d_prev_AbsorbedWeight_cutoff, N_cells * sizeof(float));
    if (e != cudaSuccess) { std::cerr << "Failed d_prev_AbsorbedWeight_cutoff: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_prev_AbsorbedWeight_nuclear, N_cells * sizeof(float));
    if (e != cudaSuccess) { std::cerr << "Failed d_prev_AbsorbedWeight_nuclear: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_prev_BoundaryLoss_weight, N_cells * sizeof(float));
    if (e != cudaSuccess) { std::cerr << "Failed d_prev_BoundaryLoss_weight: " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate outflow buckets (4 faces per cell)
    size_t bucket_bytes = N_cells * 4 * sizeof(DeviceOutflowBucket);
    e = cudaMalloc(&d_OutflowBuckets, bucket_bytes);
    if (e != cudaSuccess) { std::cerr << "Failed d_OutflowBuckets: " << bucket_bytes << " bytes - " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate audit structures
    e = cudaMalloc(&d_audit_report, N_cells * sizeof(AuditReport));
    if (e != cudaSuccess) { std::cerr << "Failed d_audit_report: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_weight_in, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_weight_in: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_weight_out, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_weight_out: " << cudaGetErrorString(e) << std::endl; return false; }

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
    if (d_prev_AbsorbedWeight_cutoff) cudaFree(d_prev_AbsorbedWeight_cutoff);
    if (d_prev_AbsorbedWeight_nuclear) cudaFree(d_prev_AbsorbedWeight_nuclear);
    if (d_prev_BoundaryLoss_weight) cudaFree(d_prev_BoundaryLoss_weight);
    if (d_OutflowBuckets) cudaFree(d_OutflowBuckets);
    if (d_theta_edges) cudaFree(d_theta_edges);
    if (d_E_edges) cudaFree(d_E_edges);
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
    d_prev_AbsorbedWeight_cutoff = nullptr;
    d_prev_AbsorbedWeight_nuclear = nullptr;
    d_prev_BoundaryLoss_weight = nullptr;
    d_OutflowBuckets = nullptr;
    d_theta_edges = nullptr;
    d_E_edges = nullptr;
    d_audit_report = nullptr;
    d_weight_in = nullptr;
    d_weight_out = nullptr;

    owns_device_memory = false;
}

bool init_pipeline_state(
    const K1K6PipelineConfig& config,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    K1K6PipelineState& state
) {
    if (!state.allocate(config.Nx, config.Nz)) {
        return false;
    }

    cudaError_t e = cudaMalloc(&state.d_theta_edges, (config.N_theta + 1) * sizeof(float));
    if (e != cudaSuccess) {
        std::cerr << "Failed d_theta_edges: " << cudaGetErrorString(e) << std::endl;
        state.cleanup();
        return false;
    }

    e = cudaMalloc(&state.d_E_edges, (config.N_E + 1) * sizeof(float));
    if (e != cudaSuccess) {
        std::cerr << "Failed d_E_edges: " << cudaGetErrorString(e) << std::endl;
        state.cleanup();
        return false;
    }

    e = cudaMemcpy(
        state.d_theta_edges,
        a_grid.edges.data(),
        (config.N_theta + 1) * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (e != cudaSuccess) {
        std::cerr << "Failed copy d_theta_edges: " << cudaGetErrorString(e) << std::endl;
        state.cleanup();
        return false;
    }

    e = cudaMemcpy(
        state.d_E_edges,
        e_grid.edges.data(),
        (config.N_E + 1) * sizeof(float),
        cudaMemcpyHostToDevice
    );
    if (e != cudaSuccess) {
        std::cerr << "Failed copy d_E_edges: " << cudaGetErrorString(e) << std::endl;
        state.cleanup();
        return false;
    }

    return true;
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
    cudaMemset(state.d_prev_AbsorbedWeight_cutoff, 0, N_cells * sizeof(float));
    cudaMemset(state.d_prev_AbsorbedWeight_nuclear, 0, N_cells * sizeof(float));
    cudaMemset(state.d_prev_BoundaryLoss_weight, 0, N_cells * sizeof(float));

    // Initialize buckets to empty
    cudaMemset(state.d_OutflowBuckets, 0xFF, N_cells * 4 * sizeof(uint32_t) * DEVICE_Kb_out);

    int zero = 0;
    cudaMemcpy(state.d_n_active, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_n_coarse, &zero, sizeof(int), cudaMemcpyHostToDevice);
}

AuditReport get_audit_report(const K1K6PipelineState& state, int Nx, int Nz) {
    (void)Nx;
    (void)Nz;
    AuditReport summary{};
    cudaMemcpy(&summary, state.d_audit_report, sizeof(AuditReport), cudaMemcpyDeviceToHost);
    return summary;
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
    K1K6PipelineState& state
) {
    if (n_coarse == 0) return true;

    K2Config k2_cfg;
    k2_cfg.E_coarse_max = config.E_coarse_max;
    k2_cfg.step_coarse = config.step_coarse;
    k2_cfg.n_steps_per_cell = config.n_steps_per_cell;
    k2_cfg.sigma_x_initial = config.sigma_x_initial;  // FIX C: Pass beam width

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
        state.d_theta_edges,
        state.d_E_edges,
        config.N_theta, config.N_E,
        config.N_theta_local, config.N_E_local,
        k2_cfg,
        config.sigma_x_initial,  // FIX C: Pass initial beam width
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
        return false;
    }

    cudaDeviceSynchronize();

    return true;
}

bool run_k3_fine_transport(
    const DevicePsiC& psi_in,
    DevicePsiC& psi_out,  // CRITICAL FIX: Output phase space
    const uint32_t* d_ActiveList,
    int n_active,
    const ::DeviceRLUT& dlut,
    const K1K6PipelineConfig& config,
    K1K6PipelineState& state
) {
    if (n_active == 0) return true;

    int threads = 256;
    int blocks = (n_active + threads - 1) / threads;

    // Call K3_FineTransport kernel
    // Physics flags: enable all physics for normal pipeline operation
    // NOTE: Lateral spreading is ALWAYS enabled (deterministic, not Monte Carlo)
    K3_FineTransport<<<blocks, threads>>>(
        psi_in.block_id,
        psi_in.value,
        d_ActiveList,
        config.Nx, config.Nz, config.dx, config.dz,
        n_active,
        dlut,
        state.d_theta_edges,
        state.d_E_edges,
        config.N_theta, config.N_E,
        config.N_theta_local, config.N_E_local,
        true,   // enable_straggling (full physics)
        true,   // enable_nuclear (full physics)
        config.sigma_x_initial,  // FIX C: Pass initial beam width
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
        return false;
    }

    cudaDeviceSynchronize();

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
    const float* d_BoundaryLoss_weight,
    const float* d_prev_AbsorbedWeight_cutoff,
    const float* d_prev_AbsorbedWeight_nuclear,
    const float* d_prev_BoundaryLoss_weight,
    AuditReport* d_report,
    int Nx, int Nz
) {
    int N_cells = Nx * Nz;
    int threads = 256;
    int blocks = (N_cells + threads - 1) / threads;

    cudaMemset(d_report, 0, N_cells * sizeof(AuditReport));

    // Call K5_WeightAudit kernel
    K5_WeightAudit<<<blocks, threads>>>(
        psi_in.block_id,
        psi_in.value,
        psi_out.block_id,
        psi_out.value,
        d_AbsorbedWeight_cutoff,
        d_AbsorbedWeight_nuclear,
        d_BoundaryLoss_weight,
        d_prev_AbsorbedWeight_cutoff,
        d_prev_AbsorbedWeight_nuclear,
        d_prev_BoundaryLoss_weight,
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
    // Compute b_E_trigger from E_trigger
    const int b_E_trigger = compute_b_E_trigger(config.E_trigger, e_grid, config.N_E_local);
    const bool summary_logging = (config.log_level >= 1);
    const bool verbose_logging = (config.log_level >= 2);
    const bool debug_dumps_enabled =
    #if defined(SM2D_ENABLE_DEBUG_DUMPS)
        true;
    #else
        false;
    #endif

    #if defined(SM2D_ENABLE_DEBUG_DUMPS)
    // DEBUG: Print b_E_trigger calculation
    std::cout << "=== E_trigger Configuration ===" << std::endl;
    std::cout << "E_trigger = " << config.E_trigger << " MeV" << std::endl;
    std::cout << "E_bin for E_trigger = " << e_grid.FindBin(config.E_trigger) << std::endl;
    std::cout << "N_E_local = " << config.N_E_local << std::endl;
    std::cout << "b_E_trigger = " << b_E_trigger << std::endl;

    // Sample b_E for 150 MeV
    int E_bin_150 = e_grid.FindBin(150.0f);
    int b_E_150 = E_bin_150 / config.N_E_local;
    std::cout << "=== 150 MeV Particle ===" << std::endl;
    std::cout << "E_bin_150 = " << E_bin_150 << std::endl;
    std::cout << "b_E_150 = " << b_E_150 << std::endl;
    std::cout << "Condition: b_E_150 < b_E_trigger → " << b_E_150 << " < " << b_E_trigger << " = " << (b_E_150 < b_E_trigger ? "TRUE (K3 active)" : "FALSE (K2 coarse)") << std::endl;
    std::cout << "==============================" << std::endl;
    #endif

    if (config.max_iterations <= 0) {
        std::cerr << "Invalid max_iterations: " << config.max_iterations << std::endl;
        return false;
    }

    // Reset pipeline state
    reset_pipeline_state(state, config.Nx, config.Nz);

    // Clear output buffer
    device_psic_clear(*psi_out);

    // Maximum iterations
    const int max_iter = config.max_iterations;

    int iter = 0;

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

        // Progress update every 50 iterations
        if (summary_logging && (iter == 1 || iter % 50 == 0)) {
            std::cout << "  Iteration " << iter << ": "
                      << state.n_active << " active, "
                      << state.n_coarse << " coarse cells" << std::endl;
        }

        // Verbose mode: print detailed early-iteration stats
        if (verbose_logging && iter <= 10) {
            std::cout << "  Iteration " << iter << ": "
                      << state.n_active << " active, "
                      << state.n_coarse << " coarse cells" << std::endl;
        }

        // Check if we're done
        if (state.n_active == 0 && state.n_coarse == 0) {
            if (summary_logging) {
                std::cout << "  Transport complete after " << iter << " iterations" << std::endl;
            }
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
            if (!run_k2_coarse_transport(*psi_in, *psi_out, state.d_ActiveMask, state.d_CoarseList,
                                          state.n_coarse, dlut, config, state)) {
                std::cerr << "K2 failed at iteration " << iter << std::endl;
                return false;
            }
        }

        // --------------------------------------------------------------------
        // K3: Fine Transport (low energy cells)
        // --------------------------------------------------------------------
        if (state.n_active > 0) {
            if (!run_k3_fine_transport(*psi_in, *psi_out, state.d_ActiveList, state.n_active,
                                       dlut, config, state)) {
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

        // Note: Buckets are cleared at START of next iteration (before K2/K3)
        // This is correct because K4 needs to read the buckets AFTER K2/K3 write to them

        // --------------------------------------------------------------------
        // DEBUG: Dump non-zero cells after K4 for selected iterations
        // --------------------------------------------------------------------
        #if defined(SM2D_ENABLE_DEBUG_DUMPS)
        if (iter <= 10 || iter == 50 || iter == 100 || iter == 150 || iter == 200 || iter == 250) {
            dump_nonzero_cells_to_csv(*psi_out, config.Nx, config.Nz, config.dx, config.dz,
                                       iter, "after_K4",
                                       a_grid.edges.data(), e_grid.edges.data(),
                                       config.N_theta, config.N_E,
                                       config.N_theta_local, config.N_E_local);
        }
        #endif

        // --------------------------------------------------------------------
        // K5: Weight Audit (conservation check)
        // --------------------------------------------------------------------
        // Note: psi_in now contains processed particles (moved from original)
        //       psi_out contains K2+K3+K4 results (cleared at start of iteration)
        if (!run_k5_weight_audit(*psi_in, *psi_out,
                                 state.d_AbsorbedWeight_cutoff,
                                 state.d_AbsorbedWeight_nuclear,
                                 state.d_BoundaryLoss_weight,
                                 state.d_prev_AbsorbedWeight_cutoff,
                                 state.d_prev_AbsorbedWeight_nuclear,
                                 state.d_prev_BoundaryLoss_weight,
                                 state.d_audit_report,
                                 config.Nx, config.Nz)) {
            std::cerr << "K5 failed at iteration " << iter << std::endl;
            return false;
        }

        if (verbose_logging) {
            AuditReport report = get_audit_report(state, config.Nx, config.Nz);
            std::cout << "  Weight audit: error=" << report.W_error
                      << " pass=" << (report.W_pass ? "yes" : "no") << std::endl;
        }

        // Update previous cumulative tallies for next iteration's delta-based K5.
        int N_cells = config.Nx * config.Nz;
        cudaMemcpy(state.d_prev_AbsorbedWeight_cutoff, state.d_AbsorbedWeight_cutoff,
                   N_cells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(state.d_prev_AbsorbedWeight_nuclear, state.d_AbsorbedWeight_nuclear,
                   N_cells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(state.d_prev_BoundaryLoss_weight, state.d_BoundaryLoss_weight,
                   N_cells * sizeof(float), cudaMemcpyDeviceToDevice);

        // --------------------------------------------------------------------
        // K6: Swap Buffers
        // --------------------------------------------------------------------
        run_k6_swap_buffers(psi_in, psi_out);

        // Clear psi_out (which will receive fresh bucket transfers in next iteration)
        // After swap, psi_in contains transferred particles to process next iteration
        // psi_out contains the old input which is now done
        device_psic_clear(*psi_out);
    }

    if (summary_logging) {
        std::cout << "K1-K6 pipeline: completed " << iter << " iterations" << std::endl;
    }

    // ========================================================================
    // Energy Conservation Report
    // ========================================================================
    if (!(verbose_logging || debug_dumps_enabled)) {
        return true;
    }

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

    // Nuclear energy (from inelastic nuclear interactions)
    std::vector<double> h_AbsorbedEnergy_nuclear(total_cells);
    cudaMemcpy(h_AbsorbedEnergy_nuclear.data(), state.d_AbsorbedEnergy_nuclear, total_cells * sizeof(double), cudaMemcpyDeviceToHost);
    double total_nuclear_energy = 0.0;
    for (int i = 0; i < total_cells; ++i) {
        total_nuclear_energy += h_AbsorbedEnergy_nuclear[i];
    }

    double total_accounted = total_edep + total_boundary_loss_energy + total_nuclear_energy;

    std::cout << "\n=== Energy Conservation Report ===" << std::endl;
    std::cout << "  Energy Deposited: " << total_edep << " MeV" << std::endl;
    std::cout << "  Nuclear Energy Deposited: " << total_nuclear_energy << " MeV" << std::endl;
    std::cout << "  Boundary Loss Energy: " << total_boundary_loss_energy << " MeV" << std::endl;
    std::cout << "  Cutoff Weight: " << total_cutoff_weight << " (particles below E < 0.1 MeV)" << std::endl;
    std::cout << "  Total Accounted Energy: " << total_accounted << " MeV" << std::endl;
    std::cout << "=====================================" << std::endl;

    return true;
}

} // namespace sm_2d
