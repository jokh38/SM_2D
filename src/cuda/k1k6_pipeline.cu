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
#include <algorithm>
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
        // Use bin center (same representative energy convention as K2/K3/K5).
        float E_lower = E_edges[i];
        float E_upper = E_edges[i + 1];
        float E_half_width = (E_upper - E_lower) * 0.5f;
        E_centers[i] = E_lower + 1.00f * E_half_width;
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
    double* d_injected_energy,
    double* d_out_of_grid_energy,
    double* d_slot_dropped_energy,
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
        if (d_out_of_grid_energy != nullptr) {
            atomicAdd(d_out_of_grid_energy, static_cast<double>(E_sample * w_sample));
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
        if (d_injected_energy != nullptr) {
            atomicAdd(d_injected_energy, static_cast<double>(E_sample * w_sample));
        }
    } else {
        if (d_slot_dropped_weight != nullptr) {
            atomicAdd(d_slot_dropped_weight, w_sample);
        }
        if (d_slot_dropped_energy != nullptr) {
            atomicAdd(d_slot_dropped_energy, static_cast<double>(E_sample * w_sample));
        }
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
    bucket.moment_A = 0.0f;
    bucket.moment_B = 0.0f;
    bucket.moment_C = 0.0f;
}

// Build compact source-cell -> bucket-base map for a single batch.
// Each source cell in CellList maps to base index (idx * 4), others stay -1.
__global__ void build_cell_to_bucket_base_kernel(
    const uint32_t* __restrict__ CellList,
    int n_cells,
    int* __restrict__ CellToBucketBase
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;

    int cell = static_cast<int>(CellList[idx]);
    CellToBucketBase[cell] = idx * 4;
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
            if (w > 0.0f) {
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

bool compute_psic_total_weight_energy(
    const DevicePsiC& psi,
    const float* d_E_edges,
    int N_E,
    int N_theta_local,
    int N_E_local,
    double& total_weight,
    double& total_energy
) {
    size_t total_slots = static_cast<size_t>(psi.N_cells) * psi.Kb;
    size_t total_values = total_slots * DEVICE_LOCAL_BINS;
    std::vector<uint32_t> h_block_ids(total_slots, DEVICE_EMPTY_BLOCK_ID);
    std::vector<float> h_values(total_values, 0.0f);
    std::vector<float> h_E_edges(static_cast<size_t>(N_E + 1), 0.0f);

    if (!device_psic_copy_to_host(psi, h_block_ids.data(), h_values.data())) {
        return false;
    }

    cudaError_t edge_copy_err = cudaMemcpy(
        h_E_edges.data(),
        d_E_edges,
        static_cast<size_t>(N_E + 1) * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    if (edge_copy_err != cudaSuccess) {
        return false;
    }

    total_weight = 0.0;
    total_energy = 0.0;

    for (int cell = 0; cell < psi.N_cells; ++cell) {
        for (int slot = 0; slot < psi.Kb; ++slot) {
            uint32_t bid = h_block_ids[static_cast<size_t>(cell) * psi.Kb + slot];
            if (bid == DEVICE_EMPTY_BLOCK_ID) {
                continue;
            }

            int b_E = static_cast<int>((bid >> 12) & 0xFFF);
            for (int lidx = 0; lidx < DEVICE_LOCAL_BINS; ++lidx) {
                size_t value_idx =
                    (static_cast<size_t>(cell) * psi.Kb + slot) * DEVICE_LOCAL_BINS + lidx;
                float w = h_values[value_idx];
                if (w <= 0.0f) {
                    continue;
                }

                int E_local = (lidx / N_theta_local) % N_E_local;
                int E_bin = b_E * N_E_local + E_local;
                E_bin = std::max(0, std::min(E_bin, N_E - 1));
                float E_lower = h_E_edges[E_bin];
                float E_upper = h_E_edges[E_bin + 1];
                float E_center = E_lower + 0.5f * (E_upper - E_lower);

                total_weight += static_cast<double>(w);
                total_energy += static_cast<double>(w) * static_cast<double>(E_center);
            }
        }
    }

    return true;
}

bool ensure_bucket_scratch_capacity(
    K1K6PipelineState& state,
    int required_cells
) {
    if (required_cells <= 0) {
        return true;
    }
    if (state.bucket_scratch_capacity_cells >= required_cells &&
        state.d_BucketScratch != nullptr) {
        return true;
    }

    if (state.d_BucketScratch != nullptr) {
        cudaFree(state.d_BucketScratch);
        state.d_BucketScratch = nullptr;
        state.bucket_scratch_capacity_cells = 0;
    }

    size_t bucket_bytes = static_cast<size_t>(required_cells) * 4 * sizeof(DeviceOutflowBucket);
    cudaError_t e = cudaMalloc(&state.d_BucketScratch, bucket_bytes);
    if (e != cudaSuccess) {
        std::cerr << "Failed d_BucketScratch: " << bucket_bytes
                  << " bytes - " << cudaGetErrorString(e) << std::endl;
        return false;
    }

    state.bucket_scratch_capacity_cells = required_cells;
    return true;
}

bool prepare_bucket_scratch_for_batch(
    K1K6PipelineState& state,
    const uint32_t* d_CellList,
    int n_cells,
    int N_cells
) {
    if (n_cells <= 0) {
        return true;
    }
    if (!ensure_bucket_scratch_capacity(state, n_cells)) {
        return false;
    }

    cudaError_t e = cudaMemset(state.d_CellToBucketBase, 0xFF, N_cells * sizeof(int));
    if (e != cudaSuccess) {
        std::cerr << "Failed memset d_CellToBucketBase: "
                  << cudaGetErrorString(e) << std::endl;
        return false;
    }

    int threads = 256;
    int blocks = (n_cells + threads - 1) / threads;
    build_cell_to_bucket_base_kernel<<<blocks, threads>>>(
        d_CellList,
        n_cells,
        state.d_CellToBucketBase
    );
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cerr << "build_cell_to_bucket_base_kernel launch failed: "
                  << cudaGetErrorString(e) << std::endl;
        return false;
    }

    int n_buckets = n_cells * 4;
    int b_threads = 256;
    int b_blocks = (n_buckets + b_threads - 1) / b_threads;
    clear_buckets_kernel<<<b_blocks, b_threads>>>(state.d_BucketScratch, n_buckets);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cerr << "clear_buckets_kernel launch failed: "
                  << cudaGetErrorString(e) << std::endl;
        return false;
    }

    cudaDeviceSynchronize();
    return true;
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
    e = cudaMalloc(&d_ActiveMask_prev, N_cells * sizeof(uint8_t));
    if (e != cudaSuccess) { std::cerr << "Failed d_ActiveMask_prev: " << N_cells << " bytes - " << cudaGetErrorString(e) << std::endl; return false; }

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
    e = cudaMalloc(&d_AbsorbedEnergy_cutoff, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_AbsorbedEnergy_cutoff: " << cudaGetErrorString(e) << std::endl; return false; }
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
    e = cudaMalloc(&d_prev_AbsorbedEnergy_cutoff, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_prev_AbsorbedEnergy_cutoff: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_prev_AbsorbedWeight_nuclear, N_cells * sizeof(float));
    if (e != cudaSuccess) { std::cerr << "Failed d_prev_AbsorbedWeight_nuclear: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_prev_BoundaryLoss_weight, N_cells * sizeof(float));
    if (e != cudaSuccess) { std::cerr << "Failed d_prev_BoundaryLoss_weight: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_prev_EdepC, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_prev_EdepC: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_prev_AbsorbedEnergy_nuclear, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_prev_AbsorbedEnergy_nuclear: " << cudaGetErrorString(e) << std::endl; return false; }
    e = cudaMalloc(&d_prev_BoundaryLoss_energy, N_cells * sizeof(double));
    if (e != cudaSuccess) { std::cerr << "Failed d_prev_BoundaryLoss_energy: " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate compact bucket-base map (batch-local buckets are allocated on-demand).
    e = cudaMalloc(&d_CellToBucketBase, N_cells * sizeof(int));
    if (e != cudaSuccess) { std::cerr << "Failed d_CellToBucketBase: " << cudaGetErrorString(e) << std::endl; return false; }

    // Allocate audit structures
    e = cudaMalloc(&d_audit_report, sizeof(AuditReport));
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
    if (d_ActiveMask_prev) cudaFree(d_ActiveMask_prev);
    if (d_ActiveList) cudaFree(d_ActiveList);
    if (d_CoarseList) cudaFree(d_CoarseList);
    if (d_n_active) cudaFree(d_n_active);
    if (d_n_coarse) cudaFree(d_n_coarse);
    if (d_EdepC) cudaFree(d_EdepC);
    if (d_AbsorbedWeight_cutoff) cudaFree(d_AbsorbedWeight_cutoff);
    if (d_AbsorbedEnergy_cutoff) cudaFree(d_AbsorbedEnergy_cutoff);
    if (d_AbsorbedWeight_nuclear) cudaFree(d_AbsorbedWeight_nuclear);
    if (d_AbsorbedEnergy_nuclear) cudaFree(d_AbsorbedEnergy_nuclear);
    if (d_BoundaryLoss_weight) cudaFree(d_BoundaryLoss_weight);
    if (d_BoundaryLoss_energy) cudaFree(d_BoundaryLoss_energy);
    if (d_prev_AbsorbedWeight_cutoff) cudaFree(d_prev_AbsorbedWeight_cutoff);
    if (d_prev_AbsorbedEnergy_cutoff) cudaFree(d_prev_AbsorbedEnergy_cutoff);
    if (d_prev_AbsorbedWeight_nuclear) cudaFree(d_prev_AbsorbedWeight_nuclear);
    if (d_prev_BoundaryLoss_weight) cudaFree(d_prev_BoundaryLoss_weight);
    if (d_prev_EdepC) cudaFree(d_prev_EdepC);
    if (d_prev_AbsorbedEnergy_nuclear) cudaFree(d_prev_AbsorbedEnergy_nuclear);
    if (d_prev_BoundaryLoss_energy) cudaFree(d_prev_BoundaryLoss_energy);
    if (d_BucketScratch) cudaFree(d_BucketScratch);
    if (d_CellToBucketBase) cudaFree(d_CellToBucketBase);
    if (d_theta_edges) cudaFree(d_theta_edges);
    if (d_E_edges) cudaFree(d_E_edges);
    if (d_audit_report) cudaFree(d_audit_report);
    if (d_weight_in) cudaFree(d_weight_in);
    if (d_weight_out) cudaFree(d_weight_out);

    d_ActiveMask = nullptr;
    d_ActiveMask_prev = nullptr;
    d_ActiveList = nullptr;
    d_CoarseList = nullptr;
    d_n_active = nullptr;
    d_n_coarse = nullptr;
    d_EdepC = nullptr;
    d_AbsorbedWeight_cutoff = nullptr;
    d_AbsorbedEnergy_cutoff = nullptr;
    d_AbsorbedWeight_nuclear = nullptr;
    d_AbsorbedEnergy_nuclear = nullptr;
    d_BoundaryLoss_weight = nullptr;
    d_BoundaryLoss_energy = nullptr;
    d_prev_AbsorbedWeight_cutoff = nullptr;
    d_prev_AbsorbedEnergy_cutoff = nullptr;
    d_prev_AbsorbedWeight_nuclear = nullptr;
    d_prev_BoundaryLoss_weight = nullptr;
    d_prev_EdepC = nullptr;
    d_prev_AbsorbedEnergy_nuclear = nullptr;
    d_prev_BoundaryLoss_energy = nullptr;
    d_BucketScratch = nullptr;
    bucket_scratch_capacity_cells = 0;
    d_CellToBucketBase = nullptr;
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
    cudaMemset(state.d_ActiveMask_prev, 0, N_cells * sizeof(uint8_t));
    cudaMemset(state.d_EdepC, 0, N_cells * sizeof(double));
    cudaMemset(state.d_AbsorbedWeight_cutoff, 0, N_cells * sizeof(float));
    cudaMemset(state.d_AbsorbedEnergy_cutoff, 0, N_cells * sizeof(double));
    cudaMemset(state.d_AbsorbedWeight_nuclear, 0, N_cells * sizeof(float));
    cudaMemset(state.d_AbsorbedEnergy_nuclear, 0, N_cells * sizeof(double));
    cudaMemset(state.d_BoundaryLoss_weight, 0, N_cells * sizeof(float));
    cudaMemset(state.d_BoundaryLoss_energy, 0, N_cells * sizeof(double));
    cudaMemset(state.d_prev_AbsorbedWeight_cutoff, 0, N_cells * sizeof(float));
    cudaMemset(state.d_prev_AbsorbedEnergy_cutoff, 0, N_cells * sizeof(double));
    cudaMemset(state.d_prev_AbsorbedWeight_nuclear, 0, N_cells * sizeof(float));
    cudaMemset(state.d_prev_BoundaryLoss_weight, 0, N_cells * sizeof(float));
    cudaMemset(state.d_prev_EdepC, 0, N_cells * sizeof(double));
    cudaMemset(state.d_prev_AbsorbedEnergy_nuclear, 0, N_cells * sizeof(double));
    cudaMemset(state.d_prev_BoundaryLoss_energy, 0, N_cells * sizeof(double));

    // Cell-to-bucket map is rebuilt per batch.
    cudaMemset(state.d_CellToBucketBase, 0xFF, N_cells * sizeof(int));

    int zero = 0;
    cudaMemcpy(state.d_n_active, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_n_coarse, &zero, sizeof(int), cudaMemcpyHostToDevice);
    state.transport_dropped_weight = 0.0;
    state.transport_dropped_energy = 0.0;
    state.transport_audit_residual_energy = 0.0;

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
    const uint8_t* d_ActiveMaskPrev,
    int Nx, int Nz,
    int b_E_fine_on,
    int b_E_fine_off,
    float weight_active_min
) {
    int threads = 256;
    int blocks = (Nx * Nz + threads - 1) / threads;

    // Call K1_ActiveMask kernel
    K1_ActiveMask<<<blocks, threads>>>(
        psi_in.block_id,      // Block IDs from DevicePsiC
        psi_in.value,         // Values from DevicePsiC
        d_ActiveMask,
        d_ActiveMaskPrev,
        Nx, Nz,
        b_E_fine_on,
        b_E_fine_off,
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
    k2_cfg.E_fine_on = config.E_fine_on;
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
        state.d_AbsorbedEnergy_cutoff,
        state.d_AbsorbedWeight_nuclear,
        state.d_AbsorbedEnergy_nuclear,
        state.d_BoundaryLoss_weight,
        state.d_BoundaryLoss_energy,
        state.d_BucketScratch,
        state.d_CellToBucketBase,
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
        state.d_AbsorbedEnergy_cutoff,
        state.d_AbsorbedWeight_nuclear,
        state.d_AbsorbedEnergy_nuclear,
        state.d_BoundaryLoss_weight,
        state.d_BoundaryLoss_energy,
        state.d_BucketScratch,
        state.d_CellToBucketBase,
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
    const int* d_CellToBucketBase,
    int Nx, int Nz,
    const float* d_E_edges,
    int N_E,
    int N_E_local
) {
    int threads = 256;
    int blocks = (Nx * Nz + threads - 1) / threads;

    // Call K4_BucketTransfer kernel
    K4_BucketTransfer<<<blocks, threads>>>(
        d_OutflowBuckets,
        d_CellToBucketBase,
        psi_out.value,
        psi_out.block_id,
        Nx, Nz,
        d_E_edges,
        N_E,
        N_E_local
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "K4 kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    cudaDeviceSynchronize();
    return true;
}

bool run_k5_conservation_audit(
    const DevicePsiC& psi_in,
    const DevicePsiC& psi_out,
    const double* d_EdepC,
    const double* d_AbsorbedEnergy_cutoff,
    const double* d_AbsorbedEnergy_nuclear,
    const double* d_BoundaryLoss_energy,
    const double* d_prev_EdepC,
    const double* d_prev_AbsorbedEnergy_cutoff,
    const double* d_prev_AbsorbedEnergy_nuclear,
    const double* d_prev_BoundaryLoss_energy,
    const float* d_AbsorbedWeight_cutoff,
    const float* d_AbsorbedWeight_nuclear,
    const float* d_BoundaryLoss_weight,
    const float* d_prev_AbsorbedWeight_cutoff,
    const float* d_prev_AbsorbedWeight_nuclear,
    const float* d_prev_BoundaryLoss_weight,
    float source_out_of_grid_weight,
    float source_slot_dropped_weight,
    double source_out_of_grid_energy,
    double source_slot_dropped_energy,
    float transport_dropped_weight,
    double transport_dropped_energy,
    int include_source_terms,
    const float* d_E_edges,
    int N_E,
    int N_E_local,
    AuditReport* d_report,
    int Nx, int Nz
) {
    int N_cells = Nx * Nz;
    int threads = 256;
    int blocks = (N_cells + threads - 1) / threads;

    cudaMemset(d_report, 0, sizeof(AuditReport));

    // Call K5_ConservationAudit kernel
    K5_ConservationAudit<<<blocks, threads>>>(
        psi_in.block_id,
        psi_in.value,
        psi_out.block_id,
        psi_out.value,
        d_EdepC,
        d_AbsorbedEnergy_nuclear,
        d_BoundaryLoss_energy,
        d_AbsorbedEnergy_cutoff,
        d_prev_EdepC,
        d_prev_AbsorbedEnergy_nuclear,
        d_prev_BoundaryLoss_energy,
        d_prev_AbsorbedEnergy_cutoff,
        d_AbsorbedWeight_cutoff,
        d_AbsorbedWeight_nuclear,
        d_BoundaryLoss_weight,
        d_prev_AbsorbedWeight_cutoff,
        d_prev_AbsorbedWeight_nuclear,
        d_prev_BoundaryLoss_weight,
        source_out_of_grid_weight,
        source_slot_dropped_weight,
        source_out_of_grid_energy,
        source_slot_dropped_energy,
        transport_dropped_weight,
        transport_dropped_energy,
        include_source_terms,
        d_E_edges,
        N_E,
        N_E_local,
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
    // Compute block thresholds for fine activation/deactivation.
    const int b_E_fine_on = compute_b_E_threshold(config.E_fine_on, e_grid, config.N_E_local);
    const int b_E_fine_off = compute_b_E_threshold(config.E_fine_off, e_grid, config.N_E_local);
    const bool summary_logging = (config.log_level >= 1);
    const bool verbose_logging = (config.log_level >= 2);
    const bool debug_dumps_enabled =
    #if defined(SM2D_ENABLE_DEBUG_DUMPS)
        true;
    #else
        false;
    #endif

    #if defined(SM2D_ENABLE_DEBUG_DUMPS)
    // DEBUG: Print fine-threshold block mapping
    std::cout << "=== Fine Threshold Configuration ===" << std::endl;
    std::cout << "E_fine_on = " << config.E_fine_on << " MeV" << std::endl;
    std::cout << "E_fine_off = " << config.E_fine_off << " MeV" << std::endl;
    std::cout << "E_bin for E_fine_on = " << e_grid.FindBin(config.E_fine_on) << std::endl;
    std::cout << "E_bin for E_fine_off = " << e_grid.FindBin(config.E_fine_off) << std::endl;
    std::cout << "N_E_local = " << config.N_E_local << std::endl;
    std::cout << "b_E_fine_on = " << b_E_fine_on << std::endl;
    std::cout << "b_E_fine_off = " << b_E_fine_off << std::endl;

    // Sample b_E for 150 MeV
    int E_bin_150 = e_grid.FindBin(150.0f);
    int b_E_150 = E_bin_150 / config.N_E_local;
    std::cout << "=== 150 MeV Particle ===" << std::endl;
    std::cout << "E_bin_150 = " << E_bin_150 << std::endl;
    std::cout << "b_E_150 = " << b_E_150 << std::endl;
    std::cout << "Condition: b_E_150 < b_E_fine_on → " << b_E_150 << " < " << b_E_fine_on << " = " << (b_E_150 < b_E_fine_on ? "TRUE (K3 active)" : "FALSE (K2 coarse)") << std::endl;
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
    const int N_cells = config.Nx * config.Nz;

    int iter = 0;
    bool k2_batching_notified = false;
    bool k3_batching_notified = false;

    // ========================================================================
    // Main Transport Loop
    // ========================================================================
    while (iter < max_iter) {
        iter++;

        // --------------------------------------------------------------------
        // K1: Active Mask Identification
        // --------------------------------------------------------------------
        cudaMemcpy(
            state.d_ActiveMask_prev,
            state.d_ActiveMask,
            config.Nx * config.Nz * sizeof(uint8_t),
            cudaMemcpyDeviceToDevice
        );

        if (!run_k1_active_mask(*psi_in, state.d_ActiveMask, state.d_ActiveMask_prev,
                                 config.Nx, config.Nz,
                                 b_E_fine_on,
                                 b_E_fine_off,
                                 config.weight_active_min)) {
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
            double terminal_residual_weight = 0.0;
            double terminal_residual_energy = 0.0;
            if (!compute_psic_total_weight_energy(
                    *psi_in,
                    state.d_E_edges,
                    config.N_E,
                    config.N_theta_local,
                    config.N_E_local,
                    terminal_residual_weight,
                    terminal_residual_energy)) {
                std::cerr << "Failed to compute terminal residual phase-space totals" << std::endl;
                return false;
            }

            if (terminal_residual_weight > 0.0 || terminal_residual_energy > 0.0) {
                state.transport_dropped_weight += terminal_residual_weight;
                state.transport_dropped_energy += terminal_residual_energy;
                if (verbose_logging) {
                    std::cout << "  Terminal residual phase space reclassified as transport drop: "
                              << "weight=" << terminal_residual_weight
                              << ", energy=" << terminal_residual_energy << " MeV" << std::endl;
                }
            }

            if (summary_logging) {
                std::cout << "  Transport complete after " << iter << " iterations" << std::endl;
            }
            break;
        }

        // CRITICAL FIX: Clear psi_out BEFORE K2/K3 write to it
        // This prevents accumulation of particles from previous iterations
        device_psic_clear(*psi_out);

        k2_reset_debug_counters();
        k3_reset_debug_counters();
        k4_reset_debug_counters();

        // --------------------------------------------------------------------
        // K2: Coarse Transport (high energy cells)
        // --------------------------------------------------------------------
        if (state.n_coarse > 0) {
            const int requested_batch = config.fine_batch_max_cells;
            const int effective_batch = (requested_batch > 0)
                ? std::min(requested_batch, state.n_coarse)
                : state.n_coarse;
            if (effective_batch <= 0) {
                std::cerr << "Invalid K2 batch size at iteration " << iter
                          << ": requested=" << requested_batch
                          << ", coarse=" << state.n_coarse << std::endl;
                return false;
            }

            if (summary_logging &&
                !k2_batching_notified &&
                requested_batch > 0 &&
                state.n_coarse > effective_batch) {
                const int n_batches = (state.n_coarse + effective_batch - 1) / effective_batch;
                std::cout << "  K2 batching enabled: " << n_batches
                          << " batches (cap=" << effective_batch << " cells/batch)" << std::endl;
                k2_batching_notified = true;
            }

            for (int batch_offset = 0; batch_offset < state.n_coarse; batch_offset += effective_batch) {
                const int batch_cells = std::min(effective_batch, state.n_coarse - batch_offset);
                const uint32_t* batch_coarse_list = state.d_CoarseList + batch_offset;

                if (!prepare_bucket_scratch_for_batch(state, batch_coarse_list, batch_cells, N_cells)) {
                    std::cerr << "Failed to prepare K2 bucket scratch at iteration " << iter
                              << ", batch_offset=" << batch_offset
                              << ", batch_cells=" << batch_cells << std::endl;
                    return false;
                }

                if (!run_k2_coarse_transport(*psi_in, *psi_out, state.d_ActiveMask, batch_coarse_list,
                                             batch_cells, dlut, config, state)) {
                    std::cerr << "K2 failed at iteration " << iter
                              << ", batch_offset=" << batch_offset
                              << ", batch_cells=" << batch_cells << std::endl;
                    return false;
                }

                if (!run_k4_bucket_transfer(*psi_out, state.d_BucketScratch, state.d_CellToBucketBase,
                                            config.Nx, config.Nz,
                                            state.d_E_edges,
                                            config.N_E,
                                            config.N_E_local)) {
                    std::cerr << "K4 failed after K2 at iteration " << iter
                              << ", batch_offset=" << batch_offset
                              << ", batch_cells=" << batch_cells << std::endl;
                    return false;
                }
            }
        }

        // --------------------------------------------------------------------
        // K3: Fine Transport (low energy cells)
        // --------------------------------------------------------------------
        if (state.n_active > 0) {
            const int requested_batch = config.fine_batch_max_cells;
            const int effective_batch = (requested_batch > 0)
                ? std::min(requested_batch, state.n_active)
                : state.n_active;
            if (effective_batch <= 0) {
                std::cerr << "Invalid K3 batch size at iteration " << iter
                          << ": requested=" << requested_batch
                          << ", active=" << state.n_active << std::endl;
                return false;
            }

            if (summary_logging &&
                !k3_batching_notified &&
                requested_batch > 0 &&
                state.n_active > effective_batch) {
                const int n_batches = (state.n_active + effective_batch - 1) / effective_batch;
                std::cout << "  K3 batching enabled: " << n_batches
                          << " batches (cap=" << effective_batch << " cells/batch)" << std::endl;
                k3_batching_notified = true;
            }

            for (int batch_offset = 0; batch_offset < state.n_active; batch_offset += effective_batch) {
                const int batch_cells = std::min(effective_batch, state.n_active - batch_offset);
                const uint32_t* batch_active_list = state.d_ActiveList + batch_offset;

                if (!prepare_bucket_scratch_for_batch(state, batch_active_list, batch_cells, N_cells)) {
                    std::cerr << "Failed to prepare K3 bucket scratch at iteration " << iter
                              << ", batch_offset=" << batch_offset
                              << ", batch_cells=" << batch_cells << std::endl;
                    return false;
                }

                if (!run_k3_fine_transport(
                        *psi_in,
                        *psi_out,
                        batch_active_list,
                        batch_cells,
                        dlut,
                        config,
                        state)) {
                    std::cerr << "K3 failed at iteration " << iter
                              << ", batch_offset=" << batch_offset
                              << ", batch_cells=" << batch_cells << std::endl;
                    return false;
                }

                if (!run_k4_bucket_transfer(*psi_out, state.d_BucketScratch, state.d_CellToBucketBase,
                                            config.Nx, config.Nz,
                                            state.d_E_edges,
                                            config.N_E,
                                            config.N_E_local)) {
                    std::cerr << "K4 failed after K3 at iteration " << iter
                              << ", batch_offset=" << batch_offset
                              << ", batch_cells=" << batch_cells << std::endl;
                    return false;
                }
            }
        }

        // Gather per-iteration drop channels from transport kernels and fold into K5.
        unsigned long long k2_slot_drop_count = 0;
        unsigned long long k2_bucket_drop_count = 0;
        unsigned long long k2_pruned_weight_count = 0;
        double k2_slot_drop_weight = 0.0;
        double k2_slot_drop_energy = 0.0;
        double k2_bucket_drop_weight = 0.0;
        double k2_bucket_drop_energy = 0.0;
        double k2_pruned_weight_sum = 0.0;
        double k2_pruned_energy_sum = 0.0;
        k2_get_debug_counters(
            k2_slot_drop_count,
            k2_slot_drop_weight,
            k2_slot_drop_energy,
            k2_bucket_drop_count,
            k2_bucket_drop_weight,
            k2_bucket_drop_energy,
            k2_pruned_weight_count,
            k2_pruned_weight_sum,
            k2_pruned_energy_sum
        );

        unsigned long long k3_slot_drop_count = 0;
        unsigned long long k3_bucket_drop_count = 0;
        unsigned long long k3_pruned_weight_count = 0;
        double k3_slot_drop_weight = 0.0;
        double k3_slot_drop_energy = 0.0;
        double k3_bucket_drop_weight = 0.0;
        double k3_bucket_drop_energy = 0.0;
        double k3_pruned_weight_sum = 0.0;
        double k3_pruned_energy_sum = 0.0;
        k3_get_debug_counters(
            k3_slot_drop_count,
            k3_slot_drop_weight,
            k3_slot_drop_energy,
            k3_bucket_drop_count,
            k3_bucket_drop_weight,
            k3_bucket_drop_energy,
            k3_pruned_weight_count,
            k3_pruned_weight_sum,
            k3_pruned_energy_sum
        );

        unsigned long long k4_slot_drop_count = 0;
        double k4_slot_drop_weight = 0.0;
        double k4_slot_drop_energy = 0.0;
        k4_get_debug_counters(
            k4_slot_drop_count,
            k4_slot_drop_weight,
            k4_slot_drop_energy
        );

        const double slot_bucket_drop_weight_iter =
            k2_slot_drop_weight +
            k2_bucket_drop_weight +
            k3_slot_drop_weight +
            k3_bucket_drop_weight +
            k4_slot_drop_weight;
        const double slot_bucket_drop_energy_iter =
            k2_slot_drop_energy +
            k2_bucket_drop_energy +
            k3_slot_drop_energy +
            k3_bucket_drop_energy +
            k4_slot_drop_energy;
        const double pruned_drop_weight_iter = k2_pruned_weight_sum + k3_pruned_weight_sum;
        const double pruned_drop_energy_iter = k2_pruned_energy_sum + k3_pruned_energy_sum;
        const double transport_drop_weight_iter = slot_bucket_drop_weight_iter + pruned_drop_weight_iter;
        const double transport_drop_energy_iter = slot_bucket_drop_energy_iter + pruned_drop_energy_iter;
        state.transport_dropped_weight += transport_drop_weight_iter;
        state.transport_dropped_energy += transport_drop_energy_iter;

        if (verbose_logging && (transport_drop_weight_iter > 0.0 ||
                                k2_pruned_weight_count > 0 || k3_pruned_weight_count > 0)) {
            std::cout << "  Transport drops: weight=" << transport_drop_weight_iter
                      << ", energy=" << transport_drop_energy_iter << " MeV"
                      << " [K2 slot=" << k2_slot_drop_count
                      << ", K2 bucket=" << k2_bucket_drop_count
                      << ", K3 slot=" << k3_slot_drop_count
                      << ", K3 bucket=" << k3_bucket_drop_count
                      << ", K4 slot=" << k4_slot_drop_count << "]" << std::endl;
            if (k2_pruned_weight_count > 0 || k3_pruned_weight_count > 0) {
                std::cout << "  Transport pruned tiny weights: "
                          << "K2 count=" << k2_pruned_weight_count
                          << ", K3 count=" << k3_pruned_weight_count
                          << ", weight=" << pruned_drop_weight_iter
                          << ", energy=" << pruned_drop_energy_iter << " MeV" << std::endl;
            }
        }

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
        // K5: Weight + Energy Audit (conservation check)
        // --------------------------------------------------------------------
        // Note: psi_in now contains processed particles (moved from original)
        //       psi_out contains K2+K3+K4 results (cleared at start of iteration)
        if (!run_k5_conservation_audit(*psi_in, *psi_out,
                                 state.d_EdepC,
                                 state.d_AbsorbedEnergy_cutoff,
                                 state.d_AbsorbedEnergy_nuclear,
                                 state.d_BoundaryLoss_energy,
                                 state.d_prev_EdepC,
                                 state.d_prev_AbsorbedEnergy_cutoff,
                                 state.d_prev_AbsorbedEnergy_nuclear,
                                 state.d_prev_BoundaryLoss_energy,
                                 state.d_AbsorbedWeight_cutoff,
                                 state.d_AbsorbedWeight_nuclear,
                                 state.d_BoundaryLoss_weight,
                                 state.d_prev_AbsorbedWeight_cutoff,
                                 state.d_prev_AbsorbedWeight_nuclear,
                                 state.d_prev_BoundaryLoss_weight,
                                 state.source_out_of_grid_weight,
                                 state.source_slot_dropped_weight,
                                 state.source_out_of_grid_energy,
                                 state.source_slot_dropped_energy,
                                 static_cast<float>(transport_drop_weight_iter),
                                 transport_drop_energy_iter,
                                 (iter == 1) ? 1 : 0,
                                 state.d_E_edges,
                                 config.N_E,
                                 config.N_E_local,
                                 state.d_audit_report,
                                 config.Nx, config.Nz)) {
            std::cerr << "K5 failed at iteration " << iter << std::endl;
            return false;
        }

        AuditReport report = get_audit_report(state, config.Nx, config.Nz);
        {
            double energy_rhs =
                report.E_out_total +
                report.E_dep_total +
                report.E_cutoff_total +
                report.E_nuclear_total +
                report.E_boundary_total +
                report.E_transport_drop_total +
                report.E_source_out_of_grid_total +
                report.E_source_slot_drop_total;
            double energy_residual = report.E_in_total - energy_rhs;
            if (std::isfinite(energy_residual)) {
                state.transport_audit_residual_energy += energy_residual;
            }
        }

        if (verbose_logging) {
            std::cout << "  Weight audit: error=" << report.W_error
                      << " pass=" << (report.W_pass ? "yes" : "no")
                      << ", Energy audit: error=" << report.E_error
                      << " pass=" << (report.E_pass ? "yes" : "no") << std::endl;
        }

        if (config.fail_fast_on_audit && (!report.W_pass || !report.E_pass)) {
            std::cerr << "K5 audit fail-fast at iteration " << iter
                      << " (W_error=" << report.W_error
                      << ", E_error=" << report.E_error
                      << ", W_pass=" << report.W_pass
                      << ", E_pass=" << report.E_pass << ")" << std::endl;
            return false;
        }

        // Update previous cumulative tallies for next iteration's delta-based K5.
        cudaMemcpy(state.d_prev_AbsorbedWeight_cutoff, state.d_AbsorbedWeight_cutoff,
                   N_cells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(state.d_prev_AbsorbedEnergy_cutoff, state.d_AbsorbedEnergy_cutoff,
                   N_cells * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(state.d_prev_AbsorbedWeight_nuclear, state.d_AbsorbedWeight_nuclear,
                   N_cells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(state.d_prev_BoundaryLoss_weight, state.d_BoundaryLoss_weight,
                   N_cells * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(state.d_prev_EdepC, state.d_EdepC,
                   N_cells * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(state.d_prev_AbsorbedEnergy_nuclear, state.d_AbsorbedEnergy_nuclear,
                   N_cells * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(state.d_prev_BoundaryLoss_energy, state.d_BoundaryLoss_energy,
                   N_cells * sizeof(double), cudaMemcpyDeviceToDevice);

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

    // Cutoff channels (particles below energy threshold)
    std::vector<float> h_AbsorbedWeight_cutoff(total_cells);
    cudaMemcpy(h_AbsorbedWeight_cutoff.data(), state.d_AbsorbedWeight_cutoff, total_cells * sizeof(float), cudaMemcpyDeviceToHost);
    double total_cutoff_weight = 0.0;
    for (int i = 0; i < total_cells; ++i) {
        total_cutoff_weight += h_AbsorbedWeight_cutoff[i];
    }
    std::vector<double> h_AbsorbedEnergy_cutoff(total_cells);
    cudaMemcpy(h_AbsorbedEnergy_cutoff.data(), state.d_AbsorbedEnergy_cutoff, total_cells * sizeof(double), cudaMemcpyDeviceToHost);
    double total_cutoff_energy = 0.0;
    for (int i = 0; i < total_cells; ++i) {
        total_cutoff_energy += h_AbsorbedEnergy_cutoff[i];
    }

    // Nuclear energy (from inelastic nuclear interactions)
    std::vector<double> h_AbsorbedEnergy_nuclear(total_cells);
    cudaMemcpy(h_AbsorbedEnergy_nuclear.data(), state.d_AbsorbedEnergy_nuclear, total_cells * sizeof(double), cudaMemcpyDeviceToHost);
    double total_nuclear_energy = 0.0;
    for (int i = 0; i < total_cells; ++i) {
        total_nuclear_energy += h_AbsorbedEnergy_nuclear[i];
    }

    const double source_injected_energy = state.source_injected_energy;
    const double source_out_of_grid_energy = state.source_out_of_grid_energy;
    const double source_slot_dropped_energy = state.source_slot_dropped_energy;
    const double source_representation_loss_energy = state.source_representation_loss_energy;
    const double transport_drop_weight = state.transport_dropped_weight;
    const double transport_drop_energy = state.transport_dropped_energy;
    const double transport_audit_residual_energy = state.transport_audit_residual_energy;
    const double source_total_energy = source_injected_energy + source_out_of_grid_energy + source_slot_dropped_energy;
    double total_accounted_transport =
        total_edep +
        total_cutoff_energy +
        total_boundary_loss_energy +
        total_nuclear_energy +
        transport_drop_energy +
        transport_audit_residual_energy;
    double total_accounted_system =
        total_accounted_transport +
        source_out_of_grid_energy +
        source_slot_dropped_energy +
        source_representation_loss_energy;

    std::cout << "\n=== Energy Conservation Report ===" << std::endl;
    std::cout << "  Source Energy (in-grid): " << source_injected_energy << " MeV" << std::endl;
    std::cout << "  Source Energy (outside grid): " << source_out_of_grid_energy << " MeV" << std::endl;
    std::cout << "  Source Energy (slot dropped): " << source_slot_dropped_energy << " MeV" << std::endl;
    std::cout << "  Source Energy (representation loss): " << source_representation_loss_energy << " MeV" << std::endl;
    std::cout << "  Source Energy (total): " << source_total_energy << " MeV" << std::endl;
    std::cout << "  Energy Deposited: " << total_edep << " MeV" << std::endl;
    std::cout << "  Cutoff Energy Deposited: " << total_cutoff_energy << " MeV" << std::endl;
    std::cout << "  Nuclear Energy Deposited: " << total_nuclear_energy << " MeV" << std::endl;
    std::cout << "  Boundary Loss Energy: " << total_boundary_loss_energy << " MeV" << std::endl;
    std::cout << "  Transport Drop Energy: " << transport_drop_energy << " MeV" << std::endl;
    std::cout << "  Transport Audit Residual Energy: " << transport_audit_residual_energy << " MeV" << std::endl;
    std::cout << "  Cutoff Weight: " << total_cutoff_weight << " (particles below E < 0.1 MeV)" << std::endl;
    std::cout << "  Transport Drop Weight: " << transport_drop_weight << std::endl;
    std::cout << "  Total Accounted Energy (transport only): " << total_accounted_transport << " MeV" << std::endl;
    std::cout << "  Total Accounted Energy (including source losses): " << total_accounted_system << " MeV" << std::endl;
    std::cout << "=====================================" << std::endl;

    return true;
}

} // namespace sm_2d
