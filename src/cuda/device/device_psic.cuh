#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"

// ============================================================================
// Device PsiC Management Structures
// ============================================================================
// Device-side PsiC (Psi Compact) management for GPU kernels
//
// This file provides GPU-accessible structures and functions for managing
// particle phase-space density on the device, mirroring the host-side PsiC
// structure defined in src/include/core/psi_storage.hpp
//
// Key differences from host-side:
// - Flattened 2D arrays instead of nested vectors
// - Device memory management with explicit init/cleanup
// - Atomic operations for thread-safe updates
// - Source particle injection functions
//
// Storage layout:
// - Kb = 32 slots per cell (device-level)
// - LOCAL_BINS = 512 per slot (4D: theta_local, E_local, x_sub, z_sub)
// - Total per cell: 32 * 512 = 16,384 values
// ============================================================================

constexpr int DEVICE_Kb = 32;           // Slots per cell on device
constexpr uint32_t DEVICE_EMPTY_SLOT = 0xFFFFFFFF;  // Empty slot marker

// ============================================================================
// Device PsiC Structure
// ============================================================================

// Device-accessible PsiC structure with flattened GPU arrays
// Matches host PsiC layout for efficient H2D/D2H transfers
struct DevicePsiC {
    int Nx;                      // Number of cells in x
    int Nz;                      // Number of cells in z
    int N_cells;                 // Total number of cells (Nx * Nz)
    int Kb;                      // Slots per cell (DEVICE_Kb = 32)

    // Flattened GPU arrays
    // Layout: cell * Kb + slot
    uint32_t* block_id;          // Block ID for each slot [N_cells * Kb]

    // Layout: cell * Kb * LOCAL_BINS + slot * LOCAL_BINS + lidx
    float* value;                // Particle weights [N_cells * Kb * LOCAL_BINS]

    // Device pointers for bucket arrays (optional, for K3/K4 kernels)
    // These are managed separately but referenced here for convenience
    // DeviceOutflowBucket* buckets;  // Per-cell outflow buckets
};

// Forward declaration
__host__ inline void device_psic_clear(DevicePsiC& psic);

// ============================================================================
// Memory Management Functions
// ============================================================================

// Initialize DevicePsiC structure and allocate GPU memory
// Returns: true on success, false on allocation failure
__host__ inline bool device_psic_init(
    DevicePsiC& psic,
    int Nx,
    int Nz,
    int Kb = DEVICE_Kb
) {
    psic.Nx = Nx;
    psic.Nz = Nz;
    psic.N_cells = Nx * Nz;
    psic.Kb = Kb;

    size_t total_slots = static_cast<size_t>(psic.N_cells) * Kb;
    size_t total_values = total_slots * LOCAL_BINS;

    size_t block_bytes = total_slots * sizeof(uint32_t);
    size_t value_bytes = total_values * sizeof(float);

    std::cout << "DevicePsiC allocation:" << std::endl;
    std::cout << "  Grid: " << Nx << "x" << Nz << " = " << psic.N_cells << " cells" << std::endl;
    std::cout << "  Kb: " << Kb << ", LOCAL_BINS: " << LOCAL_BINS << std::endl;
    std::cout << "  block_id: " << block_bytes << " bytes (" << (block_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "  value: " << value_bytes << " bytes (" << (value_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "  total per PsiC: " << ((block_bytes + value_bytes) / 1024.0 / 1024.0) << " MB" << std::endl;

    // Allocate block_id array
    cudaError_t err = cudaMalloc(&psic.block_id, block_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate block_id: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Allocate value array
    err = cudaMalloc(&psic.value, value_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate value: " << cudaGetErrorString(err) << std::endl;
        cudaFree(psic.block_id);
        return false;
    }

    std::cout << "DevicePsiC allocation successful" << std::endl;

    // Initialize to empty state
    device_psic_clear(psic);

    return true;
}

// Cleanup DevicePsiC and free GPU memory
__host__ inline void device_psic_cleanup(DevicePsiC& psic) {
    if (psic.block_id != nullptr) {
        cudaFree(psic.block_id);
        psic.block_id = nullptr;
    }
    if (psic.value != nullptr) {
        cudaFree(psic.value);
        psic.value = nullptr;
    }
}

// Clear all PsiC data on device (set to empty state)
// Launch with: clear_psic_kernel<<<grid, block>>>(psic);
__inline__ __global__ void clear_psic_kernel(DevicePsiC psic) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= psic.N_cells) return;

    size_t base_slot = cell * psic.Kb;
    size_t base_value = cell * psic.Kb * LOCAL_BINS;

    // Clear all slots for this cell
    for (int slot = 0; slot < psic.Kb; ++slot) {
        psic.block_id[base_slot + slot] = DEVICE_EMPTY_SLOT;

        // Clear all local bins
        size_t value_base = base_value + slot * LOCAL_BINS;
        for (int lidx = threadIdx.y; lidx < LOCAL_BINS; lidx += blockDim.y) {
            psic.value[value_base + lidx] = 0.0f;
        }
    }
}

// Host-side wrapper to clear PsiC
__host__ inline void device_psic_clear(DevicePsiC& psic) {
    if (psic.block_id == nullptr || psic.value == nullptr) return;

    // Launch kernel with 2D blocks for efficient clearing
    dim3 block(256, 4);  // 256 threads x 4 threads = 1024 threads per block
    int grid_size = (psic.N_cells + 255) / 256;
    clear_psic_kernel<<<grid_size, block>>>(psic);
    cudaDeviceSynchronize();
}

// ============================================================================
// Device-Side Access Functions
// ============================================================================

// Find or allocate a slot for a given block ID in a cell
// Returns: slot index [0, Kb-1] or -1 if full
__device__ inline int device_psic_find_or_allocate_slot(
    DevicePsiC& psic,
    int cell,
    uint32_t bid
) {
    if (cell < 0 || cell >= psic.N_cells) return -1;

    size_t base_slot = cell * psic.Kb;

    // First pass: try to find existing slot
    for (int slot = 0; slot < psic.Kb; ++slot) {
        if (psic.block_id[base_slot + slot] == bid) {
            return slot;
        }
    }

    // Second pass: try to allocate new slot
    for (int slot = 0; slot < psic.Kb; ++slot) {
        uint32_t expected = DEVICE_EMPTY_SLOT;
        // Atomic CAS to claim this slot
        if (atomicCAS(&psic.block_id[base_slot + slot], expected, bid) == expected) {
            // Successfully claimed - initialize values to zero
            size_t value_base = (cell * psic.Kb + slot) * LOCAL_BINS;
            for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                psic.value[value_base + lidx] = 0.0f;
            }
            return slot;
        }
    }

    // All slots full or claimed by other threads
    return -1;
}

// Get weight from a specific cell, slot, and local bin
__device__ inline float device_psic_get_weight(
    const DevicePsiC& psic,
    int cell,
    int slot,
    uint16_t lidx
) {
    if (cell < 0 || cell >= psic.N_cells) return 0.0f;
    if (slot < 0 || slot >= psic.Kb) return 0.0f;
    if (lidx >= LOCAL_BINS) return 0.0f;

    size_t idx = (cell * psic.Kb + slot) * LOCAL_BINS + lidx;
    return psic.value[idx];
}

// Set weight in a specific cell, slot, and local bin
__device__ inline void device_psic_set_weight(
    DevicePsiC& psic,
    int cell,
    int slot,
    uint16_t lidx,
    float w
) {
    if (cell < 0 || cell >= psic.N_cells) return;
    if (slot < 0 || slot >= psic.Kb) return;
    if (lidx >= LOCAL_BINS) return;

    size_t idx = (cell * psic.Kb + slot) * LOCAL_BINS + lidx;
    psic.value[idx] = w;
}

// Atomically add weight to a specific cell, slot, and local bin
__device__ inline void device_psic_add_weight(
    DevicePsiC& psic,
    int cell,
    int slot,
    uint16_t lidx,
    float w
) {
    if (cell < 0 || cell >= psic.N_cells) return;
    if (slot < 0 || slot >= psic.Kb) return;
    if (lidx >= LOCAL_BINS) return;

    size_t idx = (cell * psic.Kb + slot) * LOCAL_BINS + lidx;
    atomicAdd(&psic.value[idx], w);
}

// ============================================================================
// Source Particle Injection
// ============================================================================

// Inject source particles into a specific cell
// This is called by source initialization kernels to populate PsiC
__device__ inline bool device_psic_inject_source(
    DevicePsiC& psic,
    int cell,
    float theta,            // Polar angle [rad]
    float E,                // Energy [MeV]
    float weight,           // Statistical weight
    float x, float z,       // Position in cell [mm]
    float dx, float dz,     // Cell dimensions [mm]
    const float* __restrict__ theta_edges,  // Angular grid edges
    const float* __restrict__ E_edges,      // Energy grid edges
    int N_theta,            // Number of angular bins
    int N_E,                // Number of energy bins
    int N_theta_local,      // Local angular bins per block (8)
    int N_E_local           // Local energy bins per block (4)
) {
    if (weight <= 0.0f) return false;
    if (cell < 0 || cell >= psic.N_cells) return false;

    // Clamp position to cell bounds
    x = fmaxf(0.0f, fminf(x, dx));
    z = fmaxf(0.0f, fminf(z, dz));

    // Calculate offsets from cell center
    float x_offset = x - dx * 0.5f;
    float z_offset = z - dz * 0.5f;

    // Get sub-cell bins
    int x_sub = get_x_sub_bin(x_offset, dx);
    int z_sub = get_z_sub_bin(z_offset, dz);

    // Clamp values to grid bounds
    float theta_min = theta_edges[0];
    float theta_max = theta_edges[N_theta];
    float E_min = E_edges[0];
    float E_max = E_edges[N_E];

    theta = fmaxf(theta_min, fminf(theta, theta_max));
    E = fmaxf(E_min, fminf(E, E_max));

    // Calculate continuous bin positions
    float dtheta = (theta_max - theta_min) / N_theta;
    float theta_cont = (theta - theta_min) / dtheta;
    int theta_bin = (int)theta_cont;
    float frac_theta = theta_cont - theta_bin;

    // Option D2: Use binary search for energy bin (works for both log-spaced and piecewise-uniform)
    // Binary search in E_edges to find the bin containing energy E
    int E_bin = 0;
    if (E <= E_edges[0]) {
        E_bin = 0;
    } else if (E >= E_edges[N_E]) {
        E_bin = N_E - 1;
    } else {
        int lo = 0, hi = N_E;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (E_edges[mid + 1] <= E) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        E_bin = lo;
    }

    // For interpolation, calculate position within bin (0-1)
    float E_bin_lower = E_edges[E_bin];
    float E_bin_upper = E_edges[E_bin + 1];
    float frac_E = (E_bin_upper - E_bin_lower) > 1e-10f ? (E - E_bin_lower) / (E_bin_upper - E_bin_lower) : 0.0f;

    // Handle boundary conditions
    if (theta_bin >= N_theta - 1) {
        theta_bin = N_theta - 1;
        frac_theta = 0.0f;
    }
    if (E_bin >= N_E - 1) {
        E_bin = N_E - 1;
        frac_E = 0.0f;
    }

    // COARSE-ONLY FIX: Use single-bin emission instead of bilinear interpolation
    // This prevents particle duplication (1 particle -> 1 bin instead of 1 -> 2-4 bins)
    // Round to nearest bin instead of interpolating across 4 bins
    if (frac_theta >= 0.5f && theta_bin < N_theta - 1) theta_bin++;
    if (frac_E >= 0.5f && E_bin < N_E - 1) E_bin++;

    // Emit to single bin with full weight (no interpolation)
    uint32_t b_theta = theta_bin / N_theta_local;
    uint32_t b_E = E_bin / N_E_local;
    uint32_t bid = encode_block(b_theta, b_E);

    int theta_local = theta_bin % N_theta_local;
    int E_local = E_bin % N_E_local;

    uint16_t lidx = encode_local_idx_4d(theta_local, E_local, x_sub, z_sub);

    // Find or allocate slot and add weight
    int slot = device_psic_find_or_allocate_slot(psic, cell, bid);
    if (slot >= 0) {
        device_psic_add_weight(psic, cell, slot, lidx, weight);
        return true;
    }
    return false;
}

// ============================================================================
// Host-Device Conversion Functions
// ============================================================================

// Convert host PsiC to device format
// Copies data from host structure to pre-allocated device structure
// Returns: true on success, false on copy failure
__host__ inline bool device_psic_copy_from_host(
    DevicePsiC& psic_dev,
    const uint32_t* host_block_id,   // Host block_id array [N_cells * Kb]
    const float* host_value,         // Host value array [N_cells * Kb * LOCAL_BINS]
    cudaStream_t stream = 0
) {
    size_t total_slots = static_cast<size_t>(psic_dev.N_cells) * psic_dev.Kb;
    size_t total_values = total_slots * LOCAL_BINS;

    cudaError_t err;

    // Copy block_id
    err = cudaMemcpyAsync(psic_dev.block_id, host_block_id,
                         total_slots * sizeof(uint32_t),
                         cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return false;
    }

    // Copy value
    err = cudaMemcpyAsync(psic_dev.value, host_value,
                         total_values * sizeof(float),
                         cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return false;
    }

    return true;
}

// Convert device PsiC to host format
// Copies data from device structure to pre-allocated host structure
// Returns: true on success, false on copy failure
__host__ inline bool device_psic_copy_to_host(
    const DevicePsiC& psic_dev,
    uint32_t* host_block_id,   // Host block_id array [N_cells * Kb]
    float* host_value,         // Host value array [N_cells * Kb * LOCAL_BINS]
    cudaStream_t stream = 0
) {
    size_t total_slots = static_cast<size_t>(psic_dev.N_cells) * psic_dev.Kb;
    size_t total_values = total_slots * LOCAL_BINS;

    cudaError_t err;

    // Copy block_id
    err = cudaMemcpyAsync(host_block_id, psic_dev.block_id,
                         total_slots * sizeof(uint32_t),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return false;
    }

    // Copy value
    err = cudaMemcpyAsync(host_value, psic_dev.value,
                         total_values * sizeof(float),
                         cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return false;
    }

    return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

// Compute total PsiC (sum of all weights) for a cell
// Launch with: sum_psic_cell_kernel<<<1, 256>>>(psic, cell, d_result);
__inline__ __global__ void sum_psic_cell_kernel(
    const DevicePsiC psic,
    int cell,
    float* __restrict__ d_result
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    size_t base_value = cell * psic.Kb * LOCAL_BINS;
    size_t total_values = psic.Kb * LOCAL_BINS;

    // Load data with reduction
    float sum = 0.0f;
    for (size_t i = tid; i < total_values; i += blockDim.x) {
        sum += psic.value[base_value + i];
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
        d_result[0] = sdata[0];
    }
}

// Host-side wrapper to compute sum for a cell
__host__ inline float device_psic_sum_cell(const DevicePsiC& psic, int cell) {
    if (cell < 0 || cell >= psic.N_cells) return 0.0f;

    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    dim3 block(256);
    sum_psic_cell_kernel<<<1, block, static_cast<size_t>(block.x) * sizeof(float)>>>(psic, cell, d_result);

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return result;
}

// Compute global sum of PsiC across all cells
// Launch with: sum_psic_global_kernel<<<grid, block, shared>>>(psic, d_partial);
__inline__ __global__ void sum_psic_global_kernel(
    const DevicePsiC psic,
    float* __restrict__ d_partial  // Partial results per block
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    size_t total_values = psic.N_cells * psic.Kb * LOCAL_BINS;

    // Each block processes a portion of the data
    size_t block_start = bid * blockDim.x;
    float sum = 0.0f;

    for (size_t i = block_start + tid; i < total_values; i += blockDim.x * gridDim.x) {
        sum += psic.value[i];
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

    // Write partial result
    if (tid == 0) {
        d_partial[bid] = sdata[0];
    }
}

// Host-side wrapper to compute global sum
__host__ inline float device_psic_sum_global(const DevicePsiC& psic) {
    const int n_blocks = 256;
    dim3 block(256);
    dim3 grid(n_blocks);

    float* d_partial;
    cudaMalloc(&d_partial, n_blocks * sizeof(float));

    size_t shmem_size = block.x * sizeof(float);
    sum_psic_global_kernel<<<grid, block, shmem_size>>>(
        psic, d_partial
    );

    // Copy partial results to host using std::vector for automatic cleanup
    std::vector<float> h_partial(n_blocks);
    cudaMemcpy(h_partial.data(), d_partial, n_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_partial);

    // Final reduction on host
    float total = 0.0f;
    for (int i = 0; i < n_blocks; ++i) {
        total += h_partial[i];
    }

    return total;
}

// ============================================================================
// Debug/Validation Functions
// ============================================================================

// Validate PsiC structure (check for NaN, Inf, negative values)
__inline__ __global__ void validate_psic_kernel(
    const DevicePsiC psic,
    int* __restrict__ d_error_count,
    int* __restrict__ d_first_error_cell
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= psic.N_cells) return;

    size_t base_value = cell * psic.Kb * LOCAL_BINS;
    bool has_error = false;

    for (size_t i = 0; i < psic.Kb * LOCAL_BINS; ++i) {
        float val = psic.value[base_value + i];
        if (isnan(val) || isinf(val) || val < 0.0f) {
            has_error = true;
            break;
        }
    }

    if (has_error) {
        atomicAdd(d_error_count, 1);
        atomicMin(d_first_error_cell, cell);
    }
}

// Host-side wrapper to validate PsiC
// Returns: number of cells with errors
__host__ inline int device_psic_validate(
    const DevicePsiC& psic,
    int& first_error_cell
) {
    int* d_error_count;
    int* d_first_error_cell;

    cudaMalloc(&d_error_count, sizeof(int));
    cudaMalloc(&d_first_error_cell, sizeof(int));

    cudaMemset(d_error_count, 0, sizeof(int));
    cudaMemset(d_first_error_cell, psic.N_cells, sizeof(int));

    dim3 block(256);
    dim3 grid((psic.N_cells + 255) / 256);

    validate_psic_kernel<<<grid, block>>>(psic, d_error_count, d_first_error_cell);

    int h_error_count, h_first_error_cell;
    cudaMemcpy(&h_error_count, d_error_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_first_error_cell, d_first_error_cell, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_error_count);
    cudaFree(d_first_error_cell);

    first_error_cell = h_first_error_cell;
    return h_error_count;
}
