#include "kernels/k4_transfer.cuh"
#include "device/device_bucket.cuh"
#include "device/device_psic.cuh"  // For DEVICE_Kb
#include "core/psi_storage.hpp"
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"

// ============================================================================
// P3 FIX: Complete K4 Bucket Transfer Implementation
// ============================================================================
// Previously: Only partial implementation
// Now: Full transfer from buckets to neighbor cells with proper slot allocation
// ============================================================================

// Global counters for optional debugging bucket transfers
#if defined(SM2D_ENABLE_DEBUG_DUMPS)
__device__ int d_transferred_particles = 0;
__device__ int d_transferred_weight = 0;
#endif

// Debug counters for K4 slot allocation failures.
__device__ unsigned long long g_k4_slot_drop_count = 0;
__device__ double g_k4_slot_drop_weight = 0.0;
__device__ double g_k4_slot_drop_energy = 0.0;

__device__ inline float k4_energy_from_bin(
    const float* __restrict__ E_edges,
    int N_E,
    uint32_t b_E,
    int E_local,
    int N_E_local
) {
    int E_bin = static_cast<int>(b_E) * N_E_local + E_local;
    if (E_bin < 0) E_bin = 0;
    if (E_bin > N_E - 1) E_bin = N_E - 1;
    float E_lower = E_edges[E_bin];
    float E_upper = E_edges[E_bin + 1];
    float E_half_width = 0.5f * (E_upper - E_lower);
    return E_lower + 1.00f * E_half_width;
}

__global__ void K4_BucketTransfer(
    const DeviceOutflowBucket* __restrict__ OutflowBuckets,
    float* __restrict__ values_out,
    uint32_t* __restrict__ block_ids_out,
    int Nx, int Nz,
    const float* __restrict__ E_edges,
    int N_E,
    int N_E_local
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    // Reset debug counters (first thread does this)
    #if defined(SM2D_ENABLE_DEBUG_DUMPS)
    if (cell == 0) {
        d_transferred_particles = 0;
        d_transferred_weight = 0;
    }
    __syncthreads();
    #endif

    // P3 FIX: Define max slots per cell (must match PsiC structure)
    constexpr int max_slots_per_cell = DEVICE_Kb;  // = 8

    // Each cell receives buckets from ALL 4 neighbors
    // Process all 4 faces
    for (int face = 0; face < 4; ++face) {
        // Find which neighbor would send to this cell through this face
        // If face=0 (+z), neighbor is at cell-Nx (sends to +z direction)
        int source_cell = -1;
        int source_face = -1;

        int ix = cell % Nx;
        int iz = cell / Nx;

        switch (face) {
            case 0:  // Receiving from -z neighbor
                if (iz > 0) {
                    source_cell = cell - Nx;
                    source_face = 0;  // +z face of source
                }
                break;
            case 1:  // Receiving from +z neighbor
                if (iz + 1 < Nz) {
                    source_cell = cell + Nx;
                    source_face = 1;  // -z face of source
                }
                break;
            case 2:  // Receiving from -x neighbor
                if (ix > 0) {
                    source_cell = cell - 1;
                    source_face = 2;  // +x face of source
                }
                break;
            case 3:  // Receiving from +x neighbor
                if (ix + 1 < Nx) {
                    source_cell = cell + 1;
                    source_face = 3;  // -x face of source
                }
                break;
        }

        if (source_cell < 0) continue;

        // Get the bucket from the source cell
        int bucket_idx = source_cell * 4 + source_face;
        const DeviceOutflowBucket& bucket = OutflowBuckets[bucket_idx];

        // Transfer all slots from bucket to this cell
        for (int slot = 0; slot < DEVICE_Kb_out; ++slot) {
            uint32_t bid = bucket.block_id[slot];
            if (bid == DEVICE_EMPTY_BLOCK_ID) continue;
            uint32_t b_E = (bid >> 12) & 0xFFF;

            // Find/allocate output slot. If saturated, reuse the closest existing
            // slot block instead of dropping the entire bucket slot.
            uint32_t selected_bid = bid;
            int out_slot = device_psic_find_or_allocate_slot_with_fallback(
                &block_ids_out[cell * max_slots_per_cell],
                max_slots_per_cell,
                bid,
                selected_bid
            );
            if (out_slot < 0) {
                double dropped_weight = 0.0;
                double dropped_energy = 0.0;
                for (int lidx = 0; lidx < DEVICE_LOCAL_BINS; ++lidx) {
                    float w = bucket.value[slot][lidx];
                    if (w > 0.0f) {
                        dropped_weight += static_cast<double>(w);
                        int theta_local_unused;
                        int E_local;
                        int x_sub_unused;
                        int z_sub_unused;
                        decode_local_idx_4d(
                            lidx,
                            theta_local_unused,
                            E_local,
                            x_sub_unused,
                            z_sub_unused
                        );
                        float E = k4_energy_from_bin(E_edges, N_E, b_E, E_local, N_E_local);
                        dropped_energy += static_cast<double>(E) * static_cast<double>(w);
                    }
                }
                atomicAdd(&g_k4_slot_drop_count, 1ULL);
                atomicAdd(&g_k4_slot_drop_weight, dropped_weight);
                atomicAdd(&g_k4_slot_drop_energy, dropped_energy);
                continue;
            }

            // Transfer all local bins
            if (out_slot >= 0) {
                for (int lidx = 0; lidx < DEVICE_LOCAL_BINS; ++lidx) {
                    float w = bucket.value[slot][lidx];
                    if (w > 0) {
                        uint16_t lidx_emit = static_cast<uint16_t>(lidx);
                        if (selected_bid != bid) {
                            lidx_emit = device_remap_lidx_to_bid(
                                static_cast<uint16_t>(lidx),
                                bid,
                                selected_bid
                            );
                        }
                        int global_idx = (cell * max_slots_per_cell + out_slot) * DEVICE_LOCAL_BINS + lidx_emit;
                        atomicAdd(&values_out[global_idx], w);
                        #if defined(SM2D_ENABLE_DEBUG_DUMPS)
                        atomicAdd(&d_transferred_weight, 1);
                        #endif
                    }
                }
                #if defined(SM2D_ENABLE_DEBUG_DUMPS)
                atomicAdd(&d_transferred_particles, 1);
                #endif
            }
        }
    }

    #if defined(SM2D_ENABLE_DEBUG_DUMPS)
    // DEBUG: Print transfer stats (first thread in block 0)
    __shared__ int s_transfer_count;
    if (threadIdx.x == 0) s_transfer_count = 0;
    __syncthreads();
    if (cell == 0) {
        s_transfer_count = d_transferred_particles;
    }
    __syncthreads();

    // Only thread 0 of block 0 prints (to avoid multiple prints)
    if (blockIdx.x == 0 && threadIdx.x == 0 && s_transfer_count > 0) {
        printf("  K4: Transferred %d particles (weight bins: %d)\n",
               s_transfer_count, d_transferred_weight);
    }
    #endif
}

void k4_reset_debug_counters() {
    constexpr unsigned long long zero_count = 0ULL;
    constexpr double zero_value = 0.0;
    cudaMemcpyToSymbol(g_k4_slot_drop_count, &zero_count, sizeof(zero_count));
    cudaMemcpyToSymbol(g_k4_slot_drop_weight, &zero_value, sizeof(zero_value));
    cudaMemcpyToSymbol(g_k4_slot_drop_energy, &zero_value, sizeof(zero_value));
}

void k4_get_debug_counters(
    unsigned long long& slot_drop_count,
    double& slot_drop_weight,
    double& slot_drop_energy
) {
    cudaMemcpyFromSymbol(&slot_drop_count, g_k4_slot_drop_count, sizeof(slot_drop_count));
    cudaMemcpyFromSymbol(&slot_drop_weight, g_k4_slot_drop_weight, sizeof(slot_drop_weight));
    cudaMemcpyFromSymbol(&slot_drop_energy, g_k4_slot_drop_energy, sizeof(slot_drop_energy));
}

// ============================================================================
// CPU wrapper implementation (updated)
// ============================================================================
void run_K4_BucketTransfer(
    const DeviceOutflowBucket* buckets,
    PsiC& psi_out,
    int cell,
    int face
) {
    const DeviceOutflowBucket& bucket = buckets[cell * 4 + face];

    for (int slot = 0; slot < DEVICE_Kb_out; ++slot) {
        uint32_t bid = bucket.block_id[slot];
        if (bid == DEVICE_EMPTY_BLOCK_ID) continue;

        int ix = cell % psi_out.Nx;
        int iz = cell / psi_out.Nx;

        int neighbor = -1;
        switch (face) {
            case 0:  // +z
                if (iz + 1 < psi_out.Nz) neighbor = cell + psi_out.Nx;
                break;
            case 1:  // -z
                if (iz > 0) neighbor = cell - psi_out.Nx;
                break;
            case 2:  // +x
                if (ix + 1 < psi_out.Nx) neighbor = cell + 1;
                break;
            case 3:  // -x
                if (ix > 0) neighbor = cell - 1;
                break;
        }

        if (neighbor >= 0) {
            int out_slot = psi_out.find_or_allocate_slot(neighbor, bid);
            if (out_slot >= 0) {
                for (int lidx = 0; lidx < DEVICE_LOCAL_BINS; ++lidx) {
                    float w = bucket.value[slot][lidx];
                    if (w > 0) {
                        psi_out.value[neighbor][out_slot][lidx] += w;
                    }
                }
            }
        }
    }
}
