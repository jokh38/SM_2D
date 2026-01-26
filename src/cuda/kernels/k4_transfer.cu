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

__device__ int get_neighbor(int cell, int face, int Nx, int Nz) {
    int ix = cell % Nx;
    int iz = cell / Nx;

    switch (face) {
        case 0:  // +z
            if (iz + 1 >= Nz) return -1;
            return cell + Nx;
        case 1:  // -z
            if (iz <= 0) return -1;
            return cell - Nx;
        case 2:  // +x
            if (ix + 1 >= Nx) return -1;
            return cell + 1;
        case 3:  // -x
            if (ix <= 0) return -1;
            return cell - 1;
    }
    return -1;
}

// P3 FIX: Device function to transfer bucket contents to PsiC output
__device__ inline void device_transfer_bucket_to_psi(
    const DeviceOutflowBucket& bucket,
    int cell,
    int Nx, int Nz,
    uint32_t* __restrict__ block_ids_out,
    float* __restrict__ values_out,
    int max_slots_per_cell
) {
    for (int slot = 0; slot < DEVICE_Kb_out; ++slot) {
        uint32_t bid = bucket.block_id[slot];
        if (bid == DEVICE_EMPTY_BLOCK_ID) continue;

        // Find neighbor cell
        // Note: bucket index encodes the source cell and face
        // We need to determine the destination based on the face
        // This is handled by the caller; here we just process the bucket
    }
}

__global__ void K4_BucketTransfer(
    const DeviceOutflowBucket* __restrict__ OutflowBuckets,
    float* __restrict__ values_out,
    uint32_t* __restrict__ block_ids_out,
    int Nx, int Nz
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    // P3 FIX: Define max slots per cell (must match PsiC structure)
    // FIX: Use DEVICE_Kb instead of hardcoded 32
    constexpr int max_slots_per_cell = DEVICE_Kb;  // = 8

    // DEBUG: Track total weight transferred for cells 100, 300, 700
    float debug_total_weight = 0.0f;

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

            // Find or allocate slot in this cell's output
            // Note: This requires atomic operations for thread safety
            int out_slot = -1;

            // First try to find existing slot
            for (int s = 0; s < max_slots_per_cell; ++s) {
                uint32_t existing_bid = block_ids_out[cell * max_slots_per_cell + s];
                if (existing_bid == bid) {
                    out_slot = s;
                    break;
                }
            }

            // If not found, try to allocate new slot
            if (out_slot < 0) {
                for (int s = 0; s < max_slots_per_cell; ++s) {
                    uint32_t existing_bid = block_ids_out[cell * max_slots_per_cell + s];
                    uint32_t expected = DEVICE_EMPTY_BLOCK_ID;

                    // Atomic swap to allocate slot
                    if (existing_bid == expected) {
                        if (atomicCAS(&block_ids_out[cell * max_slots_per_cell + s],
                                      expected, bid) == expected) {
                            out_slot = s;
                            break;
                        }
                    }
                }
            }

            // Transfer all local bins
            if (out_slot >= 0) {
                for (int lidx = 0; lidx < DEVICE_LOCAL_BINS; ++lidx) {
                    float w = bucket.value[slot][lidx];
                    if (w > 0) {
                        int global_idx = (cell * max_slots_per_cell + out_slot) * DEVICE_LOCAL_BINS + lidx;
                        atomicAdd(&values_out[global_idx], w);
                        debug_total_weight += w;
                    }
                }
            }
        }
    }

    // DEBUG: Print for key cells
    if ((cell == 100 || cell == 300 || cell == 500) && debug_total_weight > 0) {
        printf("K4: cell=%d (ix=%d, iz=%d) received total_weight=%.6f\n", cell, cell % Nx, cell / Nx, debug_total_weight);
    }
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
