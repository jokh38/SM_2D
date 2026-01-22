#include "kernels/k4_transfer.cuh"
#include "core/psi_storage.hpp"

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

__global__ void K4_BucketTransfer(
    const OutflowBucket* __restrict__ OutflowBuckets,
    float* __restrict__ values_out,
    uint32_t* __restrict__ block_ids_out,
    int Nx, int Nz
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    for (int face = 0; face < 4; ++face) {
        const OutflowBucket& bucket = OutflowBuckets[cell * 4 + face];

        for (int slot = 0; slot < Kb_out; ++slot) {
            uint32_t bid = bucket.block_id[slot];
            if (bid == EMPTY_BLOCK_ID) continue;

            int neighbor = get_neighbor(cell, face, Nx, Nz);
            // Transfer to neighbor (simplified)
        }
    }
}

// CPU wrapper implementation
void run_K4_BucketTransfer(
    const OutflowBucket* buckets,
    PsiC& psi_out,
    int cell,
    int face
) {
    // Simplified CPU stub for testing
    const OutflowBucket& bucket = buckets[cell * 4 + face];

    for (int slot = 0; slot < Kb_out; ++slot) {
        uint32_t bid = bucket.block_id[slot];
        if (bid == EMPTY_BLOCK_ID) continue;

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
                for (int lidx = 0; lidx < 32; ++lidx) {
                    float w = bucket.value[slot][lidx];
                    if (w > 0) {
                        psi_out.value[neighbor][out_slot][lidx] += w;
                    }
                }
            }
        }
    }
}
