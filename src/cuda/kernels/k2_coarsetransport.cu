#include "kernels/k2_coarsetransport.cuh"
#include "device/device_lut.cuh"
#include "device/device_physics.cuh"
#include "device/device_bucket.cuh"
#include "physics/step_control.hpp"
#include "physics/highland.hpp"
#include "physics/nuclear.hpp"
#include "physics/physics.hpp"
#include "physics/energy_straggling.hpp"
#include "lut/r_lut.hpp"
#include <cstdint>
#include <mutex>
#include <cmath>

// Forward declaration for DeviceOutflowBucket (defined in device_bucket.cuh)
// This is already included via device_bucket.cuh

// ============================================================================
// FIX Problem 4: Coarse Transport Implementation
// ============================================================================
// Purpose: Handle high-energy particles where ActiveMask=0
//
// Physics:
// - Energy loss with stopping power (dE/dx)
// - Simplified MCS: use mean scattering angle (no random sampling)
// - Nuclear attenuation (same as fine transport)
// - Larger step sizes for efficiency
//
// Flow: K2 → K4 (bucket transfer)
// ============================================================================

__global__ void K2_CoarseTransport(
    // Inputs
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint8_t* __restrict__ ActiveMask,
    // Grid
    int Nx, int Nz, float dx, float dz,
    int n_coarse,
    // Device LUT
    const DeviceRLUT dlut,
    // Grid edges
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,
    // Config
    K2Config config,
    // Outputs
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,
    // Outflow buckets
    DeviceOutflowBucket* __restrict__ OutflowBuckets
) {
    // Thread ID maps to coarse cell (or use linear iteration)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Linear scan through cells to find those needing coarse transport
    int coarse_count = 0;
    int cell = -1;

    for (int c = 0; c < Nx * Nz; ++c) {
        if (ActiveMask[c] == 0) {
            if (coarse_count == idx) {
                cell = c;
                break;
            }
            coarse_count++;
        }
    }

    if (cell < 0 || coarse_count >= n_coarse) return;

    // Accumulators for this cell
    double cell_edep = 0.0;
    float cell_w_cutoff = 0.0f;
    float cell_w_nuclear = 0.0f;
    double cell_E_nuclear = 0.0f;
    float cell_boundary_weight = 0.0f;
    double cell_boundary_energy = 0.0f;

    // Coarse step size: use full cell size or configured step
    float coarse_step = fminf(config.step_coarse, fminf(dx, dz));

    // Process all slots in this cell
    for (int slot = 0; slot < 32; ++slot) {
        uint32_t bid = block_ids_in[cell * 32 + slot];
        if (bid == DEVICE_EMPTY_BLOCK_ID) continue;

        // Decode block ID
        uint32_t b_theta = bid & 0xFFF;
        uint32_t b_E = (bid >> 12) & 0xFFF;

        // Process all local bins
        for (int lidx = 0; lidx < DEVICE_LOCAL_BINS; ++lidx) {
            int global_idx = (cell * 32 + slot) * DEVICE_LOCAL_BINS + lidx;
            float weight = values_in[global_idx];
            if (weight < 1e-12f) continue;

            // Decode 4D local index
            int theta_local, E_local, x_sub, z_sub;
            decode_local_idx_4d(lidx, theta_local, E_local, x_sub, z_sub);

            // Get phase space values
            int theta_bin = b_theta * N_theta_local + theta_local;
            int E_bin = b_E * N_E_local + E_local;

            float theta_min = theta_edges[0];
            float theta_max = theta_edges[N_theta];
            float dtheta = (theta_max - theta_min) / N_theta;
            float theta = theta_min + (theta_bin + 0.5f) * dtheta;

            float E_min = E_edges[0];
            float E_max = E_edges[N_E];
            float log_E_min = logf(E_min);
            float dlog = (logf(E_max) - log_E_min) / N_E;
            float E = expf(log_E_min + (E_bin + 0.5f) * dlog);

            // Cutoff check
            if (E <= 0.1f) {
                cell_edep += E * weight;
                cell_w_cutoff += weight;
                continue;
            }

            // Starting position (sub-cell offsets)
            float x_offset = get_x_offset_from_bin(x_sub, dx);
            float z_offset = get_z_offset_from_bin(z_sub, dz);
            float x_cell = x_offset;
            float z_cell = z_offset;

            // Direction
            float mu = cosf(theta);
            float eta = sinf(theta);

            // Coarse transport: take one large step through the cell
            // Energy loss for coarse step
            float mean_dE = device_compute_energy_deposition(dlut, E, coarse_step);
            float dE = mean_dE;  // Coarse: use mean (no straggling)
            dE = fminf(dE, E);

            float E_new = E - dE;
            float edep = dE * weight;

            // Nuclear attenuation
            float w_rem, E_rem;
            float w_new = device_apply_nuclear_attenuation(weight, E, coarse_step, w_rem, E_rem);
            edep += E_rem;

            // Coarse MCS: use RMS angle (no random sampling for efficiency)
            float sigma_mcs = device_highland_sigma(E, coarse_step);
            // Apply RMS scattering as systematic angular spread
            // In coarse mode, we bias the angle toward the mean (zero deflection)
            // This represents the "average" trajectory
            float theta_new = theta;  // Coarse: no random scattering, just energy loss

            float mu_new = cosf(theta_new);
            float eta_new = sinf(theta_new);

            // Position update
            float x_new = x_cell + eta_new * coarse_step;
            float z_new = z_cell + mu_new * coarse_step;

            // Clamp to cell bounds
            x_new = fmaxf(0.0f, fminf(x_new, dx));
            z_new = fmaxf(0.0f, fminf(z_new, dz));

            // Check boundary crossing
            int exit_face = device_determine_exit_face(x_cell, z_cell, x_new, z_new, dx, dz);

            if (exit_face >= 0) {
                // Calculate sub-cell bins for neighbor
                float x_offset_neighbor = device_get_neighbor_x_offset(x_new, exit_face, dx);
                int x_sub_neighbor = get_x_sub_bin(x_offset_neighbor, dx);

                float z_offset_neighbor;
                if (exit_face == 0) {
                    z_offset_neighbor = -dz * 0.5f + dz * 0.125f;
                } else if (exit_face == 1) {
                    z_offset_neighbor = dz * 0.5f - dz * 0.125f;
                } else {
                    z_offset_neighbor = z_new - dz * 0.5f;
                }
                int z_sub_neighbor = get_z_sub_bin(z_offset_neighbor, dz);
                z_sub_neighbor = device_transform_z_sub_for_neighbor(z_sub_neighbor, exit_face);

                int bucket_idx = device_bucket_index(cell, exit_face, Nx, Nz);
                DeviceOutflowBucket& bucket = OutflowBuckets[bucket_idx];
                device_emit_component_to_bucket_4d(
                    bucket, theta_new, E_new, w_new, x_sub_neighbor, z_sub_neighbor,
                    theta_edges, E_edges, N_theta, N_E,
                    N_theta_local, N_E_local
                );

                cell_boundary_weight += w_new;
                cell_boundary_energy += E_new * w_new;
                cell_boundary_energy += E_rem;
            } else {
                // Remains in cell
                cell_edep += edep;
                cell_w_nuclear += w_rem;
                cell_E_nuclear += E_rem;

                if (E_new <= 0.1f) {
                    cell_edep += E_new * w_new;
                    cell_w_cutoff += w_new;
                }
            }
        }
    }

    // Write accumulators
    atomicAdd(&EdepC[cell], cell_edep);
    atomicAdd(&AbsorbedWeight_cutoff[cell], cell_w_cutoff);
    atomicAdd(&AbsorbedWeight_nuclear[cell], cell_w_nuclear);
    atomicAdd(&AbsorbedEnergy_nuclear[cell], cell_E_nuclear);
    atomicAdd(&BoundaryLoss_weight[cell], cell_boundary_weight);
    atomicAdd(&BoundaryLoss_energy[cell], cell_boundary_energy);
}

// ============================================================================
// CPU Wrapper Implementation
// ============================================================================

void run_K2_CoarseTransport(
    const uint32_t* block_ids_in,
    const float* values_in,
    const uint8_t* ActiveMask,
    int Nx, int Nz, float dx, float dz,
    int n_coarse,
    DeviceRLUT dlut,
    const float* theta_edges,
    const float* E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,
    K2Config config,
    double* EdepC,
    float* AbsorbedWeight_cutoff,
    float* AbsorbedWeight_nuclear,
    double* AbsorbedEnergy_nuclear,
    float* BoundaryLoss_weight,
    double* BoundaryLoss_energy,
    DeviceOutflowBucket* OutflowBuckets
) {
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n_coarse + threads_per_block - 1) / threads_per_block;

    K2_CoarseTransport<<<blocks, threads_per_block>>>(
        block_ids_in, values_in, ActiveMask,
        Nx, Nz, dx, dz, n_coarse,
        dlut,
        theta_edges, E_edges,
        N_theta, N_E, N_theta_local, N_E_local,
        config,
        EdepC, AbsorbedWeight_cutoff, AbsorbedWeight_nuclear, AbsorbedEnergy_nuclear,
        BoundaryLoss_weight, BoundaryLoss_energy,
        OutflowBuckets
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling would go here
    }
}
