#include "kernels/k2_coarsetransport.cuh"
#include "device/device_lut.cuh"
#include "device/device_physics.cuh"
#include "device/device_bucket.cuh"
#include "device/device_psic.cuh"  // For DEVICE_Kb
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
    const uint32_t* __restrict__ CoarseList,  // CRITICAL FIX: Use CoarseList instead of scanning
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
    DeviceOutflowBucket* __restrict__ OutflowBuckets,
    // CRITICAL FIX: Output phase space for particles remaining in cell
    uint32_t* __restrict__ block_ids_out,
    float* __restrict__ values_out
) {
    // Thread ID maps to coarse cell index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // CRITICAL FIX: Use CoarseList directly instead of scanning ActiveMask
    // This changes from O(N_cells * n_coarse) to O(n_coarse)
    if (idx >= n_coarse) return;

    int cell = CoarseList[idx];

    if (idx < 10 || (idx >= 95 && idx < 105)) {
        constexpr int Kb = DEVICE_Kb;
        printf("K2: Thread %d assigned to cell %d via CoarseList\n", idx, cell);
    }

    if (cell < 0 || cell >= Nx * Nz) return;

    // Accumulators for this cell
    double cell_edep = 0.0;
    float cell_w_cutoff = 0.0f;
    float cell_w_nuclear = 0.0f;
    double cell_E_nuclear = 0.0f;
    float cell_boundary_weight = 0.0f;
    double cell_boundary_energy = 0.0f;

    // DEBUG: Count particles processed in this cell
    int particles_in = 0;
    int particles_out = 0;

    // Coarse step size: limit to cell size for accurate dose deposition
    // With N_E=1024, finer energy grid reduces binning error
    float coarse_step = fminf(config.step_coarse, fminf(dx, dz));

    // FIX: Use DEVICE_Kb instead of hardcoded 32
    constexpr int Kb = DEVICE_Kb;  // = 8

    // Process all slots in this cell
    for (int slot = 0; slot < Kb; ++slot) {
        uint32_t bid = block_ids_in[cell * Kb + slot];
        if (bid == DEVICE_EMPTY_BLOCK_ID) continue;

        // Decode block ID
        uint32_t b_theta = bid & 0xFFF;
        uint32_t b_E = (bid >> 12) & 0xFFF;

        // Process all local bins
        for (int lidx = 0; lidx < DEVICE_LOCAL_BINS; ++lidx) {
            int global_idx = (cell * Kb + slot) * DEVICE_LOCAL_BINS + lidx;
            float weight = values_in[global_idx];
            if (weight < 1e-12f) continue;

            particles_in++;  // DEBUG: Count input particle

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
            // PER SPEC.md:76: Use geometric mean for energy representation
            // E_rep[i] = sqrt(E_edges[i] * E_edges[i+1])
            // Geometric mean approximation: E_edges[i] * exp(0.5 * dlog)
            float E = expf(log_E_min + (E_bin + 0.5f) * dlog);  // Geometric mean

            // H7 DEBUG: Print energy being read from bin
            if (cell % 200 == 100 && (cell / 200) < 5 && weight > 0.01f) {
                printf("K2 READ: cell=%d, E_bin=%d, b_E=%d, E=%.3f (expected ~150)\n",
                       cell, E_bin, b_E, E);
            }

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

            // FIX: Calculate distance to boundary and limit step to avoid crossing
            float half_dz = dz * 0.5f;
            float half_dx = dx * 0.5f;
            float step_to_z_plus = (mu > 0) ? (half_dz - z_cell) / mu : 1e30f;
            float step_to_z_minus = (mu < 0) ? (-half_dz - z_cell) / mu : 1e30f;
            float step_to_x_plus = (eta > 0) ? (half_dx - x_cell) / eta : 1e30f;
            float step_to_x_minus = (eta < 0) ? (-half_dx - x_cell) / eta : 1e30f;
            float step_to_boundary = fminf(fminf(step_to_z_plus, step_to_z_minus),
                                           fminf(step_to_x_plus, step_to_x_minus));
            step_to_boundary = fmaxf(step_to_boundary, 0.0f);

            // CRITICAL FIX: step_to_boundary is a path length, coarse_step is geometric distance
            // Convert path length to geometric distance for comparison
            float mu_abs = fmaxf(fabsf(mu), 1e-6f);  // Avoid division by zero

            // BUG FIX: Don't limit step by boundary distance
            // Boundary detection handles crossing; limiting step causes particles to get stuck
            // Use the full coarse_step regardless of distance to boundary
            float coarse_step_limited = coarse_step;

            // Convert geometric distance to path length for energy calculation
            float coarse_range_step = coarse_step_limited / mu_abs;

            // Energy loss for coarse range step
            float mean_dE = device_compute_energy_deposition(dlut, E, coarse_range_step);
            float dE = mean_dE;  // Coarse: use mean (no straggling)
            dE = fminf(dE, E);

            float E_new = E - dE;
            float edep = dE * weight;

            // Nuclear attenuation for coarse range step
            float w_rem, E_rem;
            float w_new = device_apply_nuclear_attenuation(weight, E, coarse_range_step, w_rem, E_rem);
            edep += E_rem;

            // Coarse MCS: use RMS angle (no random sampling for efficiency)
            float sigma_mcs = device_highland_sigma(E, coarse_range_step);
            // Apply RMS scattering as systematic angular spread
            // In coarse mode, we bias the angle toward the mean (zero deflection)
            // This represents the "average" trajectory
            float theta_new = theta;  // Coarse: no random scattering, just energy loss

            float mu_new = cosf(theta_new);
            float eta_new = sinf(theta_new);

            // Position update (use limited step to avoid boundary crossing)
            float x_new = x_cell + eta_new * coarse_step_limited;
            float z_new = z_cell + mu_new * coarse_step_limited;

            // Check boundary crossing FIRST (using unclamped position)
            int exit_face = device_determine_exit_face(x_cell, z_cell, x_new, z_new, dx, dz);

            // THEN clamp to cell bounds for emission calculations
            x_new = fmaxf(-half_dx, fminf(x_new, half_dx));
            z_new = fmaxf(-half_dz, fminf(z_new, half_dz));

            if (exit_face >= 0) {
                // CRITICAL FIX: Check E_new directly against cutoff BEFORE emitting
                // Binned phase space causes E to be "reset" to geometric mean,
                // so particles can never reach E <= 0.1 MeV if we check after binning.
                if (E_new <= 0.1f) {
                    // Particle should be absorbed - deposit remaining energy locally
                    cell_edep += edep + E_new * w_new;  // Remaining energy from step
                    cell_w_cutoff += w_new;
                } else {
                    // DEBUG: See which particles are crossing boundaries - ALWAYS print for key cells
                    int z_cell_idx = cell / 200;
                    if (cell % 200 == 100 && z_cell_idx < 10) {
                        printf("K2 CROSS: cell=%d (z=%d), face=%d, E_new=%.3f, w=%.6e, z_old=%.4f, z_new=%.4f\n", cell, z_cell_idx, exit_face, E_new, w_new, z_cell, z_new);
                    }
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
                    // COARSE-ONLY FIX: Use single-bin emission instead of interpolation
                    // Interpolation causes exponential particle splitting, leading to
                    // weights of ~10^-18 after 60 iterations. For coarse transport,
                    // single-bin emission is sufficient and prevents fragmentation.
                    device_emit_component_to_bucket_4d(
                        bucket, theta_new, E_new, w_new, x_sub_neighbor, z_sub_neighbor,
                        theta_edges, E_edges, N_theta, N_E,
                        N_theta_local, N_E_local
                    );

                    // FIX: Deposit energy in current cell before particle leaves
                    // Both electronic (dE * weight) and nuclear (E_rem) energy are
                    // deposited locally in this cell, not carried across boundary.
                    cell_edep += edep;
                    cell_w_nuclear += w_rem;
                    cell_E_nuclear += E_rem;

                    // Account for energy/weight carried out by surviving particle
                    cell_boundary_weight += w_new;
                    cell_boundary_energy += E_new * w_new;
                }
            } else {
                // CRITICAL FIX: Particle remains in cell - MUST write to output phase space!
                // Previously: particles were lost if they didn't cross boundaries
                cell_edep += edep;
                cell_w_nuclear += w_rem;
                cell_E_nuclear += E_rem;

                // Cutoff check - don't write to output if below cutoff
                if (E_new <= 0.1f) {
                    cell_edep += E_new * w_new;
                    cell_w_cutoff += w_new;
                } else {
                    // DEBUG: Track particles at different depths - ALWAYS print for key cells
                    // cell = z * Nx + x, with Nx = 200
                    // cell 100: z=0, x=100 (source)
                    // cell 300: z=1, x=100
                    // cell 500: z=2, x=100
                    int z_cell_idx = cell / 200;
                    if (cell % 200 == 100 && z_cell_idx < 10) {
                        printf("K2 STAY: cell=%d (z=%d), E=%.3f, w=%.6e, z_pos=%.4f, stay_reason=boundary\n", cell, z_cell_idx, E_new, w_new, z_new);
                    }
                    // CRITICAL: Write particle to output phase space so it persists!
                    // Get updated position in centered coordinates
                    int x_sub = get_x_sub_bin(x_new, dx);
                    int z_sub = get_z_sub_bin(z_new, dz);

                    // Find theta and E bins for new energy/angle
                    float theta_min = theta_edges[0];
                    float theta_max = theta_edges[N_theta];
                    float dtheta = (theta_max - theta_min) / N_theta;
                    int theta_bin_new = static_cast<int>((theta_new - theta_min) / dtheta);
                    theta_bin_new = fmaxf(0, fminf(theta_bin_new, N_theta - 1));

                    float log_E_min = logf(E_edges[0]);
                    float log_E_max = logf(E_edges[N_E]);
                    float dlog = (log_E_max - log_E_min) / N_E;
                    int E_bin_new = static_cast<int>((logf(E_new) - log_E_min) / dlog);
                    E_bin_new = fmaxf(0, fminf(E_bin_new, N_E - 1));

                    // DEBUG: Track binning for all low energies
                    if (E_new < 12.0f) {
                        uint32_t b_E_new_debug = static_cast<uint32_t>(E_bin_new) / N_E_local;
                        printf("K2 WRITE LOW: cell=%d, E_new=%.3f, log_E=%.3f, E_bin=%d, b_E=%u\n",
                               cell, E_new, logf(E_new), E_bin_new, b_E_new_debug);
                    }

                    // Compute new block_id
                    uint32_t b_theta_new = static_cast<uint32_t>(theta_bin_new) / N_theta_local;
                    uint32_t b_E_new = static_cast<uint32_t>(E_bin_new) / N_E_local;
                    uint32_t bid_new = (b_E_new << 12) | (b_theta_new & 0xFFF);

                    // Find or allocate slot in output
                    constexpr int Kb = DEVICE_Kb;
                    int out_slot = -1;
                    for (int s = 0; s < Kb; ++s) {
                        uint32_t existing_bid = block_ids_out[cell * Kb + s];
                        if (existing_bid == bid_new) {
                            out_slot = s;
                            break;
                        }
                    }

                    // Allocate new slot if needed
                    if (out_slot < 0) {
                        for (int s = 0; s < Kb; ++s) {
                            uint32_t expected = DEVICE_EMPTY_BLOCK_ID;
                            if (atomicCAS(&block_ids_out[cell * Kb + s], expected, bid_new) == expected) {
                                out_slot = s;
                                break;
                            }
                        }
                    }

                    // Write weight to local bin
                    if (out_slot >= 0 && E_new > 0.1f) {
                        // Compute new local bin index
                        int theta_local_new = theta_bin_new % N_theta_local;
                        int E_local_new = E_bin_new % N_E_local;
                        int lidx_new = encode_local_idx_4d(theta_local_new, E_local_new, x_sub, z_sub);
                        int global_idx_out = (cell * Kb + out_slot) * DEVICE_LOCAL_BINS + lidx_new;
                        atomicAdd(&values_out[global_idx_out], w_new);
                        particles_out++;  // DEBUG: Count output particle
                    }
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
    const uint32_t* CoarseList,  // CRITICAL FIX: Pass CoarseList to kernel
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
    DeviceOutflowBucket* OutflowBuckets,
    uint32_t* block_ids_out,
    float* values_out
) {
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (n_coarse + threads_per_block - 1) / threads_per_block;

    K2_CoarseTransport<<<blocks, threads_per_block>>>(
        block_ids_in, values_in, ActiveMask, CoarseList,
        Nx, Nz, dx, dz, n_coarse,
        dlut,
        theta_edges, E_edges,
        N_theta, N_E, N_theta_local, N_E_local,
        config,
        EdepC, AbsorbedWeight_cutoff, AbsorbedWeight_nuclear, AbsorbedEnergy_nuclear,
        BoundaryLoss_weight, BoundaryLoss_energy,
        OutflowBuckets,
        block_ids_out, values_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling would go here
    }
}
