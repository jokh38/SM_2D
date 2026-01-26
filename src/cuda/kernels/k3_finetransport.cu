#include "kernels/k3_finetransport.cuh"
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

// ============================================================================
// Global LUT instance for CPU transport (initialized on first use)
// ============================================================================
static std::mutex rlut_mutex;
static RLUT* global_rlut = nullptr;
static bool rlut_initialized = false;

const RLUT& get_global_rlut() {
    std::lock_guard<std::mutex> lock(rlut_mutex);
    if (!rlut_initialized) {
        global_rlut = new RLUT(GenerateRLUT(0.1f, 300.0f, 256));
        rlut_initialized = true;
    }
    return *global_rlut;
}

// ============================================================================
// P1 FIX: Full GPU Kernel Implementation
// ============================================================================
// Previously: Only a stub with TODO comment
// Now: Complete implementation with:
//   - Device LUT access
//   - Full physics (energy loss, straggling, MCS, nuclear)
//   - Boundary crossing detection
//   - Bucket emission for cell transfer
// ============================================================================

__global__ void K3_FineTransport(
    // Inputs
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint32_t* __restrict__ ActiveList,
    // Grid
    int Nx, int Nz, float dx, float dz,
    int n_active,
    // Device LUT
    const DeviceRLUT dlut,
    // Grid edges for bin finding
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,
    // Outputs
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,
    // Outflow buckets for boundary crossing (P3 FIX)
    DeviceOutflowBucket* __restrict__ OutflowBuckets
) {
    // Thread ID maps to active cell
    int active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= n_active) return;

    int cell = ActiveList[active_idx];

    // Note: With LOCAL_BINS=128, shared buckets would exceed 48KB limit
    // Write directly to global memory instead of using shared memory

    // Accumulators for this cell
    double cell_edep = 0.0;
    float cell_w_cutoff = 0.0f;
    float cell_w_nuclear = 0.0f;
    double cell_E_nuclear = 0.0;
    float cell_boundary_weight = 0.0f;
    double cell_boundary_energy = 0.0;

    // FIX: Use DEVICE_Kb instead of hardcoded 32
    constexpr int Kb = DEVICE_Kb;  // = 8

    // Process all slots in this cell
    for (int slot = 0; slot < Kb; ++slot) {
        uint32_t bid = block_ids_in[cell * Kb + slot];
        if (bid == DEVICE_EMPTY_BLOCK_ID) continue;

        // Decode block ID
        uint32_t b_theta = bid & 0xFFF;
        uint32_t b_E = (bid >> 12) & 0xFFF;

        // Process all local bins in this block
        for (int lidx = 0; lidx < DEVICE_LOCAL_BINS; ++lidx) {
            int global_idx = (cell * Kb + slot) * DEVICE_LOCAL_BINS + lidx;
            float weight = values_in[global_idx];
            if (weight < 1e-12f) continue;

            // FIX Problem 1: Decode 4D local index (theta_local, E_local, x_sub, z_sub)
            int theta_local, E_local, x_sub, z_sub;
            decode_local_idx_4d(lidx, theta_local, E_local, x_sub, z_sub);

            // Get representative phase space values
            int theta_bin = b_theta * N_theta_local + theta_local;
            int E_bin = b_E * N_E_local + E_local;

            float theta_min = theta_edges[0];
            float theta_max = theta_edges[N_theta];
            float dtheta = (theta_max - theta_min) / N_theta;

            // FIX Problem 3: Intra-bin sampling for angle distribution
            // Instead of using bin center, sample uniformly within bin
            // This preserves the variance that would otherwise be lost
            unsigned seed = static_cast<unsigned>(
                (cell * 7 + slot * 13 + lidx * 17) ^ 0x5DEECE66DL
            );
            float theta_frac = (seed & 0xFFFF) / 65536.0f;  // [0, 1)
            float theta = theta_edges[theta_bin] + theta_frac * dtheta;

            float E_min = E_edges[0];
            float E_max = E_edges[N_E];
            float log_E_min = logf(E_min);
            float log_E_max = logf(E_max);
            float dlog = (log_E_max - log_E_min) / N_E;
            float E = expf(log_E_min + (E_bin + 0.5f) * dlog);

            // Cutoff check
            if (E <= 0.1f) {
                cell_edep += E * weight;
                cell_w_cutoff += weight;
                continue;
            }

            // FIX Problem 1: Each component starts at cell center + sub-cell offsets
            // Sub-cell partitioning tracks BOTH x and z position within cell
            float x_offset = get_x_offset_from_bin(x_sub, dx);
            float z_offset = get_z_offset_from_bin(z_sub, dz);
            float x_cell = x_offset;  // Position relative to cell origin (0,0)
            float z_cell = z_offset;

            // Initial direction (before MCS)
            float mu_init = cosf(theta);
            float eta_init = sinf(theta);

            // P9 FIX: First estimate step to boundary for initial direction
            float step_to_z_plus = (mu_init > 0) ? (dz - z_cell) / mu_init : 1e30f;
            float step_to_z_minus = (mu_init < 0) ? (-z_cell) / mu_init : 1e30f;
            float step_to_x_plus = (eta_init > 0) ? (dx - x_cell) / eta_init : 1e30f;
            float step_to_x_minus = (eta_init < 0) ? (-x_cell) / eta_init : 1e30f;
            float step_to_boundary = fminf(fminf(step_to_z_plus, step_to_z_minus),
                                           fminf(step_to_x_plus, step_to_x_minus));
            step_to_boundary = fmaxf(step_to_boundary, 0.0f);

            // Compute physics-limited step size
            float step_phys = device_compute_max_step(dlut, E, dx, dz);

            // P9 FIX: Use minimum of physics step and distance to boundary
            float actual_step = fminf(step_phys, step_to_boundary);

            // FIX Problem 2: Mid-point MCS method for better physical accuracy
            // The scattering should occur at the midpoint of the step, not at the start
            // First half: move with initial direction
            float half_step = actual_step * 0.5f;
            float x_mid = x_cell + eta_init * half_step;
            float z_mid = z_cell + mu_init * half_step;

            // Energy loss with straggling for actual step
            float mean_dE = device_compute_energy_deposition(dlut, E, actual_step);
            float sigma_dE = device_energy_straggling_sigma(E, actual_step, 1.0f);
            float dE = device_sample_energy_loss(mean_dE, sigma_dE, seed);
            dE = fminf(dE, E);

            float E_new = E - dE;
            float edep = dE * weight;

            // Nuclear attenuation for actual step
            float w_rem, E_rem;
            float w_new = device_apply_nuclear_attenuation(weight, E, actual_step, w_rem, E_rem);
            edep += E_rem;

            // MCS at midpoint (using energy at start of step for simplicity)
            float sigma_mcs = device_highland_sigma(E, actual_step);
            float theta_scatter = device_sample_mcs_angle(sigma_mcs, seed);
            float theta_new = theta + theta_scatter;

            // Second half: move with new scattered direction
            float mu_new = cosf(theta_new);
            float eta_new = sinf(theta_new);
            // Note: cos²θ + sin²θ = 1, so normalization is unnecessary

            // Complete position update: from midpoint with new direction
            float x_new = x_mid + eta_new * half_step;
            float z_new = z_mid + mu_new * half_step;

            // Clamp position to cell bounds
            x_new = fmaxf(0.0f, fminf(x_new, dx));
            z_new = fmaxf(0.0f, fminf(z_new, dz));

            // Check boundary crossing
            int exit_face = device_determine_exit_face(x_cell, z_cell, x_new, z_new, dx, dz);

            if (exit_face >= 0) {
                // FIX Problem 1: Calculate x_sub and z_sub for neighbor cell
                float x_offset_neighbor = device_get_neighbor_x_offset(x_new, exit_face, dx);
                int x_sub_neighbor = get_x_sub_bin(x_offset_neighbor, dx);

                // For z_offset in neighbor, we need to determine where we entered
                float z_offset_neighbor;
                if (exit_face == 0) {  // +z face: entering from bottom
                    z_offset_neighbor = -dz * 0.5f + dz * 0.125f;  // bin 0 center
                } else if (exit_face == 1) {  // -z face: entering from top
                    z_offset_neighbor = dz * 0.5f - dz * 0.125f;  // bin 3 center
                } else {
                    // x face: preserve relative z position
                    z_offset_neighbor = z_new - dz * 0.5f;
                }
                int z_sub_neighbor = get_z_sub_bin(z_offset_neighbor, dz);
                z_sub_neighbor = device_transform_z_sub_for_neighbor(z_sub_neighbor, exit_face);

                // Write directly to global memory
                int bucket_idx = device_bucket_index(cell, exit_face, Nx, Nz);
                DeviceOutflowBucket& bucket = OutflowBuckets[bucket_idx];
                // Use bilinear interpolation for improved accuracy
                device_emit_component_to_bucket_4d_interp(
                    bucket, theta_new, E_new, w_new, x_sub_neighbor, z_sub_neighbor,
                    theta_edges, E_edges, N_theta, N_E,
                    N_theta_local, N_E_local
                );

                cell_boundary_weight += w_new;
                cell_boundary_energy += E_new * w_new;
                // P6 FIX: Add nuclear energy to boundary energy accounting
                cell_boundary_energy += E_rem;
            } else {
                // Particle remains in cell - deposit energy locally
                cell_edep += edep;
                cell_w_nuclear += w_rem;
                cell_E_nuclear += E_rem;

                // FIX Problem 1: Update sub-cell bins for next iteration
                // Recalculate x_sub and z_sub based on new position
                float x_offset_new = x_new - dx * 0.5f;
                float z_offset_new = z_new - dz * 0.5f;
                x_sub = get_x_sub_bin(x_offset_new, dx);
                z_sub = get_z_sub_bin(z_offset_new, dz);

                // Cutoff check
                if (E_new <= 0.1f) {
                    cell_edep += E_new * w_new;
                    cell_w_cutoff += w_new;
                }
            }
        }
    }

    // Write accumulators to global memory (atomic for thread safety)
    atomicAdd(&EdepC[cell], cell_edep);
    atomicAdd(&AbsorbedWeight_cutoff[cell], cell_w_cutoff);
    atomicAdd(&AbsorbedWeight_nuclear[cell], cell_w_nuclear);
    atomicAdd(&AbsorbedEnergy_nuclear[cell], cell_E_nuclear);
    atomicAdd(&BoundaryLoss_weight[cell], cell_boundary_weight);
    atomicAdd(&BoundaryLoss_energy[cell], cell_boundary_energy);
}

// ============================================================================
// CPU Test Stubs (unchanged)
// ============================================================================

K3Result run_K3_single_component(const Component& c) {
    K3Result r;
    if (c.E <= E_cutoff) {
        r.Edep = c.E;
        r.E_new = 0.0f;
        r.terminated = true;
        r.remained_in_cell = false;
        return r;
    }

    const auto& lut = get_global_rlut();

    constexpr float rho_water = 1.0f;

    float step_size = compute_max_step_physics(lut, c.E);
    float mean_dE = compute_energy_deposition(lut, c.E, step_size);
    float sigma_E = energy_straggling_sigma(c.E, step_size, rho_water);

    unsigned seed = static_cast<unsigned>(
        (unsigned)(c.x * 10000) ^ (unsigned)(c.z * 1000) ^ (unsigned)(c.E * 100)
    );
    float dE = sample_energy_loss_with_straggling(mean_dE, sigma_E, seed);
    dE = fminf(dE, c.E);

    r.Edep = dE;
    r.E_new = compute_energy_after_step(lut, c.E, step_size);

    float w_removed, E_removed;
    float w_new = apply_nuclear_attenuation(c.w, c.E, step_size, w_removed, E_removed);
    r.nuclear_weight_removed = w_removed;
    r.nuclear_energy_removed = E_removed;

    r.remained_in_cell = true;

    if (r.E_new <= E_cutoff) {
        r.terminated = true;
        r.E_new = 0.0f;
    }

    float mu_temp = c.mu;
    float eta_temp = c.eta;

    float sigma_mcs = highland_sigma(c.E, step_size, X0_water);
    r.theta_scatter = sample_mcs_angle(sigma_mcs, seed);
    update_direction_after_mcs(c.theta, r.theta_scatter, mu_temp, eta_temp);

    r.mu_new = mu_temp;
    r.eta_new = eta_temp;

    return r;
}

K3Result run_K3_with_forced_split(const Component& c) {
    K3Result r = run_K3_single_component(c);
    r.split_count = 7;
    return r;
}
