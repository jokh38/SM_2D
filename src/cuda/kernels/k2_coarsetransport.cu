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
#include <stdexcept>
#include <string>

// ============================================================================
// Debug Measurements for Weight/Variance Conservation (PLAN_MCS Phase A-5)
// ============================================================================
// Set to 1 to enable debug output for weight and variance tracking
#define DEBUG_MCS_CONSERVATION 0
// ============================================================================

// PLAN_MCS Configuration Constants
// Sigma-based spread radius calculation (Phase A-4)
constexpr float K_SIGMA_SPREAD = 3.0f;  // Cover ±3σ (99.7% of Gaussian)
constexpr int MIN_SPREAD_RADIUS = 1;    // Minimum radius (cells)
constexpr int MAX_SPREAD_RADIUS = 50;   // Maximum radius to limit exponential growth

// PLAN_MCS Phase B-5: Moment-based K2 to K3 transition criteria
// These thresholds define when particles should transfer from coarse (K2) to fine (K3) transport
//
// K2 remains valid when:
//   1. sqrt(A) < THETA_K2_MAX           : Angular spread is small
//   2. sqrt(C)/dx < SIGMA_X_MAX_BINS    : Lateral spread fits in reasonable bins
//   3. sqrt(A) * step < SMALL_ANGLE_MAX : Small-angle approximation valid
//
// If any condition is violated, particle should be flagged for K3 transfer
//
// NOTE: The actual transfer is managed by ActiveMask on the host side.
// This kernel calculates moments but does not directly perform K2->K3 transfers.
// Future enhancement: Transfer moment_A, moment_B, moment_C to K3 state.
//
constexpr float THETA_K2_MAX = 0.02f;      // 20 mrad threshold for angular spread
constexpr float SIGMA_X_MAX_BINS = 3.0f;   // Maximum sigma_x in bin widths
constexpr float SMALL_ANGLE_MAX = 0.1f;    // Small-angle approximation limit

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
#if DEBUG_MCS_CONSERVATION
    // Debug outputs for conservation tracking (Phase A-5)
    , float* __restrict__ debug_weight_in
    , float* __restrict__ debug_weight_out
    , float* __restrict__ debug_variance_in
    , float* __restrict__ debug_variance_out
#endif
) {
    // Thread ID maps to coarse cell index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // CRITICAL FIX: Use CoarseList directly instead of scanning ActiveMask
    // This changes from O(N_cells * n_coarse) to O(n_coarse)
    if (idx >= n_coarse) return;

    int cell = CoarseList[idx];

    if (cell < 0 || cell >= Nx * Nz) return;

    // Accumulators for this cell
    constexpr int Kb = DEVICE_Kb;
    double cell_edep = 0.0;
    float cell_w_cutoff = 0.0f;
    float cell_w_nuclear = 0.0f;
    double cell_E_nuclear = 0.0f;
    float cell_boundary_weight = 0.0f;
    double cell_boundary_energy = 0.0f;

#if DEBUG_MCS_CONSERVATION
    // Debug accumulators for Phase A-5: weight and variance conservation
    float debug_w_in = 0.0f;
    float debug_w_out = 0.0f;
    float debug_var_in = 0.0f;  // ⟨x²⟩ variance before transport
    float debug_var_out = 0.0f;  // ⟨x²⟩ variance after transport
#endif

    // Coarse step size: limit to cell size for accurate dose deposition
    // IMPORTANT: Coarse transport only supports single-cell transfers per iteration
    // Using step > cell size causes particles to "jump" over cells without proper transport
    float coarse_step = fminf(config.step_coarse, fminf(dx, dz));

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

#if DEBUG_MCS_CONSERVATION
            // Track input weight for conservation check (Phase A-5)
            debug_w_in += weight;
            // Initial x² variance (before transport) - centered in cell
            float x_in_variance = x_offset * x_offset;
            debug_var_in += weight * x_in_variance;
#endif

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

            // CRITICAL: Use the same energy calculation as K3 for consistency
            // K3 uses lower edge + 20% of half-width to ensure energy actually decreases
            // K2 must use the same calculation to avoid energy jumps when transitioning
            float E_lower = E_edges[E_bin];
            float E_upper = E_edges[E_bin + 1];
            float E_half_width = (E_upper - E_lower) * 0.5f;
            float E = E_lower + 0.50f * E_half_width;  // 50% of half-width from lower edge

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

            // ========================================================================
            // PLAN_MCS Phase A & B: Fermi-Eyges Moment Tracking + Lateral Spreading
            // ========================================================================
            // IMPLEMENTATION (A-4, B-1, B-4):
            // 1. Track Fermi-Eyges moments (A, B, C) for each particle
            // 2. Use accumulated C moment for sigma_x = sqrt(C) (B-4)
            // 3. Calculate sigma-based spread radius to limit exponential growth (A-4)
            // 4. Apply lateral spreading with calculated radius
            // ========================================================================

            // Initialize Fermi-Eyges moments for this particle
            // Note: In a full implementation, these would be accumulated across iterations
            // For K2 coarse transport, we calculate per-step moments
            float moment_A = 0.0f;  // ⟨θ²⟩ - Angular variance
            float moment_B = 0.0f;  // ⟨xθ⟩ - Position-angle covariance
            float moment_C = 0.0f;  // ⟨x²⟩ - Position variance

            // Calculate scattering power T for this step
            float T = device_scattering_power_T(E, coarse_step_limited);

            // Update Fermi-Eyges moments for this step
            device_fermi_eyges_step(moment_A, moment_B, moment_C, T, coarse_step_limited);

            // Calculate sigma_x from accumulated C moment (B-4)
            float sigma_x = device_accumulated_sigma_x(moment_C);

            // PLAN_MCS Phase B-5: Check moment-based K2->K3 transition criteria
            // K2 validity check using accumulated moments
            float sqrt_A = device_accumulated_sigma_theta(moment_A);
            float sigma_x_bins = sigma_x / dx;

            // K2 remains valid only if all conditions are met
            bool k2_valid =
                (sqrt_A < THETA_K2_MAX) &&              // Angular spread < 20 mrad
                (sigma_x_bins < SIGMA_X_MAX_BINS) &&    // Lateral spread fits in bins
                (sqrt_A * coarse_step_limited < SMALL_ANGLE_MAX);  // Small-angle valid

            // Note: If k2_valid is false, particle should be flagged for K3 transfer
            // This is handled externally by ActiveMask, not within this kernel

            // Calculate sigma-based spread radius (A-4)
            // radius = k_sigma * (sigma_x / dx), clamped to [MIN, MAX]
            float radius_sigma = sigma_x / dx;
            int spread_radius = static_cast<int>(ceilf(K_SIGMA_SPREAD * radius_sigma));
            spread_radius = max(spread_radius, MIN_SPREAD_RADIUS);
            spread_radius = min(spread_radius, min(MAX_SPREAD_RADIUS, Nx / 2));

            // Ensure even number for symmetric spreading
            if (spread_radius % 2 != 0) spread_radius++;

            // Theta remains unchanged (no random sampling in K2)
            float theta_new = theta;
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
                    // ====================================================================
                    // PLAN_MCS Phase A-4: Sigma-Based Lateral Spreading
                    // ====================================================================
                    // Use sigma-based radius for lateral spreading:
                    // - Calculate sigma_x from Fermi-Eyges C moment
                    // - Determine spread radius: R = k_sigma * (sigma_x / dx)
                    // - Clamp R to [MIN_SPREAD_RADIUS, MAX_SPREAD_RADIUS]
                    // - Use lateral spreading only for +z direction (primary beam)
                    // ====================================================================

                    if (exit_face == FACE_Z_PLUS) {
                        // Forward emission with lateral spreading
                        int iz_source = cell / Nx;
                        int target_z = iz_source + 1;

                        // Use lateral spreading with sigma-based radius
                        device_emit_lateral_spread(
                            OutflowBuckets,
                            cell,
                            target_z,
                            theta_new,
                            E_new,
                            w_new,
                            x_new,  // Lateral position in cell-centered coordinates
                            sigma_x,
                            dx,
                            Nx, Nz,
                            theta_edges, E_edges,
                            N_theta, N_E,
                            N_theta_local, N_E_local,
                            spread_radius  // Sigma-based radius
                        );
                    } else {
                        // Other faces - single-cell emission (no lateral spreading)
                        float x_offset_neighbor = device_get_neighbor_x_offset(x_new, exit_face, dx);
                        int x_sub_neighbor = get_x_sub_bin(x_offset_neighbor, dx);

                        float z_offset_neighbor;
                        if (exit_face == FACE_Z_MINUS) {
                            z_offset_neighbor = -dz * 0.5f + dz * 0.125f;
                        } else if (exit_face == FACE_Z_PLUS) {
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
                    }

                    // FIX: Deposit energy in current cell before particle leaves
                    // Both electronic (dE * weight) and nuclear (E_rem) energy are
                    // deposited locally in this cell, not carried across boundary.
                    cell_edep += edep;
                    cell_w_nuclear += w_rem;
                    cell_E_nuclear += E_rem;

                    // Account for energy/weight carried out by surviving particle
                    cell_boundary_weight += w_new;
                    cell_boundary_energy += E_new * w_new;

#if DEBUG_MCS_CONSERVATION
                    // Track output variance for particles leaving cell (Phase A-5)
                    debug_w_out += w_new;
                    // x² variance after transport includes lateral spreading (sigma_x)
                    float x_out_variance = moment_C;  // ⟨x²⟩ from Fermi-Eyges
                    debug_var_out += w_new * x_out_variance;
#endif
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

                    // Option D2: Use binary search with E_edges for piecewise-uniform grid
                    int E_bin_new = device_find_bin_edges(E_edges, N_E, E_new);
                    E_bin_new = fmaxf(0, fminf(E_bin_new, N_E - 1));

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

#if DEBUG_MCS_CONSERVATION
                        // Track output variance for particles remaining in cell (Phase A-5)
                        debug_w_out += w_new;
                        float x_out_variance = moment_C;  // ⟨x²⟩ from Fermi-Eyges
                        debug_var_out += w_new * x_out_variance;
#endif
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

#if DEBUG_MCS_CONSERVATION
    // Write debug accumulators for conservation tracking (Phase A-5)
    if (debug_weight_in) atomicAdd(&debug_weight_in[cell], debug_w_in);
    if (debug_weight_out) atomicAdd(&debug_weight_out[cell], debug_w_out);
    if (debug_variance_in) atomicAdd(&debug_variance_in[cell], debug_var_in);
    if (debug_variance_out) atomicAdd(&debug_variance_out[cell], debug_var_out);
#endif
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
#if DEBUG_MCS_CONSERVATION
    , float* debug_weight_in
    , float* debug_weight_out
    , float* debug_variance_in
    , float* debug_variance_out
#endif
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
#if DEBUG_MCS_CONSERVATION
        , debug_weight_in
        , debug_weight_out
        , debug_variance_in
        , debug_variance_out
#endif
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("K2_CoarseTransport kernel failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}
