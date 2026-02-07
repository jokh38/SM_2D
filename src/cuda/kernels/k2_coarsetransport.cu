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

// Forward declaration for DeviceOutflowBucket (defined in device_bucket.cuh)
// This is already included via device_bucket.cuh

// ============================================================================
// Profiling Counters (ITERATION 3)
// ============================================================================
// Track moment-based enhancement statistics for verification
// Enable by defining ENABLE_MCS_PROFILING during compilation
// ============================================================================

#ifdef ENABLE_MCS_PROFILING
// Global counters for moment-based enhancement tracking
__device__ unsigned long long g_mcs_enhancement_count = 0;      // Number of enhancements applied
__device__ unsigned long long g_mcs_total_evaluations = 0;       // Total moment evaluations
__device__ unsigned long long g_mcs_sqrt_A_exceeds = 0;          // Count: sqrt(A) >= 0.02
__device__ unsigned long long g_mcs_sqrt_C_exceeds = 0;          // Count: sqrt(C)/dx >= 3.0
__device__ double g_mcs_total_sqrt_A = 0.0;                      // Accumulated sqrt(A) values
__device__ double g_mcs_total_sqrt_C_dx = 0.0;                   // Accumulated sqrt(C)/dx values
#endif

// Debug counters for output-slot allocation failures (K2).
__device__ unsigned long long g_k2_slot_drop_count = 0;
__device__ double g_k2_slot_drop_weight = 0.0;
__device__ double g_k2_slot_drop_energy = 0.0;
__device__ unsigned long long g_k2_bucket_drop_count = 0;
__device__ double g_k2_bucket_drop_weight = 0.0;
__device__ double g_k2_bucket_drop_energy = 0.0;

// ============================================================================
// K2 Coarse Transport: Energy Loss with Fermi-Eyges Moment-Based Lateral Spreading
// ============================================================================
// Purpose: Handle high-energy particles where ActiveMask=0
//
// Physics:
// - Energy loss with stopping power (dE/dx)
// - Nuclear attenuation (same as fine transport)
// - Fermi-Eyges moment-based lateral spreading (O(z^(3/2)) scaling)
// - Moment-based K2→K3 transition criteria (via enhancement)
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
    // FIX C: Initial beam width (now from config instead of hardcoded)
    float sigma_x_initial,
    // Outputs
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    double* __restrict__ AbsorbedEnergy_cutoff,
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

    if (cell < 0 || cell >= Nx * Nz) return;

    // Accumulators for this cell
    constexpr int Kb = DEVICE_Kb;
    double cell_edep = 0.0;
    float cell_w_cutoff = 0.0f;
    double cell_E_cutoff = 0.0;
    float cell_w_nuclear = 0.0f;
    double cell_E_nuclear = 0.0f;
    float cell_boundary_weight = 0.0f;
    double cell_boundary_energy = 0.0f;

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

            // CRITICAL: Use the same representative energy calculation as K3 for consistency.
            // K2 and K3 both use the bin center to avoid transition discontinuities.
            float E_lower = E_edges[E_bin];
            float E_upper = E_edges[E_bin + 1];
            float E_half_width = (E_upper - E_lower) * 0.5f;
            float E = E_lower + 1.00f * E_half_width;  // Bin center

            // Cutoff check
            if (E <= 0.1f) {
                cell_w_cutoff += weight;
                cell_E_cutoff += E * weight;
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

            // BUG FIX: Don't limit step by boundary distance.
            // Boundary detection handles crossing; limiting step causes particles to get stuck.
            float coarse_step_limited = coarse_step;

            // Convert geometric distance to path length for energy calculation
            float coarse_range_step = coarse_step_limited / mu_abs;

            // Crossing guard: if this coarse step would cross the fine threshold,
            // split at E_fine_on so K2 does not overshoot into low-energy transport.
            bool split_at_fine_threshold = false;
            if (E > config.E_fine_on + 1e-6f) {
                float R_current = device_lookup_R(dlut, E);
                float R_fine_on = device_lookup_R(dlut, fmaxf(config.E_fine_on, dlut.E_min));
                float range_to_fine = R_current - R_fine_on;

                if (range_to_fine > 0.0f && range_to_fine < coarse_range_step) {
                    coarse_range_step = range_to_fine;
                    coarse_step_limited = coarse_range_step * mu_abs;
                    split_at_fine_threshold = true;
                }
            }

            // Energy loss for coarse range step
            float mean_dE = device_compute_energy_deposition(dlut, E, coarse_range_step);
            float dE = mean_dE;  // Coarse: use mean (no straggling)
            dE = fminf(dE, E);

            float E_new = E - dE;
            if (split_at_fine_threshold) {
                E_new = fminf(E_new, config.E_fine_on);
            }
            float edep = dE * weight;

            // Nuclear attenuation for coarse range step
            float w_rem, E_rem;
            float w_new = device_apply_nuclear_attenuation(weight, E, coarse_range_step, w_rem, E_rem);
            edep += E_rem;

            // ========================================================================
            // K2 Coarse Transport: Fermi-Eyges Moment-Based Lateral Spreading
            // ========================================================================
            // FIX: Use depth-based sigma_x for correct Fermi-Eyges O(z^(3/2)) scaling
            // Previous implementation reset moments to zero each iteration, causing
            // sigma_x << dx and no lateral spreading. New calculation uses depth from surface.
            // ========================================================================

            // Calculate depth from surface for accumulated lateral spread
            int iz = cell / Nx;
            float depth_from_surface_mm = iz * dz + z_cell;  // Total depth from surface

            // Calculate accumulated lateral spread using Fermi-Eyges theory
            // sigma_x(z) ≈ theta_0 * z / sqrt(3) for thin target
            float sigma_theta_depth = device_highland_sigma(E, depth_from_surface_mm);
            float sigma_x_depth = sigma_theta_depth * depth_from_surface_mm / 1.7320508f;  // / sqrt(3)

            // Combine with initial beam width (FIX C: now from input parameter)
            // sigma_x_initial is passed from config (sim.ini sigma_x_mm)
            float sigma_x = sqrtf(sigma_x_initial * sigma_x_initial + sigma_x_depth * sigma_x_depth);

            // Ensure minimum sigma_x
            sigma_x = fmaxf(sigma_x, 0.01f);

            // ========================================================================
            // Iteration 2: Moment-Based Spreading Enhancement
            // ========================================================================
            // Implements design specification B-5 (moment-based K2→K3 criteria)
            // within architectural constraints by enhancing spreading when moments
            // exceed thresholds. This approximates K3 behavior without requiring
            // pipeline architecture changes.
            //
            // Design thresholds (PLAN_MCS.md B-5):
            //   sqrt_A = sqrt(⟨θ²⟩) < 0.02 rad (20 mrad)
            //   sqrt_C_over_dx = sqrt(⟨x²⟩) / dx < 3.0 bins
            //
            // When thresholds exceeded: Apply 2x spreading to simulate K3 transport
            //
            // ITERATION 3: Added profiling counters for verification
            // ========================================================================

            // Calculate sqrt_A from depth (angular spread at this depth)
            float sqrt_A = sigma_theta_depth;  // Already computed above
            float sqrt_C_over_dx = sigma_x / dx;  // σₓ in bin units

#ifdef ENABLE_MCS_PROFILING
            atomicAdd(&g_mcs_total_evaluations, 1);
            atomicAdd(&g_mcs_total_sqrt_A, sqrt_A);
            atomicAdd(&g_mcs_total_sqrt_C_dx, sqrt_C_over_dx);
            if (sqrt_A >= 0.02f) atomicAdd(&g_mcs_sqrt_A_exceeds, 1);
            if (sqrt_C_over_dx >= 3.0f) atomicAdd(&g_mcs_sqrt_C_exceeds, 1);
#endif

            // Moment-based validity check (design spec B-5)
            bool k2_moments_valid =
                (sqrt_A < 0.02f) &&           // θ_RMS < 20 mrad
                (sqrt_C_over_dx < 3.0f);      // σₓ < 3 bins

            // Apply moment-based spreading enhancement
            // When moments exceed K2 validity thresholds, enhance spreading
            if (!k2_moments_valid) {
#ifdef ENABLE_MCS_PROFILING
                atomicAdd(&g_mcs_enhancement_count, 1);
#endif
                // Enhance spreading to approximate K3 behavior
                // This compensates for not transferring to K3 by using wider sigma_x
                sigma_x *= 2.0f;  // 2x spreading for large moment cases
            }

            // Theta remains unchanged (direction doesn't change, weight spreads instead)
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
                    cell_edep += edep;
                    cell_w_cutoff += w_new;
                    cell_E_cutoff += E_new * w_new;  // Remaining energy from step
                } else {
                    // ====================================================================
                    // Single-cell emission (K2: deterministic lateral spreading)
                    // ====================================================================
                    // For boundary crossing, use simple emission. Lateral spreading
                    // is applied when particles remain in the cell (see else block).
                    // ====================================================================

                    float x_offset_neighbor = device_get_neighbor_x_offset(x_new, exit_face, dx);
                    int x_sub_neighbor = get_x_sub_bin(x_offset_neighbor, dx);

                    float z_offset_neighbor;
                    if (exit_face == FACE_Z_MINUS) {
                        z_offset_neighbor = -dz * 0.5f + dz * 0.125f;
                    } else if (exit_face == FACE_Z_PLUS) {
                        z_offset_neighbor = dz * 0.5f - dz * 0.125f;
                    } else {
                        z_offset_neighbor = z_new;
                    }
                    int z_sub_neighbor = get_z_sub_bin(z_offset_neighbor, dz);
                    z_sub_neighbor = device_transform_z_sub_for_neighbor(z_sub_neighbor, exit_face);

                    int bucket_idx = device_bucket_index(cell, exit_face, Nx, Nz);
                    DeviceOutflowBucket& bucket = OutflowBuckets[bucket_idx];
                    float dropped_boundary_weight = device_emit_component_to_bucket_4d(
                        bucket, theta_new, E_new, w_new, x_sub_neighbor, z_sub_neighbor,
                        theta_edges, E_edges, N_theta, N_E,
                        N_theta_local, N_E_local
                    );
                    if (dropped_boundary_weight > 0.0f) {
                        atomicAdd(&g_k2_bucket_drop_count, 1ULL);
                        atomicAdd(&g_k2_bucket_drop_weight, static_cast<double>(dropped_boundary_weight));
                        atomicAdd(&g_k2_bucket_drop_energy, static_cast<double>(E_new * dropped_boundary_weight));
                    }

                    // Deposit energy in current cell before particle leaves
                    cell_edep += edep;
                    cell_w_nuclear += w_rem;
                    cell_E_nuclear += E_rem;

                    // Only count as boundary loss if particle is leaving simulation domain.
                    if (device_get_neighbor(cell, exit_face, Nx, Nz) < 0) {
                        cell_boundary_weight += w_new;
                        cell_boundary_energy += E_new * w_new;
                    }
                }
            } else {
                // CRITICAL FIX: Particle remains in cell - MUST write to output phase space!
                // Previously: particles were lost if they didn't cross boundaries
                cell_edep += edep;
                cell_w_nuclear += w_rem;
                cell_E_nuclear += E_rem;

                // Cutoff check - don't write to output if below cutoff
                if (E_new <= 0.1f) {
                    cell_w_cutoff += w_new;
                    cell_E_cutoff += E_new * w_new;
                } else {
                    // ========================================================================
                    // DETERMINISTIC LATERAL SPREADING: Gaussian weight distribution
                    // ========================================================================
                    // The particle weight is distributed across x positions using a Gaussian
                    // distribution with sigma_x calculated from the Highland formula.
                    // This is NOT Monte Carlo - it's a deterministic weight distribution.
                    // ========================================================================

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

                    // ====================================================================
                    // Apply Gaussian lateral spreading within the cell
                    // FIX B: Use sub-cell spreading function for correct spatial scale
                    // ====================================================================
                    // Distribute weight across x_sub bins using Gaussian distribution
                    // with proper sub-cell spacing (dx/8 instead of dx)
                    // ====================================================================

                    if (out_slot < 0 && E_new > 0.1f) {
                        atomicAdd(&g_k2_slot_drop_count, 1ULL);
                        atomicAdd(&g_k2_slot_drop_weight, static_cast<double>(w_new));
                        atomicAdd(&g_k2_slot_drop_energy, static_cast<double>(E_new * w_new));
                        continue;
                    }

                    if (E_new > 0.1f) {
                        // FIX B: Use sub-cell Gaussian spreading with correct dx/8 spacing
                        constexpr int N_x_sub = 8;  // Number of sub-cell bins

                        // Calculate weights for each sub-bin using Gaussian CDF
                        float weights[N_x_sub];
                        float x_center = x_new;  // Center of Gaussian within cell

                        device_gaussian_spread_weights_subcell(weights, x_center, sigma_x, dx, N_x_sub);

                        // Get cell coordinates
                        int ix = cell % Nx;
                        int z_sub_target = get_z_sub_bin(z_new, dz);

                        // FIX: For large sigma_x, we must account for weight that goes to neighbors
                        // The fraction of the Gaussian distribution within the cell
                        float left_boundary = -dx * 0.5f;
                        float right_boundary = dx * 0.5f;
                        float w_in_cell = device_gaussian_cdf(right_boundary, x_center, sigma_x) -
                                         device_gaussian_cdf(left_boundary, x_center, sigma_x);

                        // Scale weights by the fraction within the cell to conserve total weight
                        // (device_gaussian_spread_weights_subcell normalizes to sum=1.0)
                        float w_cell_fraction = fminf(1.0f, fmaxf(0.0f, w_in_cell));  // Clamp for safety

                        // Normalize sub-cell weights in non-negative space for robustness.
                        float w_sub_sum = 0.0f;
                        for (int x_sub_target = 0; x_sub_target < N_x_sub; ++x_sub_target) {
                            weights[x_sub_target] = fmaxf(0.0f, weights[x_sub_target]);
                            w_sub_sum += weights[x_sub_target];
                        }
                        if (w_sub_sum < 1e-12f) {
                            for (int x_sub_target = 0; x_sub_target < N_x_sub; ++x_sub_target) {
                                weights[x_sub_target] = 0.0f;
                            }
                            weights[N_x_sub / 2] = 1.0f;
                            w_sub_sum = 1.0f;
                        }

                        // Distribute weight across all x_sub bins
                        for (int x_sub_target = 0; x_sub_target < N_x_sub; ++x_sub_target) {
                            float w_frac = (weights[x_sub_target] / w_sub_sum) * w_cell_fraction;
                            if (w_frac < 1e-10f) continue;  // Skip negligible weights

                            float w_spread = w_new * w_frac;

                            // Compute new local bin index
                            int theta_local_new = theta_bin_new % N_theta_local;
                            int E_local_new = E_bin_new % N_E_local;
                            int lidx_new = encode_local_idx_4d(theta_local_new, E_local_new, x_sub_target, z_sub_target);

                            // Write to output phase space (all within same cell)
                            int global_idx_out = (cell * Kb + out_slot) * DEVICE_LOCAL_BINS + lidx_new;
                            atomicAdd(&values_out[global_idx_out], w_spread);
                        }

                        // FIX B extended: For large sigma_x (when spread exceeds cell boundary),
                        // also emit to neighbor cells via buckets
                        if (w_cell_fraction < 1.0f - 1e-6f) {
                            // Left/right tail weights from Gaussian CDF.
                            // Re-normalize to guarantee: w_cell_fraction + w_left + w_right = 1.
                            float w_left_raw = fmaxf(0.0f, device_gaussian_cdf(left_boundary, x_center, sigma_x));
                            float w_right_raw = fmaxf(0.0f, 1.0f - device_gaussian_cdf(right_boundary, x_center, sigma_x));
                            float w_tail_total = fmaxf(0.0f, 1.0f - w_cell_fraction);
                            float w_tail_sum = w_left_raw + w_right_raw;
                            float w_left = 0.0f;
                            float w_right = 0.0f;
                            if (w_tail_total > 0.0f && w_tail_sum > 1e-12f) {
                                float tail_scale = w_tail_total / w_tail_sum;
                                w_left = w_left_raw * tail_scale;
                                w_right = w_right_raw * tail_scale;
                            }

                            // Emit left tail to left neighbor (or boundary if no neighbor)
                            if (w_left > 1e-6f) {
                                float w_spread_left = w_new * w_left;
                                if (ix > 0) {
                                    int lateral_face = FACE_X_MINUS;
                                    int x_sub_neighbor = N_x_sub - 1;  // Rightmost bin of left neighbor
                                    int z_sub_neighbor = get_z_sub_bin(z_new, dz);

                                    int bucket_idx = device_bucket_index(cell, lateral_face, Nx, Nz);
                                    DeviceOutflowBucket& lateral_bucket = OutflowBuckets[bucket_idx];

                                    float dropped_left_tail = device_emit_component_to_bucket_4d(
                                        lateral_bucket, theta_new, E_new, w_spread_left,
                                        x_sub_neighbor, z_sub_neighbor,
                                        theta_edges, E_edges, N_theta, N_E,
                                        N_theta_local, N_E_local
                                    );
                                    if (dropped_left_tail > 0.0f) {
                                        atomicAdd(&g_k2_bucket_drop_count, 1ULL);
                                        atomicAdd(&g_k2_bucket_drop_weight, static_cast<double>(dropped_left_tail));
                                        atomicAdd(&g_k2_bucket_drop_energy, static_cast<double>(E_new * dropped_left_tail));
                                    }
                                } else {
                                    // No left neighbor: treat lateral tail as domain boundary loss.
                                    cell_boundary_weight += w_spread_left;
                                    cell_boundary_energy += E_new * w_spread_left;
                                }
                            }

                            // Emit right tail to right neighbor (or boundary if no neighbor)
                            if (w_right > 1e-6f) {
                                float w_spread_right = w_new * w_right;
                                if (ix < Nx - 1) {
                                    int lateral_face = FACE_X_PLUS;
                                    int x_sub_neighbor = 0;  // Leftmost bin of right neighbor
                                    int z_sub_neighbor = get_z_sub_bin(z_new, dz);

                                    int bucket_idx = device_bucket_index(cell, lateral_face, Nx, Nz);
                                    DeviceOutflowBucket& lateral_bucket = OutflowBuckets[bucket_idx];

                                    float dropped_right_tail = device_emit_component_to_bucket_4d(
                                        lateral_bucket, theta_new, E_new, w_spread_right,
                                        x_sub_neighbor, z_sub_neighbor,
                                        theta_edges, E_edges, N_theta, N_E,
                                        N_theta_local, N_E_local
                                    );
                                    if (dropped_right_tail > 0.0f) {
                                        atomicAdd(&g_k2_bucket_drop_count, 1ULL);
                                        atomicAdd(&g_k2_bucket_drop_weight, static_cast<double>(dropped_right_tail));
                                        atomicAdd(&g_k2_bucket_drop_energy, static_cast<double>(E_new * dropped_right_tail));
                                    }
                                } else {
                                    // No right neighbor: treat lateral tail as domain boundary loss.
                                    cell_boundary_weight += w_spread_right;
                                    cell_boundary_energy += E_new * w_spread_right;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Write accumulators
    atomicAdd(&EdepC[cell], cell_edep);
    atomicAdd(&AbsorbedWeight_cutoff[cell], cell_w_cutoff);
    atomicAdd(&AbsorbedEnergy_cutoff[cell], cell_E_cutoff);
    atomicAdd(&AbsorbedWeight_nuclear[cell], cell_w_nuclear);
    atomicAdd(&AbsorbedEnergy_nuclear[cell], cell_E_nuclear);
    atomicAdd(&BoundaryLoss_weight[cell], cell_boundary_weight);
    atomicAdd(&BoundaryLoss_energy[cell], cell_boundary_energy);
}

void k2_reset_debug_counters() {
    constexpr unsigned long long zero_count = 0ULL;
    constexpr double zero_value = 0.0;
    cudaMemcpyToSymbol(g_k2_slot_drop_count, &zero_count, sizeof(zero_count));
    cudaMemcpyToSymbol(g_k2_slot_drop_weight, &zero_value, sizeof(zero_value));
    cudaMemcpyToSymbol(g_k2_slot_drop_energy, &zero_value, sizeof(zero_value));
    cudaMemcpyToSymbol(g_k2_bucket_drop_count, &zero_count, sizeof(zero_count));
    cudaMemcpyToSymbol(g_k2_bucket_drop_weight, &zero_value, sizeof(zero_value));
    cudaMemcpyToSymbol(g_k2_bucket_drop_energy, &zero_value, sizeof(zero_value));
}

void k2_get_debug_counters(
    unsigned long long& slot_drop_count,
    double& slot_drop_weight,
    double& slot_drop_energy,
    unsigned long long& bucket_drop_count,
    double& bucket_drop_weight,
    double& bucket_drop_energy
) {
    cudaMemcpyFromSymbol(&slot_drop_count, g_k2_slot_drop_count, sizeof(slot_drop_count));
    cudaMemcpyFromSymbol(&slot_drop_weight, g_k2_slot_drop_weight, sizeof(slot_drop_weight));
    cudaMemcpyFromSymbol(&slot_drop_energy, g_k2_slot_drop_energy, sizeof(slot_drop_energy));
    cudaMemcpyFromSymbol(&bucket_drop_count, g_k2_bucket_drop_count, sizeof(bucket_drop_count));
    cudaMemcpyFromSymbol(&bucket_drop_weight, g_k2_bucket_drop_weight, sizeof(bucket_drop_weight));
    cudaMemcpyFromSymbol(&bucket_drop_energy, g_k2_bucket_drop_energy, sizeof(bucket_drop_energy));
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
    float sigma_x_initial,  // FIX C: Initial beam width from config
    double* EdepC,
    float* AbsorbedWeight_cutoff,
    double* AbsorbedEnergy_cutoff,
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
        sigma_x_initial,  // FIX C: Pass initial beam width
        EdepC, AbsorbedWeight_cutoff, AbsorbedEnergy_cutoff, AbsorbedWeight_nuclear, AbsorbedEnergy_nuclear,
        BoundaryLoss_weight, BoundaryLoss_energy,
        OutflowBuckets,
        block_ids_out, values_out
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("K2_CoarseTransport kernel failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

// ============================================================================
// Profiling Helper Functions (ITERATION 3)
// ============================================================================

#ifdef ENABLE_MCS_PROFILING

/**
 * @brief Reset profiling counters to zero
 *
 * Call this before starting a simulation to clear previous profiling data.
 */
void k2_reset_profiling_counters() {
    unsigned long long zero_ull = 0;
    double zero_d = 0.0;

    cudaMemcpyToSymbol(g_mcs_enhancement_count, &zero_ull, sizeof(unsigned long long));
    cudaMemcpyToSymbol(g_mcs_total_evaluations, &zero_ull, sizeof(unsigned long long));
    cudaMemcpyToSymbol(g_mcs_sqrt_A_exceeds, &zero_ull, sizeof(unsigned long long));
    cudaMemcpyToSymbol(g_mcs_sqrt_C_exceeds, &zero_ull, sizeof(unsigned long long));
    cudaMemcpyToSymbol(g_mcs_total_sqrt_A, &zero_d, sizeof(double));
    cudaMemcpyToSymbol(g_mcs_total_sqrt_C_dx, &zero_d, sizeof(double));
}

/**
 * @brief Retrieve profiling counters from device
 *
 * @param enhancement_count Number of enhancements applied
 * @param total_evaluations Total moment evaluations
 * @param sqrt_A_exceeds Count: sqrt(A) >= 0.02
 * @param sqrt_C_exceeds Count: sqrt(C)/dx >= 3.0
 * @param avg_sqrt_A Average sqrt(A) value
 * @param avg_sqrt_C_dx Average sqrt(C)/dx value
 */
void k2_get_profiling_counters(
    unsigned long long& enhancement_count,
    unsigned long long& total_evaluations,
    unsigned long long& sqrt_A_exceeds,
    unsigned long long& sqrt_C_exceeds,
    double& avg_sqrt_A,
    double& avg_sqrt_C_dx
) {
    unsigned long long total_sqrt_A_count = 0;
    unsigned long long total_sqrt_C_count = 0;
    double sum_sqrt_A = 0.0;
    double sum_sqrt_C_dx = 0.0;

    cudaMemcpyFromSymbol(&enhancement_count, g_mcs_enhancement_count, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&total_evaluations, g_mcs_total_evaluations, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&sqrt_A_exceeds, g_mcs_sqrt_A_exceeds, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&sqrt_C_exceeds, g_mcs_sqrt_C_exceeds, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&sum_sqrt_A, g_mcs_total_sqrt_A, sizeof(double));
    cudaMemcpyFromSymbol(&sum_sqrt_C_dx, g_mcs_total_sqrt_C_dx, sizeof(double));

    // Calculate averages
    avg_sqrt_A = (total_evaluations > 0) ? sum_sqrt_A / total_evaluations : 0.0;
    avg_sqrt_C_dx = (total_evaluations > 0) ? sum_sqrt_C_dx / total_evaluations : 0.0;
}

/**
 * @brief Print profiling summary to stdout
 *
 * Call this after simulation completes to see moment-based enhancement statistics.
 */
void k2_print_profiling_summary() {
    unsigned long long enhancement_count, total_evaluations, sqrt_A_exceeds, sqrt_C_exceeds;
    double avg_sqrt_A, avg_sqrt_C_dx;

    k2_get_profiling_counters(
        enhancement_count, total_evaluations,
        sqrt_A_exceeds, sqrt_C_exceeds,
        avg_sqrt_A, avg_sqrt_C_dx
    );

    std::cout << "\n=== K2 MCS Profiling Summary (Iteration 3) ===" << std::endl;
    std::cout << "Total moment evaluations:    " << total_evaluations << std::endl;
    std::cout << "Enhancement triggers:        " << enhancement_count
              << " (" << (total_evaluations > 0 ? (100.0 * enhancement_count / total_evaluations) : 0.0)
              << "% of evaluations)" << std::endl;
    std::cout << "sqrt(A) >= 0.02 triggers:    " << sqrt_A_exceeds << std::endl;
    std::cout << "sqrt(C)/dx >= 3.0 triggers:  " << sqrt_C_exceeds << std::endl;
    std::cout << "Average sqrt(A):             " << (avg_sqrt_A * 1000.0) << " mrad" << std::endl;
    std::cout << "Average sqrt(C)/dx:          " << avg_sqrt_C_dx << " bins" << std::endl;
    std::cout << "==============================================\n" << std::endl;
}

#endif // ENABLE_MCS_PROFILING
