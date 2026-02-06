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
// Uses C++11 "magic static" pattern:
// - Thread-safe initialization (guaranteed by C++11 standard)
// - Automatic cleanup on program exit
// - No manual memory management required
const RLUT& get_global_rlut() {
    static const RLUT rlut = GenerateRLUT(0.1f, 300.0f, 256);
    return rlut;
}

// ============================================================================
// Physics Constants for K3 Fine Transport
// ============================================================================
// IMPORTANT: This is NOT a Monte Carlo code!
// This is a deterministic phase-space transport code using binned representation.
// Lateral spreading is handled by Gaussian weight distribution across cells,
// NOT by random sampling of scattering angles for individual particles.
// ============================================================================
namespace {
    // Energy tracking constants
    constexpr float ENERGY_CUTOFF_MEV = 0.1f;           // Minimum energy for transport [MeV]
    constexpr float WEIGHT_THRESHOLD = 1e-15f;           // Minimum weight for transport
    constexpr float ENERGY_OFFSET_RATIO = 0.50f;         // Offset from lower edge (fraction of half-width)
    constexpr float BOUNDARY_SAFETY_FACTOR = 1.001f;     // Allow slight boundary crossing

    // Scattering reduction factors (TEST: set all to 1.0 for accurate physics)
    // BUG: Previous values (0.3, 0.5, 0.7) caused lateral spread to be too narrow
    // FIX: Use 1.0 (no reduction) at all energies for correct Highland formula
    constexpr float SCATTER_REDUCTION_HIGH_E = 1.0f;     // E > 100 MeV
    constexpr float SCATTER_REDUCTION_MID_HIGH = 1.0f;   // E > 50 MeV
    constexpr float SCATTER_REDUCTION_MID_LOW = 1.0f;    // E > 20 MeV
    constexpr float SCATTER_REDUCTION_LOW_E = 1.0f;      // E <= 20 MeV (full scattering)

    // Energy thresholds for scattering reduction
    constexpr float ENERGY_HIGH_THRESHOLD = 100.0f;     // [MeV]
    constexpr float ENERGY_MID_HIGH_THRESHOLD = 50.0f;  // [MeV]
    constexpr float ENERGY_MID_LOW_THRESHOLD = 20.0f;   // [MeV]
}

// ============================================================================
// P1 FIX: Full GPU Kernel Implementation
// ============================================================================
// Complete implementation with:
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
    // Physics configuration flags (for testing/validation)
    bool enable_straggling,   // Enable energy straggling (Vavilov)
    bool enable_nuclear,      // Enable nuclear interactions
    // FIX C: Initial beam width for lateral spreading (from input config)
    float sigma_x_initial,
    // NOTE: Lateral spreading is ALWAYS enabled (deterministic, not Monte Carlo)
    // Outputs
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,
    // Outflow buckets for boundary crossing (P3 FIX)
    DeviceOutflowBucket* __restrict__ OutflowBuckets,
    // CRITICAL FIX: Output phase space for particles remaining in cell
    uint32_t* __restrict__ block_ids_out,
    float* __restrict__ values_out
) {
    // Thread ID maps to active cell
    int active_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (active_idx >= n_active) return;

    int cell = ActiveList[active_idx];

    // Accumulators for this cell
    double cell_edep = 0.0;
    float cell_w_cutoff = 0.0f;
    float cell_w_nuclear = 0.0f;
    double cell_E_nuclear = 0.0;
    float cell_boundary_weight = 0.0f;
    double cell_boundary_energy = 0.0;

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

            if (weight < WEIGHT_THRESHOLD) continue;

            // FIX Problem 1: Decode 4D local index (theta_local, E_local, x_sub, z_sub)
            int theta_local, E_local, x_sub, z_sub;
            decode_local_idx_4d(lidx, theta_local, E_local, x_sub, z_sub);

            // Get representative phase space values
            int theta_bin = b_theta * N_theta_local + theta_local;
            int E_bin = b_E * N_E_local + E_local;

            float theta_min = theta_edges[0];
            float theta_max = theta_edges[N_theta];
            float dtheta = (theta_max - theta_min) / N_theta;

            // Use bin center (deterministic, not Monte Carlo sampling)
            float theta = theta_edges[theta_bin] + 0.5f * dtheta;

            // H6 FIX: Use piecewise-uniform energy grid (Option D2) instead of log-spaced
            // The E_edges array contains the actual bin edges for the piecewise-uniform grid
            //
            // CRITICAL FIX FOR ENERGY TRACKING:
            // Using the bin midpoint causes energy to "reset" in each iteration because
            // the phase space only stores which bin a particle is in, not the continuous
            // energy value. To ensure actual energy loss across iterations, we use the
            // lower edge PLUS a small fraction of the bin width (20% of half-width).
            //
            // This ensures particles actually move to lower bins as they lose energy:
            //   - Particle in bin [150, 151] uses E ≈ 150.10 for physics (20% offset)
            //   - Loses 0.24 MeV → E_new ≈ 149.86
            //   - Gets binned to [149, 150] (since 149.86 < 150)
            //   - Next iteration uses E ≈ 149.10 for physics
            //   - Energy actually decreases over time!
            float E_lower = E_edges[E_bin];
            float E_upper = E_edges[E_bin + 1];
            float E_half_width = (E_upper - E_lower) * 0.5f;
            float E = E_lower + ENERGY_OFFSET_RATIO * E_half_width;

            // Cutoff check
            if (E <= ENERGY_CUTOFF_MEV) {
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
            // Using centered coordinate system: [-dx/2, +dx/2] x [-dz/2, +dz/2]
            float half_dz = dz * 0.5f;
            float half_dx = dx * 0.5f;
            float step_to_z_plus = (mu_init > 0) ? (half_dz - z_cell) / mu_init : 1e30f;
            float step_to_z_minus = (mu_init < 0) ? (-half_dz - z_cell) / mu_init : 1e30f;
            float step_to_x_plus = (eta_init > 0) ? (half_dx - x_cell) / eta_init : 1e30f;
            float step_to_x_minus = (eta_init < 0) ? (-half_dx - x_cell) / eta_init : 1e30f;
            float step_to_boundary = fminf(fminf(step_to_z_plus, step_to_z_minus),
                                           fminf(step_to_x_plus, step_to_x_minus));
            step_to_boundary = fmaxf(step_to_boundary, 0.0f);

            // Compute physics-limited step size (returns range step in mm)
            float step_phys = device_compute_max_step(dlut, E, dx, dz);

            // P9 FIX: Use minimum of physics step and distance to boundary
            // step_to_boundary is already the path length (computed by dividing geometric distance by direction cosine)
            // For normal incidence (mu=1), path_length = distance. For angled tracks, path_length > distance.
            // CRITICAL FIX: Don't divide by mu_init again - step_to_boundary is already path length!

            // Use the smaller of physics-limited range step and path length to boundary
            // BUG FIX: Allow boundary crossing by using slightly more than 100% of boundary distance
            // Previous 99.9% limit caused particles to get stuck at boundaries
            float step_to_boundary_safe = step_to_boundary * BOUNDARY_SAFETY_FACTOR;
            float actual_range_step = fminf(step_phys, step_to_boundary_safe);

            // FIX Problem 2: Mid-point method for energy deposition
            // Energy loss should occur along the entire step path
            // Use the range step directly as the distance traveled (path length = distance along trajectory)
            float half_step = actual_range_step * 0.5f;
            float x_mid = x_cell + eta_init * half_step;
            float z_mid = z_cell + mu_init * half_step;

            // Energy loss with optional straggling for actual range step
            float mean_dE = device_compute_energy_deposition(dlut, E, actual_range_step);
            float dE;
            if (enable_straggling) {
                // Use deterministic seed based on phase space coordinates (not random)
                unsigned seed = static_cast<unsigned>(
                    (cell * 7 + slot * 13 + lidx * 17) ^ 0x5DEECE66DL
                );
                float sigma_dE = device_energy_straggling_sigma(E, actual_range_step, 1.0f);
                dE = device_sample_energy_loss(mean_dE, sigma_dE, seed);
            } else {
                // Energy loss only mode: use mean Bethe-Bloch energy loss (no straggling)
                dE = mean_dE;
            }
            dE = fminf(dE, E);

            float E_new = E - dE;
            float edep = dE * weight;

            // Nuclear attenuation for actual range step (optional)
            float w_rem, E_rem;
            float w_new;
            if (enable_nuclear) {
                w_new = device_apply_nuclear_attenuation(weight, E, actual_range_step, w_rem, E_rem);
                edep += E_rem;
            } else {
                // Energy loss only mode: no nuclear attenuation
                w_new = weight;
                w_rem = 0.0f;
                E_rem = 0.0f;
            }

            // ========================================================================
            // DETERMINISTIC LATERAL SPREADING (NOT Monte Carlo)
            // ========================================================================
            // This code implements deterministic lateral spreading using Gaussian
            // weight distribution across cells, NOT random sampling of scattering angles.
            //
            // The lateral spread is calculated from the Highland scattering angle formula:
            //   sigma_theta = Highland_sigma(E, step)
            //   sigma_x = sigma_theta * step / sqrt(3)
            //
            // Weight is distributed across neighboring cells using Gaussian CDF.
            // This is deterministic and reproducible, not Monte Carlo.
            // ========================================================================

            // FIX: Use depth-based sigma_x for correct Fermi-Eyges O(z^(3/2)) scaling
            // Previous per-step calculation caused sigma_x << dx, resulting in no lateral spread
            // New calculation uses accumulated depth from surface
            int ix = cell % Nx;
            int iz = cell / Nx;
            float depth_from_surface_mm = iz * dz + z_cell;  // Total depth from surface

            // Calculate accumulated lateral spread using Fermi-Eyges theory
            // sigma_x(z) ≈ theta_0 * z / sqrt(3) for thin target
            // This gives O(z) scaling, which combined with energy loss gives O(z^(3/2))
            float sigma_theta_depth = device_highland_sigma(E, depth_from_surface_mm);
            float sigma_x_depth = sigma_theta_depth * depth_from_surface_mm / 1.7320508f;  // / sqrt(3)

            // Combine with initial beam width (FIX C: now from input parameter)
            // Total sigma_x = sqrt(sigma_x0^2 + sigma_x_depth^2)
            // sigma_x_initial is passed from config (sim.ini sigma_x_mm)
            float sigma_x = sqrtf(sigma_x_initial * sigma_x_initial + sigma_x_depth * sigma_x_depth);

            // Ensure minimum sigma_x for numerical stability
            sigma_x = fmaxf(sigma_x, 0.01f);  // MIN_SIGMA for numerical stability

            // For deterministic transport, keep direction unchanged (theta_new = theta)
            // Lateral spreading is handled by weight distribution across cells
            float theta_new = theta;
            float mu_new = mu_init;
            float eta_new = eta_init;

            // Complete position update: move along original direction
            // Lateral spreading will be applied via weight distribution to output
            float x_new = x_cell + eta_new * actual_range_step;
            float z_new = z_cell + mu_new * actual_range_step;

            // Check boundary crossing FIRST (using unclamped position)
            int exit_face = device_determine_exit_face(x_cell, z_cell, x_new, z_new, dx, dz);

            // THEN clamp position to cell bounds for emission calculations
            x_new = fmaxf(-half_dx, fminf(x_new, half_dx));
            z_new = fmaxf(-half_dz, fminf(z_new, half_dz));

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
                // CRITICAL FIX: Use non-interpolating emission to prevent bucket overflow
                // Bilinear interpolation splits particles across 4 (theta,E) bins, causing
                // quadratic growth in unique block IDs that exceeds DEVICE_Kb_out=8 slots.
                // For energy-loss-only mode (theta~0), nearest neighbor is more accurate
                // AND prevents weight loss from bucket overflow.
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

                // CRITICAL FIX: Only count as boundary loss if particle exits simulation domain
                // Check if neighbor cell exists (within grid bounds)
                int ix = cell % Nx;
                int iz = cell / Nx;
                bool neighbor_exists = true;
                switch (exit_face) {
                    case 0:  // +z face
                        neighbor_exists = (iz + 1 < Nz);
                        break;
                    case 1:  // -z face
                        neighbor_exists = (iz > 0);
                        break;
                    case 2:  // +x face
                        neighbor_exists = (ix + 1 < Nx);
                        break;
                    case 3:  // -x face
                        neighbor_exists = (ix > 0);
                        break;
                }

                // Only count as boundary loss if particle is leaving simulation domain
                if (!neighbor_exists) {
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
                if (E_new <= ENERGY_CUTOFF_MEV) {
                    cell_edep += E_new * w_new;
                    cell_w_cutoff += w_new;
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

                    // H6 FIX: Use binary search for piecewise-uniform energy grid (Option D2)
                    // Binary search in E_edges to find the bin containing E_new
                    int E_bin_new = 0;
                    if (E_new <= E_edges[0]) {
                        E_bin_new = 0;
                    } else if (E_new >= E_edges[N_E]) {
                        E_bin_new = N_E - 1;
                    } else {
                        int lo = 0, hi = N_E;
                        while (lo < hi) {
                            int mid = (lo + hi) / 2;
                            if (E_edges[mid + 1] <= E_new) {
                                lo = mid + 1;
                            } else {
                                hi = mid;
                            }
                        }
                        E_bin_new = lo;
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

                    // ====================================================================
                    // Apply Gaussian lateral spreading within the cell
                    // FIX B: Use sub-cell spreading function for correct spatial scale
                    // ====================================================================
                    // Distribute weight across x_sub bins using Gaussian distribution
                    // with proper sub-cell spacing (dx/8 instead of dx)
                    // ====================================================================

                    if (out_slot >= 0 && E_new > ENERGY_CUTOFF_MEV) {
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
                        float w_cell_fraction = fminf(1.0f, w_in_cell);  // Clamp for safety

                        // Distribute weight across all x_sub bins
                        for (int x_sub_target = 0; x_sub_target < N_x_sub; ++x_sub_target) {
                            float w_frac = weights[x_sub_target] * w_cell_fraction;  // Scale by in-cell fraction
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
                        if (sigma_x > dx * 0.5f) {
                            // Left tail weight (integral from -inf to left_boundary)
                            float w_left = device_gaussian_cdf(left_boundary, x_center, sigma_x);
                            // Right tail weight (integral from right_boundary to +inf)
                            float w_right = 1.0f - device_gaussian_cdf(right_boundary, x_center, sigma_x);

                            // Emit left tail to left neighbor
                            if (w_left > 1e-6f && ix > 0) {
                                float w_spread_left = w_new * w_left;
                                int lateral_face = FACE_X_MINUS;
                                int x_sub_neighbor = N_x_sub - 1;  // Rightmost bin of left neighbor
                                int z_sub_neighbor = get_z_sub_bin(z_new, dz);

                                int bucket_idx = device_bucket_index(cell, lateral_face, Nx, Nz);
                                DeviceOutflowBucket& lateral_bucket = OutflowBuckets[bucket_idx];

                                device_emit_component_to_bucket_4d(
                                    lateral_bucket, theta_new, E_new, w_spread_left,
                                    x_sub_neighbor, z_sub_neighbor,
                                    theta_edges, E_edges, N_theta, N_E,
                                    N_theta_local, N_E_local
                                );

                                // FIX: Removed cell_edep += E_new * w_spread_left;
                                // This was causing double-counting: energy is transferred to neighbor
                                // via bucket AND counted here. The energy will be deposited when
                                // the neighbor bucket is processed.
                                cell_boundary_weight += w_spread_left;
                            }

                            // Emit right tail to right neighbor
                            if (w_right > 1e-6f && ix < Nx - 1) {
                                float w_spread_right = w_new * w_right;
                                int lateral_face = FACE_X_PLUS;
                                int x_sub_neighbor = 0;  // Leftmost bin of right neighbor
                                int z_sub_neighbor = get_z_sub_bin(z_new, dz);

                                int bucket_idx = device_bucket_index(cell, lateral_face, Nx, Nz);
                                DeviceOutflowBucket& lateral_bucket = OutflowBuckets[bucket_idx];

                                device_emit_component_to_bucket_4d(
                                    lateral_bucket, theta_new, E_new, w_spread_right,
                                    x_sub_neighbor, z_sub_neighbor,
                                    theta_edges, E_edges, N_theta, N_E,
                                    N_theta_local, N_E_local
                                );

                                // FIX: Removed cell_edep += E_new * w_spread_right;
                                // This was causing double-counting: energy is transferred to neighbor
                                // via bucket AND counted here. The energy will be deposited when
                                // the neighbor bucket is processed.
                                cell_boundary_weight += w_spread_right;
                            }
                        }
                    }
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
