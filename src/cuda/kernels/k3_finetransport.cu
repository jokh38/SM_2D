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
    // Use bin center as representative energy to reduce systematic range bias.
    constexpr float ENERGY_OFFSET_RATIO = 1.00f;         // Offset from lower edge (fraction of half-width)
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

    constexpr unsigned THETA_SEED_A = 2654435761u;
    constexpr unsigned THETA_SEED_B = 2246822519u;
    constexpr unsigned THETA_SEED_C = 3266489917u;
    // Depth-scaled carry-over of unresolved accumulated variance.
    // This compensates quantization loss more strongly at mid/deep depth
    // without applying the full accumulated C at every step.
    constexpr float UNRESOLVED_C_BASE_FRACTION = 0.04f;
    constexpr float UNRESOLVED_C_DEPTH_SLOPE = 0.003f;   // per mm
    constexpr float UNRESOLVED_C_MAX_FRACTION = 0.35f;
}

// Debug counters for output-slot allocation failures (K3).
__device__ unsigned long long g_k3_slot_drop_count = 0;
__device__ double g_k3_slot_drop_weight = 0.0;
__device__ double g_k3_slot_drop_energy = 0.0;
__device__ unsigned long long g_k3_bucket_drop_count = 0;
__device__ double g_k3_bucket_drop_weight = 0.0;
__device__ double g_k3_bucket_drop_energy = 0.0;
__device__ unsigned long long g_k3_pruned_weight_count = 0;
__device__ double g_k3_pruned_weight_sum = 0.0;
__device__ double g_k3_pruned_energy_sum = 0.0;

// Initial beam energy for total accumulated scattering (fix for energy-dependent scattering)
// For 150 MeV protons, scattering should be computed at initial beam energy,
// not current energy (which decreases due to energy loss).
// Using lower energy overestimates scattering, causing too-wide beams.
constexpr float E_MEAN_150MEV = 150.0f;  // Initial beam energy for this validation

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
    // Precomputed Fermi-Eyges moments per depth (for z^3 lateral spreading)
    const float* __restrict__ d_FE_moment_A,  // [Nz] angular variance
    const float* __restrict__ d_FE_moment_B,  // [Nz] position-angle covariance
    const float* __restrict__ d_FE_sigma_total,  // [Nz] total lateral sigma
    // Outputs
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    double* __restrict__ AbsorbedEnergy_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,
    // Outflow buckets for boundary crossing (P3 FIX)
    DeviceOutflowBucket* __restrict__ OutflowBuckets,
    const int* __restrict__ CellToBucketBase,
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
    double cell_E_cutoff = 0.0;
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
            if (weight <= 0.0f) continue;

            // FIX Problem 1: Decode 4D local index (theta_local, E_local, x_sub, z_sub)
            int theta_local, E_local, x_sub, z_sub;
            decode_local_idx_4d(lidx, theta_local, E_local, x_sub, z_sub);

            // Get representative phase space values
            int theta_bin = b_theta * N_theta_local + theta_local;
            int E_bin = b_E * N_E_local + E_local;
            theta_bin = max(0, min(theta_bin, N_theta - 1));
            E_bin = max(0, min(E_bin, N_E - 1));

            if (weight < WEIGHT_THRESHOLD) {
                float E_lower_pruned = E_edges[E_bin];
                float E_upper_pruned = E_edges[E_bin + 1];
                float E_half_width_pruned = (E_upper_pruned - E_lower_pruned) * 0.5f;
                float E_pruned = E_lower_pruned + ENERGY_OFFSET_RATIO * E_half_width_pruned;
                atomicAdd(&g_k3_pruned_weight_count, 1ULL);
                atomicAdd(&g_k3_pruned_weight_sum, static_cast<double>(weight));
                atomicAdd(&g_k3_pruned_energy_sum, static_cast<double>(E_pruned * weight));
                continue;
            }

            float theta_min = theta_edges[0];
            float theta_max = theta_edges[N_theta];
            float dtheta = (theta_max - theta_min) / N_theta;

            // Use bin center (deterministic, not Monte Carlo sampling)
            float theta = theta_edges[theta_bin] + 0.5f * dtheta;

            // H6 FIX: Use piecewise-uniform energy grid (Option D2) instead of log-spaced
            // The E_edges array contains the actual bin edges for the piecewise-uniform grid
            //
            // CRITICAL FIX FOR ENERGY TRACKING:
            // The phase space stores discrete bins, not a continuous per-particle energy.
            // Use the bin center as representative energy for transport/audit consistency.
            //
            // This ensures particles move to lower bins as they lose energy:
            //   - Particle in bin [150, 151] uses E ≈ 150.50 for physics (bin center)
            //   - Loses 0.24 MeV → E_new ≈ 149.86
            //   - Gets binned to [149, 150] (since 149.86 < 150)
            //   - Next iteration uses E ≈ 149.50 for physics
            //   - Energy actually decreases over time!
            float E_lower = E_edges[E_bin];
            float E_upper = E_edges[E_bin + 1];
            float E_half_width = (E_upper - E_lower) * 0.5f;
            float E = E_lower + ENERGY_OFFSET_RATIO * E_half_width;

            // Cutoff check
            if (E <= ENERGY_CUTOFF_MEV) {
                cell_w_cutoff += weight;
                cell_E_cutoff += E * weight;
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

            // ====================================================================
            // FERMI-EYGES MOMENTS: Precomputed A(z), B(z) for z^3 lateral spread
            // ====================================================================
            // Previous approach (BROKEN): Applied total accumulated sigma as a
            // convolution kernel at every step. Since sigma_total ~= sigma_initial,
            // this acted as a constant box filter giving depth-independent spread.
            //
            // New approach (CORRECT): Use precomputed moments A(iz), B(iz) that
            // encode the accumulated scattering history including energy degradation.
            // delta_C per step = 2*B*ds + A*ds^2 + T*ds^3/3
            // The B and A terms encode the "lever arm" effect: scattering acquired
            // early in the path causes larger lateral displacement because the
            // particle has more remaining path to drift. This gives z^3 scaling.
            // ====================================================================
            int iz = cell / Nx;
            float step_mm = fmaxf(actual_range_step, 0.0f);

            // Read precomputed moments at current depth
            float A_precomp = (d_FE_moment_A != nullptr && iz >= 0 && iz < Nz) ? d_FE_moment_A[iz] : 0.0f;
            float B_precomp = (d_FE_moment_B != nullptr && iz >= 0 && iz < Nz) ? d_FE_moment_B[iz] : 0.0f;

            // Current step scattering power
            float E_mid = fmaxf(E - 0.5f * dE, dlut.E_min);
            float T_step = device_scattering_power_T(E_mid, step_mm);

            // Per-step angular spread (for theta sampling)
            float sigma_theta_step = sqrtf(fmaxf(T_step * step_mm, 0.0f));

            // INCREMENTAL spatial variance for this step (Fermi-Eyges with lever arm)
            // delta_C = 2*B*ds + A*ds^2 + T*ds^3/3
            // - 2*B*ds: existing angular-position covariance propagates into position variance
            // - A*ds^2: existing angular variance causes additional displacement
            // - T*ds^3/3: new scattering in this step
            float delta_C = 2.0f * B_precomp * step_mm
                          + A_precomp * step_mm * step_mm
                          + T_step * step_mm * step_mm * step_mm / 3.0f;
            delta_C = fmaxf(delta_C, 0.0f);

            // sigma_x for per-step lateral emission (incremental spread only)
            float sigma_x = fmaxf(sqrtf(delta_C), 0.01f);

            // POST-PROCESSING APPROACH: Disable in-transport lateral spreading.
            // MCS lateral spread will be applied as a single post-processing
            // convolution on EdepC after transport completes. This avoids the
            // double-counting problem where applying total sigma at every step
            // re-shapes the beam to constant width.
            float sigma_x_transport = 0.001f;  // Near-zero: kernel -> delta function

            unsigned path_seed = static_cast<unsigned>(fminf(static_cast<float>(iz) * dz * 1000.0f, 4294967295.0f));
            unsigned theta_seed =
                (static_cast<unsigned>(cell + 1) * THETA_SEED_A) ^
                (static_cast<unsigned>(slot + 1) * THETA_SEED_B) ^
                (static_cast<unsigned>(lidx + 1) * THETA_SEED_C) ^
                static_cast<unsigned>(E_bin + 17) ^
                (path_seed * 747796405u);
            float theta_scatter = device_sample_mcs_angle(sigma_theta_step, theta_seed);
            float theta_new = theta + theta_scatter;
            theta_new = fmaxf(theta_edges[0], fminf(theta_new, theta_edges[N_theta]));
            float mu_new = cosf(theta_new);
            float eta_new = sinf(theta_new);

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
                // CRITICAL FIX: Check E_new directly against cutoff BEFORE emitting.
                // Binned phase space can raise representative energy after transfer,
                // so crossing components must be absorbed here once below cutoff.
                if (E_new <= ENERGY_CUTOFF_MEV) {
                    cell_edep += edep;
                    cell_w_nuclear += w_rem;
                    cell_E_nuclear += E_rem;
                    cell_w_cutoff += w_new;
                    cell_E_cutoff += E_new * w_new;
                } else {
                    // Deposit step-local channels in current cell before transfer.
                    cell_edep += edep;
                    cell_w_nuclear += w_rem;
                    cell_E_nuclear += E_rem;

                    int neighbor_cell = device_get_neighbor(cell, exit_face, Nx, Nz);
                    if (neighbor_cell < 0) {
                        // Particle leaves the simulation domain.
                        cell_boundary_weight += w_new;
                        cell_boundary_energy += E_new * w_new;
                    } else {
                        // Re-binned transport uses representative bin-center energies.
                        // Fold E_new->E_emit mismatch back into local deposition to keep
                        // step-wise energy accounting closed.
                        int E_bin_transport = device_find_bin_edges(E_edges, N_E, E_new);
                        float E_lower_transport = E_edges[E_bin_transport];
                        float E_upper_transport = E_edges[E_bin_transport + 1];
                        float E_emit_transport =
                            E_lower_transport + 0.5f * (E_upper_transport - E_lower_transport);

                        if (exit_face == FACE_Z_PLUS) {
                            // Forward transport is the dominant path in proton runs:
                            // use robust multi-cell Gaussian emission for this branch.
                            int target_z = cell / Nx + 1;
                            float lateral_boundary_weight = 0.0f;
                            float dropped_boundary_weight = device_emit_lateral_spread(
                                OutflowBuckets,
                                CellToBucketBase,
                                cell,
                                target_z,
                                theta_new,
                                E_new,
                                w_new,
                                x_new,
                                sigma_x_transport,
                                dx,
                                Nx,
                                Nz,
                                theta_edges,
                                E_edges,
                                N_theta,
                                N_E,
                                N_theta_local,
                                N_E_local,
                                11,
                                &lateral_boundary_weight
                            );
                            if (lateral_boundary_weight > 0.0f) {
                                cell_boundary_weight += lateral_boundary_weight;
                                cell_boundary_energy += E_new * lateral_boundary_weight;
                            }
                            if (dropped_boundary_weight > 0.0f) {
                                atomicAdd(&g_k3_bucket_drop_count, 1ULL);
                                atomicAdd(&g_k3_bucket_drop_weight, static_cast<double>(dropped_boundary_weight));
                                atomicAdd(&g_k3_bucket_drop_energy, static_cast<double>(E_new * dropped_boundary_weight));
                            }

                            float transported_weight = fmaxf(
                                0.0f,
                                w_new - lateral_boundary_weight - dropped_boundary_weight
                            );
                            if (transported_weight > 0.0f) {
                                cell_edep += static_cast<double>(
                                    (E_new - E_emit_transport) * transported_weight
                                );
                            }
                        } else {
                            float x_offset_neighbor = device_get_neighbor_x_offset(x_new, exit_face, dx);
                            int x_sub_neighbor = get_x_sub_bin(x_offset_neighbor, dx);

                            float z_offset_neighbor;
                            if (exit_face == FACE_Z_MINUS) {
                                z_offset_neighbor = dz * 0.5f - dz * 0.125f;
                            } else {
                                z_offset_neighbor = z_new;
                            }
                            int z_sub_neighbor = get_z_sub_bin(z_offset_neighbor, dz);
                            z_sub_neighbor = device_transform_z_sub_for_neighbor(z_sub_neighbor, exit_face);

                            int bucket_idx = device_bucket_index_from_base(
                                CellToBucketBase,
                                cell,
                                exit_face,
                                Nx,
                                Nz
                            );
                            if (bucket_idx < 0) {
                                atomicAdd(&g_k3_bucket_drop_count, 1ULL);
                                atomicAdd(&g_k3_bucket_drop_weight, static_cast<double>(w_new));
                                atomicAdd(&g_k3_bucket_drop_energy, static_cast<double>(E_new * w_new));
                                continue;
                            }
                            DeviceOutflowBucket& bucket = OutflowBuckets[bucket_idx];
                            float dropped_boundary_weight = device_emit_component_to_bucket_4d(
                                bucket, theta_new, E_new, w_new, x_sub_neighbor, z_sub_neighbor,
                                theta_edges, E_edges, N_theta, N_E,
                                N_theta_local, N_E_local
                            );
                            if (dropped_boundary_weight > 0.0f) {
                                atomicAdd(&g_k3_bucket_drop_count, 1ULL);
                                atomicAdd(&g_k3_bucket_drop_weight, static_cast<double>(dropped_boundary_weight));
                                atomicAdd(&g_k3_bucket_drop_energy, static_cast<double>(E_new * dropped_boundary_weight));
                            }

                            float transported_weight = fmaxf(0.0f, w_new - dropped_boundary_weight);
                            if (transported_weight > 0.0f) {
                                cell_edep += static_cast<double>(
                                    (E_new - E_emit_transport) * transported_weight
                                );
                            }
                        }
                    }
                }
            } else {
                // CRITICAL FIX: Particle remains in cell - MUST write to output phase space!
                // Previously: particles were lost if they didn't cross boundaries
                cell_edep += edep;
                cell_w_nuclear += w_rem;
                cell_E_nuclear += E_rem;

                // Cutoff check - don't write to output if below cutoff
                if (E_new <= ENERGY_CUTOFF_MEV) {
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

                    // H6 FIX: Use binary search for piecewise-uniform energy grid (Option D2)
                    // Binary search in E_edges to find the bin containing E_new
                    int E_bin_new = device_find_bin_edges(E_edges, N_E, E_new);

                    // Compute new block_id
                    uint32_t b_theta_new = static_cast<uint32_t>(theta_bin_new) / N_theta_local;
                    uint32_t b_E_new = static_cast<uint32_t>(E_bin_new) / N_E_local;
                    uint32_t bid_new = (b_E_new << 12) | (b_theta_new & 0xFFF);

                    // Find/allocate output slot, and if saturated, reuse the closest
                    // existing block instead of dropping the entire component.
                    constexpr int Kb = DEVICE_Kb;
                    uint32_t selected_bid = bid_new;
                    int out_slot = device_psic_find_or_allocate_slot_with_fallback(
                        &block_ids_out[cell * Kb],
                        Kb,
                        bid_new,
                        selected_bid
                    );

                    // ====================================================================
                    // Apply Gaussian lateral spreading within the cell
                    // FIX B: Use sub-cell spreading function for correct spatial scale
                    // ====================================================================
                    // Distribute weight across x_sub bins using Gaussian distribution
                    // with proper sub-cell spacing (dx/8 instead of dx)
                    // ====================================================================

                    if (out_slot < 0 && E_new > ENERGY_CUTOFF_MEV) {
                        atomicAdd(&g_k3_slot_drop_count, 1ULL);
                        atomicAdd(&g_k3_slot_drop_weight, static_cast<double>(w_new));
                        atomicAdd(&g_k3_slot_drop_energy, static_cast<double>(E_new * w_new));
                        continue;
                    }

                    if (E_new > ENERGY_CUTOFF_MEV) {
                        int theta_bin_emit = theta_bin_new;
                        int E_bin_emit = E_bin_new;
                        if (selected_bid != bid_new) {
                            int b_theta_emit = static_cast<int>(selected_bid & 0xFFF);
                            int b_E_emit = static_cast<int>((selected_bid >> 12) & 0xFFF);
                            int theta_min_emit = b_theta_emit * N_theta_local;
                            int theta_max_emit = theta_min_emit + N_theta_local - 1;
                            int E_min_emit = b_E_emit * N_E_local;
                            int E_max_emit = E_min_emit + N_E_local - 1;
                            theta_bin_emit = max(theta_min_emit, min(theta_bin_emit, theta_max_emit));
                            E_bin_emit = max(E_min_emit, min(E_bin_emit, E_max_emit));
                        }
                        int theta_local_new = theta_bin_emit % N_theta_local;
                        int E_local_new = E_bin_emit % N_E_local;
                        float E_lower_emit = E_edges[E_bin_emit];
                        float E_upper_emit = E_edges[E_bin_emit + 1];
                        float E_emit = E_lower_emit + 0.5f * (E_upper_emit - E_lower_emit);

                        // FIX B: Use sub-cell Gaussian spreading with correct dx/8 spacing
                        constexpr int N_x_sub = 8;  // Number of sub-cell bins

                        // Calculate weights for each sub-bin using Gaussian CDF
                        float weights[N_x_sub];
                        float x_center = x_new;  // Center of Gaussian within cell

                        device_gaussian_spread_weights_subcell(weights, x_center, sigma_x_transport, dx, N_x_sub);

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
                        float w_left = 0.0f;
                        float w_right = 0.0f;

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
                            if (w_frac <= 0.0f) continue;

                            float w_spread = w_new * w_frac;

                            // Compute new local bin index
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
                            if (w_tail_total > 0.0f && w_tail_sum > 1e-12f) {
                                float tail_scale = w_tail_total / w_tail_sum;
                                w_left = w_left_raw * tail_scale;
                                w_right = w_right_raw * tail_scale;
                            }

                            // Emit left tail to left neighbor (or boundary if no neighbor)
                            if (w_left > 0.0f) {
                                float w_spread_left = w_new * w_left;
                                if (ix > 0) {
                                    int lateral_face = FACE_X_MINUS;
                                    int x_sub_neighbor = N_x_sub - 1;  // Rightmost bin of left neighbor
                                    int z_sub_neighbor = get_z_sub_bin(z_new, dz);

                                    int bucket_idx = device_bucket_index_from_base(
                                        CellToBucketBase,
                                        cell,
                                        lateral_face,
                                        Nx,
                                        Nz
                                    );
                                    if (bucket_idx < 0) {
                                        atomicAdd(&g_k3_bucket_drop_count, 1ULL);
                                        atomicAdd(&g_k3_bucket_drop_weight, static_cast<double>(w_spread_left));
                                        atomicAdd(&g_k3_bucket_drop_energy, static_cast<double>(E_new * w_spread_left));
                                        continue;
                                    }
                                    DeviceOutflowBucket& lateral_bucket = OutflowBuckets[bucket_idx];

                                    float dropped_left_tail = device_emit_component_to_bucket_4d(
                                        lateral_bucket, theta_new, E_new, w_spread_left,
                                        x_sub_neighbor, z_sub_neighbor,
                                        theta_edges, E_edges, N_theta, N_E,
                                        N_theta_local, N_E_local
                                    );
                                    if (dropped_left_tail > 0.0f) {
                                        atomicAdd(&g_k3_bucket_drop_count, 1ULL);
                                        atomicAdd(&g_k3_bucket_drop_weight, static_cast<double>(dropped_left_tail));
                                        atomicAdd(&g_k3_bucket_drop_energy, static_cast<double>(E_new * dropped_left_tail));
                                    }
                                } else {
                                    // No left neighbor: treat lateral tail as domain boundary loss.
                                    cell_boundary_weight += w_spread_left;
                                    cell_boundary_energy += E_new * w_spread_left;
                                }
                            }

                            // Emit right tail to right neighbor (or boundary if no neighbor)
                            if (w_right > 0.0f) {
                                float w_spread_right = w_new * w_right;
                                if (ix < Nx - 1) {
                                    int lateral_face = FACE_X_PLUS;
                                    int x_sub_neighbor = 0;  // Leftmost bin of right neighbor
                                    int z_sub_neighbor = get_z_sub_bin(z_new, dz);

                                    int bucket_idx = device_bucket_index_from_base(
                                        CellToBucketBase,
                                        cell,
                                        lateral_face,
                                        Nx,
                                        Nz
                                    );
                                    if (bucket_idx < 0) {
                                        atomicAdd(&g_k3_bucket_drop_count, 1ULL);
                                        atomicAdd(&g_k3_bucket_drop_weight, static_cast<double>(w_spread_right));
                                        atomicAdd(&g_k3_bucket_drop_energy, static_cast<double>(E_new * w_spread_right));
                                        continue;
                                    }
                                    DeviceOutflowBucket& lateral_bucket = OutflowBuckets[bucket_idx];

                                    float dropped_right_tail = device_emit_component_to_bucket_4d(
                                        lateral_bucket, theta_new, E_new, w_spread_right,
                                        x_sub_neighbor, z_sub_neighbor,
                                        theta_edges, E_edges, N_theta, N_E,
                                        N_theta_local, N_E_local
                                    );
                                    if (dropped_right_tail > 0.0f) {
                                        atomicAdd(&g_k3_bucket_drop_count, 1ULL);
                                        atomicAdd(&g_k3_bucket_drop_weight, static_cast<double>(dropped_right_tail));
                                        atomicAdd(&g_k3_bucket_drop_energy, static_cast<double>(E_new * dropped_right_tail));
                                    }
                                } else {
                                    // No right neighbor: treat lateral tail as domain boundary loss.
                                    cell_boundary_weight += w_spread_right;
                                    cell_boundary_energy += E_new * w_spread_right;
                                }
                            }
                        }

                        float rebinned_fraction = w_cell_fraction;
                        if (ix > 0) rebinned_fraction += w_left;
                        if (ix < Nx - 1) rebinned_fraction += w_right;
                        rebinned_fraction = fminf(1.0f, fmaxf(0.0f, rebinned_fraction));
                        float rebinned_weight = w_new * rebinned_fraction;
                        if (rebinned_weight > 0.0f) {
                            cell_edep += static_cast<double>((E_new - E_emit) * rebinned_weight);
                        }
                    }
                }
            }
        }
    }

    // Write accumulators to global memory (atomic for thread safety)
    atomicAdd(&EdepC[cell], cell_edep);
    atomicAdd(&AbsorbedWeight_cutoff[cell], cell_w_cutoff);
    atomicAdd(&AbsorbedEnergy_cutoff[cell], cell_E_cutoff);
    atomicAdd(&AbsorbedWeight_nuclear[cell], cell_w_nuclear);
    atomicAdd(&AbsorbedEnergy_nuclear[cell], cell_E_nuclear);
    atomicAdd(&BoundaryLoss_weight[cell], cell_boundary_weight);
    atomicAdd(&BoundaryLoss_energy[cell], cell_boundary_energy);
}

void k3_reset_debug_counters() {
    constexpr unsigned long long zero_count = 0ULL;
    constexpr double zero_value = 0.0;
    cudaMemcpyToSymbol(g_k3_slot_drop_count, &zero_count, sizeof(zero_count));
    cudaMemcpyToSymbol(g_k3_slot_drop_weight, &zero_value, sizeof(zero_value));
    cudaMemcpyToSymbol(g_k3_slot_drop_energy, &zero_value, sizeof(zero_value));
    cudaMemcpyToSymbol(g_k3_bucket_drop_count, &zero_count, sizeof(zero_count));
    cudaMemcpyToSymbol(g_k3_bucket_drop_weight, &zero_value, sizeof(zero_value));
    cudaMemcpyToSymbol(g_k3_bucket_drop_energy, &zero_value, sizeof(zero_value));
    cudaMemcpyToSymbol(g_k3_pruned_weight_count, &zero_count, sizeof(zero_count));
    cudaMemcpyToSymbol(g_k3_pruned_weight_sum, &zero_value, sizeof(zero_value));
    cudaMemcpyToSymbol(g_k3_pruned_energy_sum, &zero_value, sizeof(zero_value));
}

void k3_get_debug_counters(
    unsigned long long& slot_drop_count,
    double& slot_drop_weight,
    double& slot_drop_energy,
    unsigned long long& bucket_drop_count,
    double& bucket_drop_weight,
    double& bucket_drop_energy,
    unsigned long long& pruned_weight_count,
    double& pruned_weight_sum,
    double& pruned_energy_sum
) {
    cudaMemcpyFromSymbol(&slot_drop_count, g_k3_slot_drop_count, sizeof(slot_drop_count));
    cudaMemcpyFromSymbol(&slot_drop_weight, g_k3_slot_drop_weight, sizeof(slot_drop_weight));
    cudaMemcpyFromSymbol(&slot_drop_energy, g_k3_slot_drop_energy, sizeof(slot_drop_energy));
    cudaMemcpyFromSymbol(&bucket_drop_count, g_k3_bucket_drop_count, sizeof(bucket_drop_count));
    cudaMemcpyFromSymbol(&bucket_drop_weight, g_k3_bucket_drop_weight, sizeof(bucket_drop_weight));
    cudaMemcpyFromSymbol(&bucket_drop_energy, g_k3_bucket_drop_energy, sizeof(bucket_drop_energy));
    cudaMemcpyFromSymbol(&pruned_weight_count, g_k3_pruned_weight_count, sizeof(pruned_weight_count));
    cudaMemcpyFromSymbol(&pruned_weight_sum, g_k3_pruned_weight_sum, sizeof(pruned_weight_sum));
    cudaMemcpyFromSymbol(&pruned_energy_sum, g_k3_pruned_energy_sum, sizeof(pruned_energy_sum));
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
