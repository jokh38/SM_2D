#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "core/psi_storage.hpp"
#include "core/grids.hpp"
#include "device/device_lut.cuh"
#include "device/device_bucket.cuh"
#include "device/device_psic.cuh"
#include "kernels/k5_audit.cuh"  // For AuditReport

// ============================================================================
// K1-K6 Transport Pipeline Header
// ============================================================================
// Complete particle transport pipeline with energy-based transport selection
// - K1: Active mask identification
// - K2: Coarse transport (high energy, fast physics)
// - K3: Fine transport (low energy, detailed physics)
// - K4: Bucket transfer (boundary crossing)
// - K5: Weight + energy audit (conservation check)
// - K6: Buffer swap (prepare for next iteration)
// ============================================================================

// Forward declarations
struct PsiC;
struct DevicePsiC;
struct EnergyGrid;
struct AngularGrid;

namespace sm_2d {

// ============================================================================
// Pipeline Configuration
// ============================================================================

struct K1K6PipelineConfig {
    // Active mask thresholds (K1)
    float E_fine_on;           // Fine transport activation threshold [MeV]
    float E_fine_off;          // Fine transport deactivation threshold [MeV]
    float weight_active_min;   // Minimum weight for active cell

    // Coarse transport config (K2)
    float E_coarse_max;        // Maximum energy for coarse transport [MeV]
    float step_coarse;         // Coarse step size [mm]
    int n_steps_per_cell;      // Number of sub-steps per cell

    // Fine transport is used when E <= E_fine_on (with optional hysteresis)

    // FIX C: Initial beam width for lateral spreading
    float sigma_x_initial;     // Initial beam width [mm] (from sim.ini sigma_x_mm)

    // Grid dimensions
    int Nx, Nz;                // Spatial grid
    float dx, dz;              // Cell sizes [mm]

    // Phase-space grid
    int N_theta, N_E;          // Global bins
    int N_theta_local, N_E_local;  // Local bins per block

    // Loop control + logging
    int fine_batch_max_cells = 0; // 0 => process all active fine cells at once
    int max_iterations;        // Maximum transport iterations
    int log_level;             // 0: quiet, 1: summary, 2: verbose
    bool fail_fast_on_audit = false;   // Stop immediately when K5 W/E pass flags fail
};

// ============================================================================
// Helper Kernel Declarations
// ============================================================================

// Inject source particle into PsiC (pencil beam - single particle)
__global__ void inject_source_kernel(
    DevicePsiC psi,
    int Nx, int Nz, float dx, float dz, float x_min, float z_min,  // Grid info for multi-cell injection
    float x0, float z0,           // Global source position (mm)
    float theta0, float sigma_theta,
    float E0, float W_total,
    float sigma_x,
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local
);

// Inject Gaussian beam source into PsiC (multiple sampled particles)
// Each thread processes one sample from the Gaussian distribution
__global__ void inject_gaussian_source_kernel(
    DevicePsiC psi,
    float x0, float z0,              // Central beam position [mm]
    float theta0, float E0, float W_total,  // Central angle, energy, total weight
    float sigma_x, float sigma_theta, float sigma_E,  // Gaussian spreads
    int n_samples,                    // Number of samples to draw
    unsigned int random_seed,         // Seed for RNG
    float x_min, float z_min,         // Grid origin
    float dx, float dz,               // Cell size
    int Nx, int Nz,                   // Grid dimensions
    // Optional accounting counters (all in units of weight)
    float* d_injected_weight,
    float* d_out_of_grid_weight,
    float* d_slot_dropped_weight,
    // Optional accounting counters (all in units of MeV)
    double* d_injected_energy,
    double* d_out_of_grid_energy,
    double* d_slot_dropped_energy,
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local
);

// Compact active list from mask
// Converts ActiveMask binary array to compact list of active cell indices
// Input:  ActiveMask[Nx*Nz] (uint8_t, 1 = active)
// Output: ActiveList[Nx*Nz] (uint32_t, compacted indices)
// Returns: Number of active cells (n_active)
__global__ void compact_active_list(
    const uint8_t* __restrict__ ActiveMask,
    uint32_t* __restrict__ ActiveList,
    int Nx, int Nz,
    int* d_n_active
);

// Count coarse cells from mask (for K2 scheduling)
// Input:  ActiveMask[Nx*Nz] (uint8_t, 0 = needs coarse transport)
//         block_ids, values (check for actual weight)
// Output: CoarseList[Nx*Nz] (uint32_t, compacted indices)
// Returns: Number of coarse cells (n_coarse) with actual weights
__global__ void compact_coarse_list(
    const uint8_t* __restrict__ ActiveMask,
    const uint32_t* __restrict__ block_ids,
    const float* __restrict__ values,
    uint32_t* __restrict__ CoarseList,
    int Nx, int Nz,
    int* d_n_coarse
);

// Compute total weight in phase space
// Used for weight conservation audit
// Input:  block_ids, values from PsiStorage
// Output: total_weight (reduced sum)
__global__ void compute_total_weight(
    const uint32_t* __restrict__ block_ids,
    const float* __restrict__ values,
    double* __restrict__ total_weight,
    int N_cells
);

// ============================================================================
// Pipeline State Structure
// ============================================================================

struct K1K6PipelineState {
    // Input/output buffers
    DevicePsiC* psi_in;
    DevicePsiC* psi_out;

    // Active cell tracking
    uint8_t* d_ActiveMask;     // Binary mask [Nx*Nz]
    uint8_t* d_ActiveMask_prev;// Previous iteration mask [Nx*Nz] for hysteresis
    uint32_t* d_ActiveList;    // Compacted list [Nx*Nz]
    uint32_t* d_CoarseList;    // Coarse cells [Nx*Nz]

    // Counters (host and device)
    int n_active;              // Number of active cells
    int n_coarse;              // Number of coarse cells
    int* d_n_active;           // Device counter
    int* d_n_coarse;           // Device counter

    // Energy deposition tallies
    double* d_EdepC;           // Energy deposition [Nx*Nz]
    float* d_AbsorbedWeight_cutoff;
    double* d_AbsorbedEnergy_cutoff;
    float* d_AbsorbedWeight_nuclear;
    double* d_AbsorbedEnergy_nuclear;
    float* d_BoundaryLoss_weight;
    double* d_BoundaryLoss_energy;
    // Previous cumulative tallies for iteration-delta audit
    float* d_prev_AbsorbedWeight_cutoff;
    double* d_prev_AbsorbedEnergy_cutoff;
    float* d_prev_AbsorbedWeight_nuclear;
    float* d_prev_BoundaryLoss_weight;
    double* d_prev_EdepC;
    double* d_prev_AbsorbedEnergy_nuclear;
    double* d_prev_BoundaryLoss_energy;

    // Batch-local bucket scratch (P1 memory reduction path)
    DeviceOutflowBucket* d_BucketScratch;  // [bucket_scratch_capacity_cells * 4]
    int bucket_scratch_capacity_cells;     // Number of cells currently allocated in scratch
    int* d_CellToBucketBase;               // [Nx*Nz], -1 for cells not in current batch

    // Reusable phase-space grid edges on device
    float* d_theta_edges;      // [N_theta + 1]
    float* d_E_edges;          // [N_E + 1]

    // Audit
    AuditReport* d_audit_report;
    double* d_weight_in;
    double* d_weight_out;
    // Source accounting (host-side scalars captured before transport loop)
    float source_injected_weight;
    float source_out_of_grid_weight;
    float source_slot_dropped_weight;
    double source_injected_energy;
    double source_out_of_grid_energy;
    double source_slot_dropped_energy;
    double source_representation_loss_energy;
    // Transport-side drop accounting (accumulated across iterations).
    double transport_dropped_weight;
    double transport_dropped_energy;
    // Per-iteration K5 signed energy residual accumulated across transport.
    double transport_audit_residual_energy;

    // Device memory ownership
    bool owns_device_memory;

    K1K6PipelineState()
        : psi_in(nullptr), psi_out(nullptr)
        , d_ActiveMask(nullptr), d_ActiveMask_prev(nullptr), d_ActiveList(nullptr), d_CoarseList(nullptr)
        , n_active(0), n_coarse(0)
        , d_n_active(nullptr), d_n_coarse(nullptr)
        , d_EdepC(nullptr)
        , d_AbsorbedWeight_cutoff(nullptr), d_AbsorbedEnergy_cutoff(nullptr), d_AbsorbedWeight_nuclear(nullptr)
        , d_AbsorbedEnergy_nuclear(nullptr)
        , d_BoundaryLoss_weight(nullptr), d_BoundaryLoss_energy(nullptr)
        , d_prev_AbsorbedWeight_cutoff(nullptr)
        , d_prev_AbsorbedEnergy_cutoff(nullptr)
        , d_prev_AbsorbedWeight_nuclear(nullptr)
        , d_prev_BoundaryLoss_weight(nullptr)
        , d_prev_EdepC(nullptr)
        , d_prev_AbsorbedEnergy_nuclear(nullptr)
        , d_prev_BoundaryLoss_energy(nullptr)
        , d_BucketScratch(nullptr)
        , bucket_scratch_capacity_cells(0)
        , d_CellToBucketBase(nullptr)
        , d_theta_edges(nullptr), d_E_edges(nullptr)
        , d_audit_report(nullptr)
        , d_weight_in(nullptr), d_weight_out(nullptr)
        , source_injected_weight(0.0f), source_out_of_grid_weight(0.0f)
        , source_slot_dropped_weight(0.0f)
        , source_injected_energy(0.0), source_out_of_grid_energy(0.0)
        , source_slot_dropped_energy(0.0)
        , source_representation_loss_energy(0.0)
        , transport_dropped_weight(0.0), transport_dropped_energy(0.0)
        , transport_audit_residual_energy(0.0)
        , owns_device_memory(false)
    {}

    // Allocate device memory for pipeline state
    bool allocate(int Nx, int Nz);

    // Free device memory
    void cleanup();

    ~K1K6PipelineState() {
        cleanup();
    }
};

// ============================================================================
// Main Pipeline Function
// ============================================================================

// Execute complete K1-K6 transport pipeline
//
// Flow:
// 1. K1: Identify active cells based on E_fine_on/E_fine_off and weight thresholds
// 2. Compact active list and coarse list
// 3. K2: Coarse transport for high-energy cells (ActiveMask = 0)
// 4. K3: Fine transport for low-energy cells (ActiveMask = 1)
// 5. K4: Bucket transfer for boundary crossings
// 6. K5: Weight + energy audit (conservation check)
// 7. K6: Swap buffers for next iteration
//
// Parameters:
//   psi_in:   Input phase space (GPU memory)
//   psi_out:  Output phase space (GPU memory)
//   dlut:     Device range lookup table
//   config:   Pipeline configuration
//   state:    Pipeline state (allocated externally or internally)
//
// Returns:
//   true if pipeline executed successfully
//   false if errors occurred (check CUDA status)
bool run_k1k6_pipeline_transport(
    DevicePsiC* psi_in,
    DevicePsiC* psi_out,
    const ::DeviceRLUT& dlut,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    const K1K6PipelineConfig& config,
    K1K6PipelineState& state
);

// ============================================================================
// Individual Kernel Wrappers (for testing/debugging)
// ============================================================================

// Run K1 only: Generate active mask
bool run_k1_active_mask(
    const DevicePsiC& psi_in,
    uint8_t* d_ActiveMask,
    const uint8_t* d_ActiveMaskPrev,
    int Nx, int Nz,
    int b_E_fine_on,
    int b_E_fine_off,
    float weight_active_min
);

// Run K2 only: Coarse transport
bool run_k2_coarse_transport(
    const DevicePsiC& psi_in,
    DevicePsiC& psi_out,  // CRITICAL FIX: Output phase space
    const uint8_t* d_ActiveMask,
    const uint32_t* d_CoarseList,
    int n_coarse,
    const ::DeviceRLUT& dlut,
    const K1K6PipelineConfig& config,
    K1K6PipelineState& state
);

// Run K3 only: Fine transport
bool run_k3_fine_transport(
    const DevicePsiC& psi_in,
    DevicePsiC& psi_out,  // CRITICAL FIX: Output phase space
    const uint32_t* d_ActiveList,
    int n_active,
    const ::DeviceRLUT& dlut,
    const K1K6PipelineConfig& config,
    K1K6PipelineState& state
);

// Run K4 only: Bucket transfer
bool run_k4_bucket_transfer(
    DevicePsiC& psi_out,
    const DeviceOutflowBucket* d_OutflowBuckets,
    const int* d_CellToBucketBase,
    int Nx, int Nz,
    const float* d_E_edges,
    int N_E,
    int N_E_local
);

// Run K5 only: Weight + energy conservation audit
bool run_k5_conservation_audit(
    const DevicePsiC& psi_in,
    const DevicePsiC& psi_out,
    const double* d_EdepC,
    const double* d_AbsorbedEnergy_cutoff,
    const double* d_AbsorbedEnergy_nuclear,
    const double* d_BoundaryLoss_energy,
    const double* d_prev_EdepC,
    const double* d_prev_AbsorbedEnergy_cutoff,
    const double* d_prev_AbsorbedEnergy_nuclear,
    const double* d_prev_BoundaryLoss_energy,
    const float* d_AbsorbedWeight_cutoff,
    const float* d_AbsorbedWeight_nuclear,
    const float* d_BoundaryLoss_weight,
    const float* d_prev_AbsorbedWeight_cutoff,
    const float* d_prev_AbsorbedWeight_nuclear,
    const float* d_prev_BoundaryLoss_weight,
    float source_out_of_grid_weight,
    float source_slot_dropped_weight,
    double source_out_of_grid_energy,
    double source_slot_dropped_energy,
    float transport_dropped_weight,
    double transport_dropped_energy,
    int include_source_terms,
    const float* d_E_edges,
    int N_E,
    int N_E_local,
    AuditReport* d_report,
    int Nx, int Nz
);

// Run K6 only: Swap buffers
void run_k6_swap_buffers(
    DevicePsiC*& psi_in,
    DevicePsiC*& psi_out
);

// ============================================================================
// Utility Functions
// ============================================================================

// Initialize pipeline state from configuration
bool init_pipeline_state(
    const K1K6PipelineConfig& config,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    K1K6PipelineState& state
);

// Reset pipeline state for new iteration
// Clears counters, tallies, and batch maps
void reset_pipeline_state(
    K1K6PipelineState& state,
    int Nx, int Nz
);

// Copy audit report from device to host
AuditReport get_audit_report(
    const K1K6PipelineState& state,
    int Nx, int Nz
);

// Compute coarse energy-block threshold from energy in MeV.
inline int compute_b_E_threshold(float E_threshold, const EnergyGrid& e_grid, int N_E_local) {
    int E_bin = e_grid.FindBin(E_threshold);
    return E_bin / N_E_local;
}

// Legacy helper alias (deprecated): use compute_b_E_threshold.
inline int compute_b_E_trigger(float E_trigger, const EnergyGrid& e_grid, int N_E_local) {
    return compute_b_E_threshold(E_trigger, e_grid, N_E_local);
}

// ============================================================================
// CUDA Error Checking
// ============================================================================

// Macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

// Version that doesn't return (for void functions)
#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

} // namespace sm_2d
