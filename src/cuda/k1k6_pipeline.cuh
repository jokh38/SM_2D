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
// - K5: Weight audit (conservation check)
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
    float E_trigger;           // Energy threshold for fine transport [MeV]
    float weight_active_min;   // Minimum weight for active cell

    // Coarse transport config (K2)
    float E_coarse_max;        // Maximum energy for coarse transport [MeV]
    float step_coarse;         // Coarse step size [mm]
    int n_steps_per_cell;      // Number of sub-steps per cell

    // Fine transport is implicitly used when E < E_trigger

    // Grid dimensions
    int Nx, Nz;                // Spatial grid
    float dx, dz;              // Cell sizes [mm]

    // Phase-space grid
    int N_theta, N_E;          // Global bins
    int N_theta_local, N_E_local;  // Local bins per block
};

// ============================================================================
// Helper Kernel Declarations
// ============================================================================

// Inject source particle into PsiC
__global__ void inject_source_kernel(
    DevicePsiC psi,
    int source_cell,
    float theta0, float E0, float W_total,
    float x_in_cell, float z_in_cell,
    float dx, float dz,
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
// Output: CoarseList[Nx*Nz] (uint32_t, compacted indices)
// Returns: Number of coarse cells (n_coarse)
__global__ void compact_coarse_list(
    const uint8_t* __restrict__ ActiveMask,
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
    float* d_AbsorbedWeight_nuclear;
    double* d_AbsorbedEnergy_nuclear;
    float* d_BoundaryLoss_weight;
    double* d_BoundaryLoss_energy;

    // Bucket system
    DeviceOutflowBucket* d_OutflowBuckets;  // [Nx*Nz*4]

    // Audit
    AuditReport* d_audit_report;
    double* d_weight_in;
    double* d_weight_out;

    // Device memory ownership
    bool owns_device_memory;

    K1K6PipelineState()
        : psi_in(nullptr), psi_out(nullptr)
        , d_ActiveMask(nullptr), d_ActiveList(nullptr), d_CoarseList(nullptr)
        , n_active(0), n_coarse(0)
        , d_n_active(nullptr), d_n_coarse(nullptr)
        , d_EdepC(nullptr)
        , d_AbsorbedWeight_cutoff(nullptr), d_AbsorbedWeight_nuclear(nullptr)
        , d_AbsorbedEnergy_nuclear(nullptr)
        , d_BoundaryLoss_weight(nullptr), d_BoundaryLoss_energy(nullptr)
        , d_OutflowBuckets(nullptr)
        , d_audit_report(nullptr)
        , d_weight_in(nullptr), d_weight_out(nullptr)
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
// 1. K1: Identify active cells based on E_trigger and weight thresholds
// 2. Compact active list and coarse list
// 3. K2: Coarse transport for high-energy cells (ActiveMask = 0)
// 4. K3: Fine transport for low-energy cells (ActiveMask = 1)
// 5. K4: Bucket transfer for boundary crossings
// 6. K5: Weight audit (conservation check)
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
    int Nx, int Nz,
    int b_E_trigger,
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
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
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
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    K1K6PipelineState& state
);

// Run K4 only: Bucket transfer
bool run_k4_bucket_transfer(
    DevicePsiC& psi_out,
    const DeviceOutflowBucket* d_OutflowBuckets,
    int Nx, int Nz
);

// Run K5 only: Weight audit
bool run_k5_weight_audit(
    const DevicePsiC& psi_in,
    const DevicePsiC& psi_out,
    const float* d_AbsorbedWeight_cutoff,
    const float* d_AbsorbedWeight_nuclear,
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
    K1K6PipelineState& state
);

// Reset pipeline state for new iteration
// Clears buckets, counters, and tallies
void reset_pipeline_state(
    K1K6PipelineState& state,
    int Nx, int Nz
);

// Copy audit report from device to host
AuditReport get_audit_report(
    const K1K6PipelineState& state
);

// Compute b_E_trigger from E_trigger
// Helper to convert energy threshold to coarse block index
inline int compute_b_E_trigger(float E_trigger, const EnergyGrid& e_grid, int N_E_local) {
    int E_bin = e_grid.FindBin(E_trigger);
    return E_bin / N_E_local;
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
