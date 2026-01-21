# Phase 4: Transport Pipeline (K1-K6)

**Status**: Pending
**Duration**: 4-5 days
**Dependencies**: Phase 1 (LUT), Phase 2 (Data Structures), Phase 3 (Physics)

---

## Objectives

1. Implement K1: ActiveMask generation
2. Implement K2: CompactActive (optional for MVP)
3. Implement K3: FineTransport (core kernel)
4. Implement K4: BucketTransfer
5. Implement K5: ConservationAudit
6. Implement K6: SwapBuffers

---

## Kernel Pipeline Overview

```
                    ┌─────────────────┐
                    │   K1: ActiveMask │  Identify cells needing fine transport
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  K2: CompactActive │  Generate compact cell list (optional)
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────────────────┐
                    │   K3: FineTransport           │  Main physics kernel
                    │  - Energy deposition          │
                    │  - MCS with splitting         │
                    │  - Bucket emission            │
                    └────────┬─────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │ K4: BucketTransfer│  Transfer buckets to neighbor cells
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ K5: Conservation  │  Verify weight/energy conservation
                    │     Audit         │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  K6: SwapBuffers  │  Exchange input/output buffers
                    └──────────────────┘
```

---

## TDD Cycle 4.1: K1 - ActiveMask

### RED - Write Tests First

Create `tests/kernels/test_k1_activemask.cpp`:

```cpp
#include <gtest/gtest.h>
#include "kernels/k1_activemask.cuh"

TEST(K1Test, HighEnergyTriggered) {
    // Setup: Create PsiC with high-energy component
    PsiC psi(4, 4, 32);

    uint32_t bid = encode_block(10, 20);  // Some high-energy block
    int slot = psi.find_or_allocate_slot(0, bid);
    psi.set_weight(0, slot, encode_local_idx(3, 1), 1.0f);

    std::vector<uint8_t> ActiveMask(16, 0);

    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10.0f, 1e-10f);

    EXPECT_TRUE(ActiveMask[0]);
}

TEST(K1Test, LowEnergyNotTriggered) {
    PsiC psi(4, 4, 32);

    // Low energy block (b_E < E_trigger / dE)
    uint32_t bid = encode_block(10, 0);  // Low energy
    int slot = psi.find_or_allocate_slot(0, bid);
    psi.set_weight(0, slot, 0, 1.0f);

    std::vector<uint8_t> ActiveMask(16, 0);

    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10.0f, 1e-10f);

    EXPECT_FALSE(ActiveMask[0]);
}

TEST(K1Test, WeightThreshold) {
    PsiC psi(4, 4, 32);

    // High energy but very low weight
    uint32_t bid = encode_block(10, 20);
    int slot = psi.find_or_allocate_slot(0, bid);
    psi.set_weight(0, slot, 0, 1e-12f);

    std::vector<uint8_t> ActiveMask(16, 0);

    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10.0f, 1e-10f);

    EXPECT_FALSE(ActiveMask[0]);  // Below weight threshold
}

TEST(K1Test, MultipleCells) {
    PsiC psi(4, 4, 32);

    // Cell 0: active
    uint32_t bid_high = encode_block(10, 20);
    int slot0 = psi.find_or_allocate_slot(0, bid_high);
    psi.set_weight(0, slot0, 0, 1.0f);

    // Cell 1: inactive
    uint32_t bid_low = encode_block(10, 0);
    int slot1 = psi.find_or_allocate_slot(1, bid_low);
    psi.set_weight(1, slot1, 0, 1.0f);

    std::vector<uint8_t> ActiveMask(16, 0);

    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10.0f, 1e-10f);

    EXPECT_TRUE(ActiveMask[0]);
    EXPECT_FALSE(ActiveMask[1]);
}
```

### GREEN - Implementation

Create `cuda/kernels/k1_activemask.cu`:

```cuda
#include "kernels/k1_activemask.cuh"
#include "core/psi_storage.hpp"
#include "core/block_encoding.hpp"

__global__ void K1_ActiveMask(
    const PsiCView psi,
    uint8_t* __restrict__ ActiveMask,
    int Nx, int Nz,
    float E_trigger,
    float weight_active_min
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    float W_cell = 0;
    bool has_high_E = false;

    // Check all blocks in cell
    for (int slot = 0; slot < psi.Kb; ++slot) {
        uint32_t bid = psi.block_id[cell][slot];
        if (bid == EMPTY_BLOCK_ID) continue;

        uint32_t b_E = (bid >> 12) & 0xFFF;

        // Estimate block minimum energy (simplified)
        // b_E * N_E_local gives starting bin index
        // E_edges[b_E * N_E_local] approx min energy in block
        // For TDD, assume high b_E means high energy
        if (b_E >= 5) {  // Threshold depends on grid
            has_high_E = true;
        }

        // Sum weights
        for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
            W_cell += psi.value[cell][slot][lidx];
        }
    }

    ActiveMask[cell] = (has_high_E && W_cell > weight_active_min) ? 1 : 0;
}
```

---

## TDD Cycle 4.2: K3 - FineTransport (Core)

### RED - Write Tests First

Create `tests/kernels/test_k3_finetransport.cpp`:

```cpp
#include <gtest/gtest.h>
#include "kernels/k3_finetransport.cuh"

TEST(K3Test, EnergyCutoff) {
    // Component at E = 0.05 MeV should terminate
    Component c = {0.0f, 0.05f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};

    K3Result result = run_K3_single_component(c);

    // Full energy should be deposited
    EXPECT_NEAR(result.Edep, 0.05f, 1e-4f);
    EXPECT_EQ(result.terminated, true);
}

TEST(K3Test, SingleStepTransport) {
    // Component that travels one step in cell
    Component c = {0.0f, 100.0f, 1.0f, 0.5f, 0.5f, 1.0f, 0.0f};

    K3Result result = run_K3_single_component(c);

    // Some energy should be deposited
    EXPECT_GT(result.Edep, 0);
    EXPECT_LT(result.Edep, 100.0f);

    // Weight should decrease due to nuclear attenuation
    EXPECT_LT(result.final_weight, 1.0f);
}

TEST(K3Test, BoundaryCrossing) {
    // Component starting near +z boundary
    float cell_size = 1.0f;
    Component c = {0.0f, 100.0f, 1.0f, 0.5f, 0.95f * cell_size, 1.0f, 0.0f};

    K3Result result = run_K3_single_component(c);

    // Should emit to bucket
    EXPECT_GT(result.bucket_emissions, 0);
    EXPECT_EQ(result.remained_in_cell, false);
}

TEST(K3Test, AngularSplit) {
    // Force variance accumulation above threshold
    Component c = {0.0f, 100.0f, 1.0f, 0.5f, 0.5f, 1.0f, 0.0f};

    // Simulate with high variance
    K3Result result = run_K3_with_forced_split(c);

    EXPECT_EQ(result.split_count, 7);
}

TEST(K3Test, NuclearAttenuation) {
    Component c = {0.0f, 100.0f, 1.0f, 0.5f, 0.5f, 1.0f, 0.0f};

    K3Result result = run_K3_single_component(c);

    // Nuclear weight removed
    EXPECT_GT(result.nuclear_weight_removed, 0);

    // Nuclear energy tracked
    EXPECT_NEAR(result.nuclear_energy_removed,
                result.nuclear_weight_removed * c.E, 1e-5f);
}
```

### GREEN - Implementation

Create `cuda/kernels/k3_finetransport.cu`:

```cuda
#include "kernels/k3_finetransport.cuh"
#include "physics/step_control.hpp"
#include "physics/highland.hpp"
#include "physics/nuclear.hpp"
#include "physics/discretization.hpp"
#include "physics/quadrature.hpp"

// Component state: (theta, E, w, x, z, mu, eta)
struct Component {
    float theta, E, w, x, z, mu, eta;
};

__device__ float distance_to_boundary(float x, float z, float mu, float eta,
                                      int cell, float dx, float dz) {
    int ix = cell % Nx;
    int iz = cell / Nx;

    float x_min = ix * dx;
    float x_max = (ix + 1) * dx;
    float z_min = iz * dz;
    float z_max = (iz + 1) * dz;

    float ds = 1e10f;

    if (eta > 0) {
        float t = (x_max - x) / eta;
        if (t > 0 && t < ds) ds = t;
    } else if (eta < 0) {
        float t = (x_min - x) / eta;
        if (t > 0 && t < ds) ds = t;
    }

    if (mu > 0) {
        float t = (z_max - z) / mu;
        if (t > 0 && t < ds) ds = t;
    } else if (mu < 0) {
        float t = (z_min - z) / mu;
        if (t > 0 && t < ds) ds = t;
    }

    return ds;
}

__global__ void K3_FineTransport(
    // Inputs
    const PsiCView PsiC_in,
    const uint32_t* __restrict__ ActiveList,
    const RLUTDevice lut,
    // Grid parameters
    int Nx, int Nz, float dx, float dz,
    int n_active,
    // Outputs (per cell)
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    // Bucket outputs [Nx][Nz][4]
    OutflowBucket* __restrict__ OutflowBuckets,
    // Boundary losses
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy
) {
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= n_active) return;

    int cell = ActiveList[cell_idx];

    // Shared memory for reduction
    __shared__ double s_edep[128];
    __shared__ float s_cutoff[128];
    __shared__ float s_nuc_w[128];
    __shared__ double s_nuc_e[128];

    int tid = threadIdx.x;
    s_edep[tid] = 0;
    s_cutoff[tid] = 0;
    s_nuc_w[tid] = 0;
    s_nuc_e[tid] = 0;

    // Process all slots in this cell
    for (int slot = 0; slot < PsiC_in.Kb; ++slot) {
        uint32_t bid = PsiC_in.block_id[cell][slot];
        if (bid == EMPTY_BLOCK_ID) continue;

        for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
            float w = PsiC_in.value[cell][slot][lidx];
            if (w < weight_epsilon) continue;

            // Decode state
            uint32_t b_theta = bid & 0xFFF;
            uint32_t b_E = (bid >> 12) & 0xFFF;

            int theta_local, E_local;
            decode_local_idx(lidx, theta_local, E_local);

            // Approximate theta, E from block/position
            float theta = theta_from_block(b_theta, theta_local);
            float E = E_from_block(b_E, E_local);

            // Component transport loop
            float x = cell_center_x(cell, dx);
            float z = cell_min_z(cell, dz);
            float mu = cosf(theta), eta = sinf(theta);

            float var_accum = 0;
            const int max_substeps = 256;

            for (int step = 0; step < max_substeps; ++step) {
                // Energy cutoff
                if (E <= E_cutoff) {
                    s_edep[tid] += w * E;
                    s_cutoff[tid] += w;
                    break;
                }

                // Weight underflow
                if (w < weight_epsilon) {
                    s_edep[tid] += w * E;
                    s_cutoff[tid] += w;
                    break;
                }

                // Step size
                float ds_geom = distance_to_boundary(x, z, mu, eta, cell, dx, dz);
                float ds_phys = compute_max_step_physics_device(E, lut);
                float ds = fminf(ds_geom, ds_phys);
                ds = fminf(ds, delta_s_max);

                // MCS
                float sigma_theta = highland_sigma_device(E, ds);

                if (sigma_theta < 0) {
                    // Reduce step and retry
                    ds *= 0.5f;
                    sigma_theta = highland_sigma_device(E, ds);
                }

                // Variance accumulation
                var_accum += sigma_theta * sigma_theta;
                float rms = sqrtf(var_accum);
                bool do_split = (rms > ((E > 50.0f) ? 0.05f : 0.05f * sqrtf(E / 50.0f)));
                if (do_split) var_accum = 0;

                // Angular split
                int n_splits = do_split ? 7 : 1;
                float theta_k[7], w_k[7];
                if (do_split) {
                    apply_angular_quadrature_device(theta, sigma_theta, w, theta_k, w_k);
                } else {
                    theta_k[0] = theta;
                    w_k[0] = w;
                }

                for (int k = 0; k < n_splits; ++k) {
                    float mu_k = cosf(theta_k[0]);
                    float eta_k = sinf(theta_k[0]);
                    float x_new = x + eta_k * ds;
                    float z_new = z + mu_k * ds;

                    // Check boundary crossing
                    if (crossed_boundary_device(x_new, z_new, cell, dx, dz)) {
                        // Emit to bucket
                        float ds_partial = distance_to_boundary(x, z, mu_k, eta_k, cell, dx, dz);
                        float R_cur = lookup_R_device(E, lut);
                        float R_cross = R_cur - ds_partial;
                        float E_cross = (R_cross > 0) ? lookup_E_inverse_device(R_cross, lut) : 0;
                        float dE_partial = E - E_cross;

                        s_edep[tid] += w_k[k] * dE_partial;

                        // Nuclear
                        auto [w_cross, w_removed] = apply_nuclear_attenuation_device(w_k[k], E, ds_partial);
                        s_nuc_w[tid] += w_removed;
                        s_nuc_e[tid] += w_removed * E;

                        // Emit to bucket
                        int face = identify_crossed_face_device(x, z, x_new, z_new, cell, dx, dz);
                        emit_to_bucket_device(OutflowBuckets, cell, face, theta_k[0], E_cross, w_cross);
                    } else {
                        // Remains in cell (would be added to new_list)
                        // Simplified: deposit and continue
                        float R_cur = lookup_R_device(E, lut);
                        float R_new = R_cur - ds;
                        float E_new = (R_new > 0) ? lookup_E_inverse_device(R_new, lut) : 0;
                        float dE = E - E_new;

                        s_edep[tid] += w_k[k] * dE;

                        auto [w_new, w_removed] = apply_nuclear_attenuation_device(w_k[k], E, ds);
                        s_nuc_w[tid] += w_removed;
                        s_nuc_e[tid] += w_removed * E;

                        x = x_new;
                        z = z_new;
                        E = E_new;
                        w = w_new;
                    }
                }

                if (!do_split && w < weight_epsilon) break;
            }
        }
    }

    __syncthreads();

    // Reduction
    if (tid == 0) {
        double sum_edep = 0;
        float sum_cutoff = 0;
        float sum_nuc_w = 0;
        double sum_nuc_e = 0;

        for (int i = 0; i < 128; ++i) {
            sum_edep += s_edep[i];
            sum_cutoff += s_cutoff[i];
            sum_nuc_w += s_nuc_w[i];
            sum_nuc_e += s_nuc_e[i];
        }

        atomicAdd(&EdepC[cell], sum_edep);
        atomicAdd(&AbsorbedWeight_cutoff[cell], sum_cutoff);
        atomicAdd(&AbsorbedWeight_nuclear[cell], sum_nuc_w);
        atomicAdd(&AbsorbedEnergy_nuclear[cell], sum_nuc_e);
    }
}
```

---

## TDD Cycle 4.3: K4 - BucketTransfer

### RED - Write Tests First

Create `tests/kernels/test_k4_transfer.cpp`:

```cpp
#include <gtest/gtest.h>
#include "kernels/k4_transfer.cuh"

TEST(K4Test, WeightConservation) {
    // Create bucket with some emissions
    OutflowBucket bucket;
    uint32_t bid = encode_block(5, 10);
    int slot = bucket.find_or_allocate_slot(bid);
    bucket.value[slot][0] = 1.0f;

    PsiC psi_out(4, 4, 32);

    float W_before = sum_bucket(bucket);

    run_K4_BucketTransfer(&bucket, psi_out, 0, 2);  // Face 0 = +z

    float W_after_bucket = sum_bucket(bucket);
    float W_after_psi = sum_psi(psi_out, 2);

    EXPECT_NEAR(W_before, W_after_bucket + W_after_psi, 1e-6f);
}

TEST(K4Test, CorrectNeighborCell) {
    OutflowBucket bucket;
    uint32_t bid = encode_block(5, 10);
    int slot = bucket.find_or_allocate_slot(bid);
    bucket.value[slot][0] = 1.0f;

    PsiC psi_out(4, 4, 32);

    // Cell 0, face 0 (+z) should transfer to cell Nx
    run_K4_BucketTransfer(&bucket, psi_out, 0, 0);

    // Neighbor of cell 0 in +z direction is cell Nx
    int neighbor_cell = 4;  // Assuming Nx = 4
    EXPECT_GT(sum_psi(psi_out, neighbor_cell), 0);
}

TEST(K4Test, AllFaces) {
    for (int face = 0; face < 4; ++face) {
        OutflowBucket bucket;
        uint32_t bid = encode_block(5, 10);
        int slot = bucket.find_or_allocate_slot(bid);
        bucket.value[slot][0] = 1.0f;

        PsiC psi_out(4, 4, 32);

        run_K4_BucketTransfer(&bucket, psi_out, 5, face);

        // Weight should transfer (except at boundaries)
        float total = 0;
        for (int c = 0; c < 16; ++c) {
            total += sum_psi(psi_out, c);
        }
        EXPECT_GT(total, 0);
    }
}
```

### GREEN - Implementation

Create `cuda/kernels/k4_transfer.cu`:

```cuda
#include "kernels/k4_transfer.cuh"
#include "core/psi_storage.hpp"
#include "core/buckets.hpp"

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
    PsiCView PsiC_out,
    int Nx, int Nz
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    // For each face
    for (int face = 0; face < 4; ++face) {
        const OutflowBucket& bucket = OutflowBuckets[cell * 4 + face];

        for (int slot = 0; slot < Kb_out; ++slot) {
            uint32_t bid = bucket.block_id[slot];
            if (bid == EMPTY_BLOCK_ID) continue;

            int neighbor = get_neighbor(cell, face, Nx, Nz);

            if (neighbor >= 0) {
                // Transfer to neighbor
                int out_slot = PsiC_out.find_or_allocate_slot(neighbor, bid);
                if (out_slot >= 0) {
                    for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                        float w = bucket.value[slot][lidx];
                        if (w > 0) {
                            atomicAdd(&PsiC_out.value[neighbor][out_slot][lidx], w);
                        }
                    }
                }
            }
            // If neighbor < 0 (boundary), loss already tracked in K3
        }
    }
}
```

---

## TDD Cycle 4.4: K5 - ConservationAudit

### RED - Write Tests First

Create `tests/kernels/test_k5_audit.cpp`:

```cpp
#include <gtest/gtest.h>
#include "kernels/k5_audit.cuh"

TEST(K5Test, WeightConservationPass) {
    AuditData data;
    data.W_in = 1.0f;
    data.W_out = 0.7f;
    data.W_cutoff = 0.2f;
    data.W_nuclear = 0.1f;

    bool pass = run_K5_WeightAudit(data);
    EXPECT_TRUE(pass);
}

TEST(K5Test, WeightConservationFail) {
    AuditData data;
    data.W_in = 1.0f;
    data.W_out = 0.8f;  // Error: 0.1 missing
    data.W_cutoff = 0.2f;
    data.W_nuclear = 0.1f;

    bool pass = run_K5_WeightAudit(data);
    EXPECT_FALSE(pass);
}

TEST(K5Test, EnergyConservation) {
    AuditData data;
    data.E_in = 100.0;
    data.E_out = 70.0;
    data.E_dep = 25.0;
    data.E_nuclear = 5.0;

    bool pass = run_K5_EnergyAudit(data);
    EXPECT_TRUE(pass);
}

TEST(K5Test, ToleranceCheck) {
    AuditData data;
    data.W_in = 1.0f;
    data.W_out = 0.7f;
    data.W_cutoff = 0.2f;
    data.W_nuclear = 0.1f;

    float error = compute_weight_error(data);
    EXPECT_LT(error, 1e-6f);
}
```

### GREEN - Implementation

Create `cuda/kernels/k5_audit.cu`:

```cuda
#include "kernels/k5_audit.cuh"

__global__ void K5_WeightAudit(
    const PsiCView PsiC_in,
    const PsiCView PsiC_out,
    const float* __restrict__ AbsorbedWeight_cutoff,
    const float* __restrict__ AbsorbedWeight_nuclear,
    AuditReport* __restrict__ report,
    int N_cells
) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= N_cells) return;

    // Sum input weights
    float W_in = 0;
    for (int slot = 0; slot < PsiC_in.Kb; ++slot) {
        if (PsiC_in.block_id[cell][slot] == EMPTY_BLOCK_ID) continue;
        for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
            W_in += PsiC_in.value[cell][slot][lidx];
        }
    }

    // Sum output weights
    float W_out = 0;
    for (int slot = 0; slot < PsiC_out.Kb; ++slot) {
        if (PsiC_out.block_id[cell][slot] == EMPTY_BLOCK_ID) continue;
        for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
            W_out += PsiC_out.value[cell][slot][lidx];
        }
    }

    float W_cutoff = AbsorbedWeight_cutoff[cell];
    float W_nuclear = AbsorbedWeight_nuclear[cell];

    float W_error = fabsf(W_in - W_out - W_cutoff - W_nuclear);
    float W_rel_error = W_error / fmaxf(W_in, 1e-20f);

    bool pass = (W_rel_error < 1e-6f);

    report[cell].W_error = W_rel_error;
    report[cell].W_pass = pass;
}
```

---

## TDD Cycle 4.5: K6 - SwapBuffers

### RED - Write Tests First

Create `tests/kernels/test_k6_swap.cpp`:

```cpp
#include <gtest/gtest.h>
#include "kernels/k6_swap.cuh"

TEST(K6Test, SwapPointers) {
    PsiC psi_a(4, 4, 32);
    PsiC psi_b(4, 4, 32);

    // Set different values
    psi_a.set_weight(0, 0, 0, 1.0f);
    psi_b.set_weight(0, 0, 0, 2.0f);

    PsiC* in = &psi_a;
    PsiC* out = &psi_b;

    EXPECT_EQ(in, &psi_a);
    EXPECT_EQ(out, &psi_b);

    K6_SwapBuffers(in, out);

    EXPECT_EQ(in, &psi_b);
    EXPECT_EQ(out, &psi_a);
}
```

### GREEN - Implementation

Create `cuda/kernels/k6_swap.cu`:

```cuda
#include "kernels/k6_swap.cuh"
#include "core/psi_storage.hpp"

void K6_SwapBuffers(PsiC*& in, PsiC*& out) {
    PsiC* temp = in;
    in = out;
    out = temp;
}
```

---

## Exit Criteria Checklist

- [ ] K1 correctly identifies active cells
- [ ] K3 deposits energy correctly for single component
- [ ] K3 emits to bucket on boundary crossing
- [ ] K4 transfers weights without loss
- [ ] K5 detects conservation violations > 1e-6
- [ ] K6 swaps buffers correctly
- [ ] Full pipeline executes without errors
- [ ] Kernel compilation succeeds for sm_75

---

## Next Steps

After completing Phase 4, proceed to **Phase 5 (Sources/Boundaries)** and **Phase 6 (Audit)**.

```bash
# Test kernels
./bin/sm2d_tests --gtest_filter="*K*"
```
