/**
 * Test for Energy-Loss-Only Transport in K3 Fine Transport Kernel
 *
 * Purpose: Validate that the CUDA K3 kernel correctly implements
 *          Bethe-Bloch energy loss without other physics processes.
 *
 * Test Configuration:
 *   - Single 150 MeV proton pencil beam at (0, 0) mm
 *   - Physics: Bethe-Bloch ONLY (no straggling, no nuclear, no MCS)
 *   - Grid: 200 mm depth, 20 mm width
 *
 * Expected Results:
 *   - Bragg peak at ~158 mm depth
 *   - No lateral spread (all dose on central axis)
 *   - Sharp distal falloff (no straggling)
 *   - Energy conservation
 */

#include <gtest/gtest.h>
#include "kernels/k3_finetransport.cuh"
#include "kernels/k4_transfer.cuh"
#include "core/grids.hpp"
#include "physics/physics.hpp"
#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include "device/device_lut.cuh"
#include "device/device_bucket.cuh"
#include "device/device_psic.cuh"
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>

// ============================================================================
// Simple Source Injection Kernel (local to test)
// ============================================================================

__global__ void test_inject_source_kernel(
    DevicePsiC psi,
    int cell,
    float theta, float E, float weight,
    float x, float z,
    float dx, float dz,
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local
) {
    if (weight <= 0.0f) return;
    if (cell >= psi.N_cells) return;

    // Clamp position to cell bounds
    x = fmaxf(0.0f, fminf(x, dx));
    z = fmaxf(0.0f, fminf(z, dz));

    // Calculate offsets from cell center
    float x_offset = x - dx * 0.5f;
    float z_offset = z - dz * 0.5f;

    // Get sub-cell bins
    int x_sub = get_x_sub_bin(x_offset, dx);
    int z_sub = get_z_sub_bin(z_offset, dz);

    // Clamp values to grid bounds
    float theta_min = theta_edges[0];
    float theta_max = theta_edges[N_theta];
    float E_min = E_edges[0];
    float E_max = E_edges[N_E];

    theta = fmaxf(theta_min, fminf(theta, theta_max));
    E = fmaxf(E_min, fminf(E, E_max));

    // Calculate continuous bin positions
    float dtheta = (theta_max - theta_min) / N_theta;
    float theta_cont = (theta - theta_min) / dtheta;
    int theta_bin = (int)theta_cont;
    float frac_theta = theta_cont - theta_bin;

    // Binary search for energy bin
    int E_bin = 0;
    if (E <= E_edges[0]) {
        E_bin = 0;
    } else if (E >= E_edges[N_E]) {
        E_bin = N_E - 1;
    } else {
        int lo = 0, hi = N_E;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (E_edges[mid + 1] <= E) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        E_bin = lo;
    }

    float E_bin_lower = E_edges[E_bin];
    float E_bin_upper = E_edges[E_bin + 1];
    float frac_E = (E_bin_upper - E_bin_lower) > 1e-10f ? (E - E_bin_lower) / (E_bin_upper - E_bin_lower) : 0.0f;

    if (theta_bin >= N_theta - 1) {
        theta_bin = N_theta - 1;
        frac_theta = 0.0f;
    }
    if (E_bin >= N_E - 1) {
        E_bin = N_E - 1;
        frac_E = 0.0f;
    }

    // Single-bin emission
    if (frac_theta >= 0.5f && theta_bin < N_theta - 1) theta_bin++;
    if (frac_E >= 0.5f && E_bin < N_E - 1) E_bin++;

    uint32_t b_theta = theta_bin / N_theta_local;
    uint32_t b_E = E_bin / N_E_local;
    uint32_t bid = encode_block(b_theta, b_E);

    int theta_local = theta_bin % N_theta_local;
    int E_local = E_bin % N_E_local;

    uint16_t lidx = encode_local_idx_4d(theta_local, E_local, x_sub, z_sub);

    // Allocate slot and add weight
    size_t base_slot = cell * psi.Kb;
    for (int slot = 0; slot < psi.Kb; ++slot) {
        uint32_t expected = DEVICE_EMPTY_SLOT;
        if (atomicCAS(&psi.block_id[base_slot + slot], expected, bid) == expected) {
            psi.block_id[base_slot + slot] = bid;
            size_t value_base = (cell * psi.Kb + slot) * LOCAL_BINS;
            psi.value[value_base + lidx] = weight;
            break;
        }
    }
}

// ============================================================================
// Simple Bucket Clear Kernel (local to test)
// ============================================================================

__global__ void test_clear_buckets_kernel(
    DeviceOutflowBucket* buckets,
    int num_buckets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_buckets) return;

    DeviceOutflowBucket& bucket = buckets[idx];
    for (int i = 0; i < DEVICE_Kb_out; ++i) {
        bucket.block_id[i] = DEVICE_EMPTY_BLOCK_ID;
        bucket.local_count[i] = 0;
        for (int j = 0; j < DEVICE_LOCAL_BINS; ++j) {
            bucket.value[i][j] = 0.0f;
        }
    }
}

// ============================================================================
// Test Fixture
// ============================================================================

class EnergyLossOnlyTest : public ::testing::Test {
protected:
    // Grid dimensions
    static constexpr int Nx = 5;       // 5 cells in x (20 mm total)
    static constexpr int Nz = 100;     // 100 cells in z (200 mm total)
    static constexpr float dx = 4.0f;   // 4 mm per cell
    static constexpr float dz = 2.0f;   // 2 mm per cell

    // Phase space dimensions
    static constexpr int N_theta = 36;  // Angular bins
    static constexpr int N_theta_local = 4;
    static constexpr int N_E_local = 2;

    // Source particle
    static constexpr float E0 = 150.0f;  // 150 MeV protons
    static constexpr float theta0 = 0.0f;  // Normal incidence
    static constexpr float W_total = 1.0f;  // Unit weight
    static constexpr int source_cell = 0;  // First cell (z=0, x=0)

    // Device memory
    DevicePsiC psi_in, psi_out;
    uint32_t* d_ActiveList;
    double* d_EdepC;
    float* d_AbsorbedWeight_cutoff;
    float* d_AbsorbedWeight_nuclear;
    double* d_AbsorbedEnergy_nuclear;
    float* d_BoundaryLoss_weight;
    double* d_BoundaryLoss_energy;
    DeviceOutflowBucket* d_OutflowBuckets;
    float* d_theta_edges;
    float* d_E_edges;

    // Host copies for validation
    std::vector<double> h_EdepC;

    // Pointers to grids (not copyable)
    std::unique_ptr<EnergyGrid> e_grid_ptr;
    std::unique_ptr<AngularGrid> a_grid_ptr;
    std::unique_ptr<RLUT> lut_ptr;
    std::unique_ptr<::DeviceLUTWrapper> device_lut_ptr;
    int N_E;

    void SetUp() override {
        // Create grids (use new to avoid copy assignment issues)
        // Use finer energy resolution at high energies for accurate energy tracking
        // The 0.25 MeV bin width ensures particles lose energy smoothly across bins
        std::vector<std::tuple<float, float, float>> energy_groups = {
            {0.1f, 2.0f, 0.1f},    // 19 bins
            {2.0f, 20.0f, 0.2f},   // 90 bins (was 0.25)
            {20.0f, 100.0f, 0.25f}, // 320 bins (was 0.5)
            {100.0f, 250.0f, 0.25f} // 600 bins (was 1.0) - FINER for better tracking
        };

        e_grid_ptr.reset(new EnergyGrid(energy_groups));
        N_E = e_grid_ptr->N_E;
        a_grid_ptr.reset(new AngularGrid(-M_PI/2.0f, M_PI/2.0f, N_theta));

        // Generate LUT
        lut_ptr.reset(new RLUT(std::move(GenerateRLUT(*e_grid_ptr))));

        // Initialize device LUT
        device_lut_ptr.reset(new ::DeviceLUTWrapper());
        if (!device_lut_ptr->init(*lut_ptr)) {
            FAIL() << "Failed to initialize device LUT";
        }

        // Allocate device memory
        if (!device_psic_init(psi_in, Nx, Nz)) {
            FAIL() << "Failed to allocate psi_in";
        }
        if (!device_psic_init(psi_out, Nx, Nz)) {
            FAIL() << "Failed to allocate psi_out";
        }

        cudaMalloc(&d_ActiveList, Nx * Nz * sizeof(uint32_t));
        cudaMalloc(&d_EdepC, Nx * Nz * sizeof(double));
        cudaMalloc(&d_AbsorbedWeight_cutoff, Nx * Nz * sizeof(float));
        cudaMalloc(&d_AbsorbedWeight_nuclear, Nx * Nz * sizeof(float));
        cudaMalloc(&d_AbsorbedEnergy_nuclear, Nx * Nz * sizeof(double));
        cudaMalloc(&d_BoundaryLoss_weight, Nx * Nz * sizeof(float));
        cudaMalloc(&d_BoundaryLoss_energy, Nx * Nz * sizeof(double));
        cudaMalloc(&d_OutflowBuckets, Nx * Nz * 4 * sizeof(DeviceOutflowBucket));
        cudaMalloc(&d_theta_edges, (N_theta + 1) * sizeof(float));
        cudaMalloc(&d_E_edges, (N_E + 1) * sizeof(float));

        // Copy grid edges to device
        cudaMemcpy(d_theta_edges, a_grid_ptr->edges.data(), (N_theta + 1) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_E_edges, e_grid_ptr->edges.data(), (N_E + 1) * sizeof(float), cudaMemcpyHostToDevice);

        // Initialize output arrays to zero
        cudaMemset(d_EdepC, 0, Nx * Nz * sizeof(double));
        cudaMemset(d_AbsorbedWeight_cutoff, 0, Nx * Nz * sizeof(float));
        cudaMemset(d_AbsorbedWeight_nuclear, 0, Nx * Nz * sizeof(float));
        cudaMemset(d_AbsorbedEnergy_nuclear, 0, Nx * Nz * sizeof(double));
        cudaMemset(d_BoundaryLoss_weight, 0, Nx * Nz * sizeof(float));
        cudaMemset(d_BoundaryLoss_energy, 0, Nx * Nz * sizeof(double));

        // Resize host array
        h_EdepC.resize(Nx * Nz);

        // Inject source particle
        inject_source();
    }

    void TearDown() override {
        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
        cudaFree(d_ActiveList);
        cudaFree(d_EdepC);
        cudaFree(d_AbsorbedWeight_cutoff);
        cudaFree(d_AbsorbedWeight_nuclear);
        cudaFree(d_AbsorbedEnergy_nuclear);
        cudaFree(d_BoundaryLoss_weight);
        cudaFree(d_BoundaryLoss_energy);
        cudaFree(d_OutflowBuckets);
        cudaFree(d_theta_edges);
        cudaFree(d_E_edges);
    }

    void inject_source() {
        // Source at center of first cell
        float x_in_cell = dx / 2.0f;
        float z_in_cell = dz / 2.0f;

        // Launch source injection kernel
        test_inject_source_kernel<<<1, 1>>>(
            psi_in,
            source_cell,
            theta0, E0, W_total,
            x_in_cell, z_in_cell,
            dx, dz,
            d_theta_edges, d_E_edges,
            N_theta, N_E,
            N_theta_local, N_E_local
        );

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            FAIL() << "Source injection failed: " << cudaGetErrorString(err);
        }
    }

    void run_transport(bool enable_straggling, bool enable_nuclear, bool enable_mcs) {
        int max_iterations = 200;  // Enough iterations for 200 mm depth
        int threads = 256;

        for (int iter = 0; iter < max_iterations; ++iter) {
            // Build active list for this iteration
            // For simplicity, mark all non-empty cells as active
            std::vector<uint32_t> h_ActiveList;
            for (int cell = 0; cell < Nx * Nz; ++cell) {
                // Check if cell has any particles
                size_t base_slot = cell * DEVICE_Kb;
                bool has_particle = false;
                for (int slot = 0; slot < DEVICE_Kb && !has_particle; ++slot) {
                    uint32_t bid;
                    cudaMemcpy(&bid, &psi_in.block_id[base_slot + slot], sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    if (bid != DEVICE_EMPTY_SLOT) {
                        has_particle = true;
                    }
                }
                if (has_particle) {
                    h_ActiveList.push_back(cell);
                }
            }

            int n_active = h_ActiveList.size();
            if (n_active == 0) break;  // No more particles

            // Copy active list to device
            cudaMemcpy(d_ActiveList, h_ActiveList.data(), n_active * sizeof(uint32_t), cudaMemcpyHostToDevice);

            // Clear buckets
            int n_buckets = Nx * Nz * 4;
            test_clear_buckets_kernel<<<(n_buckets + 255) / 256, 256>>>(d_OutflowBuckets, n_buckets);

            // Clear output phase space
            device_psic_clear(psi_out);

            // Launch K3 kernel
            int blocks = (n_active + threads - 1) / threads;

            K3_FineTransport<<<blocks, threads>>>(
                psi_in.block_id,
                psi_in.value,
                d_ActiveList,
                Nx, Nz, dx, dz,
                n_active,
                device_lut_ptr->dlut,
                d_theta_edges,
                d_E_edges,
                N_theta, N_E,
                N_theta_local, N_E_local,
                enable_straggling,
                enable_nuclear,
                enable_mcs,
                d_EdepC,
                d_AbsorbedWeight_cutoff,
                d_AbsorbedWeight_nuclear,
                d_AbsorbedEnergy_nuclear,
                d_BoundaryLoss_weight,
                d_BoundaryLoss_energy,
                d_OutflowBuckets,
                psi_out.block_id,
                psi_out.value
            );

            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "K3 kernel launch failed at iteration " << iter << ": " << cudaGetErrorString(err) << std::endl;
                break;
            }

            // K4: Transfer particles from buckets to output phase space
            int bucket_blocks = (n_buckets + threads - 1) / threads;
            K4_BucketTransfer<<<bucket_blocks, threads>>>(
                d_OutflowBuckets,
                psi_out.value,
                psi_out.block_id,
                Nx, Nz
            );

            cudaDeviceSynchronize();

            // Swap psi_in and psi_out for next iteration
            std::swap(psi_in, psi_out);
        }

        // Copy results to host
        cudaMemcpy(h_EdepC.data(), d_EdepC, Nx * Nz * sizeof(double), cudaMemcpyDeviceToHost);
    }

    double get_total_edep() {
        double sum = 0.0;
        for (int i = 0; i < Nx * Nz; ++i) {
            sum += h_EdepC[i];
        }
        return sum;
    }

    int get_bragg_peak_cell() {
        int max_cell = 0;
        double max_edep = 0.0;
        for (int iz = 0; iz < Nz; ++iz) {
            double row_sum = 0.0;
            for (int ix = 0; ix < Nx; ++ix) {
                row_sum += h_EdepC[iz * Nx + ix];
            }
            if (row_sum > max_edep) {
                max_edep = row_sum;
                max_cell = iz;
            }
        }
        return max_cell;
    }

    double get_lateral_spread() {
        int bragg_cell = get_bragg_peak_cell();

        double sum_w = 0.0;
        double sum_wx = 0.0;
        double sum_wx2 = 0.0;

        for (int ix = 0; ix < Nx; ++ix) {
            double x = (ix + 0.5) * dx - (Nx * dx) / 2.0;
            double w = h_EdepC[bragg_cell * Nx + ix];
            sum_w += w;
            sum_wx += w * x;
            sum_wx2 += w * x * x;
        }

        if (sum_w < 1e-10) return 0.0;

        double mean_x = sum_wx / sum_w;
        double mean_x2 = sum_wx2 / sum_w;
        double sigma_x2 = mean_x2 - mean_x * mean_x;
        return (sigma_x2 > 0) ? sqrt(sigma_x2) : 0.0;
    }
};

// ============================================================================
// Tests
// ============================================================================

TEST_F(EnergyLossOnlyTest, EnergyLossOnly) {
    std::cout << "\n=== Test: Energy Loss Only ===" << std::endl;

    run_transport(false, false, false);

    double total_edep = get_total_edep();
    int bragg_cell = get_bragg_peak_cell();
    double bragg_depth = bragg_cell * dz;
    double lateral_spread = get_lateral_spread();

    std::cout << "Total energy deposited: " << total_edep << " MeV" << std::endl;
    std::cout << "Bragg peak cell: " << bragg_cell << " (depth = " << bragg_depth << " mm)" << std::endl;
    std::cout << "Lateral spread (sigma): " << lateral_spread << " mm" << std::endl;

    // Output depth-dose data for plotting
    std::ofstream dose_file("energy_loss_dose.csv");
    dose_file << "depth_mm,dose_MeV" << std::endl;
    for (int iz = 0; iz < Nz; ++iz) {
        double row_sum = 0.0;
        for (int ix = 0; ix < Nx; ++ix) {
            row_sum += h_EdepC[iz * Nx + ix];
        }
        double depth = iz * dz;
        dose_file << depth << "," << row_sum << std::endl;
    }
    dose_file.close();
    std::cout << "Dose data saved to energy_loss_dose.csv" << std::endl;

    EXPECT_NEAR(total_edep, E0, 0.1) << "Energy conservation failed";
    EXPECT_NEAR(bragg_depth, 158.0, 10.0) << "Bragg peak position incorrect";
    EXPECT_LT(lateral_spread, 0.5) << "Lateral spread should be zero without MCS";

    std::cout << "=== Test PASSED ===" << std::endl;
}

TEST_F(EnergyLossOnlyTest, FullPhysics) {
    std::cout << "\n=== Test: Full Physics (for comparison) ===" << std::endl;

    run_transport(true, true, true);

    double total_edep = get_total_edep();
    int bragg_cell = get_bragg_peak_cell();
    double bragg_depth = bragg_cell * dz;
    double lateral_spread = get_lateral_spread();

    std::cout << "Total energy deposited: " << total_edep << " MeV" << std::endl;
    std::cout << "Bragg peak cell: " << bragg_cell << " (depth = " << bragg_depth << " mm)" << std::endl;
    std::cout << "Lateral spread (sigma): " << lateral_spread << " mm" << std::endl;

    EXPECT_GT(total_edep, E0 * 0.95) << "Too much energy lost";
    EXPECT_LE(total_edep, E0) << "Energy created from nothing";
    EXPECT_NEAR(bragg_depth, 158.0, 15.0) << "Bragg peak position incorrect";
    EXPECT_GT(lateral_spread, 0.1) << "Lateral spread should be non-zero with MCS";

    std::cout << "=== Test PASSED ===" << std::endl;
}

TEST_F(EnergyLossOnlyTest, StragglingOnly) {
    std::cout << "\n=== Test: Straggling Only ===" << std::endl;

    run_transport(true, false, false);

    double total_edep = get_total_edep();
    int bragg_cell = get_bragg_peak_cell();
    double lateral_spread = get_lateral_spread();

    std::cout << "Total energy deposited: " << total_edep << " MeV" << std::endl;
    std::cout << "Bragg peak cell: " << bragg_cell << " (depth = " << bragg_cell * dz << " mm)" << std::endl;
    std::cout << "Lateral spread (sigma): " << lateral_spread << " mm" << std::endl;

    EXPECT_NEAR(total_edep, E0, 0.1) << "Energy conservation failed";
    EXPECT_LT(lateral_spread, 0.5) << "Lateral spread should be zero without MCS";

    std::cout << "=== Test PASSED ===" << std::endl;
}

TEST_F(EnergyLossOnlyTest, NuclearOnly) {
    std::cout << "\n=== Test: Nuclear Only ===" << std::endl;

    run_transport(false, true, false);

    double total_edep = get_total_edep();
    int bragg_cell = get_bragg_peak_cell();
    double lateral_spread = get_lateral_spread();

    std::cout << "Total energy deposited: " << total_edep << " MeV" << std::endl;
    std::cout << "Bragg peak cell: " << bragg_cell << " (depth = " << bragg_cell * dz << " mm)" << std::endl;
    std::cout << "Lateral spread (sigma): " << lateral_spread << " mm" << std::endl;

    EXPECT_LT(total_edep, E0) << "Nuclear attenuation should reduce energy";
    EXPECT_LT(lateral_spread, 0.5) << "Lateral spread should be zero without MCS";

    std::cout << "=== Test PASSED ===" << std::endl;
}

TEST_F(EnergyLossOnlyTest, MCSOnly) {
    std::cout << "\n=== Test: MCS Only ===" << std::endl;

    run_transport(false, false, true);

    double total_edep = get_total_edep();
    int bragg_cell = get_bragg_peak_cell();
    double bragg_depth = bragg_cell * dz;
    double lateral_spread = get_lateral_spread();

    std::cout << "Total energy deposited: " << total_edep << " MeV" << std::endl;
    std::cout << "Bragg peak cell: " << bragg_cell << " (depth = " << bragg_depth << " mm)" << std::endl;
    std::cout << "Lateral spread (sigma): " << lateral_spread << " mm" << std::endl;

    EXPECT_NEAR(total_edep, E0, 0.1) << "Energy conservation failed";
    EXPECT_GT(lateral_spread, 0.5) << "Lateral spread should be non-zero with MCS";

    std::cout << "=== Test PASSED ===" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
