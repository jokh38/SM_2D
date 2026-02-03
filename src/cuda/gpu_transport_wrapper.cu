#include "cuda/gpu_transport_wrapper.hpp"
#include "k1k6_pipeline.cuh"
#include "kernels/k3_finetransport.cuh"
#include "device/device_psic.cuh"
#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>
#include <tuple>
#include <fstream>
#include <iomanip>

// Bring CUDA types into sm_2d namespace for compatibility
using ::DeviceLUTWrapper;

namespace sm_2d {

// ============================================================================
// Debug: Dump non-zero cell information to CSV (initial state)
// ============================================================================
static void dump_initial_cells_to_csv(
    const DevicePsiC& psi,
    int Nx, int Nz, float dx, float dz
) {
    int N_cells = Nx * Nz;
    size_t total_values = N_cells * psi.Kb * LOCAL_BINS;

    // Copy all values from device
    std::vector<float> h_all_values(total_values);
    std::vector<uint32_t> h_all_block_ids(N_cells * psi.Kb);

    cudaMemcpy(h_all_values.data(), psi.value, total_values * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_block_ids.data(), psi.block_id, N_cells * psi.Kb * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Collect non-zero cells
    struct CellInfo {
        int cell;
        int ix;
        int iz;
        float x_center;
        float z_center;
        float total_weight;
    };
    std::vector<CellInfo> nonzero_cells;

    for (int cell = 0; cell < N_cells; ++cell) {
        float cell_weight = 0.0f;

        for (int slot = 0; slot < psi.Kb; ++slot) {
            uint32_t bid = h_all_block_ids[cell * psi.Kb + slot];
            if (bid == 0xFFFFFFFF) continue;

            for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                size_t idx = (cell * psi.Kb + slot) * LOCAL_BINS + lidx;
                float w = h_all_values[idx];
                if (w > 1e-12f) {
                    cell_weight += w;
                }
            }
        }

        if (cell_weight > 1e-12f) {
            int ix = cell % Nx;
            int iz = cell / Nx;
            float x_center = (ix + 0.5f) * dx;
            float z_center = (iz + 0.5f) * dz;

            nonzero_cells.push_back({cell, ix, iz, x_center, z_center, cell_weight});
        }
    }

    // Write CSV
    std::ofstream ofs("results/debug_cells_iter_00_initial.csv");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open debug_cells_iter_00_initial.csv for writing" << std::endl;
        return;
    }

    ofs << "cell,ix,iz,x_mm,z_mm,total_weight\n";
    ofs << std::fixed << std::setprecision(6);

    for (const auto& info : nonzero_cells) {
        ofs << info.cell << ","
            << info.ix << ","
            << info.iz << ","
            << info.x_center << ","
            << info.z_center << ","
            << info.total_weight << "\n";
    }

    ofs.close();
    std::cout << "  DEBUG: Wrote " << nonzero_cells.size() << " initial non-zero cells to debug_cells_iter_00_initial.csv" << std::endl;
}

// ============================================================================

// Implementation of DeviceLUTWrapper (PIMPL pattern for CUDA types)
class DeviceLUTWrapperImpl {
public:
    // This is the actual CUDA DeviceLUTWrapper from k3_finetransport.cuh
    ::DeviceLUTWrapper impl;
};

DeviceLUTWrapper::DeviceLUTWrapper()
    : p_impl(new DeviceLUTWrapperImpl()), dlut_ptr(nullptr)
{}

DeviceLUTWrapper::~DeviceLUTWrapper() {
    delete p_impl;
}

bool init_device_lut(const RLUT& cpu_lut, DeviceLUTWrapper& wrapper) {
    // Cast from sm_2d::RLUT to ::RLUT (same type, different namespace)
    const ::RLUT& native_lut = reinterpret_cast<const ::RLUT&>(cpu_lut);
    bool result = wrapper.p_impl->impl.init(native_lut);
    if (result) {
        wrapper.dlut_ptr = reinterpret_cast<const DeviceRLUT*>(&wrapper.p_impl->impl.dlut);
    }
    return result;
}

void run_k1k6_pipeline_transport(
    float x0, float z0, float theta0, float sigma_theta, float sigma_x,
    float E0, float W_total,
    int Nx, int Nz, float dx, float dz,
    float x_min, float z_min,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,
    const float* theta_edges,
    const float* E_edges,
    const DeviceRLUT& dlut,
    std::vector<std::vector<double>>& edep
) {
    std::cout << "Running K1-K6 GPU pipeline wrapper..." << std::endl;
    std::cout << "  Grid: " << Nx << " x " << Nz << " cells" << std::endl;
    std::cout << "  Source: (" << x0 << ", " << z0 << ") mm, " << E0 << " MeV" << std::endl;

    // Log beam parameters
    if (sigma_theta > 0.0001f) {
        std::cout << "  Angular divergence: " << sigma_theta << " rad" << std::endl;
    }
    if (sigma_x > 0.01f) {
        std::cout << "  Spatial beam width (sigma_x): " << sigma_x << " mm" << std::endl;
    }

    // Cast from sm_2d::DeviceRLUT to ::DeviceRLUT (same type, different namespace)
    const ::DeviceRLUT& native_dlut = reinterpret_cast<const ::DeviceRLUT&>(dlut);

    // Initialize pipeline configuration
    K1K6PipelineConfig config;
    config.Nx = Nx;
    config.Nz = Nz;
    config.dx = dx;
    config.dz = dz;

    // Energy thresholds
    // TEST: Use very high E_trigger to force ALL particles to use K3 fine transport
    // This enables lateral scattering from the beginning
    config.E_trigger = 1000.0f;        // K3 for E < 1000 MeV (essentially all particles)
    config.weight_active_min = 1e-12f;  // Lowered from 1e-6 to fix transport gap

    // Coarse transport settings
    config.E_coarse_max = 300.0f;       // Up to 300 MeV (original value)
    // Option D2: Adaptive step size for coarse transport
    // To ensure particles cross energy bins: dE/step > bin_width
    // At 150 MeV: dE/dx ≈ 0.54 MeV/mm, bin_width = 1 MeV
    // Need step > 1 / 0.54 ≈ 2mm for particles to cross bins
    // Using 5mm step gives dE ≈ 2.7 MeV > bin_width ✓
    config.step_coarse = 5.0f;  // Adaptive step for high energy (100-250 MeV)
    config.n_steps_per_cell = 1;        // One step per cell for coarse

    // Phase-space dimensions
    config.N_theta = N_theta;
    config.N_E = N_E;
    config.N_theta_local = N_theta_local;
    config.N_E_local = N_E_local;

    // Create grids
    // Option D2: Piecewise-uniform energy grid (must match gpu_transport_runner.cpp)
    // Use finer energy resolution at high energies for accurate energy tracking
    std::vector<std::tuple<float, float, float>> energy_groups = {
        {0.1f, 2.0f, 0.1f},      // 19 bins - Bragg peak core
        {2.0f, 20.0f, 0.2f},     // 90 bins - Bragg peak falloff (finer)
        {20.0f, 100.0f, 0.25f},  // 320 bins - Mid-energy plateau (finer)
        {100.0f, 250.0f, 0.25f}   // 600 bins - High energy (MUCH FINER for tracking)
    };
    EnergyGrid e_grid = EnergyGrid::CreatePiecewise(energy_groups);  // Option D2: piecewise-uniform
    AngularGrid a_grid(-M_PI/2.0f, M_PI/2.0f, N_theta);

    // ========================================================================
    // STEP 1: Allocate Device PsiC Buffers
    // ========================================================================

    DevicePsiC psi_in, psi_out;

    // Set CUDA device
    cudaError_t device_err = cudaSetDevice(0);
    if (device_err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(device_err) << std::endl;
        return;
    }

    if (!device_psic_init(psi_in, Nx, Nz)) {
        std::cerr << "Failed to allocate psi_in" << std::endl;
        return;
    }

    if (!device_psic_init(psi_out, Nx, Nz)) {
        std::cerr << "Failed to allocate psi_out" << std::endl;
        device_psic_cleanup(psi_in);
        return;
    }

    // ========================================================================
    // STEP 2: Allocate Auxiliary Arrays
    // ========================================================================

    K1K6PipelineState state;
    if (!init_pipeline_state(config, state)) {
        std::cerr << "Failed to allocate pipeline state" << std::endl;
        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
        return;
    }

    // ========================================================================
    // STEP 3: Inject Source Particle
    // ========================================================================

    std::cout << "  Source: (" << x0 << ", " << z0 << ") mm" << std::endl;
    std::cout << "  Energy: " << E0 << " MeV, Angle: " << theta0 << " rad" << std::endl;

    // Inject source particle into psi_in with angular and spatial distribution
    // The kernel now handles multi-cell distribution internally
    inject_source_kernel<<<1, 1>>>(
        psi_in,
        Nx, Nz, dx, dz, x_min, z_min,
        x0, z0,
        theta0, sigma_theta,
        E0, W_total,
        sigma_x,
        theta_edges, E_edges,
        N_theta, N_E,
        N_theta_local, N_E_local
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Source injection failed: " << cudaGetErrorString(err) << std::endl;
        state.cleanup();
        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
        return;
    }

    // Verify source injection - sum weights across all cells
    std::vector<float> h_all_values(psi_in.N_cells * psi_in.Kb * LOCAL_BINS);
    cudaMemcpy(h_all_values.data(), psi_in.value,
               psi_in.N_cells * psi_in.Kb * LOCAL_BINS * sizeof(float), cudaMemcpyDeviceToHost);

    float total_weight = 0.0f;
    int nonzero_cells = 0;
    for (int cell = 0; cell < psi_in.N_cells; ++cell) {
        float cell_weight = 0.0f;
        for (size_t i = 0; i < psi_in.Kb * LOCAL_BINS; ++i) {
            size_t idx = cell * psi_in.Kb * LOCAL_BINS + i;
            if (h_all_values[idx] > 0.0f) {
                cell_weight += h_all_values[idx];
            }
        }
        if (cell_weight > 0.0f) {
            nonzero_cells++;
            total_weight += cell_weight;
        }
    }
    std::cout << "  Source injection verification:" << std::endl;
    std::cout << "    Total weight: " << total_weight << " (expected: " << W_total << ")" << std::endl;
    std::cout << "    Non-zero cells: " << nonzero_cells << " / " << psi_in.N_cells << std::endl;

    // DEBUG: Dump initial cell state
    dump_initial_cells_to_csv(psi_in, Nx, Nz, dx, dz);

    // ========================================================================
    // STEP 4: Run Main Pipeline
    // ========================================================================

    if (!run_k1k6_pipeline_transport(&psi_in, &psi_out, native_dlut, e_grid, a_grid, config, state)) {
        std::cerr << "Pipeline failed" << std::endl;
        state.cleanup();
        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
        return;
    }

    // ========================================================================
    // STEP 5: Extract Energy Deposition
    // ========================================================================

    // Copy energy deposition from device to host
    std::vector<double> h_EdepC(Nx * Nz);
    cudaMemcpy(h_EdepC.data(), state.d_EdepC, Nx * Nz * sizeof(double), cudaMemcpyDeviceToHost);

    // Convert to 2D output format
    for (int iz = 0; iz < Nz; ++iz) {
        for (int ix = 0; ix < Nx; ++ix) {
            edep[iz][ix] = h_EdepC[iz * Nx + ix];
        }
    }

    std::cout << "  Energy deposition extracted" << std::endl;

    // ========================================================================
    // STEP 6: Cleanup
    // ========================================================================

    state.cleanup();
    device_psic_cleanup(psi_in);
    device_psic_cleanup(psi_out);

    std::cout << "K1-K6 pipeline wrapper complete." << std::endl;
}

std::string get_gpu_name_internal() {
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);
    return prop.name;
}

} // namespace sm_2d
