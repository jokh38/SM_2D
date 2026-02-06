#include "cuda/gpu_transport_wrapper.hpp"
#include "k1k6_pipeline.cuh"
#include "kernels/k3_finetransport.cuh"
#include "device/device_psic.cuh"
#include "core/incident_particle_config.hpp"
#include "lut/r_lut.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
#include <iomanip>

namespace sm_2d {

// ============================================================================
// Optional debug dump: non-zero cell information to CSV (initial state)
// ============================================================================
#if defined(SM2D_ENABLE_DEBUG_DUMPS)
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
#endif

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

bool init_device_lut(const ::RLUT& cpu_lut, DeviceLUTWrapper& wrapper) {
    bool result = wrapper.p_impl->impl.init(cpu_lut);
    if (result) {
        wrapper.dlut_ptr = &wrapper.p_impl->impl.dlut;
    }
    return result;
}

bool run_k1k6_pipeline_transport(
    float x0, float z0, float theta0, float E0, float W_total,
    float sigma_x, float sigma_theta, float sigma_E,  // Gaussian beam parameters
    int n_samples,                                     // Number of samples
    unsigned int random_seed,                          // RNG seed
    int Nx, int Nz, float dx, float dz,
    float x_min, float z_min,
    int N_theta_local, int N_E_local,
    const AngularGrid& theta_grid,
    const EnergyGrid& e_grid,
    const ::DeviceRLUT& dlut,
    const TransportConfig& transport,
    std::vector<std::vector<double>>& edep
) {
    const bool summary_logging = transport.log_level >= 1;
    const bool verbose_logging = transport.log_level >= 2;
    const bool debug_dumps_enabled =
    #if defined(SM2D_ENABLE_DEBUG_DUMPS)
        true;
    #else
        false;
    #endif

    const int N_theta = theta_grid.N_theta;
    const int N_E = e_grid.N_E;

    if (summary_logging) {
        std::cout << "Running K1-K6 GPU pipeline wrapper..." << std::endl;
        std::cout << "  Grid: " << Nx << " x " << Nz << " cells" << std::endl;
        std::cout << "  Source: (" << x0 << ", " << z0 << ") mm, " << E0 << " MeV" << std::endl;
    }

    // Determine if using Gaussian or pencil beam
    bool use_gaussian = (n_samples > 1) || (sigma_x > 0) || (sigma_theta > 0);
    if (summary_logging && use_gaussian) {
        std::cout << "  Using GAUSSIAN beam: sigma_x=" << sigma_x << " mm, sigma_theta=" << sigma_theta << " rad, n_samples=" << n_samples << std::endl;
    } else if (summary_logging) {
        std::cout << "  Using PENCIL beam (single particle)" << std::endl;
    }

    // Initialize pipeline configuration
    K1K6PipelineConfig config;
    config.Nx = Nx;
    config.Nz = Nz;
    config.dx = dx;
    config.dz = dz;
    config.E_trigger = transport.E_trigger;
    config.weight_active_min = transport.weight_active_min;
    config.E_coarse_max = transport.E_coarse_max;
    config.step_coarse = transport.step_coarse;
    config.n_steps_per_cell = transport.n_steps_per_cell;
    config.max_iterations = transport.max_iterations;
    config.log_level = transport.log_level;

    // Phase-space dimensions
    config.N_theta = N_theta;
    config.N_E = N_E;
    config.N_theta_local = N_theta_local;
    config.N_E_local = N_E_local;

    // FIX C: Set initial beam width from input parameter
    // This is passed to K2 and K3 kernels for lateral spreading calculation
    config.sigma_x_initial = sigma_x;

    // ========================================================================
    // STEP 1: Allocate Device PsiC Buffers
    // ========================================================================

    DevicePsiC psi_in, psi_out;

    // Set CUDA device
    cudaError_t device_err = cudaSetDevice(0);
    if (device_err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(device_err) << std::endl;
        return false;
    }

    if (!device_psic_init(psi_in, Nx, Nz)) {
        std::cerr << "Failed to allocate psi_in" << std::endl;
        return false;
    }

    if (!device_psic_init(psi_out, Nx, Nz)) {
        std::cerr << "Failed to allocate psi_out" << std::endl;
        device_psic_cleanup(psi_in);
        return false;
    }

    // ========================================================================
    // STEP 2: Allocate Auxiliary Arrays
    // ========================================================================

    K1K6PipelineState state;
    if (!init_pipeline_state(config, e_grid, theta_grid, state)) {
        std::cerr << "Failed to allocate pipeline state" << std::endl;
        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
        return false;
    }

    // ========================================================================
    // STEP 3: Inject Source Particle
    // ========================================================================

    if (summary_logging) {
        std::cout << "  Source: (" << x0 << ", " << z0 << ") mm" << std::endl;
        std::cout << "  Energy: " << E0 << " MeV, Angle: " << theta0 << " rad" << std::endl;
    }

    // Inject source particle(s) into psi_in
    // Use Gaussian sampling if parameters indicate, otherwise pencil beam
    float* d_injected_weight = nullptr;
    float* d_out_of_grid_weight = nullptr;
    float* d_slot_dropped_weight = nullptr;

    if (use_gaussian) {
        cudaMalloc(&d_injected_weight, sizeof(float));
        cudaMalloc(&d_out_of_grid_weight, sizeof(float));
        cudaMalloc(&d_slot_dropped_weight, sizeof(float));
        cudaMemset(d_injected_weight, 0, sizeof(float));
        cudaMemset(d_out_of_grid_weight, 0, sizeof(float));
        cudaMemset(d_slot_dropped_weight, 0, sizeof(float));

        // Gaussian beam: launch multiple threads to sample particles
        int threads = 256;
        int blocks = (n_samples + threads - 1) / threads;

        inject_gaussian_source_kernel<<<blocks, threads>>>(
            psi_in,
            x0, z0, theta0, E0, W_total,
            sigma_x, sigma_theta, sigma_E,
            n_samples, random_seed,
            x_min, z_min,
            dx, dz, Nx, Nz,
            d_injected_weight,
            d_out_of_grid_weight,
            d_slot_dropped_weight,
            state.d_theta_edges, state.d_E_edges,
            N_theta, N_E,
            N_theta_local, N_E_local
        );
    } else {
        // Pencil beam: single particle at center
        inject_source_kernel<<<1, 1>>>(
            psi_in,
            Nx, Nz, dx, dz, x_min, z_min,  // Grid info
            x0, z0,                        // Source position
            theta0, sigma_theta,           // Angle and spread
            E0, W_total,                   // Energy and weight
            sigma_x,                       // Lateral spread
            state.d_theta_edges, state.d_E_edges,  // Bin edges
            N_theta, N_E,                  // Global bins
            N_theta_local, N_E_local       // Local bins
        );
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Source injection failed: " << cudaGetErrorString(err) << std::endl;
        if (d_injected_weight) cudaFree(d_injected_weight);
        if (d_out_of_grid_weight) cudaFree(d_out_of_grid_weight);
        if (d_slot_dropped_weight) cudaFree(d_slot_dropped_weight);
        state.cleanup();
        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
        return false;
    }

    if (use_gaussian) {
        float h_injected_weight = 0.0f;
        float h_out_of_grid_weight = 0.0f;
        float h_slot_dropped_weight = 0.0f;
        cudaMemcpy(&h_injected_weight, d_injected_weight, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_out_of_grid_weight, d_out_of_grid_weight, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_slot_dropped_weight, d_slot_dropped_weight, sizeof(float), cudaMemcpyDeviceToHost);

        if (verbose_logging) {
            float denom = (W_total > 1e-20f) ? W_total : 1.0f;
            std::cout << "  Source injection accounting:" << std::endl;
            std::cout << "    Injected in-grid: " << h_injected_weight
                      << " (" << (100.0f * h_injected_weight / denom) << "%)" << std::endl;
            std::cout << "    Outside grid:     " << h_out_of_grid_weight
                      << " (" << (100.0f * h_out_of_grid_weight / denom) << "%)" << std::endl;
            std::cout << "    Slot dropped:     " << h_slot_dropped_weight
                      << " (" << (100.0f * h_slot_dropped_weight / denom) << "%)" << std::endl;
        }

        cudaFree(d_injected_weight);
        cudaFree(d_out_of_grid_weight);
        cudaFree(d_slot_dropped_weight);
    }

    if (verbose_logging || debug_dumps_enabled) {
        // Optional heavy verification path for source injection accounting.
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

        if (verbose_logging) {
            std::cout << "  Source injection verification:" << std::endl;
            std::cout << "    Total weight: " << total_weight << " (expected: " << W_total << ")" << std::endl;
            std::cout << "    Non-zero cells: " << nonzero_cells << " / " << psi_in.N_cells << std::endl;
        }
    }

    #if defined(SM2D_ENABLE_DEBUG_DUMPS)
    // DEBUG: Dump initial cell state
    dump_initial_cells_to_csv(psi_in, Nx, Nz, dx, dz);
    #endif

    // ========================================================================
    // STEP 4: Run Main Pipeline
    // ========================================================================

    if (!run_k1k6_pipeline_transport(&psi_in, &psi_out, dlut, e_grid, theta_grid, config, state)) {
        std::cerr << "Pipeline failed" << std::endl;
        state.cleanup();
        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
        return false;
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

    if (summary_logging) {
        std::cout << "  Energy deposition extracted" << std::endl;
    }

    // ========================================================================
    // STEP 6: Cleanup
    // ========================================================================

    state.cleanup();
    device_psic_cleanup(psi_in);
    device_psic_cleanup(psi_out);

    if (summary_logging) {
        std::cout << "K1-K6 pipeline wrapper complete." << std::endl;
    }
    return true;
}

std::string get_gpu_name_internal() {
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);
    return prop.name;
}

} // namespace sm_2d
