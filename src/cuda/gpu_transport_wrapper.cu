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
#include <cmath>
#include <algorithm>

namespace sm_2d {

// ============================================================================
// Optional debug dump: raw phase-space rows to CSV (initial state)
// ============================================================================
#if defined(SM2D_ENABLE_DEBUG_DUMPS)
static void dump_initial_cells_to_csv(
    const DevicePsiC& psi,
    int Nx, int Nz, float dx, float dz,
    const float* d_theta_edges,
    const float* d_E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local
) {
    int N_cells = Nx * Nz;
    const size_t total_slots = static_cast<size_t>(N_cells) * static_cast<size_t>(psi.Kb);
    const size_t total_values = total_slots * static_cast<size_t>(LOCAL_BINS);

    std::vector<float> h_all_values(total_values, 0.0f);
    std::vector<uint32_t> h_all_block_ids(total_slots, DEVICE_EMPTY_SLOT);
    std::vector<float> h_theta_edges(static_cast<size_t>(N_theta + 1), 0.0f);
    std::vector<float> h_E_edges(static_cast<size_t>(N_E + 1), 0.0f);

    cudaMemcpy(h_all_values.data(), psi.value, total_values * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_block_ids.data(), psi.block_id, total_slots * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_theta_edges.data(), d_theta_edges, static_cast<size_t>(N_theta + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_E_edges.data(), d_E_edges, static_cast<size_t>(N_E + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream ofs("results/debug_ps_raw_iter_00_initial.csv");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open results/debug_ps_raw_iter_00_initial.csv for writing" << std::endl;
        return;
    }

    ofs << "iter,cell,ix,iz,slot,bid,lidx,weight,"
        << "b_theta,b_E,theta_local,E_local,x_sub,z_sub,"
        << "theta_bin,E_bin,theta_rep,E_rep,x_offset_mm,z_offset_mm\n";
    ofs << std::fixed << std::setprecision(6);

    size_t nonzero_rows = 0;
    for (int cell = 0; cell < N_cells; ++cell) {
        int ix = cell % Nx;
        int iz = cell / Nx;

        for (int slot = 0; slot < psi.Kb; ++slot) {
            uint32_t bid = h_all_block_ids[static_cast<size_t>(cell) * psi.Kb + slot];
            if (bid == DEVICE_EMPTY_SLOT) {
                continue;
            }

            int b_theta = static_cast<int>(bid & 0xFFF);
            int b_E = static_cast<int>((bid >> 12) & 0xFFF);

            for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                size_t idx = (static_cast<size_t>(cell) * psi.Kb + slot) * LOCAL_BINS + lidx;
                float w = h_all_values[idx];
                if (w <= 1e-12f) {
                    continue;
                }

                int theta_local, E_local, x_sub, z_sub;
                decode_local_idx_4d(static_cast<uint16_t>(lidx), theta_local, E_local, x_sub, z_sub);

                int theta_bin = b_theta * N_theta_local + theta_local;
                int E_bin = b_E * N_E_local + E_local;
                theta_bin = std::max(0, std::min(theta_bin, N_theta - 1));
                E_bin = std::max(0, std::min(E_bin, N_E - 1));

                float theta_rep = h_theta_edges[theta_bin] +
                    0.5f * (h_theta_edges[theta_bin + 1] - h_theta_edges[theta_bin]);
                float E_rep = h_E_edges[E_bin] +
                    0.5f * (h_E_edges[E_bin + 1] - h_E_edges[E_bin]);

                float x_offset = get_x_offset_from_bin(x_sub, dx);
                float z_offset = get_z_offset_from_bin(z_sub, dz);

                ofs << 0 << ","                  // iter
                    << cell << ","
                    << ix << ","
                    << iz << ","
                    << slot << ","
                    << bid << ","
                    << lidx << ","
                    << w << ","
                    << b_theta << ","
                    << b_E << ","
                    << theta_local << ","
                    << E_local << ","
                    << x_sub << ","
                    << z_sub << ","
                    << theta_bin << ","
                    << E_bin << ","
                    << theta_rep << ","
                    << E_rep << ","
                    << x_offset << ","
                    << z_offset << "\n";
                ++nonzero_rows;
            }
        }
    }

    ofs.close();
    std::cout << "  DEBUG: Wrote " << nonzero_rows
              << " raw phase-space rows to results/debug_ps_raw_iter_00_initial.csv" << std::endl;
}
#endif

// ============================================================================

static bool compute_psic_represented_energy(
    const DevicePsiC& psi,
    const float* d_E_edges,
    int N_E,
    int N_theta_local,
    int N_E_local,
    double& represented_energy_out
) {
    size_t total_slots = static_cast<size_t>(psi.N_cells) * psi.Kb;
    size_t total_values = total_slots * LOCAL_BINS;
    std::vector<uint32_t> h_block_ids(total_slots, DEVICE_EMPTY_SLOT);
    std::vector<float> h_values(total_values, 0.0f);
    std::vector<float> h_E_edges(static_cast<size_t>(N_E + 1), 0.0f);

    if (!device_psic_copy_to_host(psi, h_block_ids.data(), h_values.data())) {
        return false;
    }

    cudaError_t edge_err = cudaMemcpy(
        h_E_edges.data(),
        d_E_edges,
        static_cast<size_t>(N_E + 1) * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    if (edge_err != cudaSuccess) {
        return false;
    }

    represented_energy_out = 0.0;
    for (int cell = 0; cell < psi.N_cells; ++cell) {
        for (int slot = 0; slot < psi.Kb; ++slot) {
            uint32_t bid = h_block_ids[static_cast<size_t>(cell) * psi.Kb + slot];
            if (bid == DEVICE_EMPTY_SLOT) {
                continue;
            }

            int b_E = static_cast<int>((bid >> 12) & 0xFFF);
            for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                size_t idx = (static_cast<size_t>(cell) * psi.Kb + slot) * LOCAL_BINS + lidx;
                float w = h_values[idx];
                if (w <= 0.0f) {
                    continue;
                }

                int E_local = (lidx / N_theta_local) % N_E_local;
                int E_bin = b_E * N_E_local + E_local;
                E_bin = std::max(0, std::min(E_bin, N_E - 1));
                float E_lower = h_E_edges[E_bin];
                float E_upper = h_E_edges[E_bin + 1];
                float E_center = E_lower + 0.5f * (E_upper - E_lower);
                represented_energy_out += static_cast<double>(w) * static_cast<double>(E_center);
            }
        }
    }
    return true;
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
        transport.debug_dumps;
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
    config.E_fine_on = transport.E_fine_on;
    config.E_fine_off = transport.E_fine_off;
    config.weight_active_min = transport.weight_active_min;
    config.E_coarse_max = transport.E_coarse_max;
    config.step_coarse = transport.step_coarse;
    config.n_steps_per_cell = transport.n_steps_per_cell;
    config.fine_batch_max_cells = transport.fine_batch_max_cells;
    config.max_iterations = transport.max_iterations;
    config.log_level = transport.log_level;
    config.fail_fast_on_audit = transport.fail_fast_on_audit || transport.validation_mode;
    config.debug_dumps_enabled = debug_dumps_enabled;

    // Phase-space dimensions
    config.N_theta = N_theta;
    config.N_E = N_E;
    config.N_theta_local = N_theta_local;
    config.N_E_local = N_E_local;

    // FIX C: Set initial beam width from input parameter
    // This is passed to K2 and K3 kernels for lateral spreading calculation
    config.sigma_x_initial = sigma_x;

    // Set initial beam energy for Fermi-Eyges moment precomputation
    config.E0_MeV = E0;

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
    double* d_injected_energy = nullptr;
    double* d_out_of_grid_energy = nullptr;
    double* d_slot_dropped_energy = nullptr;

    float h_injected_weight = use_gaussian ? 0.0f : W_total;
    float h_out_of_grid_weight = 0.0f;
    float h_slot_dropped_weight = 0.0f;
    double h_injected_energy = static_cast<double>(E0) * static_cast<double>(h_injected_weight);
    double h_out_of_grid_energy = 0.0;
    double h_slot_dropped_energy = 0.0;

    if (use_gaussian) {
        cudaMalloc(&d_injected_weight, sizeof(float));
        cudaMalloc(&d_out_of_grid_weight, sizeof(float));
        cudaMalloc(&d_slot_dropped_weight, sizeof(float));
        cudaMalloc(&d_injected_energy, sizeof(double));
        cudaMalloc(&d_out_of_grid_energy, sizeof(double));
        cudaMalloc(&d_slot_dropped_energy, sizeof(double));
        cudaMemset(d_injected_weight, 0, sizeof(float));
        cudaMemset(d_out_of_grid_weight, 0, sizeof(float));
        cudaMemset(d_slot_dropped_weight, 0, sizeof(float));
        cudaMemset(d_injected_energy, 0, sizeof(double));
        cudaMemset(d_out_of_grid_energy, 0, sizeof(double));
        cudaMemset(d_slot_dropped_energy, 0, sizeof(double));

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
            d_injected_energy,
            d_out_of_grid_energy,
            d_slot_dropped_energy,
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
        if (d_injected_energy) cudaFree(d_injected_energy);
        if (d_out_of_grid_energy) cudaFree(d_out_of_grid_energy);
        if (d_slot_dropped_energy) cudaFree(d_slot_dropped_energy);
        state.cleanup();
        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
        return false;
    }

    if (use_gaussian) {
        cudaMemcpy(&h_injected_weight, d_injected_weight, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_out_of_grid_weight, d_out_of_grid_weight, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_slot_dropped_weight, d_slot_dropped_weight, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_injected_energy, d_injected_energy, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_out_of_grid_energy, d_out_of_grid_energy, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_slot_dropped_energy, d_slot_dropped_energy, sizeof(double), cudaMemcpyDeviceToHost);

        if (verbose_logging) {
            float denom = (W_total > 1e-20f) ? W_total : 1.0f;
            double E_denom = (std::abs(h_injected_energy + h_out_of_grid_energy + h_slot_dropped_energy) > 1e-20)
                             ? (h_injected_energy + h_out_of_grid_energy + h_slot_dropped_energy)
                             : 1.0;
            std::cout << "  Source injection accounting:" << std::endl;
            std::cout << "    Injected in-grid: " << h_injected_weight
                      << " (" << (100.0f * h_injected_weight / denom) << "%)" << std::endl;
            std::cout << "    Outside grid:     " << h_out_of_grid_weight
                      << " (" << (100.0f * h_out_of_grid_weight / denom) << "%)" << std::endl;
            std::cout << "    Slot dropped:     " << h_slot_dropped_weight
                      << " (" << (100.0f * h_slot_dropped_weight / denom) << "%)" << std::endl;
            std::cout << "  Source energy accounting:" << std::endl;
            std::cout << "    Injected in-grid: " << h_injected_energy
                      << " MeV (" << (100.0 * h_injected_energy / E_denom) << "%)" << std::endl;
            std::cout << "    Outside grid:     " << h_out_of_grid_energy
                      << " MeV (" << (100.0 * h_out_of_grid_energy / E_denom) << "%)" << std::endl;
            std::cout << "    Slot dropped:     " << h_slot_dropped_energy
                      << " MeV (" << (100.0 * h_slot_dropped_energy / E_denom) << "%)" << std::endl;
        }

        cudaFree(d_injected_weight);
        cudaFree(d_out_of_grid_weight);
        cudaFree(d_slot_dropped_weight);
        cudaFree(d_injected_energy);
        cudaFree(d_out_of_grid_energy);
        cudaFree(d_slot_dropped_energy);
    }

    double represented_injected_energy = 0.0;
    if (!compute_psic_represented_energy(
            psi_in,
            state.d_E_edges,
            N_E,
            N_theta_local,
            N_E_local,
            represented_injected_energy)) {
        std::cerr << "Failed to compute represented injected energy from PsiC" << std::endl;
        state.cleanup();
        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
        return false;
    }

    double source_representation_loss_energy = h_injected_energy - represented_injected_energy;
    if (std::abs(source_representation_loss_energy) < 1e-12) {
        source_representation_loss_energy = 0.0;
    }

    state.source_injected_weight = h_injected_weight;
    state.source_out_of_grid_weight = h_out_of_grid_weight;
    state.source_slot_dropped_weight = h_slot_dropped_weight;
    state.source_injected_energy = h_injected_energy;
    state.source_out_of_grid_energy = h_out_of_grid_energy;
    state.source_slot_dropped_energy = h_slot_dropped_energy;
    state.source_representation_loss_energy = source_representation_loss_energy;

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
            std::cout << "    Injected energy (sampled): " << h_injected_energy << " MeV" << std::endl;
            std::cout << "    Injected energy (represented): " << represented_injected_energy << " MeV" << std::endl;
            std::cout << "    Source representation loss: " << source_representation_loss_energy << " MeV" << std::endl;
        }
    }

    #if defined(SM2D_ENABLE_DEBUG_DUMPS)
    if (debug_dumps_enabled) {
        // DEBUG: Dump initial cell state
        dump_initial_cells_to_csv(
            psi_in,
            Nx, Nz, dx, dz,
            state.d_theta_edges,
            state.d_E_edges,
            N_theta, N_E,
            N_theta_local, N_E_local
        );
    }
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

    // ========================================================================
    // STEP 5b: Post-processing MCS lateral spread convolution
    // ========================================================================
    // During transport, lateral spreading was intentionally disabled to avoid
    // the double-counting problem (applying total sigma at every z-step
    // re-shapes the beam to constant width instead of growing with depth).
    //
    // Instead, we apply the MCS lateral spread as a SINGLE post-processing
    // convolution on each depth row of EdepC:
    //   sigma_MCS(z) = sqrt(sigma_total(z)^2 - sigma_initial^2)
    // where sigma_total comes from Fermi-Eyges theory (precomputed).
    //
    // Result: sigma_measured(z) = sqrt(sigma_source^2 + sigma_MCS(z)^2)
    //       = sigma_total(z)  -- correct depth-dependent lateral spread
    // ========================================================================
    if (state.d_FE_sigma_total != nullptr) {
        std::vector<float> h_sigma_total(Nz);
        cudaMemcpy(h_sigma_total.data(), state.d_FE_sigma_total,
                   Nz * sizeof(float), cudaMemcpyDeviceToHost);

        float sigma_init = sigma_x;  // Initial beam width [mm]
        float sigma_init_sq = sigma_init * sigma_init;
        std::vector<double> row_buf(Nx);

        int rows_convolved = 0;
        for (int iz = 0; iz < Nz; ++iz) {
            float sigma_total_z = h_sigma_total[iz];
            float sigma_mcs_sq = sigma_total_z * sigma_total_z - sigma_init_sq;
            if (sigma_mcs_sq <= 1e-6f) continue;  // No MCS spread at this depth

            float sigma_mcs = std::sqrt(sigma_mcs_sq);
            float sigma_cells = sigma_mcs / dx;  // Convert to cell units

            // Build Gaussian kernel (truncated at +-3 sigma)
            int half_k = static_cast<int>(std::ceil(3.0f * sigma_cells));
            half_k = std::max(1, std::min(half_k, Nx / 2));
            int kernel_size = 2 * half_k + 1;
            std::vector<double> kernel(kernel_size);
            double kernel_sum = 0.0;
            for (int k = -half_k; k <= half_k; ++k) {
                double val = std::exp(-0.5 * (static_cast<double>(k) * k) /
                                      (static_cast<double>(sigma_cells) * sigma_cells));
                kernel[k + half_k] = val;
                kernel_sum += val;
            }
            for (auto& v : kernel) v /= kernel_sum;  // Normalize

            // Copy row to buffer
            for (int ix = 0; ix < Nx; ++ix) {
                row_buf[ix] = h_EdepC[iz * Nx + ix];
            }

            // 1D convolution (zero-padding at boundaries)
            for (int ix = 0; ix < Nx; ++ix) {
                double conv = 0.0;
                for (int k = -half_k; k <= half_k; ++k) {
                    int jx = ix + k;
                    if (jx >= 0 && jx < Nx) {
                        conv += row_buf[jx] * kernel[k + half_k];
                    }
                }
                h_EdepC[iz * Nx + ix] = conv;
            }
            ++rows_convolved;
        }

        if (summary_logging) {
            std::cout << "  Applied MCS lateral spread convolution to "
                      << rows_convolved << " / " << Nz << " depth rows" << std::endl;
            // Log a few representative values
            for (int iz_sample : {0, Nz/4, Nz/2, 3*Nz/4, Nz-1}) {
                if (iz_sample < Nz) {
                    float st = h_sigma_total[iz_sample];
                    float sm_sq = st * st - sigma_init_sq;
                    float sm = (sm_sq > 0.0f) ? std::sqrt(sm_sq) : 0.0f;
                    std::cout << "    z=" << iz_sample * dz << "mm: sigma_total="
                              << st << " sigma_MCS=" << sm << " mm" << std::endl;
                }
            }
        }
    }

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
