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

// Bring CUDA types into sm_2d namespace for compatibility
using ::DeviceLUTWrapper;

namespace sm_2d {

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
    float x0, float z0, float theta0, float E0, float W_total,
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

    // Cast from sm_2d::DeviceRLUT to ::DeviceRLUT (same type, different namespace)
    const ::DeviceRLUT& native_dlut = reinterpret_cast<const ::DeviceRLUT&>(dlut);

    // Initialize pipeline configuration
    K1K6PipelineConfig config;
    config.Nx = Nx;
    config.Nz = Nz;
    config.dx = dx;
    config.dz = dz;

    // Energy thresholds
    // COARSE-ONLY TEST: Set E_trigger below minimum energy (0.1 MeV) to force coarse-only transport
    // This makes b_E_trigger = 0, so fine transport never activates
    // Particles will only use coarse transport (K2) for approximate but reasonable results
    config.E_trigger = 0.05f;          // Below min energy (0.1 MeV) → b_E_trigger=0 → coarse-only
    config.weight_active_min = 1e-12f;  // FIX: Lowered from 1e-6 to fix transport gap (per debug report)

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
    // [0.1-2 MeV]: 0.1 MeV/bin (19 bins), [2-20 MeV]: 0.25 MeV/bin (72 bins)
    // [20-100 MeV]: 0.5 MeV/bin (160 bins), [100-250 MeV]: 1 MeV/bin (150 bins)
    std::vector<std::tuple<float, float, float>> energy_groups = {
        {0.1f, 2.0f, 0.1f},      // 19 bins - Bragg peak core
        {2.0f, 20.0f, 0.25f},    // 72 bins - Bragg peak falloff
        {20.0f, 100.0f, 0.5f},   // 160 bins - Mid-energy plateau
        {100.0f, 250.0f, 1.0f}    // 150 bins - High energy
    };
    EnergyGrid e_grid(energy_groups);  // Option D2: piecewise-uniform
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

    // Convert source position to cell coordinates
    float x_rel = x0 - x_min;
    float z_rel = z0 - z_min;

    int source_cell_x = static_cast<int>(x_rel / dx);
    int source_cell_z = static_cast<int>(z_rel / dz);
    int source_cell = source_cell_z * Nx + source_cell_x;

    // Clamp to valid range
    source_cell_x = (source_cell_x < 0) ? 0 : (source_cell_x >= Nx) ? Nx - 1 : source_cell_x;
    source_cell_z = (source_cell_z < 0) ? 0 : (source_cell_z >= Nz) ? Nz - 1 : source_cell_z;
    source_cell = source_cell_z * Nx + source_cell_x;

    // Position within cell
    float x_in_cell = x_rel - source_cell_x * dx;
    float z_in_cell = z_rel - source_cell_z * dz;

    std::cout << "  Source: cell (" << source_cell_x << ", " << source_cell_z
              << ") at (" << x0 << ", " << z0 << ") mm" << std::endl;
    std::cout << "  Energy: " << E0 << " MeV, Angle: " << theta0 << " rad" << std::endl;

    // Inject source particle into psi_in
    // Create a simple kernel to inject the source
    inject_source_kernel<<<1, 1>>>(
        psi_in,
        source_cell,
        theta0, E0, W_total,
        x_in_cell, z_in_cell,
        dx, dz,
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

    // Verify source injection by summing weights in source cell
    size_t source_cell_slots = psi_in.Kb * LOCAL_BINS;
    std::vector<float> h_source_values(source_cell_slots);
    size_t source_offset = source_cell * psi_in.Kb * LOCAL_BINS;
    cudaMemcpy(h_source_values.data(), psi_in.value + source_offset,
               source_cell_slots * sizeof(float), cudaMemcpyDeviceToHost);

    float total_weight = 0.0f;
    int nonzero_count = 0;
    for (size_t i = 0; i < source_cell_slots; ++i) {
        if (h_source_values[i] > 0.0f) {
            total_weight += h_source_values[i];
            nonzero_count++;
        }
    }
    std::cout << "  Source injection verification:" << std::endl;
    std::cout << "    Total weight in source cell: " << total_weight << " (expected: " << W_total << ")" << std::endl;
    std::cout << "    Non-zero bins: " << nonzero_count << " / " << source_cell_slots << std::endl;

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
