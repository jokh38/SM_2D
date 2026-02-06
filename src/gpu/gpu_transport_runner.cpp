#include "gpu/gpu_transport_runner.hpp"

#ifdef SM2D_HAS_CUDA
#include "cuda/gpu_transport_wrapper.hpp"
#include "core/local_bins.hpp"  // For N_theta_local, N_E_local
#include <cuda_runtime.h>
#endif

#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include "core/grids.hpp"
#include <iostream>
#include <stdexcept>

namespace sm_2d {

#ifdef SM2D_HAS_CUDA

bool GPUTransportRunner::is_gpu_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

std::string GPUTransportRunner::get_gpu_name() {
    return get_gpu_name_internal();
}

SimulationResult GPUTransportRunner::run(const IncidentParticleConfig& config) {
    if (!is_gpu_available()) {
        throw std::runtime_error("GPU not available. CUDA device not found.");
    }

    SimulationResult result;
    result.Nx = config.grid.Nx;
    result.Nz = config.grid.Nz;
    result.dx = config.grid.dx;
    result.dz = config.grid.dz;

    // Allocate energy deposition grid
    result.edep.resize(config.grid.Nz, std::vector<double>(config.grid.Nx, 0.0));

    // Create coordinate centers
    result.x_centers.resize(config.grid.Nx);
    result.z_centers.resize(config.grid.Nz);

    for (int i = 0; i < config.grid.Nx; ++i) {
        result.x_centers[i] = (i + 0.5f) * config.grid.dx - (config.grid.Nx * config.grid.dx) / 2.0f;
    }

    for (int j = 0; j < config.grid.Nz; ++j) {
        result.z_centers[j] = j * config.grid.dz;
    }

    // Create phase-space grids for K1-K6 pipeline
    // TODO: Make these configurable via IncidentParticleConfig
    const int N_theta = 36;           // Global angular bins
    // Option D2: Piecewise-uniform energy grid (401 bins total)
    // [0.1-2 MeV]: 0.1 MeV → 19 bins (Bragg peak core)
    // [2-20 MeV]: 0.25 MeV → 72 bins (Bragg peak falloff)
    // [20-100 MeV]: 0.5 MeV → 160 bins (Mid-energy plateau)
    // [100-250 MeV]: 1 MeV → 150 bins (High energy)
    // Local bins must match compile-time constants in local_bins.hpp
    // NOTE: Using reduced values for memory (SPEC requires 8, 4)
    const int N_theta_local = ::N_theta_local;  // = 4 (memory optimized, SPEC wants 8)
    const int N_E_local = ::N_E_local;          // = 2 (memory optimized, SPEC wants 4)

    // Angular grid: [-pi/2, pi/2] for full angular coverage
    AngularGrid theta_grid(-M_PI/2.0f, M_PI/2.0f, N_theta);

    // Energy grid: piecewise-uniform (Option D2)
    // Format: {{E_start, E_end, resolution}, ...}
    // Use finer resolution at high energies for accurate energy tracking
    std::vector<std::tuple<float, float, float>> energy_groups = {
        {0.1f, 2.0f, 0.1f},      // 19 bins - Bragg peak core
        {2.0f, 20.0f, 0.2f},     // 90 bins - Bragg peak falloff (finer)
        {20.0f, 100.0f, 0.25f},  // 320 bins - Mid-energy plateau (finer)
        {100.0f, 250.0f, 0.25f}   // 600 bins - High energy (MUCH FINER for tracking)
    };
    EnergyGrid E_grid = EnergyGrid::CreatePiecewise(energy_groups);
    const int N_E = E_grid.N_E;  // Will be 1029

    std::cout << "=== Option D2 Energy Grid ===" << std::endl;
    std::cout << "  N_E = " << N_E << " bins" << std::endl;
    std::cout << "  [0.1-2 MeV]: 0.1 MeV/bin (19 bins)" << std::endl;
    std::cout << "  [2-20 MeV]: 0.2 MeV/bin (90 bins)" << std::endl;
    std::cout << "  [20-100 MeV]: 0.25 MeV/bin (320 bins)" << std::endl;
    std::cout << "  [100-250 MeV]: 0.25 MeV/bin (600 bins)" << std::endl;
    std::cout << "=============================" << std::endl;

    // Generate NIST range LUT using the piecewise-uniform energy grid (Option D2)
    // This ensures LUT uses the same energy bins as the transport
    auto lut = GenerateRLUT(E_grid);

    // DEBUG: Verify LUT values
    std::cout << "=== LUT Verification ===" << std::endl;
    std::cout << "  R(0.1 MeV) = " << lut.lookup_R(0.1f) << " mm (expected ~0.0016)" << std::endl;
    std::cout << "  R(150 MeV) = " << lut.lookup_R(150.0f) << " mm (expected ~158)" << std::endl;
    std::cout << "  R(250 MeV) = " << lut.lookup_R(250.0f) << " mm (max range in LUT)" << std::endl;
    std::cout << "  S(150 MeV) = " << lut.lookup_S(150.0f) << " MeV*cm^2/g" << std::endl;

    // Test energy loss calculation
    float E_test = 150.0f;
    float step = 2.0f;  // Use 2mm step for Option D2
    float R_before = lut.lookup_R(E_test);
    float R_after = R_before - step;
    float E_after = lut.lookup_E_inverse(std::max(0.001f, R_after));
    float dE = E_test - E_after;
    std::cout << "  Energy loss test: E=" << E_test << " -> E_new=" << E_after
              << ", dE=" << dE << " MeV (step=" << step << "mm)" << std::endl;
    std::cout << "=======================" << std::endl;

    // Create device LUT wrapper
    DeviceLUTWrapper device_lut;
    if (!init_device_lut(reinterpret_cast<const RLUT&>(lut), device_lut)) {
        throw std::runtime_error("Failed to initialize device LUT");
    }

    // Grid origin
    float x_min = -(config.grid.Nx * config.grid.dx) / 2.0f;
    float z_min = 0.0f;

    // Allocate device memory for grid edges
    float* d_theta_edges = nullptr;
    float* d_E_edges = nullptr;

    cudaMalloc(&d_theta_edges, (N_theta + 1) * sizeof(float));
    cudaMalloc(&d_E_edges, (N_E + 1) * sizeof(float));

    cudaMemcpy(d_theta_edges, theta_grid.edges.data(), (N_theta + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E_edges, E_grid.edges.data(), (N_E + 1) * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Running K1-K6 GPU pipeline transport..." << std::endl;
    std::cout << "  GPU: " << get_gpu_name() << std::endl;
    std::cout << "  Energy: " << config.get_energy_MeV() << " MeV" << std::endl;
    std::cout << "  Phase-space: " << N_theta << " x " << N_E << " bins" << std::endl;

    // Run K1-K6 pipeline transport with angular divergence and spatial spread
    run_k1k6_pipeline_transport(
        config.get_position_x_mm(),
        config.get_position_z_mm(),
        config.get_angle_rad(),
        config.angular.sigma_theta,  // Angular divergence from config
        config.spatial.sigma_x,       // Spatial beam width from config
        config.get_energy_MeV(),
        config.W_total,
        config.spatial.sigma_x,        // Lateral beam spread
        config.angular.sigma_theta,    // Angular divergence
        config.energy.sigma_E,         // Energy spread
        config.sampling.n_samples,     // Number of samples
        config.sampling.random_seed,   // RNG seed
        config.grid.Nx, config.grid.Nz,
        config.grid.dx, config.grid.dz,
        x_min, z_min,
        N_theta, N_E,
        N_theta_local, N_E_local,
        d_theta_edges,
        d_E_edges,
        *device_lut.get(),
        result.edep
    );

    // Cleanup device edges
    cudaFree(d_theta_edges);
    cudaFree(d_E_edges);

    std::cout << "GPU transport complete." << std::endl;

    return result;
}

#else // !SM2D_HAS_CUDA (CPU-only build)

bool GPUTransportRunner::is_gpu_available() {
    return false;  // No CUDA support in this build
}

std::string GPUTransportRunner::get_gpu_name() {
    return "N/A (CPU-only build)";
}

SimulationResult GPUTransportRunner::run(const IncidentParticleConfig& config) {
    throw std::runtime_error("GPU transport not available. This binary was built without CUDA support. Please use the CPU deterministic transport path.");
}

#endif // SM2D_HAS_CUDA

} // namespace sm_2d
