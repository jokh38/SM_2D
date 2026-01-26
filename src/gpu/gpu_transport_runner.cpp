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

    // Generate NIST range LUT
    auto lut = GenerateRLUT(0.1f, 300.0f, 256);

    // Create device LUT wrapper
    DeviceLUTWrapper device_lut;
    if (!init_device_lut(reinterpret_cast<const RLUT&>(lut), device_lut)) {
        throw std::runtime_error("Failed to initialize device LUT");
    }

    // Grid origin
    float x_min = -(config.grid.Nx * config.grid.dx) / 2.0f;
    float z_min = 0.0f;

    // Create phase-space grids for K1-K6 pipeline
    // TODO: Make these configurable via IncidentParticleConfig
    const int N_theta = 36;           // Global angular bins
    const int N_E = 32;               // Global energy bins
    // Local bins must match compile-time constants in local_bins.hpp
    const int N_theta_local = ::N_theta_local;  // = 4
    const int N_E_local = ::N_E_local;          // = 2

    // Angular grid: [-pi/2, pi/2] for full angular coverage
    AngularGrid theta_grid(-M_PI/2.0f, M_PI/2.0f, N_theta);

    // Energy grid: log-spaced from 0.1 MeV to 300 MeV
    EnergyGrid E_grid(0.1f, 300.0f, N_E);

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

    // Run K1-K6 pipeline transport
    run_k1k6_pipeline_transport(
        config.get_position_x_mm(),
        config.get_position_z_mm(),
        config.get_angle_rad(),
        config.get_energy_MeV(),
        config.W_total,
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
