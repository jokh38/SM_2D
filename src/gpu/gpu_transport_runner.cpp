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
    // H7 FIX: E_max changed from 300.0 to 250.0 MeV
    // R(300 MeV) returns NaN due to NIST data range limitation (capped at 250 MeV)
    auto lut = GenerateRLUT(0.1f, 250.0f, 256);

    // DEBUG: Verify LUT values
    std::cout << "=== LUT Verification ===" << std::endl;
    std::cout << "  R(0.1 MeV) = " << lut.lookup_R(0.1f) << " mm (expected ~0.0016)" << std::endl;
    std::cout << "  R(150 MeV) = " << lut.lookup_R(150.0f) << " mm (expected ~158)" << std::endl;
    std::cout << "  R(250 MeV) = " << lut.lookup_R(250.0f) << " mm (max range in LUT)" << std::endl;
    std::cout << "  S(150 MeV) = " << lut.lookup_S(150.0f) << " MeV*cm^2/g" << std::endl;

    // Test energy loss calculation
    float E_test = 150.0f;
    float step = 0.5f;
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

    // Create phase-space grids for K1-K6 pipeline
    // TODO: Make these configurable via IncidentParticleConfig
    const int N_theta = 36;           // Global angular bins
    const int N_E = 1280;             // Global energy bins (~1 MeV resolution at high energy)
                                      // Bin width: 0.92 MeV at 150 MeV, 1.53 MeV at 250 MeV
                                      // This reduces energy loss from geometric mean rounding
                                      // Memory: ~2.1 GB for phase space (fits in 8GB VRAM)
    // Local bins must match compile-time constants in local_bins.hpp
    // NOTE: Using reduced values for memory (SPEC requires 8, 4)
    const int N_theta_local = ::N_theta_local;  // = 4 (memory optimized, SPEC wants 8)
    const int N_E_local = ::N_E_local;          // = 2 (memory optimized, SPEC wants 4)

    // Angular grid: [-pi/2, pi/2] for full angular coverage
    AngularGrid theta_grid(-M_PI/2.0f, M_PI/2.0f, N_theta);

    // Energy grid: log-spaced from 0.1 MeV to 250 MeV (SPEC v0.8 requirement)
    EnergyGrid E_grid(0.1f, 250.0f, N_E);

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
