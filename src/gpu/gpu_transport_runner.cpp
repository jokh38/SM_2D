#include "gpu/gpu_transport_runner.hpp"
#include "cuda/gpu_transport_wrapper.hpp"
#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace sm_2d {

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

    // Number of particles (use n_samples from config)
    int n_particles = config.sampling.n_samples;

    std::cout << "Running GPU transport with " << n_particles << " particles..." << std::endl;
    std::cout << "  GPU: " << get_gpu_name() << std::endl;
    std::cout << "  Energy: " << config.get_energy_MeV() << " MeV" << std::endl;

    // Run GPU transport
    run_gpu_transport(
        config.get_position_x_mm(),
        config.get_position_z_mm(),
        config.get_angle_rad(),
        config.get_energy_MeV(),
        config.W_total,
        n_particles,
        config.grid.Nx, config.grid.Nz,
        config.grid.dx, config.grid.dz,
        x_min, z_min,
        *device_lut.get(),
        result.edep
    );

    std::cout << "GPU transport complete." << std::endl;

    return result;
}

} // namespace sm_2d
