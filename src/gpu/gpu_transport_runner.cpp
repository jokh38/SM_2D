#include "gpu/gpu_transport_runner.hpp"
#include "cuda/gpu_transport_wrapper.hpp"
#include "core/local_bins.hpp"  // For N_theta_local, N_E_local
#include <cuda_runtime.h>

#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include "core/grids.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>

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
    if (config.transport.N_theta_local != ::N_theta_local ||
        config.transport.N_E_local != ::N_E_local) {
        throw std::invalid_argument(
            "Transport local bin config must match compile-time constants: "
            "N_theta_local=" + std::to_string(::N_theta_local) +
            ", N_E_local=" + std::to_string(::N_E_local)
        );
    }

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

    TransportConfig runtime_transport = config.transport;
    if (runtime_transport.max_iterations == 0) {
        runtime_transport.max_iterations = config.grid.max_steps;
    }
    // Legacy compatibility: if only E_trigger was customized, lift it into fine-on/off policy.
    const TransportConfig default_transport{};
    if (std::abs(runtime_transport.E_trigger - runtime_transport.E_fine_on) > 1e-6f &&
        std::abs(runtime_transport.E_fine_on - default_transport.E_fine_on) < 1e-6f &&
        std::abs(runtime_transport.E_fine_off - default_transport.E_fine_off) < 1e-6f) {
        runtime_transport.E_fine_on = runtime_transport.E_trigger;
        runtime_transport.E_fine_off = std::max(runtime_transport.E_fine_off, runtime_transport.E_fine_on);
    }
    runtime_transport.E_trigger = runtime_transport.E_fine_on;

    // Create phase-space grids for K1-K6 pipeline from runtime transport config.
    const int N_theta = runtime_transport.N_theta;
    const int N_theta_local = runtime_transport.N_theta_local;
    const int N_E_local = runtime_transport.N_E_local;

    // Angular grid: [-pi/2, pi/2] for full angular coverage
    AngularGrid theta_grid(-M_PI/2.0f, M_PI/2.0f, N_theta);

    // Energy grid: piecewise-uniform from transport configuration.
    std::vector<std::tuple<float, float, float>> energy_groups;
    energy_groups.reserve(runtime_transport.energy_groups.size());
    for (const auto& group : runtime_transport.energy_groups) {
        energy_groups.emplace_back(group.E_min_MeV, group.E_max_MeV, group.dE_MeV);
    }
    EnergyGrid E_grid = EnergyGrid::CreatePiecewise(energy_groups);
    const int N_E = E_grid.N_E;

    if (runtime_transport.log_level >= 2) {
        std::cout << "=== Energy Grid ===" << std::endl;
        std::cout << "  N_E = " << N_E << " bins" << std::endl;
        for (const auto& group : runtime_transport.energy_groups) {
            std::cout << "  [" << group.E_min_MeV << "-" << group.E_max_MeV
                      << " MeV]: " << group.dE_MeV << " MeV/bin" << std::endl;
        }
        std::cout << "===================" << std::endl;
    }

    // Generate NIST range LUT using the piecewise-uniform energy grid (Option D2)
    // This ensures LUT uses the same energy bins as the transport
    auto lut = GenerateRLUT(E_grid);

    if (runtime_transport.log_level >= 2) {
        std::cout << "=== LUT Verification ===" << std::endl;
        std::cout << "  R(0.1 MeV) = " << lut.lookup_R(0.1f) << " mm (expected ~0.0016)" << std::endl;
        std::cout << "  R(150 MeV) = " << lut.lookup_R(150.0f) << " mm (expected ~158)" << std::endl;
        std::cout << "  R(250 MeV) = " << lut.lookup_R(250.0f) << " mm (max range in LUT)" << std::endl;
        std::cout << "  S(150 MeV) = " << lut.lookup_S(150.0f) << " MeV*cm^2/g" << std::endl;

        float E_test = 150.0f;
        float step = 2.0f;
        float R_before = lut.lookup_R(E_test);
        float R_after = R_before - step;
        float E_after = lut.lookup_E_inverse(std::max(0.001f, R_after));
        float dE = E_test - E_after;
        std::cout << "  Energy loss test: E=" << E_test << " -> E_new=" << E_after
                  << ", dE=" << dE << " MeV (step=" << step << "mm)" << std::endl;
        std::cout << "=======================" << std::endl;
    }

    // Create device LUT wrapper
    DeviceLUTWrapper device_lut;
    if (!init_device_lut(lut, device_lut)) {
        throw std::runtime_error("Failed to initialize device LUT");
    }

    // Grid origin
    float x_min = -(config.grid.Nx * config.grid.dx) / 2.0f;
    float z_min = 0.0f;

    if (runtime_transport.log_level >= 1) {
        std::cout << "Running K1-K6 GPU pipeline transport..." << std::endl;
        std::cout << "  GPU: " << get_gpu_name() << std::endl;
        std::cout << "  Energy: " << config.get_energy_MeV() << " MeV" << std::endl;
        std::cout << "  Phase-space: " << N_theta << " x " << N_E << " bins" << std::endl;
        std::cout << "  max_iterations: " << runtime_transport.max_iterations << std::endl;
    }

    // Run K1-K6 pipeline transport with angular divergence and spatial spread
    if (!run_k1k6_pipeline_transport(
        config.get_position_x_mm(),      // x0
        config.get_position_z_mm(),      // z0
        config.get_angle_rad(),          // theta0
        config.get_energy_MeV(),         // E0 (beam energy)
        config.W_total,                  // W_total (total weight)
        config.spatial.sigma_x,          // sigma_x (lateral beam spread)
        config.angular.sigma_theta,      // sigma_theta (angular divergence)
        config.energy.sigma_E,           // sigma_E (energy spread)
        config.sampling.n_samples,       // n_samples
        config.sampling.random_seed,     // random_seed
        config.grid.Nx, config.grid.Nz,  // Nx, Nz
        config.grid.dx, config.grid.dz,  // dx, dz
        x_min, z_min,                    // x_min, z_min
        N_theta_local, N_E_local,        // N_theta_local, N_E_local
        theta_grid,
        E_grid,
        *device_lut.get(),               // dlut
        runtime_transport,
        result.edep                      // edep output
    )) {
        throw std::runtime_error("GPU pipeline execution failed");
    }

    if (runtime_transport.log_level >= 1) {
        std::cout << "GPU transport complete." << std::endl;
    }

    return result;
}

} // namespace sm_2d
