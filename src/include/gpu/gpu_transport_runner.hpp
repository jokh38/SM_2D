#pragma once
#include "core/incident_particle_config.hpp"
#include "validation/deterministic_beam.hpp"
#include <memory>
#include <vector>

namespace sm_2d {

/**
 * @brief GPU-based deterministic transport runner
 *
 * This runs the full K1-K6 CUDA kernel pipeline for accurate proton
 * dose calculation with proper energy straggling (Vavilov model).
 */
class GPUTransportRunner {
public:
    /**
     * @brief Run GPU-based transport simulation
     *
     * @param config Simulation configuration
     * @return SimulationResult Dose distribution result
     */
    static SimulationResult run(const IncidentParticleConfig& config);

    /**
     * @brief Check if GPU is available
     */
    static bool is_gpu_available();

    /**
     * @brief Get GPU device name
     */
    static std::string get_gpu_name();
};

} // namespace sm_2d
