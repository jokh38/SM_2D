#pragma once
#include "source/pencil_source.hpp"
#include "source/gaussian_source.hpp"
#include "core/incident_particle_config.hpp"
#include "core/grids.hpp"
#include "core/psi_storage.hpp"

namespace sm_2d {

/**
 * @brief Adapter to convert IncidentParticleConfig to legacy source types
 *
 * This allows using the new centralized configuration system with existing
 * source injection code without modifications.
 */

/**
 * @brief Create PencilSource from IncidentParticleConfig
 */
inline PencilSource make_pencil_source_legacy(const IncidentParticleConfig& config) {
    PencilSource src;
    src.x0 = config.spatial.x0;
    src.z0 = config.spatial.z0;
    src.theta0 = config.angular.theta0;
    src.E0 = config.energy.mean_E0;
    src.W_total = config.W_total;
    return src;
}

/**
 * @brief Create GaussianSource from IncidentParticleConfig
 */
inline GaussianSource make_gaussian_source_legacy(const IncidentParticleConfig& config) {
    GaussianSource src;
    src.x0 = config.spatial.x0;
    src.theta0 = config.angular.theta0;
    src.sigma_x = config.spatial.sigma_x;
    src.sigma_theta = config.angular.sigma_theta;
    src.E0 = config.energy.mean_E0;
    src.sigma_E = config.energy.sigma_E;
    src.W_total = config.W_total;
    src.n_samples = config.sampling.n_samples;
    return src;
}

/**
 * @brief Unified source injector that selects source type based on config
 */
inline void inject_incident_particle(
    PsiC& psi,
    const IncidentParticleConfig& config,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid
) {
    switch (config.beam_profile) {
        case BeamProfileType::GAUSSIAN: {
            auto src = make_gaussian_source_legacy(config);
            inject_source(psi, src, e_grid, a_grid);
            break;
        }
        case BeamProfileType::PENCIL:
        case BeamProfileType::FLAT:  // Treat FLAT as PENCIL for now
        case BeamProfileType::CUSTOM: // Treat CUSTOM as PENCIL for now
        default: {
            auto src = make_pencil_source_legacy(config);
            inject_source(psi, src, e_grid, a_grid);
            break;
        }
    }
}

/**
 * @brief Builder class for fluent configuration construction
 */
class IncidentParticleBuilder {
public:
    IncidentParticleConfig config;

    IncidentParticleBuilder& particle(ParticleType type) {
        config.particle_type = type;
        return *this;
    }

    IncidentParticleBuilder& mass(float amu) {
        config.particle_mass_amu = amu;
        return *this;
    }

    IncidentParticleBuilder& charge(float e) {
        config.particle_charge_e = e;
        return *this;
    }

    IncidentParticleBuilder& beam_profile(BeamProfileType profile) {
        config.beam_profile = profile;
        return *this;
    }

    IncidentParticleBuilder& energy(float mean_E, float sigma_E = 0.0f) {
        config.energy.mean_E0 = mean_E;
        config.energy.sigma_E = sigma_E;
        return *this;
    }

    IncidentParticleBuilder& position(float x, float z) {
        config.spatial.x0 = x;
        config.spatial.z0 = z;
        return *this;
    }

    IncidentParticleBuilder& angle(float theta) {
        config.angular.theta0 = theta;
        return *this;
    }

    IncidentParticleBuilder& spatial_spread(float sigma_x, float sigma_z = 0.0f) {
        config.spatial.sigma_x = sigma_x;
        config.spatial.sigma_z = sigma_z;
        return *this;
    }

    IncidentParticleBuilder& angular_spread(float sigma_theta) {
        config.angular.sigma_theta = sigma_theta;
        return *this;
    }

    IncidentParticleBuilder& samples(int n) {
        config.sampling.n_samples = n;
        return *this;
    }

    IncidentParticleBuilder& weight(float w) {
        config.W_total = w;
        return *this;
    }

    IncidentParticleBuilder& seed(unsigned int s) {
        config.sampling.random_seed = s;
        return *this;
    }

    IncidentParticleConfig build() const {
        IncidentParticleConfig result = config;
        result.validate();
        return result;
    }
};

/**
 * @brief Convenience functions for common preset configurations
 */
namespace presets {

inline IncidentParticleConfig proton_70MeV_pencil() {
    IncidentParticleConfig config;
    config.particle_type = ParticleType::PROTON;
    config.beam_profile = BeamProfileType::PENCIL;
    config.energy.mean_E0 = 70.0f;
    config.spatial.x0 = 30.0f;  // Center of typical grid
    config.spatial.z0 = 0.0f;
    config.W_total = 1.0f;
    return config;
}

inline IncidentParticleConfig proton_150MeV_pencil() {
    IncidentParticleConfig config;
    config.particle_type = ParticleType::PROTON;
    config.beam_profile = BeamProfileType::PENCIL;
    config.energy.mean_E0 = 150.0f;
    config.W_total = 1.0f;
    return config;
}

inline IncidentParticleConfig proton_150MeV_gaussian() {
    IncidentParticleConfig config;
    config.particle_type = ParticleType::PROTON;
    config.beam_profile = BeamProfileType::GAUSSIAN;
    config.energy.mean_E0 = 150.0f;
    config.energy.sigma_E = 1.0f;
    config.spatial.sigma_x = 5.0f;
    config.angular.sigma_theta = 0.01f;
    config.sampling.n_samples = 1000;
    config.W_total = 1.0f;
    return config;
}

inline IncidentParticleConfig electron_20MeV_pencil() {
    IncidentParticleConfig config;
    config.particle_type = ParticleType::ELECTRON;
    config.beam_profile = BeamProfileType::PENCIL;
    config.energy.mean_E0 = 20.0f;
    config.particle_mass_amu = 0.00054858f;
    config.particle_charge_e = -1.0f;
    config.W_total = 1.0f;
    return config;
}

} // namespace presets

} // namespace sm_2d
