#pragma once
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace sm_2d {

/**
 * @brief Particle type enumeration
 */
enum class ParticleType {
    PROTON,
    ELECTRON,
    POSITRON,
    ALPHA,
    CARBON_ION,
    CUSTOM
};

/**
 * @brief Convert particle type to string
 */
inline std::string particle_type_to_string(ParticleType type) {
    switch (type) {
        case ParticleType::PROTON:     return "proton";
        case ParticleType::ELECTRON:   return "electron";
        case ParticleType::POSITRON:   return "positron";
        case ParticleType::ALPHA:      return "alpha";
        case ParticleType::CARBON_ION: return "carbon_ion";
        case ParticleType::CUSTOM:     return "custom";
        default:                       return "unknown";
    }
}

/**
 * @brief Convert string to particle type
 */
inline ParticleType string_to_particle_type(const std::string& str) {
    if (str == "proton")     return ParticleType::PROTON;
    if (str == "electron")   return ParticleType::ELECTRON;
    if (str == "positron")   return ParticleType::POSITRON;
    if (str == "alpha")      return ParticleType::ALPHA;
    if (str == "carbon_ion") return ParticleType::CARBON_ION;
    if (str == "custom")     return ParticleType::CUSTOM;
    return ParticleType::PROTON;  // default
}

/**
 * @brief Beam profile type enumeration
 */
enum class BeamProfileType {
    PENCIL,      // Point source, no spatial spread
    GAUSSIAN,    // Gaussian spatial/energy/angular distribution
    FLAT,        // Uniform circular/rectangular beam
    CUSTOM       // User-defined profile
};

/**
 * @brief Convert beam profile type to string
 */
inline std::string beam_profile_to_string(BeamProfileType type) {
    switch (type) {
        case BeamProfileType::PENCIL:  return "pencil";
        case BeamProfileType::GAUSSIAN: return "gaussian";
        case BeamProfileType::FLAT:     return "flat";
        case BeamProfileType::CUSTOM:   return "custom";
        default:                        return "pencil";
    }
}

/**
 * @brief Convert string to beam profile type
 */
inline BeamProfileType string_to_beam_profile(const std::string& str) {
    if (str == "pencil")  return BeamProfileType::PENCIL;
    if (str == "gaussian") return BeamProfileType::GAUSSIAN;
    if (str == "flat")     return BeamProfileType::FLAT;
    if (str == "custom")   return BeamProfileType::CUSTOM;
    return BeamProfileType::PENCIL;  // default
}

/**
 * @brief Energy spectrum configuration
 */
struct EnergySpectrum {
    float mean_E0 = 150.0f;          // Mean/central energy (MeV)
    float sigma_E = 0.0f;            // Energy spread (MeV), 0 = monoenergetic
    float min_E = 0.0f;              // Minimum energy cutoff (MeV)
    float max_E = 500.0f;            // Maximum energy cutoff (MeV)

    // Custom spectrum (for non-Gaussian distributions)
    std::vector<float> energy_bins;
    std::vector<float> probabilities;

    bool is_monoenergetic() const { return sigma_E <= 0.0f; }
    bool is_custom() const { return !energy_bins.empty(); }
};

/**
 * @brief Spatial beam configuration
 */
struct SpatialBeamConfig {
    float x0 = 0.0f;           // Central position X (mm)
    float z0 = 0.0f;           // Central position Z (mm)
    float y0 = 0.0f;           // Central position Y (mm) for 3D extension

    // Gaussian profile parameters
    float sigma_x = 0.0f;      // Spatial spread X (mm)
    float sigma_z = 0.0f;      // Spatial spread Z (mm)

    // Flat beam parameters
    float width_x = 0.0f;      // Beam width X (mm)
    float width_z = 0.0f;      // Beam width Z (mm)
    float radius = 0.0f;       // Circular beam radius (mm)

    bool has_spread() const { return sigma_x > 0.0f || width_x > 0.0f || radius > 0.0f; }
};

/**
 * @brief Angular beam configuration
 */
struct AngularBeamConfig {
    float theta0 = 0.0f;       // Central angle in X-Z plane (radians)
    float phi0 = 0.0f;         // Central angle in Y-Z plane (radians) for 3D

    float sigma_theta = 0.0f;  // Angular divergence (radians)
    float sigma_phi = 0.0f;    // Angular divergence Y (radians)

    bool has_divergence() const { return sigma_theta > 0.0f; }
};

/**
 * @brief Monte Carlo sampling configuration
 */
struct SamplingConfig {
    int n_samples = 1000;          // Number of MC samples for distributed sources
    unsigned int random_seed = 42; // Random seed for reproducibility
    bool use_stratified = false;   // Use stratified sampling
};

/**
 * @brief Unified incident particle configuration
 *
 * This structure centralizes all parameters for incident particle specification.
 * It supports multiple beam types and particle species.
 */
struct IncidentParticleConfig {
    // Particle specification
    ParticleType particle_type = ParticleType::PROTON;
    float particle_mass_amu = 1.0f;    // Mass in atomic mass units
    float particle_charge_e = 1.0f;    // Charge in elementary charge units

    // Beam profile selection
    BeamProfileType beam_profile = BeamProfileType::PENCIL;

    // Energy, spatial, and angular configuration
    EnergySpectrum energy;
    SpatialBeamConfig spatial;
    AngularBeamConfig angular;

    // Monte Carlo sampling
    SamplingConfig sampling;

    // Beam weight/intensity
    float W_total = 1.0f;              // Total weight (normalized dose)

    // Validation/defaults
    void validate() const;
    void set_defaults_for_proton();
    void set_defaults_for_electron();

    // Utility methods
    float get_energy_MeV() const { return energy.mean_E0; }
    float get_position_x_mm() const { return spatial.x0; }
    float get_position_z_mm() const { return spatial.z0; }
    float get_angle_rad() const { return angular.theta0; }
};

/**
 * @brief Validate configuration parameters
 */
inline void IncidentParticleConfig::validate() const {
    // Energy validation
    if (energy.mean_E0 <= 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: mean_E0 must be positive");
    }
    if (energy.sigma_E < 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: sigma_E must be non-negative");
    }
    if (energy.min_E < 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: min_E must be non-negative");
    }
    if (energy.max_E <= energy.min_E) {
        throw std::invalid_argument("IncidentParticleConfig: max_E must be > min_E");
    }

    // Spatial validation
    if (spatial.sigma_x < 0.0f || spatial.sigma_z < 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: sigma must be non-negative");
    }
    if (spatial.width_x < 0.0f || spatial.width_z < 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: width must be non-negative");
    }
    if (spatial.radius < 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: radius must be non-negative");
    }

    // Angular validation
    if (angular.sigma_theta < 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: sigma_theta must be non-negative");
    }

    // Sampling validation
    if (sampling.n_samples <= 0) {
        throw std::invalid_argument("IncidentParticleConfig: n_samples must be positive");
    }

    // Weight validation
    if (W_total <= 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: W_total must be positive");
    }
}

/**
 * @brief Set defaults for 150 MeV proton pencil beam
 */
inline void IncidentParticleConfig::set_defaults_for_proton() {
    particle_type = ParticleType::PROTON;
    particle_mass_amu = 1.0f;
    particle_charge_e = 1.0f;
    beam_profile = BeamProfileType::PENCIL;

    energy.mean_E0 = 150.0f;
    energy.sigma_E = 0.0f;
    energy.min_E = 0.0f;
    energy.max_E = 250.0f;

    spatial.x0 = 0.0f;
    spatial.z0 = 0.0f;
    spatial.sigma_x = 0.0f;
    spatial.sigma_z = 0.0f;

    angular.theta0 = 0.0f;
    angular.sigma_theta = 0.0f;

    sampling.n_samples = 1000;
    sampling.random_seed = 42;

    W_total = 1.0f;
}

/**
 * @brief Set defaults for electron beam
 */
inline void IncidentParticleConfig::set_defaults_for_electron() {
    particle_type = ParticleType::ELECTRON;
    particle_mass_amu = 0.00054858f;  // electron mass in amu
    particle_charge_e = -1.0f;
    beam_profile = BeamProfileType::PENCIL;

    energy.mean_E0 = 20.0f;
    energy.sigma_E = 0.0f;
    energy.min_E = 0.0f;
    energy.max_E = 30.0f;

    spatial.x0 = 0.0f;
    spatial.z0 = 0.0f;
    spatial.sigma_x = 0.0f;
    spatial.sigma_z = 0.0f;

    angular.theta0 = 0.0f;
    angular.sigma_theta = 0.0f;

    sampling.n_samples = 1000;
    sampling.random_seed = 42;

    W_total = 1.0f;
}

/**
 * @brief Create PencilSource compatible structure from IncidentParticleConfig
 */
struct PencilSourceView {
    float x0;
    float z0;
    float theta0;
    float E0;
    float W_total;
};

inline PencilSourceView make_pencil_source(const IncidentParticleConfig& config) {
    return PencilSourceView{
        config.spatial.x0,
        config.spatial.z0,
        config.angular.theta0,
        config.energy.mean_E0,
        config.W_total
    };
}

/**
 * @brief Create GaussianSource compatible structure from IncidentParticleConfig
 */
struct GaussianSourceView {
    float x0;
    float theta0;
    float sigma_x;
    float sigma_theta;
    float E0;
    float sigma_E;
    float W_total;
    int n_samples;
};

inline GaussianSourceView make_gaussian_source(const IncidentParticleConfig& config) {
    return GaussianSourceView{
        config.spatial.x0,
        config.angular.theta0,
        config.spatial.sigma_x,
        config.angular.sigma_theta,
        config.energy.mean_E0,
        config.energy.sigma_E,
        config.W_total,
        config.sampling.n_samples
    };
}

} // namespace sm_2d
