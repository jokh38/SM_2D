#pragma once
#include <string>
#include <tuple>
#include <vector>
#include <memory>
#include <stdexcept>
#include "core/local_bins.hpp"

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
 * @brief Sampling configuration for distributed sources
 */
struct SamplingConfig {
    int n_samples = 1000;          // Number of samples for distributed sources
    unsigned int random_seed = 42; // Random seed for reproducibility
    bool use_stratified = false;   // Use stratified sampling
};

/**
 * @brief Grid configuration for simulation
 */
struct GridConfig {
    int Nx = 100;                  // Number of transverse bins
    int Nz = 200;                  // Number of depth bins
    float dx = 1.0f;               // Transverse spacing (mm)
    float dz = 1.0f;               // Depth spacing (mm)
    int max_steps = 100;           // Maximum simulation steps
};

/**
 * @brief Piecewise energy-grid segment for transport
 */
struct TransportEnergyGroup {
    float E_min_MeV = 0.1f;
    float E_max_MeV = 1.0f;
    float dE_MeV = 0.1f;
};

inline std::vector<TransportEnergyGroup> default_transport_energy_groups() {
    return {
        {0.1f, 2.0f, 0.1f},
        {2.0f, 20.0f, 0.2f},
        {20.0f, 100.0f, 0.25f},
        {100.0f, 250.0f, 0.25f}
    };
}

/**
 * @brief Transport pipeline runtime configuration
 */
struct TransportConfig {
    // Phase-space grid
    int N_theta = 36;                 // Global angular bins
    int N_theta_local = ::N_theta_local;  // Local angular bins per block (must match compile-time constant)
    int N_E_local = ::N_E_local;          // Local energy bins per block (must match compile-time constant)
    std::vector<TransportEnergyGroup> energy_groups = default_transport_energy_groups();

    // K1/K2/K3 selection + transport controls
    float E_fine_on = 10.0f;          // Fine transport activation threshold [MeV]
    float E_fine_off = 11.0f;         // Fine transport deactivation threshold [MeV] (hysteresis)
    float E_trigger = 10.0f;          // Legacy alias for E_fine_on [MeV]
    float weight_active_min = 1e-12f; // Active-cell minimum weight
    float E_coarse_max = 300.0f;      // Coarse transport upper validity energy [MeV]
    float step_coarse = 5.0f;         // Coarse transport step [mm]
    int n_steps_per_cell = 1;         // K2 sub-steps
    int fine_batch_max_cells = 0;     // 0 => auto planning
    int fine_halo_cells = 1;          // Scratch halo thickness (cells)
    float preflight_vram_margin = 0.85f; // Preflight usable VRAM fraction [0,1]

    // Iteration + logging
    int max_iterations = 0;           // 0 => use grid.max_steps
    int log_level = 1;                // 0: quiet, 1: summary, 2: verbose
};

/**
 * @brief Output file configuration
 */
struct OutputConfig {
    // File paths
    std::string output_dir = "results";
    std::string dose_2d_file = "dose_2d.txt";
    std::string pdd_file = "pdd.txt";
    std::string let_file = "";     // Empty = don't output LET

    // Output format
    std::string format = "txt";    // txt, csv, or hdf5

    // Output options
    bool normalize_dose = true;    // Normalize to max dose
    bool save_2d = true;           // Save 2D dose distribution
    bool save_pdd = true;          // Save depth-dose
    bool save_lat_profiles = false;// Save lateral profiles at specified depths
    std::vector<float> lat_depths_mm;  // Depths for lateral profiles
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

    // Sampling configuration
    SamplingConfig sampling;

    // Grid configuration
    GridConfig grid;

    // Transport kernel configuration
    TransportConfig transport;

    // Output configuration
    OutputConfig output;

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

    // Grid validation
    if (grid.Nx <= 0) {
        throw std::invalid_argument("IncidentParticleConfig: Nx must be positive");
    }
    if (grid.Nz <= 0) {
        throw std::invalid_argument("IncidentParticleConfig: Nz must be positive");
    }
    if (grid.dx <= 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: dx must be positive");
    }
    if (grid.dz <= 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: dz must be positive");
    }
    if (grid.max_steps <= 0) {
        throw std::invalid_argument("IncidentParticleConfig: max_steps must be positive");
    }

    // Transport validation
    if (transport.N_theta <= 0) {
        throw std::invalid_argument("IncidentParticleConfig: transport.N_theta must be positive");
    }
    if (transport.N_theta_local <= 0) {
        throw std::invalid_argument("IncidentParticleConfig: transport.N_theta_local must be positive");
    }
    if (transport.N_E_local <= 0) {
        throw std::invalid_argument("IncidentParticleConfig: transport.N_E_local must be positive");
    }
    if (transport.E_fine_on <= 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: transport.E_fine_on must be positive");
    }
    if (transport.E_fine_off < transport.E_fine_on) {
        throw std::invalid_argument("IncidentParticleConfig: transport.E_fine_off must be >= transport.E_fine_on");
    }
    if (transport.E_trigger <= 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: transport.E_trigger must be positive");
    }
    if (transport.weight_active_min <= 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: transport.weight_active_min must be positive");
    }
    if (transport.E_coarse_max <= 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: transport.E_coarse_max must be positive");
    }
    if (transport.step_coarse <= 0.0f) {
        throw std::invalid_argument("IncidentParticleConfig: transport.step_coarse must be positive");
    }
    if (transport.n_steps_per_cell <= 0) {
        throw std::invalid_argument("IncidentParticleConfig: transport.n_steps_per_cell must be positive");
    }
    if (transport.fine_batch_max_cells < 0) {
        throw std::invalid_argument("IncidentParticleConfig: transport.fine_batch_max_cells cannot be negative");
    }
    if (transport.fine_halo_cells < 0) {
        throw std::invalid_argument("IncidentParticleConfig: transport.fine_halo_cells cannot be negative");
    }
    if (transport.preflight_vram_margin <= 0.0f || transport.preflight_vram_margin > 1.0f) {
        throw std::invalid_argument("IncidentParticleConfig: transport.preflight_vram_margin must be in (0,1]");
    }
    if (transport.max_iterations < 0) {
        throw std::invalid_argument("IncidentParticleConfig: transport.max_iterations cannot be negative");
    }
    if (transport.log_level < 0) {
        throw std::invalid_argument("IncidentParticleConfig: transport.log_level cannot be negative");
    }
    if (transport.energy_groups.empty()) {
        throw std::invalid_argument("IncidentParticleConfig: transport.energy_groups cannot be empty");
    }
    float last_max = transport.energy_groups.front().E_min_MeV;
    for (size_t i = 0; i < transport.energy_groups.size(); ++i) {
        const auto& group = transport.energy_groups[i];
        if (group.E_min_MeV <= 0.0f) {
            throw std::invalid_argument("IncidentParticleConfig: transport.energy_groups E_min must be positive");
        }
        if (group.E_max_MeV <= group.E_min_MeV) {
            throw std::invalid_argument("IncidentParticleConfig: transport.energy_groups E_max must be > E_min");
        }
        if (group.dE_MeV <= 0.0f) {
            throw std::invalid_argument("IncidentParticleConfig: transport.energy_groups dE must be positive");
        }
        if (i > 0 && group.E_min_MeV < last_max) {
            throw std::invalid_argument("IncidentParticleConfig: transport.energy_groups must be non-overlapping and ordered");
        }
        last_max = group.E_max_MeV;
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

    grid.Nx = 100;
    grid.Nz = 200;
    grid.dx = 1.0f;
    grid.dz = 1.0f;
    grid.max_steps = 100;

    output.output_dir = "results";
    output.format = "txt";
    output.save_2d = true;
    output.save_pdd = true;

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
