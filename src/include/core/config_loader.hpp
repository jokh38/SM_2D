#pragma once
#include "incident_particle_config.hpp"
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>

namespace sm_2d {

/**
 * @brief Configuration section holding key-value pairs
 */
struct ConfigSection {
    std::unordered_map<std::string, std::string> values;

    std::string get(const std::string& key, const std::string& default_val = "") const {
        auto it = values.find(key);
        return (it != values.end()) ? it->second : default_val;
    }

    float get_float(const std::string& key, float default_val = 0.0f) const {
        auto it = values.find(key);
        if (it != values.end()) {
            try {
                return std::stof(it->second);
            } catch (...) {
                return default_val;
            }
        }
        return default_val;
    }

    int get_int(const std::string& key, int default_val = 0) const {
        auto it = values.find(key);
        if (it != values.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return default_val;
            }
        }
        return default_val;
    }

    bool get_bool(const std::string& key, bool default_val = false) const {
        auto it = values.find(key);
        if (it != values.end()) {
            std::string val = it->second;
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            return (val == "true" || val == "1" || val == "yes" || val == "on");
        }
        return default_val;
    }

    bool has(const std::string& key) const {
        return values.find(key) != values.end();
    }
};

/**
 * @brief Simple key-value configuration file parser
 *
 * File format (INI-style with sections):
 * ```
 * [particle]
 * type = proton
 * mass_amu = 1.0
 * charge_e = 1.0
 *
 * [beam]
 * profile = pencil
 * weight = 1.0
 *
 * [energy]
 * mean_MeV = 150.0
 * sigma_MeV = 0.0
 * min_MeV = 0.0
 * max_MeV = 250.0
 *
 * [spatial]
 * x0_mm = 0.0
 * z0_mm = 0.0
 * sigma_x_mm = 5.0
 * sigma_z_mm = 5.0
 *
 * [angular]
 * theta0_rad = 0.0
 * sigma_theta_rad = 0.01
 *
 * [sampling]
 * n_samples = 1000
 * random_seed = 42
 * ```
 */
class ConfigLoader {
public:
    std::unordered_map<std::string, ConfigSection> sections;

    /**
     * @brief Load configuration from file
     */
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        sections.clear();
        std::string current_section = "default";

        std::string line;
        while (std::getline(file, line)) {
            // Trim whitespace
            line = trim(line);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }

            // Check for section header
            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.length() - 2);
                // Trim section name
                current_section = trim(current_section);
                continue;
            }

            // Parse key = value
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));

                // Remove quotes if present
                if (value.size() >= 2 &&
                    ((value.front() == '"' && value.back() == '"') ||
                     (value.front() == '\'' && value.back() == '\''))) {
                    value = value.substr(1, value.length() - 2);
                }

                sections[current_section].values[key] = value;
            }
        }

        return true;
    }

    /**
     * @brief Save configuration to file
     */
    bool save(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        for (const auto& section_pair : sections) {
            const std::string& section_name = section_pair.first;
            const ConfigSection& section = section_pair.second;

            file << "[" << section_name << "]" << std::endl;
            for (const auto& value_pair : section.values) {
                file << value_pair.first << " = " << value_pair.second << std::endl;
            }
            file << std::endl;
        }

        return true;
    }

    /**
     * @brief Get section by name
     */
    ConfigSection get_section(const std::string& name) const {
        auto it = sections.find(name);
        if (it != sections.end()) {
            return it->second;
        }
        return ConfigSection{};
    }

    /**
     * @brief Check if section exists
     */
    bool has_section(const std::string& name) const {
        return sections.find(name) != sections.end();
    }

private:
    static std::string trim(const std::string& str) {
        size_t start = 0;
        while (start < str.length() && std::isspace(str[start])) {
            ++start;
        }
        if (start == str.length()) {
            return "";
        }

        size_t end = str.length() - 1;
        while (end > start && std::isspace(str[end])) {
            --end;
        }

        return str.substr(start, end - start + 1);
    }
};

inline std::string trim_config_token(const std::string& str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        ++start;
    }
    if (start == str.length()) {
        return "";
    }

    size_t end = str.length() - 1;
    while (end > start && std::isspace(static_cast<unsigned char>(str[end]))) {
        --end;
    }
    return str.substr(start, end - start + 1);
}

inline std::vector<TransportEnergyGroup> parse_transport_energy_groups(
    const std::string& raw,
    const std::vector<TransportEnergyGroup>& fallback
) {
    if (raw.empty()) {
        return fallback;
    }

    std::vector<TransportEnergyGroup> groups;
    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim_config_token(token);
        if (token.empty()) {
            continue;
        }

        std::stringstream ts(token);
        std::string p0, p1, p2;
        if (!std::getline(ts, p0, ':') ||
            !std::getline(ts, p1, ':') ||
            !std::getline(ts, p2, ':')) {
            return fallback;
        }

        try {
            TransportEnergyGroup g;
            g.E_min_MeV = std::stof(trim_config_token(p0));
            g.E_max_MeV = std::stof(trim_config_token(p1));
            g.dE_MeV = std::stof(trim_config_token(p2));
            groups.push_back(g);
        } catch (...) {
            return fallback;
        }
    }

    return groups.empty() ? fallback : groups;
}

inline std::string serialize_transport_energy_groups(const std::vector<TransportEnergyGroup>& groups) {
    std::ostringstream oss;
    for (size_t i = 0; i < groups.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << groups[i].E_min_MeV << ":" << groups[i].E_max_MeV << ":" << groups[i].dE_MeV;
    }
    return oss.str();
}

/**
 * @brief Load IncidentParticleConfig from file
 */
inline IncidentParticleConfig load_incident_particle_config(const std::string& filename) {
    ConfigLoader loader;
    if (!loader.load(filename)) {
        throw std::runtime_error("Failed to load configuration file: " + filename);
    }

    IncidentParticleConfig config;

    // [particle] section
    ConfigSection particle_sec = loader.get_section("particle");
    if (particle_sec.has("type")) {
        config.particle_type = string_to_particle_type(particle_sec.get("type"));
    }
    config.particle_mass_amu = particle_sec.get_float("mass_amu", config.particle_mass_amu);
    config.particle_charge_e = particle_sec.get_float("charge_e", config.particle_charge_e);

    // [beam] section
    ConfigSection beam_sec = loader.get_section("beam");
    if (beam_sec.has("profile")) {
        config.beam_profile = string_to_beam_profile(beam_sec.get("profile"));
    }
    config.W_total = beam_sec.get_float("weight", config.W_total);

    // [energy] section
    ConfigSection energy_sec = loader.get_section("energy");
    config.energy.mean_E0 = energy_sec.get_float("mean_MeV", config.energy.mean_E0);
    config.energy.sigma_E = energy_sec.get_float("sigma_MeV", config.energy.sigma_E);
    config.energy.min_E = energy_sec.get_float("min_MeV", config.energy.min_E);
    config.energy.max_E = energy_sec.get_float("max_MeV", config.energy.max_E);

    // [spatial] section
    ConfigSection spatial_sec = loader.get_section("spatial");
    config.spatial.x0 = spatial_sec.get_float("x0_mm", config.spatial.x0);
    config.spatial.z0 = spatial_sec.get_float("z0_mm", config.spatial.z0);
    config.spatial.sigma_x = spatial_sec.get_float("sigma_x_mm", config.spatial.sigma_x);
    config.spatial.sigma_z = spatial_sec.get_float("sigma_z_mm", config.spatial.sigma_z);
    config.spatial.width_x = spatial_sec.get_float("width_x_mm", config.spatial.width_x);
    config.spatial.width_z = spatial_sec.get_float("width_z_mm", config.spatial.width_z);
    config.spatial.radius = spatial_sec.get_float("radius_mm", config.spatial.radius);

    // [angular] section
    ConfigSection angular_sec = loader.get_section("angular");
    config.angular.theta0 = angular_sec.get_float("theta0_rad", config.angular.theta0);
    config.angular.phi0 = angular_sec.get_float("phi0_rad", config.angular.phi0);
    config.angular.sigma_theta = angular_sec.get_float("sigma_theta_rad", config.angular.sigma_theta);
    config.angular.sigma_phi = angular_sec.get_float("sigma_phi_rad", config.angular.sigma_phi);

    // [sampling] section
    ConfigSection sampling_sec = loader.get_section("sampling");
    config.sampling.n_samples = sampling_sec.get_int("n_samples", config.sampling.n_samples);
    config.sampling.random_seed = static_cast<unsigned int>(
        sampling_sec.get_int("random_seed", config.sampling.random_seed));
    config.sampling.use_stratified = sampling_sec.get_bool("use_stratified", config.sampling.use_stratified);

    // [grid] section
    ConfigSection grid_sec = loader.get_section("grid");
    config.grid.Nx = grid_sec.get_int("Nx", config.grid.Nx);
    config.grid.Nz = grid_sec.get_int("Nz", config.grid.Nz);
    config.grid.dx = grid_sec.get_float("dx_mm", config.grid.dx);
    config.grid.dz = grid_sec.get_float("dz_mm", config.grid.dz);
    config.grid.max_steps = grid_sec.get_int("max_steps", config.grid.max_steps);

    // [transport] section
    ConfigSection transport_sec = loader.get_section("transport");
    config.transport.N_theta = transport_sec.get_int("N_theta", config.transport.N_theta);
    config.transport.N_theta_local = transport_sec.get_int("N_theta_local", config.transport.N_theta_local);
    config.transport.N_E_local = transport_sec.get_int("N_E_local", config.transport.N_E_local);
    const bool has_fine_on =
        transport_sec.has("E_fine_on_MeV") || transport_sec.has("E_fine_on");
    const bool has_fine_off =
        transport_sec.has("E_fine_off_MeV") || transport_sec.has("E_fine_off");
    const bool has_legacy_trigger =
        transport_sec.has("E_trigger_MeV") || transport_sec.has("E_trigger");

    if (has_fine_on) {
        config.transport.E_fine_on = transport_sec.get_float("E_fine_on_MeV", config.transport.E_fine_on);
        config.transport.E_fine_on = transport_sec.get_float("E_fine_on", config.transport.E_fine_on);
    }
    if (has_fine_off) {
        config.transport.E_fine_off = transport_sec.get_float("E_fine_off_MeV", config.transport.E_fine_off);
        config.transport.E_fine_off = transport_sec.get_float("E_fine_off", config.transport.E_fine_off);
    }
    if (has_legacy_trigger) {
        float legacy_trigger = config.transport.E_trigger;
        legacy_trigger = transport_sec.get_float("E_trigger_MeV", legacy_trigger);
        legacy_trigger = transport_sec.get_float("E_trigger", legacy_trigger);
        config.transport.E_trigger = legacy_trigger;
        if (!has_fine_on) {
            config.transport.E_fine_on = legacy_trigger;
        }
    }
    if (!has_legacy_trigger) {
        config.transport.E_trigger = config.transport.E_fine_on;
    }
    if (!has_fine_off) {
        config.transport.E_fine_off = std::max(config.transport.E_fine_off, config.transport.E_fine_on);
    }
    config.transport.weight_active_min = transport_sec.get_float("weight_active_min", config.transport.weight_active_min);
    config.transport.E_coarse_max = transport_sec.get_float("E_coarse_max_MeV", config.transport.E_coarse_max);
    config.transport.E_coarse_max = transport_sec.get_float("E_coarse_max", config.transport.E_coarse_max);
    config.transport.step_coarse = transport_sec.get_float("step_coarse_mm", config.transport.step_coarse);
    config.transport.step_coarse = transport_sec.get_float("step_coarse", config.transport.step_coarse);
    config.transport.n_steps_per_cell = transport_sec.get_int("n_steps_per_cell", config.transport.n_steps_per_cell);
    config.transport.fine_batch_max_cells = transport_sec.get_int("fine_batch_max_cells", config.transport.fine_batch_max_cells);
    config.transport.fine_halo_cells = transport_sec.get_int("fine_halo_cells", config.transport.fine_halo_cells);
    config.transport.preflight_vram_margin = transport_sec.get_float("preflight_vram_margin", config.transport.preflight_vram_margin);
    config.transport.validation_mode = transport_sec.get_bool("validation_mode", config.transport.validation_mode);
    config.transport.fail_fast_on_audit = transport_sec.get_bool("fail_fast_on_audit", config.transport.fail_fast_on_audit);
    config.transport.debug_dumps = transport_sec.get_bool("debug_dumps", config.transport.debug_dumps);
    config.transport.debug_dumps = transport_sec.get_bool("enable_debug_dumps", config.transport.debug_dumps);
    config.transport.max_iterations = transport_sec.get_int("max_iterations", config.transport.max_iterations);
    config.transport.log_level = transport_sec.get_int("log_level", config.transport.log_level);
    if (transport_sec.has("energy_groups")) {
        config.transport.energy_groups = parse_transport_energy_groups(
            transport_sec.get("energy_groups"),
            config.transport.energy_groups
        );
    }

    // [output] section
    ConfigSection output_sec = loader.get_section("output");
    config.output.output_dir = output_sec.get("output_dir", config.output.output_dir);
    config.output.dose_2d_file = output_sec.get("dose_2d_file", config.output.dose_2d_file);
    config.output.pdd_file = output_sec.get("pdd_file", config.output.pdd_file);
    config.output.let_file = output_sec.get("let_file", config.output.let_file);
    config.output.format = output_sec.get("format", config.output.format);
    config.output.normalize_dose = output_sec.get_bool("normalize_dose", config.output.normalize_dose);
    config.output.save_2d = output_sec.get_bool("save_2d", config.output.save_2d);
    config.output.save_pdd = output_sec.get_bool("save_pdd", config.output.save_pdd);
    config.output.save_lat_profiles = output_sec.get_bool("save_lat_profiles", config.output.save_lat_profiles);

    return config;
}

/**
 * @brief Save IncidentParticleConfig to file
 */
inline bool save_incident_particle_config(
    const std::string& filename,
    const IncidentParticleConfig& config
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << "# Incident Particle Configuration\n";
    file << "# Auto-generated by SM_2D\n\n";

    // [particle] section
    file << "[particle]\n";
    file << "type = " << particle_type_to_string(config.particle_type) << "\n";
    file << "mass_amu = " << config.particle_mass_amu << "\n";
    file << "charge_e = " << config.particle_charge_e << "\n\n";

    // [beam] section
    file << "[beam]\n";
    file << "profile = " << beam_profile_to_string(config.beam_profile) << "\n";
    file << "weight = " << config.W_total << "\n\n";

    // [energy] section
    file << "[energy]\n";
    file << "mean_MeV = " << config.energy.mean_E0 << "\n";
    file << "sigma_MeV = " << config.energy.sigma_E << "\n";
    file << "min_MeV = " << config.energy.min_E << "\n";
    file << "max_MeV = " << config.energy.max_E << "\n\n";

    // [spatial] section
    file << "[spatial]\n";
    file << "x0_mm = " << config.spatial.x0 << "\n";
    file << "z0_mm = " << config.spatial.z0 << "\n";
    file << "sigma_x_mm = " << config.spatial.sigma_x << "\n";
    file << "sigma_z_mm = " << config.spatial.sigma_z << "\n";
    file << "width_x_mm = " << config.spatial.width_x << "\n";
    file << "width_z_mm = " << config.spatial.width_z << "\n";
    file << "radius_mm = " << config.spatial.radius << "\n\n";

    // [angular] section
    file << "[angular]\n";
    file << "theta0_rad = " << config.angular.theta0 << "\n";
    file << "phi0_rad = " << config.angular.phi0 << "\n";
    file << "sigma_theta_rad = " << config.angular.sigma_theta << "\n";
    file << "sigma_phi_rad = " << config.angular.sigma_phi << "\n\n";

    // [sampling] section
    file << "[sampling]\n";
    file << "n_samples = " << config.sampling.n_samples << "\n";
    file << "random_seed = " << config.sampling.random_seed << "\n";
    file << "use_stratified = " << (config.sampling.use_stratified ? "true" : "false") << "\n\n";

    // [grid] section
    file << "[grid]\n";
    file << "Nx = " << config.grid.Nx << "\n";
    file << "Nz = " << config.grid.Nz << "\n";
    file << "dx_mm = " << config.grid.dx << "\n";
    file << "dz_mm = " << config.grid.dz << "\n";
    file << "max_steps = " << config.grid.max_steps << "\n\n";

    // [transport] section
    file << "[transport]\n";
    file << "N_theta = " << config.transport.N_theta << "\n";
    file << "N_theta_local = " << config.transport.N_theta_local << "\n";
    file << "N_E_local = " << config.transport.N_E_local << "\n";
    file << "E_fine_on_MeV = " << config.transport.E_fine_on << "\n";
    file << "E_fine_off_MeV = " << config.transport.E_fine_off << "\n";
    file << "E_trigger_MeV = " << config.transport.E_fine_on << "\n";
    file << "weight_active_min = " << config.transport.weight_active_min << "\n";
    file << "E_coarse_max_MeV = " << config.transport.E_coarse_max << "\n";
    file << "step_coarse_mm = " << config.transport.step_coarse << "\n";
    file << "n_steps_per_cell = " << config.transport.n_steps_per_cell << "\n";
    file << "fine_batch_max_cells = " << config.transport.fine_batch_max_cells << "\n";
    file << "fine_halo_cells = " << config.transport.fine_halo_cells << "\n";
    file << "preflight_vram_margin = " << config.transport.preflight_vram_margin << "\n";
    file << "validation_mode = " << (config.transport.validation_mode ? "true" : "false") << "\n";
    file << "fail_fast_on_audit = " << (config.transport.fail_fast_on_audit ? "true" : "false") << "\n";
    file << "debug_dumps = " << (config.transport.debug_dumps ? "true" : "false") << "\n";
    file << "max_iterations = " << config.transport.max_iterations << "\n";
    file << "log_level = " << config.transport.log_level << "\n";
    file << "energy_groups = " << serialize_transport_energy_groups(config.transport.energy_groups) << "\n\n";

    // [output] section
    file << "[output]\n";
    file << "output_dir = " << config.output.output_dir << "\n";
    file << "dose_2d_file = " << config.output.dose_2d_file << "\n";
    file << "pdd_file = " << config.output.pdd_file << "\n";
    file << "let_file = " << config.output.let_file << "\n";
    file << "format = " << config.output.format << "\n";
    file << "normalize_dose = " << (config.output.normalize_dose ? "true" : "false") << "\n";
    file << "save_2d = " << (config.output.save_2d ? "true" : "false") << "\n";
    file << "save_pdd = " << (config.output.save_pdd ? "true" : "false") << "\n";
    file << "save_lat_profiles = " << (config.output.save_lat_profiles ? "true" : "false") << "\n";

    return true;
}

/**
 * @brief Print configuration summary to stream
 */
inline void print_config_summary(std::ostream& os, const IncidentParticleConfig& config) {
    os << "=== Incident Particle Configuration ===\n";
    os << "Particle Type: " << particle_type_to_string(config.particle_type) << "\n";
    os << "Beam Profile: " << beam_profile_to_string(config.beam_profile) << "\n";
    os << "Energy: " << config.energy.mean_E0 << " MeV";
    if (config.energy.sigma_E > 0) {
        os << " (sigma = " << config.energy.sigma_E << " MeV)";
    }
    os << "\n";
    os << "Position: (" << config.spatial.x0 << ", " << config.spatial.z0 << ") mm\n";
    os << "Angle: " << config.angular.theta0 << " rad";
    if (config.angular.sigma_theta > 0) {
        os << " (sigma = " << config.angular.sigma_theta << " rad)";
    }
    os << "\n";
    os << "Weight: " << config.W_total << "\n";
    if (config.beam_profile == BeamProfileType::GAUSSIAN) {
        os << "Samples: " << config.sampling.n_samples << "\n";
    }
    os << "\n";
    os << "Grid: " << config.grid.Nx << " x " << config.grid.Nz;
    os << " (dx=" << config.grid.dx << "mm, dz=" << config.grid.dz << "mm)\n";
    int runtime_steps = (config.transport.max_iterations > 0) ? config.transport.max_iterations : config.grid.max_steps;
    os << "Transport: theta_bins=" << config.transport.N_theta
       << ", E_fine_on=" << config.transport.E_fine_on << " MeV"
       << ", E_fine_off=" << config.transport.E_fine_off << " MeV"
       << ", step_coarse=" << config.transport.step_coarse << " mm"
       << ", max_iterations=" << runtime_steps
       << ", validation_mode=" << (config.transport.validation_mode ? "on" : "off")
       << ", fail_fast_on_audit=" << (config.transport.fail_fast_on_audit ? "on" : "off")
       << ", debug_dumps=" << (config.transport.debug_dumps ? "on" : "off")
       << "\n";
    os << "Output Dir: " << config.output.output_dir << "\n";
    os << "======================================\n";
}

} // namespace sm_2d
