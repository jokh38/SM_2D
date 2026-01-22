#include "core/config_loader.hpp"
#include "validation/pencil_beam.hpp"
#include "validation/bragg_peak.hpp"
#include "source/source_adapter.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

using namespace sm_2d;

/**
 * @brief Create output directory if it doesn't exist
 */
bool create_output_directory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return (info.st_mode & S_IFDIR) != 0;
    }
    #if defined(_WIN32) || defined(_WIN64)
        return _mkdir(path.c_str()) == 0;
    #else
        return mkdir(path.c_str(), 0755) == 0;
    #endif
}

/**
 * @brief Convert IncidentParticleConfig to PencilBeamConfig
 */
PencilBeamConfig make_pencil_beam_config(const IncidentParticleConfig& config) {
    PencilBeamConfig pbc;
    pbc.E0 = config.energy.mean_E0;
    pbc.x0 = config.spatial.x0;
    pbc.z0 = config.spatial.z0;
    pbc.theta0 = config.angular.theta0;
    pbc.Nx = config.grid.Nx;
    pbc.Nz = config.grid.Nz;
    pbc.dx = config.grid.dx;
    pbc.dz = config.grid.dz;
    pbc.max_steps = config.grid.max_steps;
    pbc.W_total = config.W_total;
    pbc.random_seed = config.sampling.random_seed;
    return pbc;
}

/**
 * @brief Save 2D dose distribution to file
 */
bool save_dose_2d(const std::string& filepath, const SimulationResult& result, bool normalize) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return false;
    }

    double max_dose = 0.0;
    if (normalize) {
        for (int iz = 0; iz < result.Nz; ++iz) {
            for (int ix = 0; ix < result.Nx; ++ix) {
                max_dose = std::max(max_dose, result.edep[iz][ix]);
            }
        }
    }

    out << "# 2D Dose Distribution\n";
    out << "# x(mm) z(mm) dose(Gy)";
    if (normalize) {
        out << " dose_norm";
    }
    out << "\n";

    for (int iz = 0; iz < result.Nz; ++iz) {
        for (int ix = 0; ix < result.Nx; ++ix) {
            double x = result.x_centers[ix];
            double z = result.z_centers[iz];
            double dose = result.edep[iz][ix];
            out << std::fixed << std::setprecision(4)
                << x << "\t" << z << "\t" << dose;
            if (normalize && max_dose > 0) {
                out << "\t" << (dose / max_dose);
            }
            out << "\n";
        }
        out << "\n";
    }
    out.close();
    return true;
}

/**
 * @brief Save depth-dose (PDD) to file
 */
bool save_pdd(const std::string& filepath, const SimulationResult& result, bool normalize) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << std::endl;
        return false;
    }

    auto depth_dose = get_depth_dose(result);
    int peak_z = find_bragg_peak_z(result);
    double peak_dose = depth_dose[peak_z];

    out << "# Depth-Dose Distribution (PDD)\n";
    out << "# Depth(mm) Dose(Gy)";
    if (normalize) {
        out << " Dose_norm";
    }
    out << "\n";
    out << "# Bragg Peak at: " << (peak_z * result.dz) << " mm, " << peak_dose << " Gy\n";

    for (size_t i = 0; i < depth_dose.size(); ++i) {
        double depth = i * result.dz;
        double dose = depth_dose[i];
        out << std::fixed << std::setprecision(4)
            << depth << "\t" << dose;
        if (normalize && peak_dose > 0) {
            out << "\t" << (dose / peak_dose);
        }
        out << "\n";
    }
    out.close();
    return true;
}

int main(int argc, char* argv[]) {
    std::string config_file = "sim.ini";

    if (argc > 1) {
        config_file = argv[1];
    }

    std::cout << "========================================\n";
    std::cout << "SM_2D: Proton Therapy Simulation\n";
    std::cout << "========================================\n";
    std::cout << "Loading configuration from: " << config_file << std::endl;

    try {
        auto config = load_incident_particle_config(config_file);
        config.validate();
        print_config_summary(std::cout, config);

        if (!create_output_directory(config.output.output_dir)) {
            std::cerr << "Warning: Could not create output directory: "
                      << config.output.output_dir << std::endl;
        }

        std::cout << "\n--- Running Simulation ---" << std::endl;
        auto pbc = make_pencil_beam_config(config);
        auto result = run_pencil_beam(pbc);

        int peak_z = find_bragg_peak_z(result);
        double peak_depth = peak_z * result.dz;
        auto depth_dose = get_depth_dose(result);
        double peak_dose = depth_dose[peak_z];

        std::cout << "Simulation complete." << std::endl;
        std::cout << "  Bragg Peak: " << peak_depth << " mm depth, " << peak_dose << " Gy" << std::endl;

        std::string dose_2d_path = config.output.output_dir + "/" + config.output.dose_2d_file;
        std::string pdd_path = config.output.output_dir + "/" + config.output.pdd_file;

        std::cout << "\n--- Saving Results ---" << std::endl;

        if (config.output.save_2d) {
            if (save_dose_2d(dose_2d_path, result, config.output.normalize_dose)) {
                std::cout << "  2D dose saved to: " << dose_2d_path << std::endl;
            }
        }

        if (config.output.save_pdd) {
            if (save_pdd(pdd_path, result, config.output.normalize_dose)) {
                std::cout << "  PDD saved to: " << pdd_path << std::endl;
            }
        }

        std::cout << "\n=== Simulation Complete ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
