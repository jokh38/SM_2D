#include "core/config_loader.hpp"
#include "gpu/gpu_transport_runner.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
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
            // FIX: Replace NaN with 0 for invalid dose values
            if (std::isnan(dose) || std::isinf(dose)) {
                dose = 0.0;
            }
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
        // FIX: Replace NaN with 0 for invalid dose values
        if (std::isnan(dose) || std::isinf(dose)) {
            dose = 0.0;
        }
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

        // Resolve output directory relative to config file location
        // If config_file is a path, get its directory
        std::string output_dir = config.output.output_dir;
        if (!output_dir.empty() && output_dir[0] != '/') {
            // Relative path - resolve from config file directory
            size_t last_slash = config_file.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                std::string config_dir = config_file.substr(0, last_slash);
                output_dir = config_dir + "/" + output_dir;
            }
        }

        if (!create_output_directory(output_dir)) {
            std::cerr << "Warning: Could not create output directory: "
                      << output_dir << std::endl;
        }

        std::cout << "\n--- Running Simulation ---" << std::endl;

        SimulationResult result;

        std::cout << "Using GPU transport (Vavilov energy straggling)" << std::endl;
        result = GPUTransportRunner::run(config);

        auto depth_dose = get_depth_dose(result);
        int peak_z = find_bragg_peak_z(result);
        double peak_depth = peak_z * result.dz;
        double peak_dose = depth_dose[peak_z];

        std::cout << "Simulation complete." << std::endl;
        std::cout << "  Bragg Peak: " << peak_depth << " mm depth, " << peak_dose << " Gy" << std::endl;

        std::string dose_2d_path = output_dir + "/" + config.output.dose_2d_file;
        std::string pdd_path = output_dir + "/" + config.output.pdd_file;

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
