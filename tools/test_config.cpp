#include "core/incident_particle_config.hpp"
#include "core/config_loader.hpp"
#include "source/source_adapter.hpp"
#include <iostream>
#include <iomanip>

using namespace sm_2d;

int main(int argc, char* argv[]) {
    std::string config_file = "config/proton_70MeV_pencil.ini";

    // Allow command-line override
    if (argc > 1) {
        config_file = argv[1];
    }

    std::cout << "Loading configuration from: " << config_file << std::endl;

    try {
        // Load configuration from file
        auto config = load_incident_particle_config(config_file);

        // Validate configuration
        config.validate();

        // Print configuration summary
        print_config_summary(std::cout, config);

        // Create legacy source structures
        if (config.beam_profile == BeamProfileType::PENCIL) {
            auto pencil = make_pencil_source_legacy(config);
            std::cout << "\n--- Legacy PencilSource ---" << std::endl;
            std::cout << "x0 = " << pencil.x0 << " mm" << std::endl;
            std::cout << "z0 = " << pencil.z0 << " mm" << std::endl;
            std::cout << "theta0 = " << pencil.theta0 << " rad" << std::endl;
            std::cout << "E0 = " << pencil.E0 << " MeV" << std::endl;
            std::cout << "W_total = " << pencil.W_total << std::endl;
        } else if (config.beam_profile == BeamProfileType::GAUSSIAN) {
            auto gaussian = make_gaussian_source_legacy(config);
            std::cout << "\n--- Legacy GaussianSource ---" << std::endl;
            std::cout << "x0 = " << gaussian.x0 << " mm" << std::endl;
            std::cout << "theta0 = " << gaussian.theta0 << " rad" << std::endl;
            std::cout << "sigma_x = " << gaussian.sigma_x << " mm" << std::endl;
            std::cout << "sigma_theta = " << gaussian.sigma_theta << " rad" << std::endl;
            std::cout << "E0 = " << gaussian.E0 << " MeV" << std::endl;
            std::cout << "sigma_E = " << gaussian.sigma_E << " MeV" << std::endl;
            std::cout << "n_samples = " << gaussian.n_samples << std::endl;
            std::cout << "W_total = " << gaussian.W_total << std::endl;
        }

        // Test builder pattern
        std::cout << "\n--- Builder Pattern Example ---" << std::endl;
        auto custom_config = IncidentParticleBuilder()
            .particle(ParticleType::PROTON)
            .beam_profile(BeamProfileType::GAUSSIAN)
            .energy(120.0f, 0.5f)
            .position(25.0f, 0.0f)
            .angle(0.0f)
            .spatial_spread(3.0f)
            .angular_spread(0.005f)
            .samples(2000)
            .weight(1.0f)
            .seed(123)
            .build();

        print_config_summary(std::cout, custom_config);

        // Test presets
        std::cout << "\n--- Preset Examples ---" << std::endl;
        std::cout << "70 MeV Pencil:" << std::endl;
        auto p70 = presets::proton_70MeV_pencil();
        std::cout << "  Energy: " << p70.energy.mean_E0 << " MeV" << std::endl;
        std::cout << "  Profile: " << beam_profile_to_string(p70.beam_profile) << std::endl;

        std::cout << "\n150 MeV Gaussian:" << std::endl;
        auto p150 = presets::proton_150MeV_gaussian();
        std::cout << "  Energy: " << p150.energy.mean_E0 << " MeV" << std::endl;
        std::cout << "  Sigma_x: " << p150.spatial.sigma_x << " mm" << std::endl;
        std::cout << "  Samples: " << p150.sampling.n_samples << std::endl;

        // Save configuration
        std::string output_file = "config/output_test.ini";
        if (save_incident_particle_config(output_file, custom_config)) {
            std::cout << "\nConfiguration saved to: " << output_file << std::endl;
        }

        std::cout << "\n=== All tests passed! ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
