#include <gtest/gtest.h>
#include "validation/lateral_spread.hpp"
#include "validation/pencil_beam.hpp"

TEST(LateralSpread, GetSigmaAtZ) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 100;
    config.Nz = 200;
    config.dx = 1.0f;
    config.dz = 1.0f;

    auto result = run_pencil_beam(config);

    // Test at various depths
    for (float z : {50.0f, 100.0f, 150.0f}) {
        float sigma = get_lateral_sigma_at_z(result, z);
        EXPECT_GE(sigma, 0.0f);
    }
}

TEST(LateralSpread, GetLateralProfile) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 100;
    config.Nz = 200;
    config.dx = 1.0f;
    config.dz = 1.0f;

    auto result = run_pencil_beam(config);
    auto profile = get_lateral_profile(result, 100.0f);

    EXPECT_EQ(profile.size(), 100u);

    // Profile should be non-negative
    for (double val : profile) {
        EXPECT_GE(val, 0.0);
    }
}

TEST(LateralSpread, FermiEygesSigma) {
    // Test Fermi-Eyges calculation for different energies
    float sigma_150 = compute_fermi_eyges_sigma(150.0f, 100.0f);
    float sigma_70 = compute_fermi_eyges_sigma(70.0f, 100.0f);

    // Higher energy should have smaller lateral spread
    EXPECT_LT(sigma_150, sigma_70);

    // Both should be positive
    EXPECT_GT(sigma_150, 0.0f);
    EXPECT_GT(sigma_70, 0.0f);
}

TEST(LateralSpread, FitGaussianSigma) {
    // Create a simple Gaussian profile
    std::vector<double> profile(100);
    float dx = 1.0f;
    float true_sigma = 5.0f;
    float center = 50.0f;

    for (size_t i = 0; i < profile.size(); ++i) {
        float x = i * dx;
        profile[i] = std::exp(-(x - center) * (x - center) / (2.0f * true_sigma * true_sigma));
    }

    float fitted_sigma = fit_gaussian_sigma(profile, dx);

    // Fitted sigma should be close to true sigma (within 20%)
    EXPECT_NEAR(fitted_sigma, true_sigma, true_sigma * 0.2f);
}
