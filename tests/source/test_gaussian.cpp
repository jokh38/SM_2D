#include <gtest/gtest.h>
#include "source/gaussian_source.hpp"

TEST(GaussianSourceTest, DefaultValues) {
    GaussianSource src;
    EXPECT_FLOAT_EQ(src.x0, 0.0f);
    EXPECT_FLOAT_EQ(src.sigma_x, 5.0f);
    EXPECT_FLOAT_EQ(src.E0, 150.0f);
    EXPECT_FLOAT_EQ(src.W_total, 1.0f);
}

TEST(GaussianSourceTest, WeightSumConserved) {
    GaussianSource src;
    src.W_total = 1.0f;
    src.x0 = 5.0f;       // Center of 10-cell grid
    src.sigma_x = 1.0f;  // Narrow spread to minimize boundary loss
    src.theta0 = 0.05f;
    src.sigma_theta = 0.01f;
    src.E0 = 150.0f;
    src.sigma_E = 5.0f;
    src.n_samples = 10000;  // More samples for better statistics

    EnergyGrid e_grid(0.1f, 250.0f, 256);
    AngularGrid a_grid(-M_PI/2, M_PI/2, 512);

    PsiC psi(10, 10, 32);
    inject_source(psi, src, e_grid, a_grid);

    float total = 0;
    for (int cell = 0; cell < 100; ++cell) {
        total += sum_psi(psi, cell);
    }

    // For x0=5.0, sigma_x=1.0, in range [0,10], ~99.9999% of samples within bounds
    // Allow 2% tolerance for statistical variation
    EXPECT_NEAR(total, src.W_total, 0.02f);
}
