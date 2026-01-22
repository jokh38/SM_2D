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
    src.x0 = 5.0f;
    src.theta0 = 0.05f;

    EnergyGrid e_grid(0.1f, 250.0f, 256);
    AngularGrid a_grid(-M_PI/2, M_PI/2, 512);

    PsiC psi(10, 10, 32);
    inject_source(psi, src, e_grid, a_grid);

    float total = 0;
    for (int cell = 0; cell < 100; ++cell) {
        total += sum_psi(psi, cell);
    }

    EXPECT_NEAR(total, src.W_total, 1e-3f);
}
