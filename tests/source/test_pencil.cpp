#include <gtest/gtest.h>
#include "source/pencil_source.hpp"

TEST(PencilSourceTest, DefaultValues) {
    PencilSource src;
    EXPECT_FLOAT_EQ(src.x0, 0.0f);
    EXPECT_FLOAT_EQ(src.z0, 0.0f);
    EXPECT_FLOAT_EQ(src.E0, 150.0f);
    EXPECT_FLOAT_EQ(src.W_total, 1.0f);
}

TEST(PencilSourceTest, InjectionCorrectCell) {
    PencilSource src;
    src.x0 = 0.5f;
    src.z0 = 0.5f;

    EnergyGrid e_grid(0.1f, 250.0f, 256);
    AngularGrid a_grid(-M_PI/2, M_PI/2, 512);

    PsiC psi(4, 4, 32);
    inject_source(psi, src, e_grid, a_grid);

    EXPECT_GT(sum_psi(psi, 0), 0);
}
