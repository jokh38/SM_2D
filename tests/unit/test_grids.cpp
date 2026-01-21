#include <gtest/gtest.h>
#include "core/grids.hpp"

TEST(EnergyGridTest, BinEdgesCorrect) {
    EnergyGrid grid(0.1f, 250.0f, 256);
    EXPECT_EQ(grid.edges.size(), 257);
    EXPECT_NEAR(grid.edges[0], 0.1f, 1e-6f);
    EXPECT_NEAR(grid.edges[256], 250.0f, 1e-4f);
}

TEST(EnergyGridTest, FindEnergyBin) {
    EnergyGrid grid(0.1f, 250.0f, 256);
    int bin = grid.FindBin(150.0f);
    EXPECT_GE(bin, 0);
    EXPECT_LT(bin, 256);
}

TEST(AngularGridTest, BinEdgesCorrect) {
    AngularGrid grid(-M_PI/2, M_PI/2, 512);
    EXPECT_EQ(grid.edges.size(), 513);
    EXPECT_NEAR(grid.edges[0], -M_PI/2, 1e-6f);
    EXPECT_NEAR(grid.edges[512], M_PI/2, 1e-6f);
}
