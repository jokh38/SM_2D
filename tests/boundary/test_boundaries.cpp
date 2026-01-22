#include <gtest/gtest.h>
#include "boundary/boundaries.hpp"

TEST(BoundaryTest, AbsorbAtZmax) {
    BoundaryConfig config;
    config.z_max = BoundaryType::ABSORB;

    // For 8x8 grid, z_max boundary has iz = 7
    // Use ix = 3, iz = 7 → cell = 3 + 7 * 8 = 59
    int cell = 59;
    int face = 0;  // +z direction

    int neighbor = get_neighbor(cell, face, config, 8, 8);
    EXPECT_EQ(neighbor, -1);
}

TEST(BoundaryTest, NormalNeighbor) {
    BoundaryConfig config;
    int cell = 10;
    int face = 0;

    int neighbor = get_neighbor(cell, face, config, 8, 8);
    EXPECT_EQ(neighbor, 18);
}
