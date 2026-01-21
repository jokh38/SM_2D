#include <gtest/gtest.h>
#include "boundary/boundaries.hpp"

TEST(BoundaryTest, AbsorbAtZmax) {
    BoundaryConfig config;
    config.z_max = BoundaryType::ABSORB;

    int cell = 31;
    int face = 0;

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
