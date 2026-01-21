#include <gtest/gtest.h>
#include "validation/pencil_beam.hpp"

TEST(PencilBeam, RunSimulation) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 50;
    config.Nz = 100;
    config.dx = 2.0f;
    config.dz = 2.0f;
    config.W_total = 1.0f;

    auto result = run_pencil_beam(config);

    EXPECT_EQ(result.Nx, 50);
    EXPECT_EQ(result.Nz, 100);
    EXPECT_FLOAT_EQ(result.dx, 2.0f);
    EXPECT_FLOAT_EQ(result.dz, 2.0f);
    EXPECT_EQ(result.edep.size(), 100u);
    EXPECT_EQ(result.edep[0].size(), 50u);
}

TEST(PencilBeam, FindBraggPeak) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 50;
    config.Nz = 100;
    config.dx = 2.0f;
    config.dz = 2.0f;

    auto result = run_pencil_beam(config);
    int peak_z = find_bragg_peak_z(result);

    // Bragg peak should be somewhere in the middle of the grid
    EXPECT_GT(peak_z, 0);
    EXPECT_LT(peak_z, result.Nz);
}

TEST(PencilBeam, GetDepthDose) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 50;
    config.Nz = 100;
    config.dx = 2.0f;
    config.dz = 2.0f;

    auto result = run_pencil_beam(config);
    auto depth_dose = get_depth_dose(result);

    EXPECT_EQ(depth_dose.size(), 100u);

    // Depth dose should be non-negative
    for (double dose : depth_dose) {
        EXPECT_GE(dose, 0.0);
    }
}
