#include <gtest/gtest.h>
#include "validation/bragg_peak.hpp"
#include "validation/pencil_beam.hpp"

TEST(BraggPeak, FindPosition) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 100;
    config.Nz = 200;
    config.dx = 1.0f;
    config.dz = 1.0f;

    auto result = run_pencil_beam(config);
    float z_peak = find_bragg_peak_position_mm(result);

    // Peak should be positive and within grid
    EXPECT_GT(z_peak, 0.0f);
    EXPECT_LT(z_peak, 200.0f);
}

TEST(BraggPeak, ComputeFWHM) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 100;
    config.Nz = 200;
    config.dx = 1.0f;
    config.dz = 1.0f;

    auto result = run_pencil_beam(config);
    float fwhm = compute_bragg_peak_fwhm(result);

    // FWHM should be positive
    EXPECT_GT(fwhm, 0.0f);
}

TEST(BraggPeak, FindR80) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 100;
    config.Nz = 200;
    config.dx = 1.0f;
    config.dz = 1.0f;

    auto result = run_pencil_beam(config);
    float r80 = find_R80(result);

    // R80 should be positive and less than grid extent
    EXPECT_GT(r80, 0.0f);
    EXPECT_LT(r80, 200.0f);
}

TEST(BraggPeak, FindR20) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 100;
    config.Nz = 200;
    config.dx = 1.0f;
    config.dz = 1.0f;

    auto result = run_pencil_beam(config);
    float r20 = find_R20(result);

    // R20 should be greater than R80
    float r80 = find_R80(result);
    EXPECT_GT(r20, r80);
}

TEST(BraggPeak, DistalFalloff) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 100;
    config.Nz = 200;
    config.dx = 1.0f;
    config.dz = 1.0f;

    auto result = run_pencil_beam(config);
    float falloff = compute_distal_falloff(result);

    // Distal falloff should be positive
    EXPECT_GT(falloff, 0.0f);
}
