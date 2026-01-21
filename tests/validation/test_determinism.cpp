#include <gtest/gtest.h>
#include "validation/determinism.hpp"
#include "validation/pencil_beam.hpp"

TEST(Determinism, ComputeChecksum) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 50;
    config.Nz = 100;
    config.dx = 2.0f;
    config.dz = 2.0f;

    auto result = run_pencil_beam(config);
    uint32_t checksum = compute_checksum(result);

    // Checksum should be non-zero for a valid result
    EXPECT_NE(checksum, 0u);
}

TEST(Determinism, SameConfigSameChecksum) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 50;
    config.Nz = 100;
    config.dx = 2.0f;
    config.dz = 2.0f;

    auto result1 = run_pencil_beam(config);
    auto result2 = run_pencil_beam(config);

    uint32_t checksum1 = compute_checksum(result1);
    uint32_t checksum2 = compute_checksum(result2);

    EXPECT_EQ(checksum1, checksum2);
}

TEST(Determinism, VerifyDeterminism) {
    PencilBeamConfig config;
    config.E0 = 150.0f;
    config.Nx = 50;
    config.Nz = 100;
    config.dx = 2.0f;
    config.dz = 2.0f;

    bool is_deterministic = verify_determinism(config);

    // The stub implementation should be deterministic
    EXPECT_TRUE(is_deterministic);
}
