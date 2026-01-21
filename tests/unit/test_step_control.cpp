#include <gtest/gtest.h>
#include "physics/step_control.hpp"
#include "lut/r_lut.hpp"

TEST(StepControlTest, StepSizePositive) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);
    float step = compute_max_step_physics(lut, 150.0f);
    EXPECT_GT(step, 0.0f);
}

TEST(StepControlTest, StepSizeSmallerNearBragg) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);
    float step_150 = compute_max_step_physics(lut, 150.0f);
    float step_10 = compute_max_step_physics(lut, 10.0f);
    EXPECT_LT(step_10, step_150);
}

TEST(StepControlTest, EnergyDecreases) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);
    float E_initial = 150.0f;
    float step = 1.0f;
    float E_final = compute_energy_after_step(lut, E_initial, step);
    EXPECT_LT(E_final, E_initial);
}
