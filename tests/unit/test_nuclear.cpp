#include <gtest/gtest.h>
#include "physics/nuclear.hpp"

TEST(NuclearTest, CrossSectionPositive) {
    float sigma = Sigma_total(150.0f);
    EXPECT_GT(sigma, 0.0f);
}

TEST(NuclearTest, AttenuationReducesWeight) {
    float w_old = 1.0f;
    float w_removed, E_removed;
    float w_new = apply_nuclear_attenuation(w_old, 150.0f, 1.0f, w_removed, E_removed);
    EXPECT_LT(w_new, w_old);
    EXPECT_GT(w_removed, 0.0f);
}

TEST(NuclearTest, EnergyTracked) {
    float w_old = 1.0f;
    float E = 150.0f;
    float w_removed, E_removed;
    apply_nuclear_attenuation(w_old, E, 1.0f, w_removed, E_removed);
    EXPECT_NEAR(E_removed, w_removed * E, 1e-5f);
}
