#include <gtest/gtest.h>
#include "physics/highland.hpp"

TEST(HighlandTest, SigmaPositive) {
    float sigma = highland_sigma(150.0f, 1.0f);
    EXPECT_GT(sigma, 0.0f);
}

TEST(HighlandTest, SigmaIncreasesWithStep) {
    float sigma_1mm = highland_sigma(150.0f, 1.0f);
    float sigma_5mm = highland_sigma(150.0f, 5.0f);
    EXPECT_GT(sigma_5mm, sigma_1mm);
}

TEST(HighlandTest, SigmaDecreasesWithEnergy) {
    float sigma_10MeV = highland_sigma(10.0f, 1.0f);
    float sigma_150MeV = highland_sigma(150.0f, 1.0f);
    EXPECT_GT(sigma_10MeV, sigma_150MeV);
}
