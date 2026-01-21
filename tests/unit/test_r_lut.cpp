#include <gtest/gtest.h>
#include "lut/r_lut.hpp"

TEST(RLUTTest, R_150MeV_WithinTolerance) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);
    float R = lut.lookup_R(150.0f);
    // NIST: ~158 mm for 150 MeV in water
    float R_expected = 158.0f;
    float error = fabsf(R - R_expected) / R_expected;
    EXPECT_LT(error, 0.15f);  // ±15%
}

TEST(RLUTTest, R_70MeV_WithinTolerance) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);
    float R = lut.lookup_R(70.0f);
    // NIST: ~40.8 mm for 70 MeV in water
    float R_expected = 40.8f;
    float error = fabsf(R - R_expected) / R_expected;
    EXPECT_LT(error, 0.15f);  // ±15%
}

TEST(RLUTTest, InverseConsistency) {
    RLUT lut = GenerateRLUT(0.1f, 250.0f, 256);
    std::vector<float> test_energies = {10.0f, 50.0f, 100.0f, 150.0f};

    for (float E_test : test_energies) {
        float R = lut.lookup_R(E_test);
        float E_recovered = lut.lookup_E_inverse(R);
        float error = fabsf(E_recovered - E_test) / E_test;
        EXPECT_LT(error, 0.01f) << "Failed at E=" << E_test;
    }
}
