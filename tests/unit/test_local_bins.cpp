#include <gtest/gtest.h>
#include "core/local_bins.hpp"

TEST(LocalBinsTest, LocalBinsConfigurationMatchesBuild) {
    // LOCAL_BINS = N_theta_local * N_E_local * N_x_sub * N_z_sub
    EXPECT_EQ(LOCAL_BINS, N_theta_local * N_E_local * N_x_sub * N_z_sub);
    EXPECT_EQ(LOCAL_BINS, 256);
    EXPECT_EQ(N_theta_local, 4);
    EXPECT_EQ(N_E_local, 2);
    EXPECT_EQ(N_x_sub, 8);
    EXPECT_EQ(N_z_sub, 4);
}

TEST(LocalBinsTest, EncodeDecodeRoundTrip) {
    for (int t = 0; t < N_theta_local; ++t) {
        for (int e = 0; e < N_E_local; ++e) {
            uint16_t encoded = encode_local_idx(t, e);
            int t_decoded, e_decoded;
            decode_local_idx(encoded, t_decoded, e_decoded);
            EXPECT_EQ(t_decoded, t);
            EXPECT_EQ(e_decoded, e);
        }
    }
}

TEST(LocalBinsTest, EncodeDecode3DRoundTrip) {
    for (int t = 0; t < N_theta_local; ++t) {
        for (int e = 0; e < N_E_local; ++e) {
            for (int x = 0; x < N_x_sub; ++x) {
                uint16_t encoded = encode_local_idx_3d(t, e, x);
                int t_decoded, e_decoded, x_decoded;
                decode_local_idx_3d(encoded, t_decoded, e_decoded, x_decoded);
                EXPECT_EQ(t_decoded, t);
                EXPECT_EQ(e_decoded, e);
                EXPECT_EQ(x_decoded, x);
            }
        }
    }
}

TEST(LocalBinsTest, XOffsetToBinRoundTrip) {
    float dx = 1.0f;  // 1 mm cell size

    // Test each sub-cell bin
    for (int x = 0; x < N_x_sub; ++x) {
        float offset = get_x_offset_from_bin(x, dx);
        int bin = get_x_sub_bin(offset, dx);
        EXPECT_EQ(bin, x);
    }

    // Test boundary values
    EXPECT_EQ(get_x_sub_bin(-0.5f * dx, dx), 0);  // Left edge
    EXPECT_EQ(get_x_sub_bin(0.0f, dx), N_x_sub / 2);        // Center
    EXPECT_EQ(get_x_sub_bin(0.49f * dx, dx), N_x_sub - 1);  // Near right edge
}

TEST(LocalBinsTest, ZOffsetToBinRoundTrip) {
    float dz = 1.0f;  // 1 mm cell size

    // FIX Problem 1: Test z_sub bins
    for (int z = 0; z < N_z_sub; ++z) {
        float offset = get_z_offset_from_bin(z, dz);
        int bin = get_z_sub_bin(offset, dz);
        EXPECT_EQ(bin, z);
    }

    // Test boundary values
    EXPECT_EQ(get_z_sub_bin(-0.5f * dz, dz), 0);  // Bottom edge
    EXPECT_EQ(get_z_sub_bin(0.0f, dz), N_z_sub / 2);        // Center
    EXPECT_EQ(get_z_sub_bin(0.49f * dz, dz), N_z_sub - 1);  // Near top edge
}

TEST(LocalBinsTest, EncodeDecode4DRoundTrip) {
    // FIX Problem 1: Test 4D encoding
    for (int t = 0; t < N_theta_local; ++t) {
        for (int e = 0; e < N_E_local; ++e) {
            for (int x = 0; x < N_x_sub; ++x) {
                for (int z = 0; z < N_z_sub; ++z) {
                    uint16_t encoded = encode_local_idx_4d(t, e, x, z);
                    int t_decoded, e_decoded, x_decoded, z_decoded;
                    decode_local_idx_4d(encoded, t_decoded, e_decoded, x_decoded, z_decoded);
                    EXPECT_EQ(t_decoded, t);
                    EXPECT_EQ(e_decoded, e);
                    EXPECT_EQ(x_decoded, x);
                    EXPECT_EQ(z_decoded, z);
                }
            }
        }
    }
}

TEST(LocalBinsTest, LocalBinsCapacity) {
    // Ensure LOCAL_BINS fits in uint16_t
    EXPECT_LE(LOCAL_BINS, 65536);
}
