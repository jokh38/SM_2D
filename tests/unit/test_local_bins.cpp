#include <gtest/gtest.h>
#include "core/local_bins.hpp"

TEST(LocalBinsTest, LocalBinsEquals32) {
    EXPECT_EQ(LOCAL_BINS, 32);
    EXPECT_EQ(N_theta_local, 8);
    EXPECT_EQ(N_E_local, 4);
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
