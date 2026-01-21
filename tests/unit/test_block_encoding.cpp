#include <gtest/gtest.h>
#include "core/block_encoding.hpp"

TEST(BlockEncodingTest, EncodeDecodeRoundTrip) {
    for (uint32_t b_theta = 0; b_theta < 100; ++b_theta) {
        for (uint32_t b_E = 0; b_E < 100; ++b_E) {
            uint32_t block_id = encode_block(b_theta, b_E);
            uint32_t theta_decoded, E_decoded;
            decode_block(block_id, theta_decoded, E_decoded);
            EXPECT_EQ(theta_decoded, b_theta);
            EXPECT_EQ(E_decoded, b_E);
        }
    }
}
