#include <gtest/gtest.h>
#include "core/buckets.hpp"

TEST(BucketTest, InitializeEmpty) {
    OutflowBucket bucket;
    for (int i = 0; i < Kb_out; ++i) {
        EXPECT_EQ(bucket.block_id[i], EMPTY_BLOCK_ID);
    }
}

TEST(BucketTest, EmitToBucket) {
    OutflowBucket bucket;
    uint32_t bid = encode_block(10, 20);
    int slot = bucket.find_or_allocate_slot(bid);
    ASSERT_GE(slot, 0);

    atomic_add(bucket.value[slot][0], 1.0f);
    EXPECT_FLOAT_EQ(bucket.value[slot][0], 1.0f);
}
