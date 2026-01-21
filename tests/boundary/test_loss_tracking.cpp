#include <gtest/gtest.h>
#include "boundary/loss_tracking.hpp"

TEST(BoundaryLossTest, InitializeZero) {
    BoundaryLoss loss;
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(loss.weight[i], 0.0f);
        EXPECT_DOUBLE_EQ(loss.energy[i], 0.0);
    }
}

TEST(BoundaryLossTest, RecordLoss) {
    BoundaryLoss loss;
    record_boundary_loss(loss, 0, 1.0f, 100.0f);

    EXPECT_FLOAT_EQ(loss.weight[0], 1.0f);
    EXPECT_DOUBLE_EQ(loss.energy[0], 100.0);
}

TEST(BoundaryLossTest, TotalLoss) {
    BoundaryLoss loss;
    record_boundary_loss(loss, 0, 0.5f, 50.0f);
    record_boundary_loss(loss, 1, 0.3f, 30.0f);
    record_boundary_loss(loss, 2, 0.1f, 10.0f);
    record_boundary_loss(loss, 3, 0.1f, 10.0f);

    float total_w = total_boundary_weight_loss(loss);
    EXPECT_NEAR(total_w, 1.0f, 1e-6f);
}
