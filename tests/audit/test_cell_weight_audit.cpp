#include <gtest/gtest.h>
#include "audit/conservation.hpp"

TEST(CellWeightAudit, SingleComponentTransport) {
    CellWeightAudit audit;
    audit.W_in = 1.0f;
    audit.W_out = 0.7f;
    audit.W_cutoff = 0.2f;
    audit.W_nuclear = 0.1f;

    bool pass = check_weight_conservation(audit);
    EXPECT_TRUE(pass);
    EXPECT_LT(audit.W_error, 1e-6f);
}

TEST(CellWeightAudit, WeightConservationFail) {
    CellWeightAudit audit;
    audit.W_in = 1.0f;
    audit.W_out = 0.8f;  // Error
    audit.W_cutoff = 0.2f;
    audit.W_nuclear = 0.1f;

    bool pass = check_weight_conservation(audit);
    EXPECT_FALSE(pass);
    EXPECT_GT(audit.W_error, 1e-4f);
}
