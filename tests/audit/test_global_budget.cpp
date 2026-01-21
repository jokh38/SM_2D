#include <gtest/gtest.h>
#include "audit/global_budget.hpp"

TEST(GlobalBudget, WeightCloses) {
    GlobalAudit audit;
    audit.W_total_in = 1.0f;
    audit.W_total_out = 0.6f;
    audit.W_total_cutoff = 0.2f;
    audit.W_total_nuclear = 0.1f;
    audit.W_boundary = 0.1f;

    bool pass = check_global_weight_conservation(audit);
    EXPECT_TRUE(pass);
}

TEST(GlobalBudget, EnergyCloses) {
    GlobalAudit audit;
    audit.E_total_in = 100.0;
    audit.E_total_dep = 60.0;
    audit.E_total_nuclear = 10.0;
    audit.E_boundary = 30.0;

    bool pass = check_global_energy_conservation(audit);
    EXPECT_TRUE(pass);
}
