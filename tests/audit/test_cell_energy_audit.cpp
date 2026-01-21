#include <gtest/gtest.h>
#include "audit/conservation.hpp"

TEST(CellEnergyAudit, SingleComponentTransport) {
    CellEnergyAudit audit;
    audit.E_in = 100.0;
    audit.E_out = 70.0;
    audit.E_dep = 25.0;
    audit.E_nuclear = 5.0;

    bool pass = check_energy_conservation(audit);
    EXPECT_TRUE(pass);
    EXPECT_LT(audit.E_error, 1e-5);
}

TEST(CellEnergyAudit, EnergyDriftDetection) {
    CellEnergyAudit audit;
    audit.E_in = 100.0;
    audit.E_out = 75.0;  // Too high
    audit.E_dep = 20.0;
    audit.E_nuclear = 4.0;

    bool pass = check_energy_conservation(audit);
    EXPECT_FALSE(pass);
    EXPECT_GT(audit.E_error, 1e-5);
}
