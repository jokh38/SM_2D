#include <gtest/gtest.h>
#include "audit/reporting.hpp"
#include <sstream>

TEST(AuditReport, PrintWeightReport) {
    CellWeightAudit audit;
    audit.W_in = 1.0f;
    audit.W_out = 0.7f;
    audit.W_cutoff = 0.2f;
    audit.W_nuclear = 0.1f;

    std::ostringstream oss;
    print_cell_weight_report(oss, audit, 42);

    std::string report = oss.str();
    EXPECT_NE(report.find("Cell 42"), std::string::npos);
    EXPECT_NE(report.find("PASS"), std::string::npos);
}
