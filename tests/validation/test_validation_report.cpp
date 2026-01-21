#include <gtest/gtest.h>
#include "validation/validation_report.hpp"
#include <sstream>

TEST(ValidationReport, GenerateReport) {
    ValidationResults results;
    results.bragg_150 = {150.0f, 158.0f, 158.0f, 0.0f, true};
    results.bragg_70 = {70.0f, 40.8f, 40.8f, 0.0f, true};
    results.lateral = {100.0f, 5.0f, 5.0f, 0.0f, true};
    results.conservation = {1e-6f, 1e-6f, true};
    results.overall_pass = true;

    std::ostringstream oss;
    generate_validation_report(oss, results);

    std::string report = oss.str();

    // Check that report contains key sections
    EXPECT_NE(report.find("BRAGG PEAK VALIDATION"), std::string::npos);
    EXPECT_NE(report.find("LATERAL SPREAD VALIDATION"), std::string::npos);
    EXPECT_NE(report.find("CONSERVATION VALIDATION"), std::string::npos);
    EXPECT_NE(report.find("OVERALL RESULT"), std::string::npos);
}

TEST(ValidationReport, RunFullValidation) {
    ValidationResults results = run_full_validation();

    // Check that all fields are populated
    EXPECT_GT(results.bragg_150.energy_MeV, 0.0f);
    EXPECT_GT(results.bragg_70.energy_MeV, 0.0f);
    EXPECT_GT(results.lateral.z_mm, 0.0f);
}
