#include <gtest/gtest.h>
#include "kernels/k3_finetransport.cuh"

TEST(K3Test, EnergyCutoff) {
    Component c = {0.0f, 0.05f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    K3Result result = run_K3_single_component(c);
    EXPECT_NEAR(result.Edep, 0.05f, 1e-4f);
    EXPECT_TRUE(result.terminated);
}

TEST(K3Test, SingleStepTransport) {
    // Component that travels one step in cell
    Component c = {0.0f, 100.0f, 1.0f, 0.5f, 0.5f, 1.0f, 0.0f};

    K3Result result = run_K3_single_component(c);

    // Some energy should be deposited
    EXPECT_GT(result.Edep, 0);
    EXPECT_LT(result.Edep, 100.0f);

    // Weight should decrease due to nuclear attenuation
    EXPECT_GT(result.nuclear_weight_removed, 0);
}

TEST(K3Test, NuclearAttenuation) {
    Component c = {0.0f, 100.0f, 1.0f, 0.5f, 0.5f, 1.0f, 0.0f};

    K3Result result = run_K3_single_component(c);

    // Nuclear weight removed
    EXPECT_GT(result.nuclear_weight_removed, 0);

    // Nuclear energy tracked
    EXPECT_NEAR(result.nuclear_energy_removed,
                result.nuclear_weight_removed * c.E, 1e-5f);
}

TEST(K3Test, AngularSplit) {
    // Force variance accumulation above threshold
    Component c = {0.0f, 100.0f, 1.0f, 0.5f, 0.5f, 1.0f, 0.0f};

    // Simulate with high variance
    K3Result result = run_K3_with_forced_split(c);

    EXPECT_EQ(result.split_count, 7);
}
