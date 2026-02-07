#include <gtest/gtest.h>

#include "k1k6_pipeline.cuh"
#include "device/device_psic.cuh"
#include "core/block_encoding.hpp"

#include <cuda_runtime.h>
#include <vector>

namespace {
void check_cuda(cudaError_t err, const char* what) {
    ASSERT_EQ(err, cudaSuccess) << what << ": " << cudaGetErrorString(err);
}
}  // namespace

class K5DropChannelAuditTest : public ::testing::Test {
protected:
    DevicePsiC psi_in{};
    DevicePsiC psi_out{};

    double* d_EdepC = nullptr;
    double* d_AbsorbedEnergy_cutoff = nullptr;
    double* d_AbsorbedEnergy_nuclear = nullptr;
    double* d_BoundaryLoss_energy = nullptr;
    double* d_prev_EdepC = nullptr;
    double* d_prev_AbsorbedEnergy_cutoff = nullptr;
    double* d_prev_AbsorbedEnergy_nuclear = nullptr;
    double* d_prev_BoundaryLoss_energy = nullptr;

    float* d_AbsorbedWeight_cutoff = nullptr;
    float* d_AbsorbedWeight_nuclear = nullptr;
    float* d_BoundaryLoss_weight = nullptr;
    float* d_prev_AbsorbedWeight_cutoff = nullptr;
    float* d_prev_AbsorbedWeight_nuclear = nullptr;
    float* d_prev_BoundaryLoss_weight = nullptr;

    float* d_E_edges = nullptr;
    AuditReport* d_report = nullptr;

    void SetUp() override {
        ASSERT_TRUE(device_psic_init(psi_in, 1, 1));
        ASSERT_TRUE(device_psic_init(psi_out, 1, 1));
        device_psic_clear(psi_in);
        device_psic_clear(psi_out);

        check_cuda(cudaMalloc(&d_EdepC, sizeof(double)), "cudaMalloc d_EdepC");
        check_cuda(cudaMalloc(&d_AbsorbedEnergy_cutoff, sizeof(double)), "cudaMalloc d_AbsorbedEnergy_cutoff");
        check_cuda(cudaMalloc(&d_AbsorbedEnergy_nuclear, sizeof(double)), "cudaMalloc d_AbsorbedEnergy_nuclear");
        check_cuda(cudaMalloc(&d_BoundaryLoss_energy, sizeof(double)), "cudaMalloc d_BoundaryLoss_energy");
        check_cuda(cudaMalloc(&d_prev_EdepC, sizeof(double)), "cudaMalloc d_prev_EdepC");
        check_cuda(cudaMalloc(&d_prev_AbsorbedEnergy_cutoff, sizeof(double)), "cudaMalloc d_prev_AbsorbedEnergy_cutoff");
        check_cuda(cudaMalloc(&d_prev_AbsorbedEnergy_nuclear, sizeof(double)), "cudaMalloc d_prev_AbsorbedEnergy_nuclear");
        check_cuda(cudaMalloc(&d_prev_BoundaryLoss_energy, sizeof(double)), "cudaMalloc d_prev_BoundaryLoss_energy");

        check_cuda(cudaMalloc(&d_AbsorbedWeight_cutoff, sizeof(float)), "cudaMalloc d_AbsorbedWeight_cutoff");
        check_cuda(cudaMalloc(&d_AbsorbedWeight_nuclear, sizeof(float)), "cudaMalloc d_AbsorbedWeight_nuclear");
        check_cuda(cudaMalloc(&d_BoundaryLoss_weight, sizeof(float)), "cudaMalloc d_BoundaryLoss_weight");
        check_cuda(cudaMalloc(&d_prev_AbsorbedWeight_cutoff, sizeof(float)), "cudaMalloc d_prev_AbsorbedWeight_cutoff");
        check_cuda(cudaMalloc(&d_prev_AbsorbedWeight_nuclear, sizeof(float)), "cudaMalloc d_prev_AbsorbedWeight_nuclear");
        check_cuda(cudaMalloc(&d_prev_BoundaryLoss_weight, sizeof(float)), "cudaMalloc d_prev_BoundaryLoss_weight");

        check_cuda(cudaMalloc(&d_E_edges, 2 * sizeof(float)), "cudaMalloc d_E_edges");
        check_cuda(cudaMalloc(&d_report, sizeof(AuditReport)), "cudaMalloc d_report");

        // Single-bin energy grid centered at 100 MeV.
        const float h_E_edges[2] = {99.0f, 101.0f};
        check_cuda(cudaMemcpy(d_E_edges, h_E_edges, sizeof(h_E_edges), cudaMemcpyHostToDevice),
                   "cudaMemcpy d_E_edges");

        check_cuda(cudaMemset(d_EdepC, 0, sizeof(double)), "cudaMemset d_EdepC");
        check_cuda(cudaMemset(d_AbsorbedEnergy_cutoff, 0, sizeof(double)), "cudaMemset d_AbsorbedEnergy_cutoff");
        check_cuda(cudaMemset(d_AbsorbedEnergy_nuclear, 0, sizeof(double)), "cudaMemset d_AbsorbedEnergy_nuclear");
        check_cuda(cudaMemset(d_BoundaryLoss_energy, 0, sizeof(double)), "cudaMemset d_BoundaryLoss_energy");
        check_cuda(cudaMemset(d_prev_EdepC, 0, sizeof(double)), "cudaMemset d_prev_EdepC");
        check_cuda(cudaMemset(d_prev_AbsorbedEnergy_cutoff, 0, sizeof(double)), "cudaMemset d_prev_AbsorbedEnergy_cutoff");
        check_cuda(cudaMemset(d_prev_AbsorbedEnergy_nuclear, 0, sizeof(double)), "cudaMemset d_prev_AbsorbedEnergy_nuclear");
        check_cuda(cudaMemset(d_prev_BoundaryLoss_energy, 0, sizeof(double)), "cudaMemset d_prev_BoundaryLoss_energy");

        check_cuda(cudaMemset(d_AbsorbedWeight_cutoff, 0, sizeof(float)), "cudaMemset d_AbsorbedWeight_cutoff");
        check_cuda(cudaMemset(d_AbsorbedWeight_nuclear, 0, sizeof(float)), "cudaMemset d_AbsorbedWeight_nuclear");
        check_cuda(cudaMemset(d_BoundaryLoss_weight, 0, sizeof(float)), "cudaMemset d_BoundaryLoss_weight");
        check_cuda(cudaMemset(d_prev_AbsorbedWeight_cutoff, 0, sizeof(float)), "cudaMemset d_prev_AbsorbedWeight_cutoff");
        check_cuda(cudaMemset(d_prev_AbsorbedWeight_nuclear, 0, sizeof(float)), "cudaMemset d_prev_AbsorbedWeight_nuclear");
        check_cuda(cudaMemset(d_prev_BoundaryLoss_weight, 0, sizeof(float)), "cudaMemset d_prev_BoundaryLoss_weight");

        // One in-grid component: W=1.0, E=100 MeV representative.
        const uint32_t bid = encode_block(0, 0);
        const float weight = 1.0f;
        check_cuda(cudaMemcpy(&psi_in.block_id[0], &bid, sizeof(uint32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy psi_in.block_id[0]");
        check_cuda(cudaMemcpy(&psi_in.value[0], &weight, sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy psi_in.value[0]");
    }

    void TearDown() override {
        if (d_EdepC) cudaFree(d_EdepC);
        if (d_AbsorbedEnergy_cutoff) cudaFree(d_AbsorbedEnergy_cutoff);
        if (d_AbsorbedEnergy_nuclear) cudaFree(d_AbsorbedEnergy_nuclear);
        if (d_BoundaryLoss_energy) cudaFree(d_BoundaryLoss_energy);
        if (d_prev_EdepC) cudaFree(d_prev_EdepC);
        if (d_prev_AbsorbedEnergy_cutoff) cudaFree(d_prev_AbsorbedEnergy_cutoff);
        if (d_prev_AbsorbedEnergy_nuclear) cudaFree(d_prev_AbsorbedEnergy_nuclear);
        if (d_prev_BoundaryLoss_energy) cudaFree(d_prev_BoundaryLoss_energy);

        if (d_AbsorbedWeight_cutoff) cudaFree(d_AbsorbedWeight_cutoff);
        if (d_AbsorbedWeight_nuclear) cudaFree(d_AbsorbedWeight_nuclear);
        if (d_BoundaryLoss_weight) cudaFree(d_BoundaryLoss_weight);
        if (d_prev_AbsorbedWeight_cutoff) cudaFree(d_prev_AbsorbedWeight_cutoff);
        if (d_prev_AbsorbedWeight_nuclear) cudaFree(d_prev_AbsorbedWeight_nuclear);
        if (d_prev_BoundaryLoss_weight) cudaFree(d_prev_BoundaryLoss_weight);

        if (d_E_edges) cudaFree(d_E_edges);
        if (d_report) cudaFree(d_report);

        device_psic_cleanup(psi_in);
        device_psic_cleanup(psi_out);
    }
};

TEST_F(K5DropChannelAuditTest, IncludesTransportDropAndSourceLossTerms) {
    // Choose source-loss + transport-drop values so conservation closes exactly.
    // W_in = 1.0 (in-grid) + 0.2 + 0.3 = 1.5
    // W_rhs = 1.0 (transport drop) + 0.2 + 0.3 = 1.5
    // E_in = 100 + 20 + 30 = 150
    // E_rhs = 100 (transport drop) + 20 + 30 = 150
    ASSERT_TRUE(sm_2d::run_k5_conservation_audit(
        psi_in, psi_out,
        d_EdepC,
        d_AbsorbedEnergy_cutoff,
        d_AbsorbedEnergy_nuclear,
        d_BoundaryLoss_energy,
        d_prev_EdepC,
        d_prev_AbsorbedEnergy_cutoff,
        d_prev_AbsorbedEnergy_nuclear,
        d_prev_BoundaryLoss_energy,
        d_AbsorbedWeight_cutoff,
        d_AbsorbedWeight_nuclear,
        d_BoundaryLoss_weight,
        d_prev_AbsorbedWeight_cutoff,
        d_prev_AbsorbedWeight_nuclear,
        d_prev_BoundaryLoss_weight,
        0.2f,
        0.3f,
        20.0,
        30.0,
        1.0f,
        100.0,
        1,
        d_E_edges,
        1,
        1,
        d_report,
        1,
        1
    ));

    AuditReport report{};
    check_cuda(cudaMemcpy(&report, d_report, sizeof(AuditReport), cudaMemcpyDeviceToHost),
               "cudaMemcpy report");

    EXPECT_EQ(report.W_pass, 1);
    EXPECT_EQ(report.E_pass, 1);
    EXPECT_NEAR(report.W_in_total, 1.5f, 1e-6f);
    EXPECT_NEAR(report.E_in_total, 150.0, 1e-6);
    EXPECT_NEAR(report.W_transport_drop_total, 1.0f, 1e-6f);
    EXPECT_NEAR(report.E_transport_drop_total, 100.0, 1e-6);
    EXPECT_NEAR(report.W_source_out_of_grid_total, 0.2f, 1e-6f);
    EXPECT_NEAR(report.W_source_slot_drop_total, 0.3f, 1e-6f);
    EXPECT_NEAR(report.E_source_out_of_grid_total, 20.0, 1e-6);
    EXPECT_NEAR(report.E_source_slot_drop_total, 30.0, 1e-6);

    // With drops removed, the same state should fail conservation.
    ASSERT_TRUE(sm_2d::run_k5_conservation_audit(
        psi_in, psi_out,
        d_EdepC,
        d_AbsorbedEnergy_cutoff,
        d_AbsorbedEnergy_nuclear,
        d_BoundaryLoss_energy,
        d_prev_EdepC,
        d_prev_AbsorbedEnergy_cutoff,
        d_prev_AbsorbedEnergy_nuclear,
        d_prev_BoundaryLoss_energy,
        d_AbsorbedWeight_cutoff,
        d_AbsorbedWeight_nuclear,
        d_BoundaryLoss_weight,
        d_prev_AbsorbedWeight_cutoff,
        d_prev_AbsorbedWeight_nuclear,
        d_prev_BoundaryLoss_weight,
        0.2f,
        0.3f,
        20.0,
        30.0,
        0.0f,
        0.0,
        1,
        d_E_edges,
        1,
        1,
        d_report,
        1,
        1
    ));

    AuditReport no_drop_report{};
    check_cuda(cudaMemcpy(&no_drop_report, d_report, sizeof(AuditReport), cudaMemcpyDeviceToHost),
               "cudaMemcpy no_drop_report");
    EXPECT_EQ(no_drop_report.W_pass, 0);
    EXPECT_EQ(no_drop_report.E_pass, 0);
    EXPECT_GT(no_drop_report.W_error, 1e-6f);
    EXPECT_GT(no_drop_report.E_error, 1e-5f);
}
