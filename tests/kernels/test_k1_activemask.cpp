#include <gtest/gtest.h>
#include "kernels/k1_activemask.cuh"
#include "core/psi_storage.hpp"
#include "core/grids.hpp"

TEST(K1Test, LowEnergyTriggersFineTransport) {
    // P4 FIX: Test now correctly uses b_E_trigger (block index) instead of E_trigger (MeV)
    PsiC psi(4, 4, 32);

    // Low energy block (b_E = 5, below threshold b_E_trigger = 10)
    uint32_t bid = encode_block(10, 5);
    int slot = psi.find_or_allocate_slot(0, bid);
    psi.set_weight(0, slot, encode_local_idx(3, 1), 1.0f);

    std::vector<uint8_t> ActiveMask(16, 0);
    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10, 1e-10f);
    EXPECT_TRUE(ActiveMask[0]);  // Low energy triggers fine transport
}

TEST(K1Test, HighEnergyDoesNotTrigger) {
    // P4 FIX: High energy should NOT trigger fine transport
    PsiC psi(4, 4, 32);

    // High energy block (b_E = 15, above threshold b_E_trigger = 10)
    uint32_t bid = encode_block(10, 15);
    int slot = psi.find_or_allocate_slot(0, bid);
    psi.set_weight(0, slot, 0, 1.0f);

    std::vector<uint8_t> ActiveMask(16, 0);

    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10, 1e-10f);

    EXPECT_FALSE(ActiveMask[0]);  // High energy does NOT trigger fine transport
}

TEST(K1Test, WeightThreshold) {
    PsiC psi(4, 4, 32);

    // Low energy but very low weight
    uint32_t bid = encode_block(10, 5);
    int slot = psi.find_or_allocate_slot(0, bid);
    psi.set_weight(0, slot, 0, 1e-12f);

    std::vector<uint8_t> ActiveMask(16, 0);

    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10, 1e-10f);

    EXPECT_FALSE(ActiveMask[0]);  // Below weight threshold
}

TEST(K1Test, MultipleCells) {
    PsiC psi(4, 4, 32);

    // Cell 0: low energy (should be active)
    uint32_t bid_low = encode_block(10, 5);
    int slot0 = psi.find_or_allocate_slot(0, bid_low);
    psi.set_weight(0, slot0, 0, 1.0f);

    // Cell 1: high energy (should be inactive)
    uint32_t bid_high = encode_block(10, 15);
    int slot1 = psi.find_or_allocate_slot(1, bid_high);
    psi.set_weight(1, slot1, 0, 1.0f);

    std::vector<uint8_t> ActiveMask(16, 0);

    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10, 1e-10f);

    EXPECT_TRUE(ActiveMask[0]);   // Low energy triggers
    EXPECT_FALSE(ActiveMask[1]);  // High energy does NOT trigger
}

TEST(K1Test, ComputeBETriggerFromEnergy) {
    // P4 FIX: Test the helper function that converts E_trigger (MeV) to b_E_trigger (block index)
    EnergyGrid e_grid(0.1f, 300.0f, 256);
    constexpr int N_E_local = 4;

    // E_trigger = 10 MeV should map to a specific block index
    int b_E_trigger = compute_b_E_trigger(10.0f, e_grid, N_E_local);

    // With log-spaced grid from 0.1 to 300 MeV:
    // 10 MeV falls in bin ~54, which maps to block ~13 (54 / 4)
    EXPECT_GT(b_E_trigger, 10);
    EXPECT_LT(b_E_trigger, 20);

    // Higher E_trigger should give higher b_E_trigger
    int b_E_trigger_50 = compute_b_E_trigger(50.0f, e_grid, N_E_local);
    EXPECT_GT(b_E_trigger_50, b_E_trigger);
}
