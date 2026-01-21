#include <gtest/gtest.h>
#include "kernels/k1_activemask.cuh"
#include "core/psi_storage.hpp"

TEST(K1Test, HighEnergyTriggered) {
    PsiC psi(4, 4, 32);
    uint32_t bid = encode_block(10, 20);
    int slot = psi.find_or_allocate_slot(0, bid);
    psi.set_weight(0, slot, encode_local_idx(3, 1), 1.0f);

    std::vector<uint8_t> ActiveMask(16, 0);
    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10.0f, 1e-10f);
    EXPECT_TRUE(ActiveMask[0]);
}

TEST(K1Test, LowEnergyNotTriggered) {
    PsiC psi(4, 4, 32);

    // Low energy block (b_E < E_trigger / dE)
    uint32_t bid = encode_block(10, 0);  // Low energy
    int slot = psi.find_or_allocate_slot(0, bid);
    psi.set_weight(0, slot, 0, 1.0f);

    std::vector<uint8_t> ActiveMask(16, 0);

    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10.0f, 1e-10f);

    EXPECT_FALSE(ActiveMask[0]);
}

TEST(K1Test, WeightThreshold) {
    PsiC psi(4, 4, 32);

    // High energy but very low weight
    uint32_t bid = encode_block(10, 20);
    int slot = psi.find_or_allocate_slot(0, bid);
    psi.set_weight(0, slot, 0, 1e-12f);

    std::vector<uint8_t> ActiveMask(16, 0);

    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10.0f, 1e-10f);

    EXPECT_FALSE(ActiveMask[0]);  // Below weight threshold
}

TEST(K1Test, MultipleCells) {
    PsiC psi(4, 4, 32);

    // Cell 0: active
    uint32_t bid_high = encode_block(10, 20);
    int slot0 = psi.find_or_allocate_slot(0, bid_high);
    psi.set_weight(0, slot0, 0, 1.0f);

    // Cell 1: inactive
    uint32_t bid_low = encode_block(10, 0);
    int slot1 = psi.find_or_allocate_slot(1, bid_low);
    psi.set_weight(1, slot1, 0, 1.0f);

    std::vector<uint8_t> ActiveMask(16, 0);

    run_K1_ActiveMask(psi, ActiveMask.data(), 4, 4, 10.0f, 1e-10f);

    EXPECT_TRUE(ActiveMask[0]);
    EXPECT_FALSE(ActiveMask[1]);
}
