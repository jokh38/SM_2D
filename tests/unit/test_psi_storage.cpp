#include <gtest/gtest.h>
#include "core/psi_storage.hpp"

TEST(PsiStorageTest, InitializeEmpty) {
    PsiC psi(32, 32, 32);
    for (int cell = 0; cell < 32 * 32; ++cell) {
        for (int slot = 0; slot < 32; ++slot) {
            EXPECT_EQ(psi.block_id[cell][slot], EMPTY_BLOCK_ID);
        }
    }
}

TEST(PsiStorageTest, AllocateAndAccess) {
    PsiC psi(32, 32, 32);
    uint32_t bid = encode_block(10, 20);
    uint16_t lidx = encode_local_idx(3, 1);
    float w = 1.5f;

    int slot = psi.find_or_allocate_slot(0, bid);
    ASSERT_GE(slot, 0);

    psi.set_weight(0, slot, lidx, w);
    EXPECT_NEAR(psi.get_weight(0, slot, lidx), w, 1e-6f);
}
