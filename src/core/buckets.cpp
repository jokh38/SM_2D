#include "core/buckets.hpp"

OutflowBucket::OutflowBucket() {
    block_id.fill(EMPTY_BLOCK_ID);
    local_count.fill(0);
    for (int i = 0; i < Kb_out; ++i) {
        value[i].fill(0.0f);
    }
}

int OutflowBucket::find_or_allocate_slot(uint32_t bid) {
    for (int slot = 0; slot < Kb_out; ++slot) {
        if (block_id[slot] == bid) {
            return slot;
        }
    }
    for (int slot = 0; slot < Kb_out; ++slot) {
        if (block_id[slot] == EMPTY_BLOCK_ID) {
            block_id[slot] = bid;
            return slot;
        }
    }
    return -1;
}

void OutflowBucket::clear() {
    block_id.fill(EMPTY_BLOCK_ID);
    local_count.fill(0);
    for (int i = 0; i < Kb_out; ++i) {
        value[i].fill(0.0f);
    }
}
