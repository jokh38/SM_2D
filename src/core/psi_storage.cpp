#include "core/psi_storage.hpp"
#include <cstring>

PsiC::PsiC(int Nx, int Nz, int Kb)
    : Nx(Nx), Nz(Nz), Kb(Kb), N_cells(Nx * Nz)
{
    block_id.resize(N_cells);
    value.resize(N_cells);

    for (int cell = 0; cell < N_cells; ++cell) {
        block_id[cell].fill(EMPTY_BLOCK_ID);
        for (int slot = 0; slot < Kb; ++slot) {
            value[cell][slot].fill(0.0f);
        }
    }
}

int PsiC::find_or_allocate_slot(int cell, uint32_t bid) {
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == bid) {
            return slot;
        }
    }
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == EMPTY_BLOCK_ID) {
            block_id[cell][slot] = bid;
            return slot;
        }
    }
    return -1;
}

float PsiC::get_weight(int cell, int slot, uint16_t lidx) const {
    if (cell < 0 || cell >= N_cells) return 0.0f;
    if (slot < 0 || slot >= Kb) return 0.0f;
    if (lidx >= LOCAL_BINS) return 0.0f;
    return value[cell][slot][lidx];
}

void PsiC::set_weight(int cell, int slot, uint16_t lidx, float w) {
    if (cell >= 0 && cell < N_cells && slot >= 0 && slot < Kb && lidx < LOCAL_BINS) {
        value[cell][slot][lidx] = w;
    }
}

void PsiC::clear() {
    for (int cell = 0; cell < N_cells; ++cell) {
        block_id[cell].fill(EMPTY_BLOCK_ID);
        for (int slot = 0; slot < Kb; ++slot) {
            value[cell][slot].fill(0.0f);
        }
    }
}

float sum_psi(const PsiC& psi, int cell) {
    if (cell < 0 || cell >= psi.Nx * psi.Nz) return 0.0f;

    float total = 0.0f;
    for (int slot = 0; slot < psi.Kb; ++slot) {
        if (psi.block_id[cell][slot] != EMPTY_BLOCK_ID) {
            for (int i = 0; i < LOCAL_BINS; ++i) {
                total += psi.value[cell][slot][i];
            }
        }
    }
    return total;
}
