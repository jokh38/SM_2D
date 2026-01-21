#pragma once
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"
#include <vector>
#include <array>
#include <cstdint>

struct PsiC {
    const int Nx;
    const int Nz;
    const int Kb;

    std::vector<std::array<uint32_t, 32>> block_id;
    std::vector<std::array<std::array<float, LOCAL_BINS>, 32>> value;

    PsiC(int Nx, int Nz, int Kb);

    int find_or_allocate_slot(int cell, uint32_t bid);
    float get_weight(int cell, int slot, uint16_t lidx) const;
    void set_weight(int cell, int slot, uint16_t lidx, float w);
    void clear();

private:
    int N_cells;
};

float sum_psi(const PsiC& psi, int cell);
