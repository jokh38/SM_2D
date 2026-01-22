#pragma once
#include <cstdint>

enum class BoundaryType : int {
    ABSORB = 0,
    REFLECT = 1,
    PERIODIC = 2
};

struct BoundaryConfig {
    BoundaryType z_min = BoundaryType::ABSORB;
    BoundaryType z_max = BoundaryType::ABSORB;
    BoundaryType x_min = BoundaryType::ABSORB;
    BoundaryType x_max = BoundaryType::ABSORB;
};

int get_neighbor(int cell, int face, const BoundaryConfig& config, int Nx, int Nz);
