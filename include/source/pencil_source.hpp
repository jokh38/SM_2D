#pragma once
#include "core/grids.hpp"
#include "core/psi_storage.hpp"

struct PencilSource {
    float x0 = 0.0f;
    float z0 = 0.0f;
    float theta0 = 0.0f;
    float E0 = 150.0f;
    float W_total = 1.0f;
};

void inject_source(
    PsiC& psi,
    const PencilSource& src,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid
);
