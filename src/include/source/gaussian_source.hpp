#pragma once
#include "core/grids.hpp"
#include "core/psi_storage.hpp"
#include <random>

struct GaussianSource {
    float x0 = 0.0f;
    float theta0 = 0.0f;
    float sigma_x = 5.0f;
    float sigma_theta = 0.01f;
    float E0 = 150.0f;
    float sigma_E = 1.0f;
    float W_total = 1.0f;
    int n_samples = 1000;
};

void inject_source(
    PsiC& psi,
    const GaussianSource& src,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid
);
