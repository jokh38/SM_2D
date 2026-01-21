#pragma once
#include <array>

struct BoundaryLoss {
    std::array<float, 4> weight;
    std::array<double, 4> energy;

    BoundaryLoss() {
        weight.fill(0.0f);
        energy.fill(0.0);
    }
};

void record_boundary_loss(BoundaryLoss& loss, int face, float w, double E);
float total_boundary_weight_loss(const BoundaryLoss& loss);
double total_boundary_energy_loss(const BoundaryLoss& loss);
