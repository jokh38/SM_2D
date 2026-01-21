#include "boundary/loss_tracking.hpp"

void record_boundary_loss(BoundaryLoss& loss, int face, float w, double E) {
    if (face >= 0 && face < 4) {
        loss.weight[face] += w;
        loss.energy[face] += E;
    }
}

float total_boundary_weight_loss(const BoundaryLoss& loss) {
    float total = 0;
    for (int i = 0; i < 4; ++i) {
        total += loss.weight[i];
    }
    return total;
}

double total_boundary_energy_loss(const BoundaryLoss& loss) {
    double total = 0;
    for (int i = 0; i < 4; ++i) {
        total += loss.energy[i];
    }
    return total;
}
