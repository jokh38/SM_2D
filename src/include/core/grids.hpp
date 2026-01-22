#pragma once
#include <vector>
#include <cmath>

// Energy grid (log-spaced)
struct EnergyGrid {
    const int N_E;
    const float E_min;
    const float E_max;
    std::vector<float> edges;   // N_E + 1 bin edges
    std::vector<float> rep;     // N_E representative (geometric mean)

    EnergyGrid(float E_min, float E_max, int N_E);
    int FindBin(float E) const;  // Binary search
    float GetRepEnergy(int bin) const;
};

// Angular grid (uniform)
struct AngularGrid {
    const int N_theta;
    const float theta_min;
    const float theta_max;
    std::vector<float> edges;   // N_theta + 1 bin edges
    std::vector<float> rep;     // N_theta representative (midpoint)

    AngularGrid(float theta_min, float theta_max, int N_theta);
    int FindBin(float theta) const;
    float GetRepTheta(int bin) const;
};
