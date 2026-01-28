#pragma once
#include <vector>
#include <cmath>
#include <tuple>

// Energy grid (log-spaced or piecewise-uniform)
struct EnergyGrid {
    const int N_E;
    const float E_min;
    const float E_max;
    std::vector<float> edges;   // N_E + 1 bin edges
    std::vector<float> rep;     // N_E representative (geometric mean or midpoint)
    bool is_piecewise;          // true for piecewise-uniform grid

    // Original constructor: log-spaced grid
    EnergyGrid(float E_min, float E_max, int N_E);

    // New constructor: piecewise-uniform grid (Option D2)
    // Format: {{E_start, E_end, resolution}, ...}
    // Example: {{0.1, 2.0, 0.1}, {2.0, 20.0, 0.25}, {20.0, 100.0, 0.5}, {100.0, 250.0, 1.0}}
    EnergyGrid(const std::vector<std::tuple<float, float, float>>& groups);

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
