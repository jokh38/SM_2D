#pragma once
#include <vector>
#include <cmath>
#include <tuple>

// Energy grid (log-spaced or piecewise-uniform)
struct EnergyGrid {
    int N_E;
    float E_min;
    float E_max;
    std::vector<float> edges;   // N_E + 1 bin edges
    std::vector<float> rep;     // N_E representative (geometric mean or midpoint)
    bool is_piecewise;          // true for piecewise-uniform grid

    // Original constructor: log-spaced grid
    EnergyGrid(float E_min, float E_max, int N_E);

    // Factory function: piecewise-uniform grid (Option D2)
    // Format: {{E_start, E_end, resolution}, ...}
    // Example: {{0.1, 2.0, 0.1}, {2.0, 20.0, 0.25}, {20.0, 100.0, 0.5}, {100.0, 250.0, 1.0}}
    // Factory pattern avoids const_cast while maintaining immutable semantics
    static EnergyGrid CreatePiecewise(const std::vector<std::tuple<float, float, float>>& groups);

    // Const accessors for immutable semantics
    int GetN_E() const { return N_E; }
    float GetE_min() const { return E_min; }
    float GetE_max() const { return E_max; }

    int FindBin(float E) const;  // Binary search
    float GetRepEnergy(int bin) const;
};

// Angular grid (uniform)
struct AngularGrid {
    int N_theta;
    float theta_min;
    float theta_max;
    std::vector<float> edges;   // N_theta + 1 bin edges
    std::vector<float> rep;     // N_theta representative (midpoint)

    AngularGrid(float theta_min, float theta_max, int N_theta);
    int FindBin(float theta) const;
    float GetRepTheta(int bin) const;

    // Const accessors for immutable semantics
    int GetN_theta() const { return N_theta; }
    float GetTheta_min() const { return theta_min; }
    float GetTheta_max() const { return theta_max; }
};
