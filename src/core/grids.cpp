#include "core/grids.hpp"
#include <algorithm>
#include <tuple>

// EnergyGrid implementation - Original log-spaced constructor
EnergyGrid::EnergyGrid(float E_min, float E_max, int N_E)
    : N_E(N_E), E_min(E_min), E_max(E_max), is_piecewise(false)
{
    edges.resize(N_E + 1);
    rep.resize(N_E);

    float log_E_min = logf(E_min);
    float log_E_max = logf(E_max);
    float delta_log = (log_E_max - log_E_min) / N_E;

    for (int i = 0; i <= N_E; ++i) {
        edges[i] = expf(log_E_min + i * delta_log);
    }

    for (int i = 0; i < N_E; ++i) {
        rep[i] = sqrtf(edges[i] * edges[i+1]);
    }
}

// EnergyGrid implementation - Piecewise-uniform constructor (Option D2)
// Compute N_E first, then use delegating constructor pattern
namespace {
    struct PiecewiseGridResult {
        int N_E;
        float E_min;
        float E_max;
        std::vector<float> edges;
        std::vector<float> rep;
    };

    PiecewiseGridResult create_piecewise_grid(const std::vector<std::tuple<float, float, float>>& groups) {
        // First pass: count total bins and set E_min, E_max
        int total_bins = 0;
        float first_E_min = std::get<0>(groups[0]);
        float last_E_max = std::get<1>(groups.back());

        for (const auto& group : groups) {
            float E_start = std::get<0>(group);
            float E_end = std::get<1>(group);
            float resolution = std::get<2>(group);
            int n_bins = static_cast<int>((E_end - E_start) / resolution + 0.5f);
            total_bins += n_bins;
        }

        std::vector<float> edges(total_bins + 1);
        std::vector<float> rep(total_bins);

        // Second pass: fill edges and rep
        int bin_offset = 0;
        for (const auto& group : groups) {
            float E_start = std::get<0>(group);
            float E_end = std::get<1>(group);
            float resolution = std::get<2>(group);
            int n_bins = static_cast<int>((E_end - E_start) / resolution + 0.5f);

            for (int i = 0; i <= n_bins; ++i) {
                edges[bin_offset + i] = E_start + i * resolution;
            }

            // Representative energy: geometric mean for energy-dependent data
            // For physics quantities like stopping power that vary exponentially,
            // the geometric mean provides better interpolation than arithmetic mean.
            // Note: This is used for LUT generation. Particle tracking in K3 uses
            // lower edge to ensure energy actually decreases between iterations.
            for (int i = 0; i < n_bins; ++i) {
                rep[bin_offset + i] = sqrtf(edges[bin_offset + i] * edges[bin_offset + i + 1]);
            }

            bin_offset += n_bins;
        }

        // Ensure last edge is exact
        edges[total_bins] = last_E_max;

        return {total_bins, first_E_min, last_E_max, edges, rep};
    }
}

// Factory function for piecewise-uniform grid
// This avoids const_cast by constructing the object with computed values
EnergyGrid EnergyGrid::CreatePiecewise(const std::vector<std::tuple<float, float, float>>& groups) {
    // Create the piecewise grid data
    auto result = create_piecewise_grid(groups);

    // Construct and return the object with correct values
    EnergyGrid grid(result.E_min, result.E_max, result.N_E);
    grid.edges = std::move(result.edges);
    grid.rep = std::move(result.rep);
    grid.is_piecewise = true;
    return grid;
}

int EnergyGrid::FindBin(float E) const {
    if (E < edges[0]) return 0;
    if (E >= edges[N_E]) return N_E - 1;

    int lo = 0, hi = N_E;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (edges[mid + 1] <= E) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

float EnergyGrid::GetRepEnergy(int bin) const {
    if (bin < 0) bin = 0;
    if (bin >= N_E) bin = N_E - 1;
    return rep[bin];
}

// AngularGrid implementation
AngularGrid::AngularGrid(float theta_min, float theta_max, int N_theta)
    : N_theta(N_theta), theta_min(theta_min), theta_max(theta_max)
{
    edges.resize(N_theta + 1);
    rep.resize(N_theta);

    float delta = (theta_max - theta_min) / N_theta;
    for (int i = 0; i <= N_theta; ++i) {
        edges[i] = theta_min + i * delta;
    }

    for (int i = 0; i < N_theta; ++i) {
        rep[i] = 0.5f * (edges[i] + edges[i+1]);
    }
}

int AngularGrid::FindBin(float theta) const {
    if (theta < theta_min) return 0;
    if (theta >= theta_max) return N_theta - 1;

    float delta = (theta_max - theta_min) / N_theta;
    int bin = static_cast<int>((theta - theta_min) / delta);
    return std::max(0, std::min(bin, N_theta - 1));
}

float AngularGrid::GetRepTheta(int bin) const {
    if (bin < 0) bin = 0;
    if (bin >= N_theta) bin = N_theta - 1;
    return rep[bin];
}
