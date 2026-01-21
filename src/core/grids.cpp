#include "core/grids.hpp"
#include <algorithm>

// EnergyGrid implementation
EnergyGrid::EnergyGrid(float E_min, float E_max, int N_E)
    : N_E(N_E), E_min(E_min), E_max(E_max)
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
