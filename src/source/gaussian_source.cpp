#include "source/gaussian_source.hpp"
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"
#include <cmath>
#include <random>

void inject_source(
    PsiC& psi,
    const GaussianSource& src,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid
) {
    std::mt19937 rng(42);
    std::normal_distribution<float> x_dist(src.x0, src.sigma_x);
    std::normal_distribution<float> theta_dist(src.theta0, src.sigma_theta);
    std::normal_distribution<float> E_dist(src.E0, src.sigma_E);

    float w_per_sample = src.W_total / src.n_samples;
    float dx = 1.0f;

    for (int i = 0; i < src.n_samples; ++i) {
        float x = x_dist(rng);
        float theta = theta_dist(rng);
        float E = E_dist(rng);

        E = fmaxf(e_grid.E_min, fminf(E, e_grid.E_max));
        theta = fmaxf(a_grid.theta_min, fminf(theta, a_grid.theta_max));

        int ix = static_cast<int>(x / dx);
        if (ix < 0 || ix >= psi.Nx) continue;

        int cell = ix;
        int theta_bin = a_grid.FindBin(theta);
        int E_bin = e_grid.FindBin(E);

        uint32_t bid = encode_block(
            theta_bin / N_theta_local,
            E_bin / N_E_local
        );

        int theta_local = theta_bin % N_theta_local;
        int E_local = E_bin % N_E_local;
        uint16_t lidx = encode_local_idx(theta_local, E_local);

        int slot = psi.find_or_allocate_slot(cell, bid);
        if (slot < 0) continue;

        float current_w = psi.get_weight(cell, slot, lidx);
        psi.set_weight(cell, slot, lidx, current_w + w_per_sample);
    }
}
