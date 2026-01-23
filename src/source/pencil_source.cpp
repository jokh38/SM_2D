#include "source/pencil_source.hpp"
#include "core/local_bins.hpp"
#include "core/block_encoding.hpp"

void inject_source(
    PsiC& psi,
    const PencilSource& src,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid
) {
    float dx = 1.0f, dz = 1.0f;
    int ix = static_cast<int>(src.x0 / dx);
    int iz = static_cast<int>(src.z0 / dz);
    int cell = ix + iz * psi.Nx;

    if (cell < 0 || cell >= psi.Nx * psi.Nz) return;

    // Calculate sub-cell x position
    float x_in_cell = src.x0 - ix * dx;  // Position within cell [0, dx)
    float x_offset = x_in_cell - dx * 0.5f;  // Offset from cell center [-dx/2, +dx/2)
    int x_sub = get_x_sub_bin(x_offset, dx);  // Sub-cell bin [0, 3]

    int theta_bin = a_grid.FindBin(src.theta0);
    int E_bin = e_grid.FindBin(src.E0);

    uint32_t bid = encode_block(
        theta_bin / N_theta_local,
        E_bin / N_E_local
    );

    int theta_local = theta_bin % N_theta_local;
    int E_local = E_bin % N_E_local;

    // Use 3D encoding with x_sub
    uint16_t lidx = encode_local_idx_3d(theta_local, E_local, x_sub);

    int slot = psi.find_or_allocate_slot(cell, bid);
    if (slot < 0) return;

    float current_w = psi.get_weight(cell, slot, lidx);
    psi.set_weight(cell, slot, lidx, current_w + src.W_total);
}
