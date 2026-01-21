#include "boundary/boundaries.hpp"

int get_neighbor(int cell, int face, const BoundaryConfig& config, int Nx, int Nz) {
    int ix = cell % Nx;
    int iz = cell / Nx;

    int neighbor = -1;

    switch (face) {
        case 0:  // +z
            if (iz + 1 >= Nz) {
                neighbor = (config.z_max == BoundaryType::ABSORB) ? -1 : cell;
            } else {
                neighbor = cell + Nx;
            }
            break;
        case 1:  // -z
            if (iz <= 0) {
                neighbor = (config.z_min == BoundaryType::ABSORB) ? -1 : cell;
            } else {
                neighbor = cell - Nx;
            }
            break;
        case 2:  // +x
            if (ix + 1 >= Nx) {
                neighbor = (config.x_max == BoundaryType::ABSORB) ? -1 : cell;
            } else {
                neighbor = cell + 1;
            }
            break;
        case 3:  // -x
            if (ix <= 0) {
                neighbor = (config.x_min == BoundaryType::ABSORB) ? -1 : cell;
            } else {
                neighbor = cell - 1;
            }
            break;
    }

    return neighbor;
}
