#pragma once
#include "physics/physics.hpp"
#include "lut/r_lut.hpp"
#include <cstdint>

// Forward declaration for device LUT access
struct DeviceRLUT;

// Component state
struct Component {
    float theta, E, w, x, z, mu, eta;
};

// Result from single component transport
struct K3Result {
    float Edep = 0;
    float E_new = 0;           // Updated energy after transport (IC-2 fix)
    float nuclear_weight_removed = 0;
    float nuclear_energy_removed = 0;
    int bucket_emissions = 0;
    bool remained_in_cell = true;
    bool terminated = false;
    int split_count = 0;
};

// Fine transport kernel
__global__ void K3_FineTransport(
    // Inputs
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint32_t* __restrict__ ActiveList,
    // Grid
    int Nx, int Nz, float dx, float dz,
    int n_active,
    // Outputs
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy
);

// CPU test stubs
K3Result run_K3_single_component(const Component& c);
K3Result run_K3_with_forced_split(const Component& c);
