#include "kernels/simple_gpu_transport.cuh"
#include "device/device_physics.cuh"
#include "device/device_lut.cuh"
#include "physics/fermi_eyges.hpp"
#include <cmath>

// Physics constants (use highland.hpp's X0_water value)
constexpr float rho_water = 1.0f;   // Density of water [g/cm³]

// ============================================================================
// Device Random Sampling Functions
// ============================================================================

/**
 * Sample uniform random number in [0, 1) using a simple LCG
 * @param seed Random seed (updated in-place)
 * @return Uniform random float in [0, 1)
 */
__device__ inline float device_sample_uniform(unsigned& seed) {
    // Linear congruential generator (LCG)
    // Using Numerical Recipes constants: a = 1664525, c = 1013904223
    seed = 1664525u * seed + 1013904223u;

    // Convert to float in [0, 1)
    // Use upper bits for better uniformity
    return (seed >> 16) / 65536.0f;
}

/**
 * GPU Kernel: Simple deterministic particle transport
 *
 * Each thread transports one particle through the medium.
 * Uses proper energy straggling (Vavilov model via device_energy_straggling_sigma).
 * Now includes Fermi-Eyges lateral spreading theory.
 */
__global__ void simple_particle_transport_kernel(
    // Input: Initial particle states
    const ParticleInput* __restrict__ inputs,
    int n_particles,

    // Device LUT for energy loss
    DeviceRLUT dlut,

    // Grid parameters
    int Nx, int Nz, float dx, float dz,
    float x_min, float z_min,

    // Physics thresholds
    float E_cutoff,

    // Output: Dose grid (2D flattened)
    double* __restrict__ dose_grid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_particles) return;

    // Load initial particle state
    ParticleInput p = inputs[idx];

    float x = p.x;
    float z = p.z;
    float E = p.E;
    float theta = p.theta;
    float w = p.w;

    // Direction cosines
    float mu = cosf(theta);
    float eta = sinf(theta);

    // Initialize random seed for this particle
    unsigned seed = static_cast<unsigned>(idx);

    // Initialize Fermi-Eyges moments for lateral spreading
    FermiEygesMoments fe_moments;

    // Transport loop
    while (E > E_cutoff && z >= 0.0f && z < Nz * dz && fabsf(x - x_min) < Nx * dx) {
        // Compute step size (adaptive based on energy)
        float step = device_compute_max_step(dlut, E);
        step = fminf(step, 2.0f);  // Cap step size

        // Check remaining distance in current cell
        int iz = static_cast<int>(z / dz);
        int ix = static_cast<int>((x - x_min) / dx);

        if (iz < 0 || iz >= Nz || ix < 0 || ix >= Nx) break;

        // Distance to cell boundary
        float z_cell = iz * dz;
        float x_cell = x_min + ix * dx;
        float dz_to_boundary = (mu > 0) ? (z_cell + dz - z) : (z - z_cell);
        float dx_to_boundary = (eta > 0) ? (x_cell + dx - x) : (x - x_cell);

        if (dz_to_boundary > 0 && dz_to_boundary < step) {
            step = dz_to_boundary;
        }

        // Energy loss with proper straggling (Vavilov model)
        float mean_dE = device_compute_energy_deposition(dlut, E, step);
        float sigma_dE = device_energy_straggling_sigma(E, step, rho_water);

        // Use Box-Muller transform for Gaussian sampling
        float z1 = device_sample_gaussian(seed);
        float dE = mean_dE + z1 * sigma_dE;
        dE = fmaxf(0.0f, fminf(dE, E));  // Clamp to valid range

        float E_new = E - dE;
        float edep = dE * w;

        // Deposit energy (atomic add for thread safety)
        int grid_idx = iz * Nx + ix;
        atomicAdd(&dose_grid[grid_idx], edep);

        // Multiple Coulomb Scattering (Highland formula)
        float sigma_theta = device_highland_sigma(E, step, X0_water);

        // Sample scattering angle
        float theta_scatter = device_sample_gaussian(seed) * sigma_theta;

        // Update direction
        float theta_new = theta + theta_scatter;
        mu = cosf(theta_new);
        eta = sinf(theta_new);

        // Update Fermi-Eyges moments for lateral spreading
        device_update_fermi_eyges_moments(fe_moments, E, z, step, X0_water);

        // Update position
        x += eta * step;
        z += mu * step;

        // Apply Fermi-Eyges lateral spread correction
        device_apply_fermi_eyges_spread(x, z, fe_moments, seed);

        // Update energy and angle
        E = E_new;
        theta = theta_new;

        // Nuclear attenuation (FIXED: proper random sampling)
        float sigma_nuclear = device_nuclear_cross_section(E);
        float P_nuclear = 1.0f - expf(-sigma_nuclear * step);

        // Use proper uniform random sampling instead of (idx % 100)
        if (P_nuclear > 0.01f) {
            float u = device_sample_uniform(seed);
            if (u < P_nuclear) {
                // Nuclear interaction - particle removed
                break;
            }
        }
    }
}

/**
 * CPU wrapper: Run GPU particle transport
 */
void run_simple_gpu_transport(
    // Source configuration
    float x0, float z0, float theta0, float E0, float W_total,
    int n_particles,

    // Grid parameters
    int Nx, int Nz, float dx, float dz,
    float x_min, float z_min,

    // Device LUT
    const DeviceRLUT& dlut,

    // Output
    std::vector<std::vector<double>>& edep
) {
    // Allocate input arrays on GPU
    ParticleInput* d_inputs;
    cudaMalloc(&d_inputs, n_particles * sizeof(ParticleInput));

    // Create input array (single source particle, will be replicated)
    std::vector<ParticleInput> inputs(n_particles);
    for (int i = 0; i < n_particles; ++i) {
        inputs[i].x = x0;
        inputs[i].z = z0;
        inputs[i].E = E0;
        inputs[i].theta = theta0;
        inputs[i].w = W_total / n_particles;  // Distribute weight
    }

    cudaMemcpy(d_inputs, inputs.data(), n_particles * sizeof(ParticleInput),
               cudaMemcpyHostToDevice);

    // Allocate dose grid on GPU
    double* d_dose;
    size_t dose_size = Nx * Nz * sizeof(double);
    cudaMalloc(&d_dose, dose_size);
    cudaMemset(d_dose, 0, dose_size);

    // Launch kernel
    int threads = 256;
    int blocks = (n_particles + threads - 1) / threads;

    float E_cutoff = 0.1f;  // 100 keV cutoff

    simple_particle_transport_kernel<<<blocks, threads>>>(
        d_inputs, n_particles, dlut,
        Nx, Nz, dx, dz, x_min, z_min, E_cutoff,
        d_dose
    );

    cudaDeviceSynchronize();

    // Copy results back
    std::vector<double> dose_flat(Nx * Nz);
    cudaMemcpy(dose_flat.data(), d_dose, dose_size, cudaMemcpyDeviceToHost);

    // Convert to 2D array
    edep.resize(Nz, std::vector<double>(Nx));
    for (int iz = 0; iz < Nz; ++iz) {
        for (int ix = 0; ix < Nx; ++ix) {
            edep[iz][ix] = dose_flat[iz * Nx + ix];
        }
    }

    // Cleanup
    cudaFree(d_inputs);
    cudaFree(d_dose);
}
