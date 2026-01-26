#include "kernels/simple_gpu_transport.cuh"
#include "device/device_physics.cuh"
#include "device/device_lut.cuh"
#include <cmath>

// Physics constants
constexpr float X0_water = 363.0f;  // Radiation length of water [mm]
constexpr float rho_water = 1.0f;   // Density of water [g/cm³]

/**
 * GPU Kernel: Simple deterministic particle transport
 *
 * Each thread transports one particle through the medium.
 * Uses proper energy straggling (Vavilov model via device_energy_straggling_sigma).
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
        unsigned seed = static_cast<unsigned>(idx * 1000 + static_cast<int>(E * 100));
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
        seed = static_cast<unsigned>(idx * 1000 + static_cast<int>(E_new * 100));
        float theta_scatter = device_sample_gaussian(seed) * sigma_theta;

        // Update direction
        float theta_new = theta + theta_scatter;
        mu = cosf(theta_new);
        eta = sinf(theta_new);

        // Update position
        x += eta * step;
        z += mu * step;

        // Update energy
        E = E_new;

        // Nuclear attenuation (simplified)
        float sigma_nuclear = device_nuclear_cross_section(E);
        float P_nuclear = 1.0f - expf(-sigma_nuclear * step);

        // Use simple random check (deterministic for reproducibility)
        if (P_nuclear > 0.01f && (idx % 100) < static_cast<int>(P_nuclear * 100)) {
            // Nuclear interaction - particle removed
            break;
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
