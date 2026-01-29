#pragma once
#include <string>
#include <vector>
#include <memory>

// Forward declarations
namespace sm_2d {
struct RLUT;
struct DeviceRLUT;

// Opaque pointer for CUDA-only types
class DeviceLUTWrapperImpl;
}

namespace sm_2d {

/**
 * @brief Opaque wrapper for DeviceLUTWrapper (CUDA-only type)
 */
class DeviceLUTWrapper {
public:
    DeviceLUTWrapper();
    ~DeviceLUTWrapper();

    // Non-copyable
    DeviceLUTWrapper(const DeviceLUTWrapper&) = delete;
    DeviceLUTWrapper& operator=(const DeviceLUTWrapper&) = delete;

    // Access to underlying device LUT
    const DeviceRLUT* get() const { return dlut_ptr; }

private:
    friend bool init_device_lut(const RLUT& cpu_lut, DeviceLUTWrapper& wrapper);
    DeviceLUTWrapperImpl* p_impl;
    const DeviceRLUT* dlut_ptr;
};

/**
 * @brief Initialize device LUT from CPU LUT
 */
bool init_device_lut(const RLUT& cpu_lut, DeviceLUTWrapper& wrapper);

/**
 * @brief Run K1-K6 pipeline GPU transport with phase-space binning
 *
 * This function runs the full multi-stage transport pipeline with:
 * - K1: Active mask identification
 * - K2: Coarse transport (high-energy cells)
 * - K3: Fine transport (low-energy cells)
 * - K4: Bucket transfer (boundary crossing)
 * - K5: Weight audit (conservation check)
 * - K6: Buffer swapping
 *
 * Supports both pencil beam and Gaussian beam sources.
 *
 * @param x0, z0 Source position (mm)
 * @param theta0 Source angle (radians)
 * @param E0 Source energy (MeV)
 * @param W_total Total source weight
 * @param sigma_x Lateral beam spread (mm, 0 for pencil beam)
 * @param sigma_theta Angular divergence (rad, 0 for pencil beam)
 * @param sigma_E Energy spread (MeV, 0 for monoenergetic)
 * @param n_samples Number of samples for Gaussian beam (1 for pencil beam)
 * @param random_seed RNG seed for reproducibility
 * @param Nx, Nz Spatial grid dimensions
 * @param dx, dz Spatial grid spacing (mm)
 * @param x_min, z_min Grid origin (mm)
 * @param N_theta, N_E Global phase-space grid dimensions
 * @param N_theta_local, N_E_local Local bin dimensions per cell
 * @param theta_edges Device pointer to theta bin edges [N_theta + 1]
 * @param E_edges Device pointer to energy bin edges [N_E + 1]
 * @param dlut Device range-energy lookup table
 * @param edep Output energy deposition grid [Nz][Nx]
 */
void run_k1k6_pipeline_transport(
    float x0, float z0, float theta0, float E0, float W_total,
    float sigma_x, float sigma_theta, float sigma_E,
    int n_samples, unsigned int random_seed,
    int Nx, int Nz, float dx, float dz,
    float x_min, float z_min,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,
    const float* theta_edges,
    const float* E_edges,
    const DeviceRLUT& dlut,
    std::vector<std::vector<double>>& edep
);

/**
 * @brief Get GPU device name (internal CUDA function)
 */
std::string get_gpu_name_internal();

} // namespace sm_2d
