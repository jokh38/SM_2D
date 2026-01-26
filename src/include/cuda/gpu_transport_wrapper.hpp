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
 * @brief Run GPU transport
 */
void run_gpu_transport(
    float x0, float z0, float theta0, float E0, float W_total,
    int n_particles,
    int Nx, int Nz, float dx, float dz,
    float x_min, float z_min,
    const DeviceRLUT& dlut,
    std::vector<std::vector<double>>& edep
);

/**
 * @brief Get GPU device name (internal CUDA function)
 */
std::string get_gpu_name_internal();

} // namespace sm_2d
