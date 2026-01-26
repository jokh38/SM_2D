#include "cuda/gpu_transport_wrapper.hpp"
#include "kernels/simple_gpu_transport.cuh"
#include "kernels/k3_finetransport.cuh"
#include "lut/r_lut.hpp"
#include "lut/nist_loader.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

// Bring CUDA types into sm_2d namespace for compatibility
using ::DeviceLUTWrapper;

namespace sm_2d {

// Implementation of DeviceLUTWrapper (PIMPL pattern for CUDA types)
class DeviceLUTWrapperImpl {
public:
    // This is the actual CUDA DeviceLUTWrapper from k3_finetransport.cuh
    ::DeviceLUTWrapper impl;
};

DeviceLUTWrapper::DeviceLUTWrapper()
    : p_impl(new DeviceLUTWrapperImpl()), dlut_ptr(nullptr)
{}

DeviceLUTWrapper::~DeviceLUTWrapper() {
    delete p_impl;
}

bool init_device_lut(const RLUT& cpu_lut, DeviceLUTWrapper& wrapper) {
    // Cast from sm_2d::RLUT to ::RLUT (same type, different namespace)
    const ::RLUT& native_lut = reinterpret_cast<const ::RLUT&>(cpu_lut);
    bool result = wrapper.p_impl->impl.init(native_lut);
    if (result) {
        wrapper.dlut_ptr = reinterpret_cast<const DeviceRLUT*>(&wrapper.p_impl->impl.dlut);
    }
    return result;
}

void run_gpu_transport(
    float x0, float z0, float theta0, float E0, float W_total,
    int n_particles,
    int Nx, int Nz, float dx, float dz,
    float x_min, float z_min,
    const DeviceRLUT& dlut,
    std::vector<std::vector<double>>& edep
) {
    // Cast from sm_2d::DeviceRLUT to ::DeviceRLUT (same type, different namespace)
    const ::DeviceRLUT& native_dlut = reinterpret_cast<const ::DeviceRLUT&>(dlut);
    run_simple_gpu_transport(
        x0, z0, theta0, E0, W_total,
        n_particles,
        Nx, Nz, dx, dz,
        x_min, z_min,
        native_dlut,
        edep
    );
}

std::string get_gpu_name_internal() {
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);
    return prop.name;
}

} // namespace sm_2d
