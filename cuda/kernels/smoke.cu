#include <cuda_runtime.h>

__global__ void smoke_test_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = 1.0f;
    }
}