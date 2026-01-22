#include <gtest/gtest.h>
#include <cuda_runtime.h>

// Define kernels at file scope, not inside TEST functions
__global__ void add_one(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1.0f;
    }
}

TEST(CudaSmokeTest, DeviceAvailable) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0);
}

TEST(CudaSmokeTest, CanAllocate) {
    float* d_ptr = nullptr;
    const size_t N = 1024;
    cudaError_t err = cudaMalloc(&d_ptr, N * sizeof(float));
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_NE(d_ptr, nullptr);
    err = cudaFree(d_ptr);
    EXPECT_EQ(err, cudaSuccess);
}

TEST(CudaSmokeTest, SimpleKernel) {
    const int N = 256;
    float* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));
    add_one<<<1, 256>>>(d_data, N);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    float h_data[256] = {0};
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(h_data[i], 1.0f);
    }
    cudaFree(d_data);
}
