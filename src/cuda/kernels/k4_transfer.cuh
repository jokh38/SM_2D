#pragma once
#include "core/buckets.hpp"
#include "core/psi_storage.hpp"
#include "device/device_bucket.cuh"
#include <cstdint>

// P3 FIX: Updated K4 header with device bucket support
__global__ void K4_BucketTransfer(
    const DeviceOutflowBucket* __restrict__ OutflowBuckets,
    float* __restrict__ values_out,
    uint32_t* __restrict__ block_ids_out,
    int Nx, int Nz
);

// Debug counters for bucket-transfer slot allocation failures in K4.
void k4_reset_debug_counters();
void k4_get_debug_counters(
    unsigned long long& slot_drop_count,
    double& slot_drop_weight
);

// CPU wrapper
void run_K4_BucketTransfer(
    const DeviceOutflowBucket* buckets,
    PsiC& psi_out,
    int cell,
    int face
);

// Helper: Create device bucket array
// Allocates and initializes OutflowBuckets on GPU
struct DeviceBucketArray {
    DeviceOutflowBucket* d_buckets;
    int num_buckets;  // Nx * Nz * 4

    DeviceBucketArray() : d_buckets(nullptr), num_buckets(0) {}

    bool init(int Nx, int Nz) {
        num_buckets = Nx * Nz * 4;
        size_t bytes = num_buckets * sizeof(DeviceOutflowBucket);

        cudaMalloc(&d_buckets, bytes);

        // Initialize all buckets to empty
        // Note: Need to launch a kernel or use cudaMemset
        cudaMemset(d_buckets, 0xFF, sizeof(uint32_t) * num_buckets * DEVICE_Kb_out);  // block_id
        cudaMemset(d_buckets, 0, sizeof(uint16_t) * num_buckets * DEVICE_Kb_out);     // local_count
        cudaMemset(d_buckets, 0, sizeof(float) * num_buckets * DEVICE_Kb_out * DEVICE_LOCAL_BINS);  // value

        return true;
    }

    void cleanup() {
        if (d_buckets) cudaFree(d_buckets);
        d_buckets = nullptr;
        num_buckets = 0;
    }

    ~DeviceBucketArray() {
        cleanup();
    }
};
