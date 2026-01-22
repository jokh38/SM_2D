#pragma once
#include <cstdlib>
#include <cstring>
#include <cstddef>

// Mock CUDA runtime for environments without CUDA
// This allows the codebase to compile and link for testing purposes

namespace cuda {

// CUDA error codes
enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorLaunchFailure = 4,
    cudaErrorDeviceUninit = 5
};

// CUDA memory copy directions
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

// Mock device query
inline cudaError_t cudaGetDeviceCount(int* count) {
    if (!count) return cudaErrorInvalidValue;
    *count = 1;  // Report 1 mock device
    return cudaSuccess;
}

// Mock memory allocation - uses malloc internally
inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!devPtr) return cudaErrorInvalidValue;
    *devPtr = malloc(size);
    return (*devPtr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

// Mock memory free
inline cudaError_t cudaFree(void* devPtr) {
    if (devPtr) {
        free(devPtr);
    }
    return cudaSuccess;
}

// Mock memory set
inline cudaError_t cudaMemset(void* devPtr, int value, size_t size) {
    if (!devPtr) return cudaErrorInvalidValue;
    memset(devPtr, value, size);
    return cudaSuccess;
}

// Mock memory copy
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if (!dst || !src) return cudaErrorInvalidValue;
    memcpy(dst, src, count);
    return cudaSuccess;
}

// Mock memory info - returns realistic values
inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    if (!free || !total) return cudaErrorInvalidValue;
    *total = 8ULL * 1024 * 1024 * 1024;  // 8 GB total
    *free = 7ULL * 1024 * 1024 * 1024;   // 7 GB free
    return cudaSuccess;
}

// Mock device synchronize
inline cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

// Mock get last error
inline cudaError_t cudaGetLastError() {
    return cudaSuccess;
}

// Mock device set
inline cudaError_t cudaSetDevice(int device) {
    return (device >= 0 && device < 1) ? cudaSuccess : cudaErrorInvalidValue;
}

// Mock device get
inline cudaError_t cudaGetDevice(int* device) {
    if (!device) return cudaErrorInvalidValue;
    *device = 0;
    return cudaSuccess;
}

} // namespace cuda

// Pull into global namespace for compatibility
using cuda::cudaError_t;
using cuda::cudaSuccess;
using cuda::cudaErrorInvalidValue;
using cuda::cudaErrorMemoryAllocation;
using cuda::cudaErrorInitializationError;
using cuda::cudaErrorLaunchFailure;
using cuda::cudaMemcpyKind;
using cuda::cudaMemcpyHostToHost;
using cuda::cudaMemcpyHostToDevice;
using cuda::cudaMemcpyDeviceToHost;
using cuda::cudaMemcpyDeviceToDevice;

inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
    return cuda::cudaMalloc(devPtr, size);
}

inline cudaError_t cudaFree(void* devPtr) {
    return cuda::cudaFree(devPtr);
}

inline cudaError_t cudaMemset(void* devPtr, int value, size_t size) {
    return cuda::cudaMemset(devPtr, value, size);
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    return cuda::cudaMemcpy(dst, src, count, kind);
}

inline cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    return cuda::cudaMemGetInfo(free, total);
}

inline cudaError_t cudaDeviceSynchronize() {
    return cuda::cudaDeviceSynchronize();
}

inline cudaError_t cudaGetLastError() {
    return cuda::cudaGetLastError();
}

inline cudaError_t cudaGetDeviceCount(int* count) {
    return cuda::cudaGetDeviceCount(count);
}

inline cudaError_t cudaSetDevice(int device) {
    return cuda::cudaSetDevice(device);
}

inline cudaError_t cudaGetDevice(int* device) {
    return cuda::cudaGetDevice(device);
}

// Get error string (global namespace only to avoid ambiguity)
inline const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInitializationError: return "initialization error";
        case cudaErrorLaunchFailure: return "launch failure";
        default: return "unknown error";
    }
}
