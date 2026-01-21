#include "utils/cuda_pool.hpp"
#include <iostream>

CudaPool::CudaPool(size_t block_size) : block_size_(block_size) {}

CudaPool::~CudaPool() {
    for (void* block : blocks_) {
        cudaFree(block);
    }
}

void* CudaPool::allocate() {
    if (!free_list_.empty()) {
        void* ptr = free_list_.top();
        free_list_.pop();
        return ptr;
    }

    void* block = nullptr;
    cudaError_t err = cudaMalloc(&block, block_size_);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    blocks_.push_back(block);
    return block;
}

void CudaPool::deallocate(void* ptr) {
    if (ptr) {
        free_list_.push(ptr);
    }
}