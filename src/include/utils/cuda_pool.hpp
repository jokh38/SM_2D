#pragma once
#include <vector>
#include <stack>

#ifdef MOCK_CUDA
#include "cuda_runtime_mock.hpp"
#else
#include <cuda_runtime.h>
#endif

class CudaPool {
public:
    explicit CudaPool(size_t block_size);
    ~CudaPool();
    void* allocate();
    void deallocate(void* ptr);
    size_t total_memory() const { return blocks_.size() * block_size_; }
private:
    size_t block_size_;
    std::vector<void*> blocks_;
    std::stack<void*> free_list_;
};