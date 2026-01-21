#include <gtest/gtest.h>
#include "utils/cuda_pool.hpp"

TEST(CudaPoolTest, AllocateDeallocate) {
    CudaPool pool(1024);
    void* ptr1 = pool.allocate();
    ASSERT_NE(ptr1, nullptr);
    pool.deallocate(ptr1);
}

TEST(CudaPoolTest, MultipleAllocations) {
    CudaPool pool(1024);
    void* ptr1 = pool.allocate();
    void* ptr2 = pool.allocate();
    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr1, ptr2);
    pool.deallocate(ptr1);
    pool.deallocate(ptr2);
}