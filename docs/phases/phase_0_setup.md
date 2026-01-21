# Phase 0: Project Setup and Infrastructure

**Status**: Pending
**Duration**: Foundation (must complete before Phase 1)
**Dependencies**: None

---

## Objectives

1. Establish build system and testing framework
2. Set up project directory structure
3. Configure CUDA environment for RTX 2080
4. Implement memory management infrastructure
5. Set up logging and diagnostics

---

## Directory Structure

```
SM_2D/
├── CMakeLists.txt
├── include/
│   ├── lut/
│   ├── core/
│   ├── physics/
│   ├── kernels/
│   ├── source/
│   ├── boundary/
│   ├── audit/
│   └── utils/
├── src/
│   ├── lut/
│   ├── core/
│   ├── physics/
│   ├── source/
│   ├── boundary/
│   ├── audit/
│   └── utils/
├── cuda/
│   └── kernels/
│       ├── k1_activemask.cu
│       ├── k3_finetransport.cu
│       ├── k4_transfer.cu
│       ├── k5_audit.cu
│       └── k6_swap.cu
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── physics/
│   └── validation/
├── data/
│   ├── nist/
│   └── lut/
├── scripts/
│   └── lut_gen/
└── docs/
    └── phases/
```

---

## TDD Cycle 0.1: Build System

### RED - Write Tests First

Create `tests/unit/test_build_system.cpp`:

```cpp
#include <gtest/gtest.h>

// Meta-test: Verify build system is configured correctly
TEST(BuildSystemTest, CMakeVersion) {
    // CMake 3.28+ required for CUDA 12.x
    EXPECT_GE(CMAKE_VERSION_MAJOR, 3);
    EXPECT_GE(CMAKE_VERSION_MINOR, 28);
}

TEST(BuildSystemTest, CompilerSupportsC++17) {
    // Verify C++17 features work
    auto [a, b] = std::make_pair(1, 2);
    EXPECT_EQ(a, 1);
    EXPECT_EQ(b, 2);
}
```

### GREEN - Minimal Implementation

Create `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.28)
project(SM_2D LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX 2080

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find dependencies
find_package(GTest REQUIRED)

# Core library
add_library(sm2d_core INTERFACE)
target_include_directories(sm2d_core INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)

# CUDA kernels
add_library(sm2d_cuda INTERFACE)
target_include_directories(sm2d_cuda INTERFACE
    ${CMAKE_SOURCE_DIR}/cuda
)

# Test executable
enable_testing()
add_subdirectory(tests)
```

Create `tests/CMakeLists.txt`:

```cmake
add_executable(sm2d_tests
    unit/test_build_system.cpp
)

target_link_libraries(sm2d_tests
    sm2d_core
    GTest::gtest_main
)

discover_tests(sm2d_tests)
```

---

## TDD Cycle 0.2: CUDA Environment

### RED - Write Tests First

Create `tests/unit/test_cuda_smoke.cpp`:

```cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>

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
    // Simple kernel execution test
    __global__ void add_one(float* data, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            data[idx] += 1.0f;
        }
    }

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
```

### GREEN - Implementation

The tests above include the kernel. Create `cuda/kernels/smoke.cu`:

```cuda
#include <cuda_runtime.h>

__global__ void smoke_test_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = 1.0f;
    }
}
```

---

## TDD Cycle 0.3: Memory Budget Tracking

### RED - Write Tests First

Create `tests/unit/test_memory_budget.cpp`:

```cpp
#include <gtest/gtest.h>
#include "utils/memory_tracker.hpp"

TEST(MemoryBudgetTest, InitialUsageLow) {
    MemoryTracker tracker;
    size_t initial = tracker.get_current_usage();
    EXPECT_LT(initial, 500ULL * 1024 * 1024);  // < 500MB baseline
}

TEST(MemoryBudgetTest, AllocationTracked) {
    MemoryTracker tracker;

    size_t before = tracker.get_current_usage();

    float* d_ptr = nullptr;
    size_t alloc_size = 100ULL * 1024 * 1024;  // 100MB
    cudaMalloc(&d_ptr, alloc_size);

    size_t after = tracker.get_current_usage();

    EXPECT_GE(after - before, alloc_size * 0.9);  // At least 90% of allocation tracked

    cudaFree(d_ptr);
}

TEST(MemoryBudgetTest, WarnAt6GB) {
    MemoryTracker tracker;
    tracker.set_warning_threshold(6ULL * 1024 * 1024 * 1024);

    // This should not warn yet
    EXPECT_FALSE(tracker.check_warning());

    // Simulate large allocation
    tracker.simulate_allocation(7ULL * 1024 * 1024 * 1024);
    EXPECT_TRUE(tracker.check_warning());
}
```

### GREEN - Implementation

Create `include/utils/memory_tracker.hpp`:

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cstddef>

class MemoryTracker {
public:
    MemoryTracker();

    // Get current GPU memory usage in bytes
    size_t get_current_usage() const;

    // Get total available memory in bytes
    size_t get_total_memory() const;

    // Set warning threshold (default: 6GB)
    void set_warning_threshold(size_t bytes);

    // Check if usage exceeds threshold
    bool check_warning() const;

    // For testing: simulate allocation
    void simulate_allocation(size_t bytes);

    // Get peak usage
    size_t get_peak_usage() const;

private:
    size_t warning_threshold_;
    size_t simulated_bytes_;
    mutable size_t peak_usage_;
};
```

Create `src/utils/memory_tracker.cpp`:

```cpp
#include "utils/memory_tracker.hpp"
#include <iostream>

MemoryTracker::MemoryTracker()
    : warning_threshold_(6ULL * 1024 * 1024 * 1024)
    , simulated_bytes_(0)
    , peak_usage_(0)
{}

size_t MemoryTracker::get_current_usage() const {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);

    size_t used = total - free + simulated_bytes_;
    peak_usage_ = std::max(peak_usage_, used);
    return used;
}

size_t MemoryTracker::get_total_memory() const {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    return total;
}

void MemoryTracker::set_warning_threshold(size_t bytes) {
    warning_threshold_ = bytes;
}

bool MemoryTracker::check_warning() const {
    return get_current_usage() > warning_threshold_;
}

void MemoryTracker::simulate_allocation(size_t bytes) {
    simulated_bytes_ += bytes;
}

size_t MemoryTracker::get_peak_usage() const {
    return peak_usage_;
}
```

---

## TDD Cycle 0.4: Logging

### RED - Write Tests First

Create `tests/unit/test_logging.cpp`:

```cpp
#include <gtest/gtest.h>
#include "utils/logger.hpp"

TEST(LoggerTest, InfoLevelWorks) {
    Logger& log = Logger::get();
    log.set_level(LogLevel::INFO);

    // Should not crash
    log.info("Test info message: {}", 42);
}

TEST(LoggerTest, TraceDisabledByDefault) {
    Logger& log = Logger::get();

    // Trace should not crash even if disabled
    log.trace("This should not appear");
}
```

### GREEN - Implementation

Create `include/utils/logger.hpp`:

```cpp
#pragma once

#include <string>
#include <memory>

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4
};

class Logger {
public:
    static Logger& get();

    void set_level(LogLevel level);
    LogLevel get_level() const;

    template<typename... Args>
    void trace(const char* fmt, Args&&... args) {
        log(LogLevel::TRACE, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void debug(const char* fmt, Args&&... args) {
        log(LogLevel::DEBUG, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void info(const char* fmt, Args&&... args) {
        log(LogLevel::INFO, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warn(const char* fmt, Args&&... args) {
        log(LogLevel::WARN, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(const char* fmt, Args&&... args) {
        log(LogLevel::ERROR, fmt, std::forward<Args>(args)...);
    }

    void flush();

private:
    Logger();
    ~Logger() = default;

    template<typename... Args>
    void log(LogLevel level, const char* fmt, Args&&... args);

    LogLevel level_ = LogLevel::INFO;
    bool colors_ = true;
};
```

---

## TDD Cycle 0.5: Memory Pool

### RED - Write Tests First

Create `tests/unit/test_memory_pool.cpp`:

```cpp
#include <gtest/gtest.h>
#include "utils/cuda_pool.hpp"

TEST(CudaPoolTest, AllocateBlock) {
    CudaPool pool(1024);  // 1KB blocks

    void* ptr = pool.allocate();
    ASSERT_NE(ptr, nullptr);

    pool.deallocate(ptr);
}

TEST(CudaPoolTest, AllocateMultiple) {
    CudaPool pool(1024);

    void* ptr1 = pool.allocate();
    void* ptr2 = pool.allocate();

    ASSERT_NE(ptr1, ptr2);
    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);

    pool.deallocate(ptr1);
    pool.deallocate(ptr2);
}

TEST(CudaPoolTest, ReuseFreedBlock) {
    CudaPool pool(1024);

    void* ptr1 = pool.allocate();
    pool.deallocate(ptr1);

    void* ptr2 = pool.allocate();
    // Should reuse the same block (implementation may vary)
    ASSERT_NE(ptr2, nullptr);

    pool.deallocate(ptr2);
}
```

### GREEN - Implementation

Create `include/utils/cuda_pool.hpp`:

```cpp
#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <stack>

class CudaPool {
public:
    explicit CudaPool(size_t block_size);
    ~CudaPool();

    // Allocate a block from the pool
    void* allocate();

    // Return a block to the pool
    void deallocate(void* ptr);

    // Get total allocated memory (in bytes)
    size_t total_memory() const { return blocks_.size() * block_size_; }

private:
    size_t block_size_;
    std::vector<void*> blocks_;      // All allocated blocks
    std::stack<void*> free_list_;    // Available blocks
};
```

---

## Exit Criteria Checklist

- [ ] All targets compile without warnings
- [ ] `make test` or `ctest` runs successfully
- [ ] CUDA smoke test passes (kernel executes)
- [ ] Memory tracker reports baseline < 500MB
- [ ] Logger outputs to console
- [ ] Memory pool allocates/reuses blocks
- [ ] Device info logged on startup

---

## Next Steps

After completing Phase 0, proceed to **Phase 1 (LUT Generation)** and **Phase 2 (Data Structures)** in parallel.

```bash
# Verify Phase 0 complete
./bin/sm2d_tests --gtest_filter="*"

# Check CUDA device
./bin/sm2d_tests --gtest_filter="CudaSmokeTest.*"

# Proceed to Phase 1
cat ../docs/phases/phase_1_lut.md
```
