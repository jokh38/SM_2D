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

# CUDA kernel sources (separable compilation)
set(CUDA_KERNEL_SOURCES
    cuda/kernels/k1_activemask.cu
    cuda/kernels/k3_finetransport.cu
    cuda/kernels/k4_transfer.cu
    cuda/kernels/k5_audit.cu
    cuda/kernels/k6_swap.cu
)

# Create CUDA object library for kernel files
add_library(sm2d_kernels OBJECT ${CUDA_KERNEL_SOURCES})
target_include_directories(sm2d_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/cuda
)
set_property(TARGET sm2d_kernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Implementation sources
add_library(sm2d_impl STATIC
    src/utils/memory_tracker.cpp
    src/utils/logger.cpp
    src/utils/cuda_pool.cpp
    src/core/grids.cpp
    src/core/psi_storage.cpp
    src/core/buckets.cpp
    src/lut/nist_loader.cpp
    src/lut/r_lut.cpp
    src/source/pencil_source.cpp
    src/source/gaussian_source.cpp
    src/boundary/boundaries.cpp
    src/boundary/loss_tracking.cpp
    src/audit/conservation.cpp
    src/audit/global_budget.cpp
    src/audit/reporting.cpp
    src/validation/pencil_beam.cpp
    src/validation/bragg_peak.cpp
    src/validation/lateral_spread.cpp
    src/validation/determinism.cpp
    src/validation/validation_report.cpp
    src/perf/memory_profiler.cpp
    src/perf/kernel_profiler.cpp
    src/perf/occupancy_analyzer.cpp
)

target_link_libraries(sm2d_impl sm2d_core)

# Test executable
enable_testing()
add_subdirectory(tests)