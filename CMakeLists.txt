cmake_minimum_required(VERSION 3.20)

# Check if nvcc exists before trying to use CUDA
find_program(NVCC_EXECUTABLE nvcc)

if(NVCC_EXECUTABLE)
    # Try to find CUDA toolkit
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        project(SM_2D LANGUAGES CXX CUDA)
        message(STATUS "CUDA found - building with GPU support")
        set(CUDA_ENABLED TRUE)
    else()
        project(SM_2D LANGUAGES CXX)
        message(STATUS "nvcc found but CUDA toolkit not complete - building CPU-only version")
        set(CUDA_ENABLED FALSE)
    endif()
else()
    project(SM_2D LANGUAGES CXX)
    message(STATUS "nvcc not found - building CPU-only version")
    set(CUDA_ENABLED FALSE)
endif()

set(CMAKE_CXX_STANDARD 17)

if(CUDA_ENABLED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX 2080
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    add_compile_definitions(SM2D_HAS_CUDA)
endif()

# Find dependencies
find_package(GTest REQUIRED)

# Core library
add_library(sm2d_core INTERFACE)
target_include_directories(sm2d_core INTERFACE
    ${CMAKE_SOURCE_DIR}/src/include
)

# CUDA kernels
add_library(sm2d_cuda INTERFACE)
target_include_directories(sm2d_cuda INTERFACE
    ${CMAKE_SOURCE_DIR}/src/cuda
)

# Implementation sources (CPU-only - always compiled)
set(IMPL_CPU_SOURCES
    src/utils/logger.cpp
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
    src/validation/deterministic_beam.cpp
    src/validation/bragg_peak.cpp
    src/validation/lateral_spread.cpp
    src/validation/determinism.cpp
    src/validation/validation_report.cpp
    src/gpu/gpu_transport_runner.cpp
)

# Implementation sources (CUDA-specific - only compiled when CUDA is available)
set(IMPL_CUDA_SOURCES
    src/utils/memory_tracker.cpp
    src/utils/cuda_pool.cpp
    src/perf/memory_profiler.cpp
    src/perf/kernel_profiler.cpp
    src/perf/occupancy_analyzer.cpp
)

# Build implementation library
if(CUDA_ENABLED)
    add_library(sm2d_impl STATIC ${IMPL_CPU_SOURCES} ${IMPL_CUDA_SOURCES})
    target_include_directories(sm2d_impl BEFORE PRIVATE /usr/local/cuda-12.4/include)
    target_include_directories(sm2d_impl PRIVATE ${CMAKE_SOURCE_DIR}/src/cuda)
    target_compile_definitions(sm2d_impl PRIVATE SM2D_HAS_CUDA)
    # Add CUDA include to core for headers that need it
    target_include_directories(sm2d_core INTERFACE /usr/local/cuda-12.4/include)
else()
    add_library(sm2d_impl STATIC ${IMPL_CPU_SOURCES})
endif()

target_link_libraries(sm2d_impl sm2d_core sm2d_cuda)

# CUDA kernel sources (only if CUDA is available)
if(CUDA_ENABLED)
    set(CUDA_KERNEL_SOURCES
        src/cuda/kernels/k1_activemask.cu
        src/cuda/kernels/k2_coarsetransport.cu
        src/cuda/kernels/k3_finetransport.cu
        src/cuda/kernels/k4_transfer.cu
        src/cuda/kernels/k5_audit.cu
        src/cuda/kernels/k6_swap.cu
        src/cuda/gpu_transport_wrapper.cu
        src/cuda/k1k6_pipeline.cu
    )

    # Create CUDA object library for kernel files
    add_library(sm2d_kernels OBJECT ${CUDA_KERNEL_SOURCES})
    target_include_directories(sm2d_kernels PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/src/cuda
    )
    set_property(TARGET sm2d_kernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)
else()
    # Dummy library for CUDA when not available
    add_library(sm2d_kernels INTERFACE)
endif()

# Main simulation executable
add_executable(run_simulation run_simulation.cpp)
target_link_libraries(run_simulation sm2d_core sm2d_impl sm2d_kernels)

# Copy executable to project root after build
add_custom_command(TARGET run_simulation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:run_simulation> ${CMAKE_SOURCE_DIR}/run_simulation
    COMMENT "Copying run_simulation to project root"
)

# Test executable
enable_testing()
add_subdirectory(tests)
