# Test files common to GPU build
set(TEST_COMMON_FILES
    unit/test_build_system.cpp
    unit/test_memory_budget.cpp
    unit/test_logging.cpp
    unit/test_memory_pool.cpp
    unit/test_grids.cpp
    unit/test_r_lut.cpp
    unit/test_local_bins.cpp
    unit/test_block_encoding.cpp
    unit/test_psi_storage.cpp
    unit/test_buckets.cpp
    unit/test_step_control.cpp
    unit/test_highland.cpp
    unit/test_nuclear.cpp
    boundary/test_boundaries.cpp
    boundary/test_loss_tracking.cpp
    audit/test_cell_weight_audit.cpp
    audit/test_cell_energy_audit.cpp
    audit/test_global_budget.cpp
    audit/test_reporting.cpp
    perf/test_memory_profiling.cpp
)

# CUDA-specific test files
set(TEST_CUDA_FILES
    unit/test_cuda_smoke.cpp
    kernels/test_k1_activemask.cpp
    kernels/test_k3_finetransport.cpp
    perf/test_kernel_profiling.cpp
    perf/test_occupancy.cpp
    gpu/test_energy_loss_only_gpu.cu
)

# GPU-only test target: always include CUDA tests.
add_executable(sm2d_tests
    ${TEST_COMMON_FILES}
    ${TEST_CUDA_FILES}
)

# Set test property to treat .cpp files with CUDA kernel calls as CUDA
set_source_files_properties(
    unit/test_cuda_smoke.cpp
    kernels/test_k1_activemask.cpp
    kernels/test_k3_finetransport.cpp
    gpu/test_energy_loss_only_gpu.cu
    PROPERTIES LANGUAGE CUDA
)

target_link_libraries(sm2d_tests
    sm2d_core
    sm2d_cuda
    sm2d_impl
    sm2d_kernels
    GTest::gtest_main
    GTest::gtest
)

target_include_directories(sm2d_tests PRIVATE ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/include)

include(GoogleTest)
gtest_discover_tests(sm2d_tests)
