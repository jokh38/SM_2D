# Create test executable with CUDA enabled
add_executable(sm2d_tests
    unit/test_build_system.cpp
    unit/test_cuda_smoke.cpp
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
    kernels/test_k1_activemask.cpp
    kernels/test_k3_finetransport.cpp
    source/test_pencil.cpp
    source/test_gaussian.cpp
    boundary/test_boundaries.cpp
    boundary/test_loss_tracking.cpp
    audit/test_cell_weight_audit.cpp
    audit/test_cell_energy_audit.cpp
    audit/test_global_budget.cpp
    audit/test_reporting.cpp
    validation/test_pencil_beam.cpp
    validation/test_bragg_peak.cpp
    validation/test_lateral_spread.cpp
    validation/test_determinism.cpp
    validation/test_validation_report.cpp
    perf/test_memory_profiling.cpp
    perf/test_kernel_profiling.cpp
    perf/test_occupancy.cpp
)

# Set test property to treat .cpp files with CUDA kernel calls as CUDA
set_source_files_properties(
    unit/test_cuda_smoke.cpp
    kernels/test_k1_activemask.cpp
    kernels/test_k3_finetransport.cpp
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

# Add CUDA include directories for tests
target_include_directories(sm2d_tests PRIVATE /usr/local/cuda-12.4/include)

include(GoogleTest)
gtest_discover_tests(sm2d_tests)
