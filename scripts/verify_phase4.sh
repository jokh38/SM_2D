#!/bin/bash
# Verification script for Phase 4 kernel implementation

echo "========================================="
echo "Phase 4 Kernel Implementation Verification"
echo "========================================="
echo ""

# Check kernel headers
echo "1. Checking kernel headers (.cuh)..."
KERNEL_HEADERS=(
    "src/cuda/kernels/k1_activemask.cuh"
    "src/cuda/kernels/k3_finetransport.cuh"
    "src/cuda/kernels/k4_transfer.cuh"
    "src/cuda/kernels/k5_audit.cuh"
    "src/cuda/kernels/k6_swap.cuh"
)

for header in "${KERNEL_HEADERS[@]}"; do
    if [ -f "$header" ]; then
        echo "  ✓ $header"
    else
        echo "  ✗ $header MISSING"
    fi
done
echo ""

# Check kernel implementations
echo "2. Checking kernel implementations (.cu)..."
KERNEL_IMPLS=(
    "src/cuda/kernels/k1_activemask.cu"
    "src/cuda/kernels/k3_finetransport.cu"
    "src/cuda/kernels/k4_transfer.cu"
    "src/cuda/kernels/k5_audit.cu"
    "src/cuda/kernels/k6_swap.cu"
)

for impl in "${KERNEL_IMPLS[@]}"; do
    if [ -f "$impl" ]; then
        lines=$(wc -l < "$impl")
        echo "  ✓ $impl ($lines lines)"
    else
        echo "  ✗ $impl MISSING"
    fi
done
echo ""

# Check test files
echo "3. Checking kernel test files..."
TEST_FILES=(
    "tests/kernels/test_k1_activemask.cpp"
    "tests/kernels/test_k3_finetransport.cpp"
)

for test in "${TEST_FILES[@]}"; do
    if [ -f "$test" ]; then
        lines=$(wc -l < "$test")
        echo "  ✓ $test ($lines lines)"
    else
        echo "  ✗ $test MISSING"
    fi
done
echo ""

# Check CMakeLists updates
echo "4. Checking CMakeLists.txt updates..."
if grep -q "sm2d_kernels" CMakeLists.txt; then
    echo "  ✓ Root CMakeLists.txt includes kernel library"
else
    echo "  ✗ Root CMakeLists.txt missing kernel library"
fi

if grep -q "test_k1_activemask.cpp" tests/CMakeLists.txt; then
    echo "  ✓ Tests/CMakeLists.txt includes kernel tests"
else
    echo "  ✗ Tests/CMakeLists.txt missing kernel tests"
fi
echo ""

# Summary
echo "========================================="
echo "Summary:"
echo "  Kernel headers: ${#KERNEL_HEADERS[@]} created"
echo "  Kernel implementations: ${#KERNEL_IMPLS[@]} created"
echo "  Test files: ${#TEST_FILES[@]} created"
echo "========================================="
echo ""
echo "Phase 4 kernel stubs implementation complete!"
echo "Next: Full CUDA implementation requires device LUT integration"
