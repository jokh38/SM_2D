# Phase 8: Performance Optimization

**Status**: Pending
**Duration**: 2-3 days (parallel with Phase 7)
**Dependencies**: Phase 4 (Kernels), Phase 6 (Audit)

---

## Objectives

1. Profile kernel execution times
2. Optimize memory access patterns
3. Implement occupancy tuning
4. Verify memory budget compliance
5. Document performance characteristics

---

## Target Hardware

- **GPU**: RTX 2080 (Turing, sm_75)
- **VRAM**: 8GB total
- **Memory Budget**: < 7GB peak (leaves 1GB headroom)

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Peak memory usage | < 7GB | `nvidia-smi` during run |
| K3 kernel time | < 100ms/step | `nvprof` / `nsys` |
| Occupancy (K3) | > 40% | Nsight Compute |
| Memory bandwidth | > 70% | Nsight Compute |

---

## TDD Cycle 8.1: Memory Profiling

### RED - Write Tests First

Create `tests/perf/test_memory_profiling.cpp`:

```cpp
#include <gtest/gtest.h>
#include "perf/memory_profiler.hpp"

TEST(MemoryProfiler, BaselineRecorded) {
    MemoryProfiler profiler;

    profiler.start();

    // Allocate and use some buffers
    PsiC psi(32, 32, 32);

    profiler.stop();

    EXPECT_GT(profiler.get_peak_bytes(), 0);
    EXPECT_LT(profiler.get_peak_bytes(), 8ULL * 1024 * 1024 * 1024);  // < 8GB
}

TEST(MemoryProfiler, BudgetCheck) {
    MemoryProfiler profiler;
    profiler.set_budget(7ULL * 1024 * 1024 * 1024);  // 7GB

    profiler.start();

    // Run simulation
    PencilBeamConfig config;
    config.E0 = 150.0f;
    auto result = run_pencil_beam(config);

    profiler.stop();

    MemoryBudget status = profiler.check_budget();

    if (status.over_budget) {
        FAIL() << "Memory budget exceeded: " << status.peak_bytes << " > " << status.budget;
    }

    EXPECT_FALSE(status.over_budget);
}
```

### GREEN - Implementation

Create `include/perf/memory_profiler.hpp`:

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct MemoryBudget {
    uint64_t budget;
    uint64_t peak_bytes;
    bool over_budget;
};

class MemoryProfiler {
public:
    MemoryProfiler();

    void set_budget(uint64_t bytes);
    void start();
    void stop();

    uint64_t get_peak_bytes() const;
    uint64_t get_current_bytes() const;
    MemoryBudget check_budget() const;

    void print_report() const;

private:
    uint64_t budget_;
    uint64_t baseline_;
    uint64_t peak_;
    bool running_;
};
```

Create `src/perf/memory_profiler.cpp`:

```cpp
#include "perf/memory_profiler.hpp"
#include <iostream>
#include <algorithm>

MemoryProfiler::MemoryProfiler()
    : budget_(7ULL * 1024 * 1024 * 1024)
    , baseline_(0)
    , peak_(0)
    , running_(false)
{}

void MemoryProfiler::set_budget(uint64_t bytes) {
    budget_ = bytes;
}

void MemoryProfiler::start() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    baseline_ = total - free;
    peak_ = baseline_;
    running_ = true;
}

void MemoryProfiler::stop() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    uint64_t current = total - free;
    peak_ = std::max(peak_, current);
    running_ = false;
}

uint64_t MemoryProfiler::get_peak_bytes() const {
    return peak_;
}

uint64_t MemoryProfiler::get_current_bytes() const {
    if (running_) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        return total - free;
    }
    return peak_;
}

MemoryBudget MemoryProfiler::check_budget() const {
    MemoryBudget status;
    status.budget = budget_;
    status.peak_bytes = peak_;
    status.over_budget = peak_ > budget_;
    return status;
}

void MemoryProfiler::print_report() const {
    double peak_gb = peak_ / (1024.0 * 1024.0 * 1024.0);
    double budget_gb = budget_ / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Memory Report:\n";
    std::cout << "  Budget: " << budget_gb << " GB\n";
    std::cout << "  Peak: " << peak_gb << " GB\n";

    if (peak_ > budget_) {
        std::cout << "  Status: OVER BUDGET\n";
    } else {
        std::cout << "  Status: OK\n";
        double headroom_gb = (budget_ - peak_) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  Headroom: " << headroom_gb << " GB\n";
    }
}
```

---

## TDD Cycle 8.2: Kernel Profiling

### RED - Write Tests First

Create `tests/perf/test_kernel_profiling.cpp`:

```cpp
#include <gtest/gtest.h>
#include "perf/kernel_profiler.hpp"

TEST(KernelProfiler, TimesRecorded) {
    KernelProfiler profiler;

    profiler.start("K3_FineTransport");

    // Run kernel
    PencilBeamConfig config;
    config.E0 = 150.0f;
    auto result = run_pencil_beam(config);

    profiler.stop("K3_FineTransport");

    EXPECT_GT(profiler.get_time_ms("K3_FineTransport"), 0);
}

TEST(KernelProfiler, MultipleKernels) {
    KernelProfiler profiler;

    profiler.start("K1_ActiveMask");
    run_K1_ActiveMask(/* ... */);
    profiler.stop("K1_ActiveMask");

    profiler.start("K3_FineTransport");
    run_K3_FineTransport(/* ... */);
    profiler.stop("K3_FineTransport");

    profiler.start("K4_BucketTransfer");
    run_K4_BucketTransfer(/* ... */);
    profiler.stop("K4_BucketTransfer");

    EXPECT_GT(profiler.get_time_ms("K1_ActiveMask"), 0);
    EXPECT_GT(profiler.get_time_ms("K3_FineTransport"), 0);
    EXPECT_GT(profiler.get_time_ms("K4_BucketTransfer"), 0);

    profiler.print_report();
}

TEST(KernelProfiler, K3PerformanceTarget) {
    KernelProfiler profiler;

    profiler.start("K3_FineTransport");

    PencilBeamConfig config;
    config.E0 = 150.0f;
    auto result = run_pencil_beam(config);

    profiler.stop("K3_FineTransport");

    float k3_time = profiler.get_time_ms("K3_FineTransport");

    EXPECT_LT(k3_time, 100.0f);  // < 100ms target
}
```

### GREEN - Implementation

Create `include/perf/kernel_profiler.hpp`:

```cpp
#pragma once

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <chrono>

class KernelProfiler {
public:
    void start(const std::string& kernel_name);
    void stop(const std::string& kernel_name);

    float get_time_ms(const std::string& kernel_name) const;
    float get_total_time_ms() const;

    void print_report() const;

private:
    std::unordered_map<std::string, float> times_ms_;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> starts_;
    float total_time_ms_ = 0;
};
```

Create `src/perf/kernel_profiler.cpp`:

```cpp
#include "perf/kernel_profiler.hpp"
#include <iostream>
#include <iomanip>

void KernelProfiler::start(const std::string& kernel_name) {
    starts_[kernel_name] = std::chrono::high_resolution_clock::now();
}

void KernelProfiler::stop(const std::string& kernel_name) {
    auto it = starts_.find(kernel_name);
    if (it == starts_.end()) return;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - it->second);
    float ms = duration.count() / 1000.0f;

    times_ms_[kernel_name] += ms;
    total_time_ms_ += ms;
}

float KernelProfiler::get_time_ms(const std::string& kernel_name) const {
    auto it = times_ms_.find(kernel_name);
    return (it != times_ms_.end()) ? it->second : 0;
}

float KernelProfiler::get_total_time_ms() const {
    return total_time_ms_;
}

void KernelProfiler::print_report() const {
    std::cout << "\n=== Kernel Performance Report ===\n";

    // Sort by time
    std::vector<std::pair<std::string, float>> sorted;
    for (const auto& kv : times_ms_) {
        sorted.push_back(kv);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& kv : sorted) {
        std::cout << std::setw(20) << kv.first << ": "
                  << std::fixed << std::setprecision(2)
                  << kv.second << " ms";

        if (total_time_ms_ > 0) {
            float pct = 100.0f * kv.second / total_time_ms_;
            std::cout << " (" << pct << "%)";
        }
        std::cout << "\n";
    }

    std::cout << "-----------------------------------\n";
    std::cout << "Total: " << total_time_ms_ << " ms\n";
}
```

---

## Optimization Strategies

### Strategy 1: Occupancy Tuning

**Problem**: Low occupancy due to high register usage

**Solution**:
```cuda
// Reduce register pressure by using shared memory
__shared__ float s_component_data[MAX_COMPS * REGS_PER_COMP];

// Or launch with more threads per block
dim3 block(256);  // Instead of 128
dim3 grid((n_active + 255) / 256);
```

### Strategy 2: Memory Coalescing

**Problem**: Unaligned memory access

**Solution**:
```cuda
// Ensure threadIdx.x accesses contiguous memory
float value = psi.value[cell][slot][threadIdx.x];  // Good

// Bad: strided access
float value = psi.value[cell][threadIdx.x][lidx];  // Bad
```

### Strategy 3: Avoid Divergence

**Problem**: Thread divergence in conditionals

**Solution**:
```cuda
// Instead of per-thread conditionals:
if (w < weight_epsilon) continue;  // Divergence

// Use uniform conditions when possible
if (w < weight_epsilon) {
    // All threads execute same path
    w = 0;
}
```

### Strategy 4: Shared Memory Reduction

**Problem**: Repeated global memory access

**Solution**:
```cuda
__shared__ float s_edep[128];
s_edep[tid] = local_edep;
__syncthreads();

if (tid == 0) {
    float sum = 0;
    for (int i = 0; i < 128; ++i) sum += s_edep[i];
    atomicAdd(&EdepC[cell], sum);
}
```

---

## TDD Cycle 8.3: Occupancy Analysis

### RED - Write Tests First

Create `tests/perf/test_occupancy.cpp`:

```cpp
#include <gtest/gtest.h>
#include "perf/occupancy_analyzer.hpp"

TEST(OccupancyAnalysis, K3OccupancyTarget) {
    OccupancyAnalyzer analyzer;

    analyzer.analyze_kernel("K3_FineTransport");

    float occupancy = analyzer.get_occupancy_percent();

    EXPECT_GT(occupancy, 40.0f);  // > 40% target
}

TEST(OccupancyAnalysis, ReportLimitingFactor) {
    OccupancyAnalyzer analyzer;

    analyzer.analyze_kernel("K3_FineTransport");

    std::string factor = analyzer.get_limiting_factor();

    EXPECT_FALSE(factor.empty());
    std::cout << "Occupancy limiting factor: " << factor << "\n";
}
```

### GREEN - Implementation

Create `include/perf/occupancy_analyzer.hpp`:

```cpp
#pragma once

#include <string>

class OccupancyAnalyzer {
public:
    // Analyze kernel occupancy (requires Nsight Compute or manual calc)
    void analyze_kernel(const std::string& kernel_name);

    float get_occupancy_percent() const;
    std::string get_limiting_factor() const;

private:
    float occupancy_ = 0;
    std::string limiting_factor_;
};
```

---

## Optimization Checklist

### Before Optimization

- [ ] Profile baseline kernel times
- [ ] Measure baseline memory usage
- [ ] Check baseline occupancy
- [ ] Identify hotspots

### Optimization Iterations

- [ ] **Iter 1**: Shared memory for reduction
- [ ] **Iter 2**: Register count reduction (use `maxrregcount`)
- [ ] **Iter 3**: Memory coalescing improvements
- [ ] **Iter 4**: Thread block size tuning
- [ ] **Iter 5**: Loop unrolling (if beneficial)

### After Optimization

- [ ] Verify correctness (all tests still pass)
- [ ] Compare to baseline
- [ ] Document performance gains
- [ ] Update performance targets

---

## Profiling Commands

### nvprof (Deprecated but simple)

```bash
# Basic profiling
nvprof --print-gpu-trace ./bin/sm2d_validation

# Kernel time breakdown
nvprof --analysis-metrics - ./bin/sm2d_validation

# Memory bandwidth
nvprof --metrics dram__throughput.avg.pct_of_peak \
       ./bin/sm2d_validation
```

### Nsight Systems

```bash
# Timeline view
nsys profile --stats=true ./bin/sm2d_validation

# GPU trace
nsys profile --trace=cuda,nvtx ./bin/sm2d_validation
```

### Nsight Compute

```bash
# Kernel analysis
ncu --set full ./bin/sm2d_validation

# Occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak \
    ./bin/sm2d_validation

# Memory bandwidth utilization
ncu --metrics dram__throughput.avg.pct_of_peak \
    ./bin/sm2d_validation
```

---

## Exit Criteria Checklist

- [ ] Peak memory usage < 7GB
- [ ] K3 kernel time documented
- [ ] K3 kernel time < 100ms (or documented if slower)
- [ ] Occupancy > 40% (or bottleneck documented)
- [ ] Performance report generated
- [ ] No obvious memory bottlenecks
- [ ] Optimization opportunities documented

---

## Performance Report Template

```markdown
# SM_2D Performance Report

## Hardware
- GPU: RTX 2080 (Turing)
- Compute Capability: 7.5
- VRAM: 8GB

## Memory Usage
- Baseline: [X] GB
- Peak: [Y] GB
- Headroom: [Z] GB
- Status: OK/OVER

## Kernel Performance (per step)

| Kernel | Time (ms) | % Total | Notes |
|--------|-----------|---------|-------|
| K1_ActiveMask | X | Y% | |
| K3_FineTransport | X | Y% | |
| K4_BucketTransfer | X | Y% | |
| K5_Conservation | X | Y% | |
| **Total** | **X** | **100%** | |

## Occupancy

| Kernel | Occupancy | Limiting Factor |
|--------|-----------|-----------------|
| K3_FineTransport | X% | registers/blocks |

## Recommendations

1. [ ]
2. [ ]
```

---

## Next Steps

After completing Phase 8:

1. All phases complete - system ready for production use
2. Generate final validation report
3. Document known limitations and future work

```bash
# Final performance check
./bin/sm2d_perf

# Full validation
./bin/sm2d_validation --full --report=FINAL_REPORT.txt
```
