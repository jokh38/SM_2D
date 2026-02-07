#include "perf/memory_preflight.hpp"

#include "core/local_bins.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace sm_2d {

namespace {

constexpr std::uint64_t kPsicSlotsPerCell = 32;         // DEVICE_Kb
constexpr std::uint64_t kBucketSlotsPerFace = 32;       // DEVICE_Kb_out
constexpr std::uint64_t kBucketFacesPerCell = 4;
constexpr std::uint64_t kRuntimeOverheadBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kAuditReportConservativeBytes = 256ULL;

constexpr std::uint64_t kPipelineBytesPerCell =
    (2ULL * sizeof(std::uint8_t)) +    // ActiveMask, ActiveMask_prev
    (2ULL * sizeof(std::uint32_t)) +   // ActiveList, CoarseList
    (1ULL * sizeof(double)) +          // EdepC
    (1ULL * sizeof(float)) +           // AbsorbedWeight_cutoff
    (1ULL * sizeof(double)) +          // AbsorbedEnergy_cutoff
    (1ULL * sizeof(float)) +           // AbsorbedWeight_nuclear
    (1ULL * sizeof(double)) +          // AbsorbedEnergy_nuclear
    (1ULL * sizeof(float)) +           // BoundaryLoss_weight
    (1ULL * sizeof(double)) +          // BoundaryLoss_energy
    (1ULL * sizeof(float)) +           // prev_AbsorbedWeight_cutoff
    (1ULL * sizeof(double)) +          // prev_AbsorbedEnergy_cutoff
    (1ULL * sizeof(float)) +           // prev_AbsorbedWeight_nuclear
    (1ULL * sizeof(float)) +           // prev_BoundaryLoss_weight
    (1ULL * sizeof(double)) +          // prev_EdepC
    (1ULL * sizeof(double)) +          // prev_AbsorbedEnergy_nuclear
    (1ULL * sizeof(double)) +          // prev_BoundaryLoss_energy
    (2ULL * sizeof(double));           // weight_in, weight_out

} // namespace

DenseMemoryEstimate estimate_dense_k1k6_memory(
    int Nx,
    int Nz,
    int N_theta,
    int N_E
) {
    DenseMemoryEstimate est{};
    if (Nx <= 0 || Nz <= 0 || N_theta <= 0 || N_E <= 0) {
        return est;
    }

    const std::uint64_t n_cells = static_cast<std::uint64_t>(Nx) * static_cast<std::uint64_t>(Nz);
    const std::uint64_t psic_single_cell_bytes =
        kPsicSlotsPerCell * (sizeof(std::uint32_t) + static_cast<std::uint64_t>(LOCAL_BINS) * sizeof(float));
    const std::uint64_t bucket_face_bytes =
        kBucketSlotsPerFace *
            (sizeof(std::uint32_t) + sizeof(std::uint16_t) + static_cast<std::uint64_t>(LOCAL_BINS) * sizeof(float)) +
        (3ULL * sizeof(float));  // Fermi-Eyges moments

    est.psi_buffers_bytes = static_cast<std::size_t>(n_cells * psic_single_cell_bytes * 2ULL);
    est.outflow_buckets_bytes = static_cast<std::size_t>(n_cells * kBucketFacesPerCell * bucket_face_bytes);

    const std::uint64_t pipeline_cell_bytes = n_cells * kPipelineBytesPerCell;
    const std::uint64_t pipeline_scalar_bytes =
        (2ULL * sizeof(int)) +  // d_n_active, d_n_coarse
        kAuditReportConservativeBytes +
        (static_cast<std::uint64_t>(N_theta + 1) * sizeof(float)) +  // d_theta_edges
        (static_cast<std::uint64_t>(N_E + 1) * sizeof(float));       // d_E_edges
    est.pipeline_state_bytes = static_cast<std::size_t>(pipeline_cell_bytes + pipeline_scalar_bytes);

    // DeviceLUTWrapper: R,S,log_E,log_R,log_S arrays + E_edges.
    est.lut_bytes = static_cast<std::size_t>(
        (5ULL * static_cast<std::uint64_t>(N_E) + static_cast<std::uint64_t>(N_E + 1)) * sizeof(float)
    );

    est.runtime_overhead_bytes = static_cast<std::size_t>(kRuntimeOverheadBytes);
    est.bytes_per_dense_cell = static_cast<std::size_t>(
        (2ULL * psic_single_cell_bytes) +
        (kBucketFacesPerCell * bucket_face_bytes) +
        kPipelineBytesPerCell
    );

    est.total_required_bytes =
        est.psi_buffers_bytes +
        est.outflow_buckets_bytes +
        est.pipeline_state_bytes +
        est.lut_bytes +
        est.runtime_overhead_bytes;

    return est;
}

MemoryPreflightResult run_memory_preflight(const MemoryPreflightInput& input) {
    MemoryPreflightResult result{};

    if (input.Nx <= 0 || input.Nz <= 0 || input.N_theta <= 0 || input.N_E <= 0) {
        result.message = "invalid preflight dimensions";
        return result;
    }

    std::size_t free_bytes = input.free_vram_bytes;
    std::size_t total_bytes = input.total_vram_bytes;
    if (free_bytes == 0 || total_bytes == 0) {
        std::size_t queried_free = 0;
        std::size_t queried_total = 0;
        const cudaError_t err = cudaMemGetInfo(&queried_free, &queried_total);
        if (err != cudaSuccess) {
            result.message = std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(err);
            return result;
        }
        free_bytes = queried_free;
        total_bytes = queried_total;
    }

    const float clamped_margin = std::max(0.0f, std::min(input.preflight_vram_margin, 1.0f));
    const std::size_t usable_bytes = static_cast<std::size_t>(
        std::floor(static_cast<double>(free_bytes) * static_cast<double>(clamped_margin))
    );

    result.free_vram_bytes = free_bytes;
    result.total_vram_bytes = total_bytes;
    result.usable_vram_bytes = usable_bytes;
    result.estimate = estimate_dense_k1k6_memory(input.Nx, input.Nz, input.N_theta, input.N_E);

    result.fine_batch_plan = plan_fine_batch_cells(
        input.fine_batch_requested_cells,
        input.Nx * input.Nz,
        usable_bytes,
        result.estimate.bytes_per_dense_cell
    );

    result.ok = result.estimate.total_required_bytes <= usable_bytes;
    if (!result.ok) {
        std::ostringstream oss;
        oss << "required " << format_binary_bytes(result.estimate.total_required_bytes)
            << " exceeds usable " << format_binary_bytes(usable_bytes)
            << " (free " << format_binary_bytes(free_bytes)
            << ", margin " << std::fixed << std::setprecision(2) << clamped_margin << ")";
        result.message = oss.str();
        return result;
    }

    if (result.fine_batch_plan.clamped) {
        std::ostringstream oss;
        oss << "fine_batch_max_cells clamped from " << result.fine_batch_plan.requested_cells
            << " to " << result.fine_batch_plan.planned_cells;
        result.message = oss.str();
    } else {
        result.message = "ok";
    }
    return result;
}

std::string format_binary_bytes(std::size_t bytes) {
    static constexpr const char* kUnits[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision((unit == 0) ? 0 : 2) << value << " " << kUnits[unit];
    return oss.str();
}

} // namespace sm_2d
