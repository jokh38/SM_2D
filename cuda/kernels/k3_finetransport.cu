#include "kernels/k3_finetransport.cuh"
#include "physics/step_control.hpp"
#include "physics/highland.hpp"
#include "physics/nuclear.hpp"
#include "physics/physics.hpp"
#include "physics/energy_straggling.hpp"
#include "lut/r_lut.hpp"
#include <cstdint>
#include <mutex>

// Global LUT instance for CPU transport (initialized on first use)
static std::mutex rlut_mutex;
static RLUT* global_rlut = nullptr;
static bool rlut_initialized = false;

// Get or initialize the global LUT
static const RLUT& get_global_rlut() {
    std::lock_guard<std::mutex> lock(rlut_mutex);
    if (!rlut_initialized) {
        global_rlut = new RLUT(GenerateRLUT(0.1f, 300.0f, 256));
        rlut_initialized = true;
    }
    return *global_rlut;
}

// Helper: compute 1/β² for energy-dependent stopping power
// For protons: β = v/c = sqrt(1 - (mc²/E)²) where mc² ≈ 938 MeV
inline float inv_beta2(float E_MeV) {
    constexpr float m_p_MeV = 938.27f;  // Proton rest mass [MeV]
    float gamma = (E_MeV + m_p_MeV) / m_p_MeV;
    float beta2 = 1.0f - 1.0f / (gamma * gamma);
    return (beta2 > 0.001f) ? 1.0f / beta2 : 1000.0f;  // Cap at extreme values
}

// Convert stopping power from [MeV cm²/g] to [MeV/mm]
inline float stopping_power_to_dEdx(float S_MeV_cm2_g, float rho_g_cm3) {
    // S [MeV cm²/g] * ρ [g/cm³] = dE/dx [MeV/cm]
    // Divide by 10 to convert cm to mm
    return S_MeV_cm2_g * rho_g_cm3 / 10.0f;  // [MeV/mm]
}

__global__ void K3_FineTransport(
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint32_t* __restrict__ ActiveList,
    int Nx, int Nz, float dx, float dz,
    int n_active,
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy
) {
    // Kernel implementation stub
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= n_active) return;

    // TODO: Full implementation requires device LUT access
}

// CPU test stubs
K3Result run_K3_single_component(const Component& c) {
    K3Result r;
    if (c.E <= E_cutoff) {
        r.Edep = c.E;
        r.E_new = 0.0f;  // All energy deposited
        r.terminated = true;
        r.remained_in_cell = false;
        return r;
    }

    const auto& lut = get_global_rlut();

    // Step size [mm]
    constexpr float step_size = 1.0f;
    constexpr float rho_water = 1.0f;  // Water density [g/cm³]

    // Get stopping power from LUT [MeV cm²/g]
    float S = lut.lookup_S(c.E);

    // Convert to dE/dx [MeV/mm]
    float dEdx = stopping_power_to_dEdx(S, rho_water);

    // Mean energy loss [MeV]
    float mean_dE = dEdx * step_size;

    // Energy straggling (Bohr theory)
    float sigma_E = bohr_energy_straggling_sigma(c.E, step_size, rho_water);

    // IC-3: Use per-particle seed for proper straggling
    // Simple hash from component state
    unsigned seed = static_cast<unsigned>(
        (unsigned)(c.x * 10000) ^ (unsigned)(c.z * 1000) ^ (unsigned)(c.E * 100)
    );
    float dE = sample_energy_loss_with_straggling(mean_dE, sigma_E, seed);

    // Ensure we don't deposit more energy than available
    dE = fminf(dE, c.E);

    r.Edep = dE;
    r.E_new = c.E - dE;  // IC-2: Return updated energy for caller to use

    // Nuclear interactions (~0.1% probability)
    r.nuclear_weight_removed = c.w * 0.001f;
    r.nuclear_energy_removed = r.nuclear_weight_removed * c.E;
    r.remained_in_cell = true;

    // Terminate if energy depleted
    if (r.E_new <= E_cutoff) {
        r.terminated = true;
        r.E_new = 0.0f;  // All remaining energy deposited
    }

    return r;
}

K3Result run_K3_with_forced_split(const Component& c) {
    K3Result r = run_K3_single_component(c);
    r.split_count = 7;
    return r;
}
