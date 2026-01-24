# SM_2D API Reference

## Overview

This document provides a comprehensive reference for all public APIs in SM_2D, organized by module.

---

## Core API

### EnergyGrid

```cpp
struct EnergyGrid {
    const int N_E;              // Number of energy bins
    const float E_min;          // Minimum energy [MeV]
    const float E_max;          // Maximum energy [MeV]
    std::vector<float> edges;   // Bin edges (N_E + 1)
    std::vector<float> rep;     // Representative energies

    // Constructor
    EnergyGrid(float E_min, float E_max, int N_E);

    // Find bin for given energy (binary search)
    int FindBin(float E) const;

    // Get representative energy for bin
    float GetRepEnergy(int bin) const;
};
```

#### Usage Example

```cpp
// Create log-spaced energy grid from 0.1 to 250 MeV
EnergyGrid e_grid(0.1f, 250.0f, 256);

// Find bin for 150 MeV proton
int bin = e_grid.FindBin(150.0f);

// Get representative energy
float E_rep = e_grid.GetRepEnergy(bin);
```

---

### AngularGrid

```cpp
struct AngularGrid {
    const int N_theta;          // Number of theta bins
    const float theta_min;      // Minimum angle [degrees]
    const float theta_max;      // Maximum angle [degrees]
    std::vector<float> edges;   // Bin edges (N_theta + 1)
    std::vector<float> rep;     // Representative angles

    // Constructor
    AngularGrid(float theta_min, float theta_max, int N_theta);

    // Find bin for given angle (O(1) arithmetic)
    int FindBin(float theta) const;

    // Get representative angle for bin
    float GetRepTheta(int bin) const;
};
```

---

### PsiC (Phase-Space Container)

```cpp
struct PsiC {
    const int Nx;                     // Grid X dimension
    const int Nz;                     // Grid Z dimension
    const int Kb;                     // Max blocks per cell (32)

    // Storage arrays
    std::vector<std::array<uint32_t, 32>> block_id;
    std::vector<std::array<std::array<float, LOCAL_BINS>, 32>> value;

    // Constructor
    PsiC(int Nx, int Nz, int Kb);

    // Find existing block or allocate new slot
    int find_or_allocate_slot(int cell, uint32_t bid);

    // Access weight at specific location
    float get_weight(int cell, int slot, uint16_t lidx) const;
    void set_weight(int cell, int slot, uint16_t lidx, float w);

    // Clear all data
    void clear();

    // Total cells
    int N_cells;
};
```

---

## Block Encoding API

```cpp
// Encode (b_theta, b_E) → 24-bit block ID
__host__ __device__ inline uint32_t encode_block(
    uint32_t b_theta,  // Angular bin (0-4095)
    uint32_t b_E       // Energy bin (0-4095)
);

// Decode block ID → (b_theta, b_E)
__host__ __device__ inline void decode_block(
    uint32_t block_id,
    uint32_t& b_theta,  // Output: angular bin
    uint32_t& b_E       // Output: energy bin
);

// Special value for empty slots
constexpr uint32_t EMPTY_BLOCK_ID = 0xFFFFFFFF;
```

#### Usage Example

```cpp
// Encode energy bin 100, angle bin 50
uint32_t bid = encode_block(50, 100);

// Decode
uint32_t theta_bin, energy_bin;
decode_block(bid, theta_bin, energy_bin);
// theta_bin = 50, energy_bin = 100
```

---

## Local Bins API

```cpp
// Constants
constexpr int N_theta_local = 8;   // Local angular subdivisions
constexpr int N_E_local = 4;       // Local energy subdivisions
constexpr int N_x_sub = 4;         // X position subdivisions
constexpr int N_z_sub = 4;         // Z position subdivisions
constexpr int LOCAL_BINS = 512;    // 8 × 4 × 4 × 4

// Encode 4D coordinates to 16-bit local index
__host__ __device__ inline uint16_t encode_local_idx_4d(
    int theta_local,  // 0-7
    int E_local,      // 0-3
    int x_sub,        // 0-3
    int z_sub         // 0-3
);

// Decode local index to 4D coordinates
__host__ __device__ inline void decode_local_idx(
    uint16_t lidx,
    int& theta_local,
    int& E_local,
    int& x_sub,
    int& z_sub
);

// Position conversion
__host__ __device__ inline float get_x_offset_from_bin(int x_sub, float dx);
__host__ __device__ inline float get_z_offset_from_bin(int z_sub, float dz);
```

---

## Physics API

### Highland Formula

```cpp
// Calculate MCS angle sigma [rad]
__host__ __device__ float highland_sigma(
    float E_MeV,   // Proton energy [MeV]
    float ds,      // Step length [mm]
    float X0       // Radiation length [mm]
);

// Sample scattering angle (Box-Muller)
__device__ float sample_mcs_angle(
    float sigma_theta,     // RMS scattering angle
    unsigned& seed         // RNG state
);

// Update direction cosines after scattering
__device__ void update_direction_after_mcs(
    float& mu,       // Direction cosine X (in/out)
    float& eta,      // Direction cosine Z (in/out)
    float delta_theta  // Scattering angle [rad]
);
```

---

### Energy Straggling

```cpp
// Calculate Vavilov kappa parameter
__host__ __device__ float vavilov_kappa(
    float E_MeV,   // Proton energy [MeV]
    float ds       // Step length [mm]
);

// Bohr straggling sigma
__host__ __device__ float bohr_straggling_sigma(
    float E_MeV,   // Proton energy [MeV]
    float ds       // Step length [mm]
);

// Sample energy loss with straggling
__device__ float sample_energy_loss_with_straggling(
    float E_MeV,
    float ds,
    unsigned& seed
);
```

---

### Step Control

```cpp
// Compute maximum step size (R-based method)
__host__ __device__ float compute_max_step_physics(
    float E,          // Current energy [MeV]
    const RLUT& lut   // Range-energy lookup table
);

// Compute energy after step (R-based)
__device__ float compute_energy_after_step(
    float E_in,       // Input energy [MeV]
    float ds,         // Step length [mm]
    const RLUT& lut   // Range-energy lookup table
);

// Compute energy deposited in step
__device__ float compute_energy_deposition(
    float E_in,
    float ds,
    const RLUT& lut
);
```

---

### Nuclear Interactions

```cpp
// Total nuclear cross-section [mm⁻¹]
__host__ __device__ float Sigma_total(
    float E_MeV   // Proton energy [MeV]
);

// Apply nuclear attenuation
__device__ void apply_nuclear_attenuation(
    float& weight,       // Particle weight (modified)
    double& energy_rem,  // Energy accumulator (for audit)
    float E_MeV,         // Current energy
    float ds             // Step length
);
```

---

## LUT API

### RLUT (Range-Energy Lookup Table)

```cpp
struct RLUT {
    EnergyGrid grid;
    std::vector<float> R;        // CSDA range [mm]
    std::vector<float> S;        // Stopping power [MeV cm²/g]
    std::vector<float> log_E;    // Pre-computed log(E)
    std::vector<float> log_R;    // Pre-computed log(R)
    std::vector<float> log_S;    // Pre-computed log(S)

    // Range from energy (log-log interpolation)
    float lookup_R(float E_MeV) const;

    // Stopping power from energy
    float lookup_S(float E_MeV) const;

    // Energy from range (inverse lookup)
    float lookup_E_inverse(float R_mm) const;
};
```

---

### NIST Data Loader

```cpp
struct NistDataRow {
    float energy_MeV;           // Kinetic energy [MeV]
    float stopping_power;       // dE/dx [MeV cm²/g]
    float csda_range_g_cm2;     // CSDA range [g/cm²]
};

// Load NIST PSTAR data from file
std::vector<NistDataRow> load_nist_pstar(
    const std::string& filepath
);

// Create RLUT from NIST data
RLUT create_r_lut_from_nist(
    const std::vector<NistDataRow>& nist_data,
    float E_min,
    float E_max,
    int N_E
);
```

---

## Source API

### PencilSource

```cpp
struct PencilSource {
    float x0 = 0.0f;       // X position [mm]
    float z0 = 0.0f;       // Z position [mm]
    float theta0 = 0.0f;   // Initial angle [rad]
    float E0 = 150.0f;     // Initial energy [MeV]
    float W_total = 1.0f;  // Total weight
};

// Inject pencil source into phase-space
void inject_pencil_source(
    const PencilSource& source,
    PsiC& psi,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    int Nx, int Nz, float dx, float dz
);
```

---

### GaussianSource

```cpp
struct GaussianSource {
    float x0 = 0.0f;           // Mean X position [mm]
    float z0 = 0.0f;           // Mean Z position [mm]
    float theta0 = 0.0f;       // Mean angle [rad]
    float sigma_x = 5.0f;      // X std dev [mm]
    float sigma_theta = 0.01f; // Angle std dev [rad]
    float E0 = 150.0f;         // Mean energy [MeV]
    float sigma_E = 1.0f;      // Energy std dev [MeV]
    float W_total = 1.0f;      // Total weight
    int n_samples = 1000;      // Monte Carlo samples
};

// Inject Gaussian source into phase-space
void inject_gaussian_source(
    const GaussianSource& source,
    PsiC& psi,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    int Nx, int Nz, float dx, float dz,
    unsigned seed = 42
);
```

---

## Boundary API

### Boundary Types

```cpp
enum class BoundaryType {
    ABSORB,     // Particles absorbed at boundary
    REFLECT,    // Particles reflect back
    PERIODIC    // Particles wrap to opposite boundary
};

struct BoundaryConfig {
    BoundaryType z_min = BoundaryType::ABSORB;
    BoundaryType z_max = BoundaryType::ABSORB;
    BoundaryType x_min = BoundaryType::ABSORB;
    BoundaryType x_max = BoundaryType::ABSORB;
};
```

---

### Loss Tracking

```cpp
struct BoundaryLoss {
    float weight[4];   // Weight lost per face
    double energy[4];  // Energy lost per face

    // Face indices: 0=+z, 1=-z, 2=+x, 3=-x
};

// Record boundary loss
void record_boundary_loss(
    BoundaryLoss& loss,
    int face,      // Face index (0-3)
    float w,       // Weight
    double E       // Energy
);

// Get total boundary losses
float total_boundary_weight_loss(const BoundaryLoss& loss);
double total_boundary_energy_loss(const BoundaryLoss& loss);
```

---

## Audit API

### Cell-Level Audit

```cpp
struct CellWeightAudit {
    float W_in;              // Input weight
    float W_out;             // Output weight
    float W_cutoff;          // Weight lost to cutoff
    float W_nuclear;         // Weight lost to nuclear
    float W_error;           // Relative error

    bool check() const {
        float W_expected = W_out + W_cutoff + W_nuclear;
        float W_rel = fabs(W_in - W_expected) / fmax(W_in, 1e-20f);
        return W_rel < 1e-6f;
    }
};

struct CellEnergyAudit {
    double E_in;             // Input energy
    double E_out;            // Output energy
    double E_dep;            // Energy deposited
    double E_nuclear;        // Energy lost to nuclear
    double E_error;          // Relative error

    bool check() const {
        double E_expected = E_out + E_dep + E_nuclear;
        double E_rel = fabs(E_in - E_expected) / fmax(E_in, 1e-20);
        return E_rel < 1e-5;
    }
};
```

---

### Global Budget

```cpp
struct GlobalAudit {
    // Weight
    double W_in_total;
    double W_out_total;
    double W_cutoff_total;
    double W_nuclear_total;
    double W_boundary_total;
    double W_error_relative;

    // Energy
    double E_in_total;
    double E_out_total;
    double E_dep_total;
    double E_nuclear_total;
    double E_boundary_total;
    double E_error_relative;

    bool weight_pass() const;
    bool energy_pass() const;
    bool pass() const;
};

// Aggregate cell audits to global
GlobalAudit aggregate_cell_audits(
    const std::vector<CellWeightAudit>& weight_audits,
    const std::vector<CellEnergyAudit>& energy_audits,
    const BoundaryLoss& boundary_loss
);
```

---

## Reporting API

```cpp
// Print cell-level weight report
void print_cell_weight_report(
    const CellWeightAudit& audit,
    int cell_id
);

// Print global audit report
void print_global_report(
    const GlobalAudit& audit,
    const std::string& output_path = ""
);

// Print list of failed cells
void print_failed_cells(
    const std::vector<CellWeightAudit>& audits,
    float threshold = 1e-6f
);

// Print summary statistics
void print_summary(
    const GlobalAudit& audit,
    int total_cells,
    int n_steps
);
```

---

## Validation API

### Bragg Peak Validation

```cpp
struct BraggPeakResult {
    float position_mm;       // Peak depth [mm]
    float peak_dose;         // Maximum dose
    float fwhm_mm;           // Full width at half maximum
    float R80;               // 80% dose depth
    float R20;               // 20% dose depth
    float distal_falloff;    // R80 - R20
    float position_error;    // vs NIST reference
    bool pass;               // ±2% criterion
};

// Analyze Bragg peak from PDD data
BraggPeakResult analyze_bragg_peak(
    const std::vector<float>& z_mm,
    const std::vector<float>& dose,
    float reference_position_mm  // NIST reference
);
```

---

### Lateral Spread Validation

```cpp
struct LateralSpreadResult {
    float z_mm;               // Depth of measurement
    float sigma_sim;          // Simulated lateral sigma
    float sigma_fermi_eyges;  // Theoretical prediction
    float relative_error;     // (sim - theory) / theory
    bool pass;                // ±15% criterion
};

// Validate lateral spread at specific depth
LateralSpreadResult validate_lateral_spread(
    const PsiC& psi,
    float z_mm,
    const EnergyGrid& e_grid,
    float E0_MeV
);
```

---

### Determinism Test

```cpp
struct DeterminismResult {
    uint32_t checksum1;       // First run
    uint32_t checksum2;       // Second run
    bool match;               // Checksums equal
    bool pass;                // Deterministic
};

// Run simulation twice and compare
DeterminismResult test_determinism(
    const std::string& config_file,
    unsigned seed1 = 42,
    unsigned seed2 = 42
);
```

---

## Utility API

### Logger

```cpp
enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERROR
};

// Get singleton logger instance
Logger& get_logger();

// Set log level
void set_log_level(LogLevel level);

// Logging macros
#define LOG_TRACE(...) get_logger().log(LogLevel::TRACE, __VA_ARGS__)
#define LOG_DEBUG(...) get_logger().log(LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...)  get_logger().log(LogLevel::INFO, __VA_ARGS__)
#define LOG_WARN(...)  get_logger().log(LogLevel::WARN, __VA_ARGS__)
#define LOG_ERROR(...) get_logger().log(LogLevel::ERROR, __VA_ARGS__)
```

---

### Memory Tracker

```cpp
struct MemoryInfo {
    size_t free_bytes;      // Free memory
    size_t total_bytes;     // Total memory
    size_t used_bytes;      // Used memory
    float utilization;      // Used / Total
};

// Query current GPU memory
MemoryInfo query_gpu_memory();

// Check with warning
bool check_memory_warning(float threshold = 0.9f);

// Print memory summary
void print_memory_summary();
```

---

## Configuration API

### Config Loader

```cpp
struct SimulationConfig {
    // Particle
    std::string particle_type;
    float mass_amu;
    float charge_e;

    // Beam
    std::string beam_profile;
    float beam_weight;

    // Energy
    float E_mean_MeV;
    float E_sigma_MeV;
    float E_min_MeV;
    float E_max_MeV;

    // Spatial
    float x0_mm, z0_mm;
    float sigma_x_mm, sigma_z_mm;

    // Angular
    float theta0_rad;
    float sigma_theta_rad;

    // Grid
    int Nx, Nz;
    float dx_mm, dz_mm;
    int max_steps;

    // Output
    std::string output_dir;
    bool normalize_dose;
    bool save_2d;
    bool save_pdd;
};

// Load configuration from INI file
SimulationConfig load_config(const std::string& ini_file);
```

---

## Main Simulation API

```cpp
// Main simulation entry point
int run_simulation(
    const std::string& config_file = "sim.ini",
    const std::string& output_dir = "results"
);

// Simulation result
struct SimulationResult {
    int exit_code;                      // 0 = success
    int n_steps_completed;              // Steps run
    double wall_time_seconds;           // Runtime

    GlobalAudit audit;                  // Conservation audit
    BraggPeakResult bragg_peak;         // Bragg peak analysis

    std::string dose_file;              // Output files
    std::string pdd_file;
    std::string audit_file;
};

// Run with result return
SimulationResult run_simulation_detailed(
    const SimulationConfig& config
);
```
