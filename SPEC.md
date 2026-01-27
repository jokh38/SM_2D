# Hierarchical Deterministic Transport Solver - Specification v0.8

## Document Overview

This specification consolidates v0.7.1 with critical corrections:
1. **R-based step control** (eliminates S(E) inconsistency)
2. **Variance-based MCS accumulation** (physically correct)
3. **Bin-edge based energy discretization** (log-grid compatible)
4. **LOCAL_BINS indexing clarification** (single definition)
5. **Nuclear energy budget closure** (conservation audit fix)
6. **Initial/boundary conditions** (implementation-ready)
7. **LUT generation and test cases** (reproducibility)

Target hardware: RTX 2080-class (8GB VRAM)

---

## 1. Physical Scope

### 1.1 Included Physics
- Continuous energy loss via CSDA range-energy relation R(E)
- Multiple Coulomb Scattering (MCS) via Highland formula
- Primary particle attenuation via nuclear interaction cross-section
- Energy cutoff with local residual deposition

### 1.2 Excluded Physics (MVP)
- Nuclear elastic/inelastic secondary production
- Delta-ray transport
- Range straggling (Landau/Vavilov)
- Multi-material interfaces

### 1.3 Accuracy Targets

| Observable | Target | Achievability |
|------------|--------|---------------|
| Bragg peak position | ±1-2% of range | High |
| Lateral σₓ at mid-range | ±15% | Medium |
| Lateral σₓ near Bragg | ±20% | Medium |
| Distal falloff (R80-R20) | ±10% | Low (no straggling) |
| Fluence attenuation | ±15% | Medium |
| Weight conservation | <1e-6 relative | High |
| Energy conservation | <1e-5 relative | High |

---

## 2. State Variables and Grid Definitions

### 2.1 Primary State Variable: Energy (with R-based Control)

Component state: `(θ, E, w, x, z, μ, η)`
- θ: polar angle [rad]
- E: kinetic energy [MeV]
- w: statistical weight
- (x, z): position [mm]
- μ = cos(θ), η = sin(θ): direction cosines

Step size control uses R(E) exclusively (see Section 5.1).

### 2.2 Energy Grid Definition (Log-Spaced, Bin-Edge Based)

```cpp
// Grid parameters
const int N_E = 256;
const float E_min = 0.1;    // MeV (cutoff)
const float E_max = 250.0;  // MeV

// Bin edges (N_E + 1 values)
float E_edges[N_E + 1];
for (int i = 0; i <= N_E; ++i) {
    E_edges[i] = E_min * pow(E_max/E_min, float(i)/N_E);
}

// Representative energy per bin (geometric mean)
float E_rep[N_E];
for (int i = 0; i < N_E; ++i) {
    E_rep[i] = sqrt(E_edges[i] * E_edges[i+1]);
}
```

### 2.3 Angular Grid Definition

```cpp
const int N_theta = 512;
const float theta_min = -PI/2;
const float theta_max = +PI/2;

float theta_edges[N_theta + 1];
for (int i = 0; i <= N_theta; ++i) {
    theta_edges[i] = theta_min + (theta_max - theta_min) * float(i) / N_theta;
}

float theta_rep[N_theta];
for (int i = 0; i < N_theta; ++i) {
    theta_rep[i] = 0.5f * (theta_edges[i] + theta_edges[i+1]);
}
```

### 2.4 LOCAL_BINS Decomposition (Single Definition)

```cpp
const int N_theta_local = 8;   // Angular sub-bins per block
const int N_E_local = 4;       // Energy sub-bins per block
const int LOCAL_BINS = N_theta_local * N_E_local;  // = 32

// Local index encoding
inline uint16_t encode_local_idx(int theta_local, int E_local) {
    return theta_local * N_E_local + E_local;
}

inline void decode_local_idx(uint16_t lidx, int& theta_local, int& E_local) {
    theta_local = lidx / N_E_local;
    E_local = lidx % N_E_local;
}
```

### 2.5 Block ID Encoding

```cpp
// Global block indices
int b_theta = global_theta_index / N_theta_local;  // Block index for theta
int b_E = global_E_index / N_E_local;              // Block index for energy

// Block ID: 24-bit encoding
uint32_t block_id = (b_theta & 0xFFF) | ((b_E & 0xFFF) << 12);

// Decoding
int b_theta = block_id & 0xFFF;
int b_E = (block_id >> 12) & 0xFFF;
```

---

## 3. Kernel Pipeline

| Kernel | Purpose | Output |
|--------|---------|--------|
| K1_ActiveMask | Identify cells requiring fine transport | ActiveMask[] |
| K2_CompactActive | Generate active cell list | ActiveList[] |
| K3_FineTransport | Transport with energy deposition | EdepC[], OutflowBuckets[][] |
| K4_BucketTransfer | Transfer to neighbor cells | PsiC_out[] |
| K5_ConservationAudit | Verify conservation | Pass/Fail |
| K6_SwapBuffers | Double buffer exchange | - |

---

## 4. Memory Layout

### 4.1 Persistent Buffers

| Buffer | Size | Type |
|--------|------|------|
| PsiC_in/out | 1.1GB each | block-sparse float32 |
| EdepC | 0.5GB | float64 |
| AbsorbedWeight_cutoff | 0.25GB | float32 |
| AbsorbedWeight_nuclear | 0.25GB | float32 |
| AbsorbedEnergy_nuclear | 0.25GB | float64 |
| BoundaryLoss_weight | 0.1GB | float32 |
| ActiveMask/List | 0.5GB | uint8/uint32 |

### 4.2 Bucket Structure (Source-Cell Indexed)

```cpp
struct OutflowBucket {
    uint32_t block_id[Kb_out];        // Kb_out = 64
    uint16_t local_count[Kb_out];
    float value[Kb_out][LOCAL_BINS];
};

// 4 faces: +z(0), -z(1), +x(2), -x(3)
OutflowBucket OutflowBuckets[Nx][Nz][4];
```

---

## 5. Physics Models

### 5.1 Energy Loss: R-Based Step Control (CRITICAL FIX v0.8)

**Principle**: Step size determined exclusively in R-space. S(E) is NOT used.

**Lookup Tables** (from NIST PSTAR):
```cpp
float R_table[N_E];  // CSDA range [mm]
float E_table[N_E];  // Kinetic energy [MeV]

float lookup_R(float E) {
    float E_safe = clamp(E, E_min, E_max);
    return exp(interp1d_log(log(E_safe), log_E_table, log_R_table));
}

float lookup_E_inverse(float R) {
    float R_safe = clamp(R, R_min, R_max);
    return exp(interp1d_log(log(R_safe), log_R_table, log_E_table));
}
```

**Step Size Determination** (R-based, no S(E)):
```cpp
float compute_max_step_physics(float E) {
    float R = lookup_R(E);
    
    // Option A: Fixed fraction of remaining range
    float delta_R_max = 0.02f * R;  // Max 2% range loss per substep
    
    // Energy-dependent refinement near Bragg
    if (E < 10.0f) {
        delta_R_max = min(delta_R_max, 0.2f);  // mm
    } else if (E < 50.0f) {
        delta_R_max = min(delta_R_max, 0.5f);
    }
    
    return delta_R_max;  // This IS the step size (since dR/ds ≈ 1 in CSDA)
}
```

**Energy Update**:
```cpp
float R_current = lookup_R(E);
float R_new = R_current - step_length;

if (R_new <= 0) {
    E_new = 0;
    delta_E = E;
} else {
    E_new = lookup_E_inverse(R_new);
    delta_E = E - E_new;
}
```

### 5.2 Multiple Coulomb Scattering (Variance-Based Accumulation)

**Highland Formula**:
```cpp
float highland_sigma(float E, float ds) {
    float beta = sqrt(1.0f - pow(m_p / (E + m_p), 2));
    float p_MeV = sqrt(pow(E + m_p, 2) - m_p * m_p);
    float t = ds / X0;  // X0 = 360.8 mm for water
    
    if (t < 1e-10f) return 0.0f;
    
    float ln_term = log(t);
    float bracket = 1.0f + 0.038f * ln_term;
    
    // Step reduction if bracket becomes unphysical
    if (bracket < 0.1f) {
        return -1.0f;  // Signal: reduce step size
    }
    
    return (13.6f / (beta * p_MeV)) * sqrt(t) * bracket;
}
```

**Variance-Based Accumulation** (CRITICAL FIX v0.8):
```cpp
// WRONG (v0.7.1): sigma_accumulated += sigma_theta;
// CORRECT (v0.8):
float var_accumulated = 0.0f;

// During substep:
float sigma_theta = highland_sigma(E, ds);
var_accumulated += sigma_theta * sigma_theta;

// Split condition (RMS-based):
float rms_accumulated = sqrt(var_accumulated);
float sigma_threshold = (E > 50.0f) ? 0.05f : 0.05f * sqrt(E / 50.0f);

if (rms_accumulated > sigma_threshold) {
    do_split = true;
    var_accumulated = 0.0f;
}
```

### 5.3 Angular Quadrature (7-Point)

| k | Δθ | Weight |
|---|-----|--------|
| 0 | -3.0σ | 0.05 |
| 1 | -1.5σ | 0.10 |
| 2 | -0.5σ | 0.20 |
| 3 | 0 | 0.30 |
| 4 | +0.5σ | 0.20 |
| 5 | +1.5σ | 0.10 |
| 6 | +3.0σ | 0.05 |

### 5.4 Nuclear Attenuation with Energy Budget

**Cross-Section**:
```cpp
float Sigma_total(float E_MeV) {
    if (E_MeV > 100.0f) return 0.0050f;  // mm⁻¹
    if (E_MeV > 50.0f)  return 0.0060f;
    return 0.0075f;
}
```

**Application with Energy Tracking** (CRITICAL FIX v0.8):
```cpp
float survival = exp(-Sigma_total(E) * step_length);
float w_new = w_old * survival;
float w_removed = w_old - w_new;

// Track removed weight
AbsorbedWeight_nuclear[cell] += w_removed;

// Track removed energy (for conservation audit)
AbsorbedEnergy_nuclear[cell] += w_removed * E;
```

---

## 6. Two-Bin Energy Discretization (Bin-Edge Based)

**CRITICAL FIX v0.8**: Uses bin edges, not uniform dE assumption.

```cpp
void discretize_energy_2bin(
    float E, float w,
    int& bin_low, int& bin_high,
    float& w_low, float& w_high
) {
    // Find interval: E_edges[i] <= E < E_edges[i+1]
    bin_low = find_energy_bin(E);
    bin_high = bin_low + 1;
    
    if (bin_high >= N_E) {
        // Edge case: E near E_max
        bin_high = bin_low;
        w_low = w;
        w_high = 0.0f;
        return;
    }
    
    if (bin_low < 0) {
        // Edge case: E below E_min
        bin_low = 0;
        bin_high = 0;
        w_low = w;
        w_high = 0.0f;
        return;
    }
    
    // Linear interpolation factor
    float t = (E - E_edges[bin_low]) / (E_edges[bin_high] - E_edges[bin_low]);
    t = clamp(t, 0.0f, 1.0f);
    
    w_low = w * (1.0f - t);
    w_high = w * t;
}

int find_energy_bin(float E) {
    // Binary search in E_edges
    int lo = 0, hi = N_E;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (E_edges[mid + 1] <= E) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}
```

**Conservation Verification**:
```cpp
// Must hold: w_low * E_rep[bin_low] + w_high * E_rep[bin_high] ≈ w * E
// Note: Approximate due to E_rep being geometric mean, not exact centroid
```

---

## 7. Initial Conditions (Source Definition)

### 7.1 Source Types

**Type A: Pencil Beam** (validation baseline)
```cpp
struct PencilSource {
    float x0 = 0.0f;      // Entry position [mm]
    float z0 = 0.0f;      // Entry depth [mm]
    float theta0 = 0.0f;  // Entry angle [rad]
    float E0 = 150.0f;    // Initial energy [MeV]
    float W_total = 1.0f; // Total weight
};
```

**Type B: Gaussian Beam** (realistic)
```cpp
struct GaussianSource {
    float x0 = 0.0f;
    float z0 = 0.0f;
    float sigma_x = 5.0f;     // Spatial spread [mm]
    float sigma_theta = 0.01f; // Angular spread [rad]
    float E0 = 150.0f;
    float sigma_E = 1.0f;     // Energy spread [MeV]
    float W_total = 1.0f;
};
```

### 7.2 Source Injection

```cpp
void inject_source(PsiC& psi, const PencilSource& src) {
    int cell = get_cell(src.x0, src.z0);
    int theta_bin = find_theta_bin(src.theta0);
    int E_bin = find_energy_bin(src.E0);
    
    uint32_t bid = encode_block(theta_bin / N_theta_local, E_bin / N_E_local);
    int theta_local = theta_bin % N_theta_local;
    int E_local = E_bin % N_E_local;
    uint16_t lidx = encode_local_idx(theta_local, E_local);
    
    int slot = find_or_allocate_slot(psi[cell], bid);
    psi[cell].value[slot][lidx] += src.W_total;
}
```

---

## 8. Boundary Conditions

### 8.1 Domain Boundaries

```cpp
enum BoundaryType { ABSORB, REFLECT, PERIODIC };

struct BoundaryConfig {
    BoundaryType z_min = ABSORB;  // Source plane (injection handled separately)
    BoundaryType z_max = ABSORB;  // Distal boundary
    BoundaryType x_min = ABSORB;  // Lateral boundaries
    BoundaryType x_max = ABSORB;
};
```

### 8.2 Boundary Loss Tracking

```cpp
// Per-boundary loss buffers
float BoundaryLoss_weight[4];   // [z_min, z_max, x_min, x_max]
float BoundaryLoss_energy[4];

void handle_boundary_crossing(int face, float w, float E) {
    if (boundary_config[face] == ABSORB) {
        atomicAdd(&BoundaryLoss_weight[face], w);
        atomicAdd(&BoundaryLoss_energy[face], w * E);
    }
    // REFLECT, PERIODIC: implement as needed (post-MVP)
}
```

### 8.3 Neighbor Lookup with Boundary Check

```cpp
int get_neighbor(int cell, int face_idx) {
    int ix = cell % Nx;
    int iz = cell / Nx;
    
    switch (face_idx) {
        case 0:  // +z
            if (iz + 1 >= Nz) return -1;
            return cell + Nx;
        case 1:  // -z
            if (iz <= 0) return -1;
            return cell - Nx;
        case 2:  // +x
            if (ix + 1 >= Nx) return -1;
            return cell + 1;
        case 3:  // -x
            if (ix <= 0) return -1;
            return cell - 1;
    }
    return -1;
}
```

---

## 9. Fine Transport Kernel (K3)

### 9.1 Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| max_substeps | 256 | Loop termination |
| delta_s_max | 0.25 × cell_size | Geometric limit |
| delta_R_frac | 0.02 | Max 2% range/step |
| weight_epsilon | 1e-12 | Underflow threshold |
| E_cutoff | 0.1 MeV | Termination energy |
| mu_min, eta_min | 1e-6 | Parallel safety |
| max_comp_list | 128 | Overflow threshold |

### 9.2 K3 Pseudocode (v0.8 Consolidated)

```cpp
__global__ void K3_FineTransport(
    PsiC_in, EdepC,
    AbsorbedWeight_cutoff, AbsorbedWeight_nuclear, AbsorbedEnergy_nuclear,
    OutflowBuckets, ActiveList, BoundaryLoss_weight, BoundaryLoss_energy
) {
    int cell = ActiveList[blockIdx.x];
    int tid = threadIdx.x;
    
    __shared__ float EdepShared[128];
    __shared__ float AbsCutoffShared[128];
    __shared__ float AbsNuclearWShared[128];
    __shared__ float AbsNuclearEShared[128];
    
    float EdepLocal = 0;
    float AbsCutoffLocal = 0;
    float AbsNuclearWLocal = 0;
    float AbsNuclearELocal = 0;
    
    for (auto [slot, lidx] : thread_workload(tid)) {
        float w = PsiC_in[cell][slot][lidx];
        if (w < weight_epsilon) continue;
        
        auto [theta, E] = decode_state(cell, slot, lidx);
        float mu = cos(theta), eta = sin(theta);
        float x = cell_center_x(cell), z = cell_min_z(cell);
        
        ComponentList comp_list = {{theta, E, w, x, z, mu, eta}};
        float var_accumulated = 0;
        
        for (int step = 0; step < max_substeps; ++step) {
            if (comp_list.empty()) break;
            
            ComponentList new_list;
            
            for (auto& c : comp_list) {
                // Energy cutoff
                if (c.E <= E_cutoff) {
                    EdepLocal += c.w * c.E;
                    AbsCutoffLocal += c.w;
                    continue;
                }
                
                // Weight underflow
                if (c.w < weight_epsilon) {
                    EdepLocal += c.w * c.E;
                    AbsCutoffLocal += c.w;
                    continue;
                }
                
                // Step size: R-based (v0.8)
                float ds_geom = distance_to_boundary(c.x, c.z, c.mu, c.eta, cell);
                float ds_phys = compute_max_step_physics(c.E);
                float ds = min({ds_geom, ds_phys, delta_s_max});
                
                // MCS
                float sigma_theta = highland_sigma(c.E, ds);
                if (sigma_theta < 0) {
                    ds *= 0.5f;
                    sigma_theta = highland_sigma(c.E, ds);
                }
                
                // Variance accumulation (v0.8)
                var_accumulated += sigma_theta * sigma_theta;
                float rms = sqrt(var_accumulated);
                float threshold = (c.E > 50) ? 0.05f : 0.05f * sqrt(c.E / 50);
                bool do_split = (rms > threshold);
                if (do_split) var_accumulated = 0;
                
                // Angular split
                int n_splits = do_split ? 7 : 1;
                const float w7[7] = {0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05};
                const float d7[7] = {-3.0, -1.5, -0.5, 0.0, +0.5, +1.5, +3.0};
                
                for (int k = 0; k < n_splits; ++k) {
                    float theta_k = do_split ? 
                        clamp(c.theta + d7[k] * sigma_theta, -PI/2, PI/2) : c.theta;
                    float w_k = do_split ? c.w * w7[k] : c.w;
                    
                    float mu_k = cos(theta_k), eta_k = sin(theta_k);
                    float x_new = c.x + eta_k * ds;
                    float z_new = c.z + mu_k * ds;
                    
                    if (crossed_boundary(x_new, z_new, cell)) {
                        // Partial step to boundary
                        float ds_partial = distance_to_boundary(c.x, c.z, mu_k, eta_k, cell);
                        
                        float R_cur = lookup_R(c.E);
                        float R_cross = R_cur - ds_partial;
                        float E_cross = (R_cross > 0) ? lookup_E_inverse(R_cross) : 0;
                        float dE_partial = c.E - E_cross;
                        
                        EdepLocal += w_k * dE_partial;
                        
                        // Nuclear attenuation with energy tracking (v0.8)
                        float surv = exp(-Sigma_total(c.E) * ds_partial);
                        float w_cross = w_k * surv;
                        float w_removed = w_k * (1 - surv);
                        AbsNuclearWLocal += w_removed;
                        AbsNuclearELocal += w_removed * c.E;
                        
                        // Emit to bucket
                        int face = identify_crossed_face(c.x, c.z, x_new, z_new, cell);
                        int neighbor = get_neighbor(cell, face);
                        
                        if (neighbor < 0) {
                            // Boundary loss
                            handle_boundary_loss(face, w_cross, E_cross);
                        } else {
                            emit_to_bucket_2bin(cell, face, theta_k, E_cross, w_cross);
                        }
                    } else {
                        // Remains in cell
                        float R_cur = lookup_R(c.E);
                        float R_new = R_cur - ds;
                        float E_new = (R_new > 0) ? lookup_E_inverse(R_new) : 0;
                        float dE = c.E - E_new;
                        
                        EdepLocal += w_k * dE;
                        
                        // Nuclear attenuation (v0.8)
                        float surv = exp(-Sigma_total(c.E) * ds);
                        float w_new = w_k * surv;
                        float w_removed = w_k * (1 - surv);
                        AbsNuclearWLocal += w_removed;
                        AbsNuclearELocal += w_removed * c.E;
                        
                        if (new_list.size() < max_comp_list) {
                            new_list.push({theta_k, E_new, w_new, x_new, z_new, mu_k, eta_k});
                        } else {
                            rebin_moment_preserving(new_list);
                            new_list.push({theta_k, E_new, w_new, x_new, z_new, mu_k, eta_k});
                        }
                    }
                }
            }
            comp_list = new_list;
        }
    }
    
    // Block reduction
    EdepShared[tid] = EdepLocal;
    AbsCutoffShared[tid] = AbsCutoffLocal;
    AbsNuclearWShared[tid] = AbsNuclearWLocal;
    AbsNuclearEShared[tid] = AbsNuclearELocal;
    __syncthreads();
    
    if (tid == 0) {
        float sum_edep = 0, sum_cutoff = 0, sum_nuc_w = 0, sum_nuc_e = 0;
        for (int i = 0; i < 128; ++i) {
            sum_edep += EdepShared[i];
            sum_cutoff += AbsCutoffShared[i];
            sum_nuc_w += AbsNuclearWShared[i];
            sum_nuc_e += AbsNuclearEShared[i];
        }
        atomicAdd(&EdepC[cell], sum_edep);
        atomicAdd(&AbsorbedWeight_cutoff[cell], sum_cutoff);
        atomicAdd(&AbsorbedWeight_nuclear[cell], sum_nuc_w);
        atomicAdd(&AbsorbedEnergy_nuclear[cell], sum_nuc_e);
    }
}
```

### 9.3 Two-Bin Bucket Emission

```cpp
void emit_to_bucket_2bin(int cell, int face, float theta, float E, float w) {
    uint32_t bid = encode_block(
        find_theta_bin(theta) / N_theta_local,
        find_energy_bin(E) / N_E_local
    );
    
    int slot = find_or_allocate_slot(OutflowBuckets[cell][face], bid);
    
    int theta_local = find_theta_bin(theta) % N_theta_local;
    
    // 2-bin energy discretization (v0.8)
    int E_bin_low, E_bin_high;
    float w_low, w_high;
    discretize_energy_2bin(E, w, E_bin_low, E_bin_high, w_low, w_high);
    
    int E_local_low = E_bin_low % N_E_local;
    int E_local_high = E_bin_high % N_E_local;
    
    uint16_t lidx_low = encode_local_idx(theta_local, E_local_low);
    uint16_t lidx_high = encode_local_idx(theta_local, E_local_high);
    
    atomicAdd(&OutflowBuckets[cell][face].value[slot][lidx_low], w_low);
    if (E_local_high != E_local_low) {
        atomicAdd(&OutflowBuckets[cell][face].value[slot][lidx_high], w_high);
    }
}
```

---

## 10. Conservation Audit (K5) - Updated

### 10.1 Weight Conservation

```cpp
for each cell:
    W_in = sum(PsiC_in[cell])
    W_out = sum(PsiC_out[cell])
    W_cutoff = AbsorbedWeight_cutoff[cell]
    W_nuclear = AbsorbedWeight_nuclear[cell]
    
    W_error = |W_in - W_out - W_cutoff - W_nuclear| / max(W_in, eps)
    assert(W_error < 1e-6)
```

### 10.2 Energy Conservation (with Nuclear Energy Budget)

```cpp
for each cell:
    E_in = sum(w * E_rep) in PsiC_in[cell]
    E_out = sum(w * E_rep) in PsiC_out[cell]
    E_dep = EdepC[cell]
    E_lost_nuclear = AbsorbedEnergy_nuclear[cell]
    
    E_error = |E_in - (E_out + E_dep + E_lost_nuclear)| / max(E_in, eps)
    
    if (E_error > 1e-5) {
        log_warning("Energy drift: cell=%d, error=%.2e", cell, E_error);
    }
```

### 10.3 Global Budget

```cpp
// Total weight
W_total_in = sum(source weights)
W_total_out = sum(PsiC_out, all cells)
W_total_cutoff = sum(AbsorbedWeight_cutoff)
W_total_nuclear = sum(AbsorbedWeight_nuclear)
W_boundary = sum(BoundaryLoss_weight)

W_global_error = |W_total_in - (W_total_out + W_total_cutoff + W_total_nuclear + W_boundary)|

// Total energy
E_total_in = sum(source E * w)
E_total_dep = sum(EdepC)
E_total_nuclear = sum(AbsorbedEnergy_nuclear)
E_boundary = sum(BoundaryLoss_energy)

E_global_error = |E_total_in - (E_total_dep + E_total_nuclear + E_boundary)|
```

---

## 11. LUT Generation Procedure

### 11.1 Data Source
- NIST PSTAR database (water, protons)
- URL: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html

### 11.2 Processing Steps

```python
# 1. Download raw data
# Format: E [MeV], S_elec [MeV·cm²/g], S_nuc, S_total, CSDA_Range [g/cm²]

# 2. Convert units
rho_water = 1.0  # g/cm³
R_mm = CSDA_Range_g_cm2 * 10 / rho_water  # [mm]

# 3. Create log-spaced grid
E_grid = np.logspace(np.log10(0.1), np.log10(250), 256)

# 4. Interpolate R(E) on log-log scale
log_R_interp = np.interp(np.log(E_grid), np.log(E_raw), np.log(R_raw))
R_grid = np.exp(log_R_interp)

# 5. Create inverse table
# Sort by R, interpolate E(R)

# 6. Save with metadata
header = {
    'material': 'water',
    'E_min_MeV': 0.1,
    'E_max_MeV': 250.0,
    'N_points': 256,
    'checksum_sha256': compute_hash(data)
}
```

### 11.3 Validation
```cpp
// R(150 MeV) for water: ~158 mm (NIST reference)
assert(abs(lookup_R(150.0) - 158.0) / 158.0 < 0.01);

// R(70 MeV): ~40.8 mm
assert(abs(lookup_R(70.0) - 40.8) / 40.8 < 0.01);
```

---

## 12. Required Test Cases

### 12.1 Unit Tests

| ID | Test | Expected |
|----|------|----------|
| T1 | E = 0.2 MeV stop | Component terminates, full E deposited |
| T2 | θ = 45° boundary | Correct face identification |
| T3 | θ = 80° boundary | No tan(θ) overflow |
| T4 | Cell edge start | Partial step computed correctly |
| T5 | Highland bracket < 0.1 | Step reduced, not clamped |
| T6 | 2-bin E scatter | w_low + w_high = w |
| T7 | LOCAL_BINS decode | Round-trip encode/decode |

### 12.2 Integration Tests

| ID | Test | Criterion |
|----|------|-----------|
| I1 | Pencil 150 MeV | Bragg peak at R(150)±2% |
| I2 | Pencil 70 MeV | Bragg peak at R(70)±2% |
| I3 | Weight conservation | Global error < 1e-6 |
| I4 | Energy conservation | Global error < 1e-5 |
| I5 | Lateral σₓ at z=R/2 | Within ±15% of Fermi-Eyges |
| I6 | Determinism | Run-to-run checksum match |

### 12.3 Stress Tests

| ID | Test | Expected Behavior |
|----|------|-------------------|
| S1 | Kb overflow | Graceful fallback, logged |
| S2 | 10000 substeps | Loop terminates |
| S3 | E near E_min | No divide-by-zero |
| S4 | Parallel beam θ≈0 | No boundary artifacts |

---

## 13. Coarse-to-Fine Transition

### 13.1 Active Cell Criteria

```cpp
bool is_cell_active(int cell) {
    // Criterion 1: Contains LOW-energy components (Bragg peak region)
    // CRITICAL: Fine transport activates at LOW energy where stopping power is LARGE
    // This is the opposite of the original (incorrect) implementation
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == EMPTY) continue;

        int b_E = (block_id[cell][slot] >> 12) & 0xFFF;
        float E_block_max = E_edges[(b_E + 1) * N_E_local];

        // FIXED: Changed from > (HIGH energy) to < (LOW energy)
        // Activate fine transport when energy is BELOW threshold
        if (E_block_max < E_trigger) return true;
    }

    // Criterion 2: Total weight above threshold
    float W_cell = sum_weights(PsiC[cell]);
    if (W_cell > weight_active_min) return true;

    return false;
}
```

### 13.2 Parameters
```cpp
const float E_trigger = 20.0f;        // MeV (threshold for fine transport)
const float weight_active_min = 1e-10f;
```

### 13.3 Physics Justification

Fine transport is activated at **LOW energy** (below E_trigger) because:
- Stopping power S(E) is **larger** at low energy (Bragg peak region)
- Rapid energy deposition requires smaller step sizes for accuracy
- At high energy, S(E) is small and coarse transport is sufficient
- This is the **opposite** of the original incorrect implementation

---

## 14. Kb Overflow Handling

### 14.1 Detection
```cpp
int find_or_allocate_slot(OutflowBucket& bucket, uint32_t bid) {
    for (int i = 0; i < Kb_out; ++i) {
        uint32_t old = atomicCAS(&bucket.block_id[i], EMPTY, bid);
        if (old == EMPTY || old == bid) return i;
    }
    return -1;  // Overflow
}
```

### 14.2 Fallback Strategy
```cpp
if (slot < 0) {
    // Strategy 1: Immediate rebin
    rebin_bucket_moment_preserving(bucket);
    slot = find_or_allocate_slot(bucket, bid);
    
    if (slot < 0) {
        // Strategy 2: Emergency deposit (lose angular info)
        int fallback_slot = 0;  // Use first slot
        atomicAdd(&bucket.value[fallback_slot][0], w);
        overflow_count++;
    }
}
```

### 14.3 Logging
```cpp
// Per-step overflow statistics
struct OverflowStats {
    int count;
    int max_cell;
    int max_face;
};
```

---

## 15. Parameter Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| N_E | 256 | Energy bins |
| N_theta | 512 | Angular bins |
| N_theta_local | 8 | Sub-bins per block |
| N_E_local | 4 | Sub-bins per block |
| LOCAL_BINS | 32 | = 8 × 4 |
| Kb | 32 | Input slots/cell |
| Kb_out | 64 | Output slots/face |
| max_substeps | 256 | Loop limit |
| delta_R_frac | 0.02 | Max range loss/step |
| delta_s_max | 0.25 × cell | Geometric limit |
| E_cutoff | 0.1 MeV | Termination |
| E_trigger | 10 MeV | Fine transport |
| weight_epsilon | 1e-12 | Underflow |
| σ_split (E>50) | 0.05 rad | RMS threshold |
| Σ_nuclear (E>100) | 0.0050 mm⁻¹ | Attenuation |
| X0 (water) | 360.8 mm | Radiation length |
| m_p | 938.272 MeV/c² | Proton mass |

---

## 16. Implementation Checklist

| ID | Rule | Verification |
|----|------|--------------|
| IC-1 | R-based Δs control (no S(E)) | Code review |
| IC-2 | Variance-based MCS accumulation | Unit test |
| IC-3 | Bin-edge 2-bin energy scatter | Unit test |
| IC-4 | LOCAL_BINS = Nθ × NE = 32 | Static assert |
| IC-5 | AbsorbedEnergy_nuclear tracked | Audit pass |
| IC-6 | Boundary loss tracking | Global budget |
| IC-7 | Bucket indexed [cell][face] | Code review |
| IC-8 | Highland: reduce ds, not clamp | Unit test |
| IC-9 | E_edges log-spaced | LUT validation |
| IC-10 | Source injection defined | Test I1, I2 |

---

## 17. Change Log

| Version | Key Changes |
|---------|-------------|
| v0.7 | Bucket outflow, R(E) primary |
| v0.7.1 | Bucket indexing fix, 7-point quadrature, nuclear attenuation |
| **v0.8** | **R-based Δs (no S(E)), variance-based MCS, bin-edge 2-bin, LOCAL_BINS spec, nuclear energy budget, initial/boundary conditions, LUT procedure, test cases** |

---

## Appendix A: Key Formulas

### A.1 Relativistic Kinematics
```
β = sqrt(1 - (m_p/(E + m_p))²)
p = sqrt((E + m_p)² - m_p²)
```

### A.2 Highland Formula
```
σ_θ = (13.6 / βp) × sqrt(Δs/X0) × [1 + 0.038 × ln(Δs/X0)]
```

### A.3 Fermi-Eyges (Validation)
```
σ_x²(z) ≈ ∫₀ᶻ (z-z')² × T(z') dz'
where T = (σ_θ/Δs)²
```

---

**End of Specification v0.8**