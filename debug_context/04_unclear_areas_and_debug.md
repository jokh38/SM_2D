# Unclear Areas and Recommended Debug Messages

## Overview
This document identifies areas in the code that are unclear or would benefit from additional debug messages for troubleshooting.

---

## Areas Requiring Clarification

### 1. Coordinate System Ambiguity

#### Location
- K3 Fine Transport: lines 167-198
- K2 Coarse Transport: lines 156-175

#### Issue
The code uses a **centered coordinate system** for cell positions:
```cpp
float half_dz = dz * 0.5f;
float half_dx = dx * 0.5f;
float step_to_z_plus = (mu > 0) ? (half_dz - z_cell) / mu : 1e30f;
```

**Unclear**: Is this consistently documented? The sub-bin calculations use:
```cpp
float x_offset = get_x_offset_from_bin(x_sub, dx);  // Returns [-dx/2, dx/2]
```

**Recommendation**: Add a debug message that prints the coordinate system on startup.

#### Proposed Debug Message
```cpp
// At start of K3 kernel
if (cell == 0) {
    printf("K3: Coordinate system: centered at cell origin\n");
    printf("K3: x in [%.3f, %.3f], z in [%.3f, %.3f]\n",
           -half_dx, half_dx, -half_dz, half_dz);
}
```

---

### 2. Energy Bin Index Calculation

#### Location
- K3 Fine Transport: line 158
- K2 Coarse Transport: line 147

#### Issue
The energy bin calculation was recently fixed to use the **lower edge**:
```cpp
float E = expf(log_E_min + E_bin * dlog);  // Lower edge
```

**Unclear**: When debugging, it's easy to confuse this with the center-based version.

**Recommendation**: Add debug output showing the bin calculation details.

#### Proposed Debug Message
```cpp
if (active_idx == 0 && lidx < 4) {
    printf("K3 E-BIN DEBUG: E_bin=%d, log_E_min=%.4f, dlog=%.6f\n", E_bin, log_E_min, dlog);
    printf("K3 E-BIN DEBUG: E=%.3f MeV (using LOWER edge, not center)\n", E);
}
```

---

### 3. Weight Conservation Failure

#### Location
- K5 Weight Audit: lines 41-45
- Main Pipeline: lines 772-773

#### Issue
When weight conservation fails, the audit reports pass/fail but doesn't show the breakdown.

**Unclear**: Which term is causing the imbalance? (W_in, W_out, W_cutoff, W_nuclear)

**Recommendation**: Add detailed breakdown when audit fails.

#### Proposed Debug Message
```cpp
// In K5_WeightAudit kernel
float W_error = fabsf(W_in - W_out - W_cutoff - W_nuclear);
float W_rel_error = W_error / fmaxf(W_in, 1e-20f);

report[cell].W_error = W_rel_error;
report[cell].W_pass = (W_rel_error < 1e-6f);

// DEBUG: Print breakdown for failing cells
if (W_rel_error >= 1e-6f) {
    printf("K5 AUDIT FAIL cell=%d: W_in=%.6f, W_out=%.6f, W_cutoff=%.6f, W_nuclear=%.6f, error=%.2e\n",
           cell, W_in, W_out, W_cutoff, W_nuclear, W_rel_error);
}
```

---

### 4. Bucket Overflow Detection

#### Location
- K3 Fine Transport: lines 302-309
- K2 Coarse Transport: lines 242-249

#### Issue
The bucket emission doesn't check if the bucket is full.

**Unclear**: What happens when more than Kb_out particles are emitted to a bucket?

**Recommendation**: Add overflow detection and warning.

#### Proposed Debug Message
```cpp
// In device_emit_component_to_bucket_4d_interp
int bucket_idx = device_bucket_index(cell, exit_face, Nx, Nz);
DeviceOutflowBucket& bucket = OutflowBuckets[bucket_idx];

// Check for overflow before emitting
int current_count = 0;
for (int i = 0; i < DEVICE_Kb_out; ++i) {
    if (bucket.block_id[i] != DEVICE_EMPTY_BLOCK_ID) {
        current_count++;
    }
}

if (current_count >= DEVICE_Kb_out) {
    printf("WARNING: Bucket overflow at cell=%d face=%d (count=%d, max=%d)\n",
           cell, exit_face, current_count, DEVICE_Kb_out);
}

// Proceed with emission...
```

---

### 5. Step Size Boundary Interaction

#### Location
- K3 Fine Transport: lines 198-216
- K2 Coarse Transport: lines 166-186

#### Issue
The interaction between physics step size and cell boundary is complex.

**Unclear**: When a particle is near a boundary, which limitation dominates?

**Recommendation**: Add debug output showing the decision process.

#### Proposed Debug Message
```cpp
// In K3, after step calculation
if (active_idx == 0 && slot == 0 && lidx < 4) {
    printf("K3 STEP DECISION: step_phys=%.4f, step_boundary=%.4f, actual=%.4f, reason=%s\n",
           step_phys, step_to_boundary, actual_range_step,
           (step_phys < step_to_boundary) ? "physics_limited" : "boundary_limited");
}
```

---

### 6. Phase Space Slot Allocation

#### Location
- K3 Fine Transport: lines 369-389
- K2 Coarse Transport: lines 300-320

#### Issue
Slot allocation uses atomicCAS which can fail under contention.

**Unclear**: How often does allocation fail? Is it causing performance issues?

**Recommendation**: Add statistics on allocation attempts.

#### Proposed Debug Message
```cpp
// Add counter to track allocation attempts
__device__ int allocation_attempts = 0;
__device__ int allocation_failures = 0;

// In allocation loop
for (int s = 0; s < Kb; ++s) {
    uint32_t expected = DEVICE_EMPTY_BLOCK_ID;
    atomicAdd(&allocation_attempts, 1);
    if (atomicCAS(&block_ids_out[...], expected, bid) == expected) {
        out_slot = s;
        break;
    }
    atomicAdd(&allocation_failures, 1);
}

// Print summary occasionally
if (cell == 0 && iter % 10 == 0) {
    printf("Slot allocation: attempts=%d, failures=%d, success_rate=%.2f%%\n",
           allocation_attempts, allocation_failures,
           100.0 * (1.0 - (float)allocation_failures / allocation_attempts));
}
```

---

### 7. Initial Source Injection

#### Location
- Main Pipeline: around line 24 (inject_source_kernel)

#### Issue
The source injection is not directly visible in the main pipeline loop.

**Unclear**: Where is the source injected? What are the initial conditions?

**Recommendation**: Add debug output showing source parameters.

#### Proposed Debug Message
```cpp
// In inject_source_kernel or main pipeline before loop
printf("SOURCE INJECTION: cell=%d, E0=%.2f MeV, theta0=%.4f rad, W_total=%.6f\n",
       source_cell, E0, theta0, W_total);
printf("SOURCE POSITION: x=%.3f mm (relative to cell), z=%.3f mm\n",
       x_in_cell, z_in_cell);
```

---

### 8. Pipeline Iteration Termination

#### Location
- Main Pipeline: lines 664-667

#### Issue
The pipeline terminates when no active cells remain, but there's no indication of *why*.

**Unclear**: Did all particles stop? Or did they leave the simulation domain?

**Recommendation**: Add termination reason logging.

#### Proposed Debug Message
```cpp
// When terminating
if (state.n_active == 0 && state.n_coarse == 0) {
    // Check if particles are in boundary buckets or truly stopped
    double total_boundary_weight = 0.0;
    // ... sum BoundaryLoss_weight ...

    printf("TERMINATION: No active cells. Boundary loss=%.6f, Remaining in domain=0\n",
           total_boundary_weight);

    if (total_boundary_weight > 0.01) {
        printf("WARNING: %.2f%% of weight lost to boundaries\n",
               100.0 * total_boundary_weight / initial_weight);
    }
}
```

---

## Debug Configuration System

### Recommendation: Add Debug Flags

Instead of scattered printf statements, use a centralized debug configuration:

```cpp
// Add to pipeline config
struct DebugConfig {
    bool print_physics_steps;     // Print step size decisions
    bool print_energy_bins;        // Print energy bin calculations
    bool print_weight_audit;       // Print weight conservation details
    bool print_boundary_crossings; // Print boundary crossing events
    bool print_slot_allocation;    // Print slot allocation stats
    int print_frequency;           // Print every N iterations
};
```

---

## Summary Table

| Area | File | Line(s) | Recommended Debug Action |
|------|------|---------|--------------------------|
| Coordinate system | k3_finetransport.cu | 167-198 | Print coordinate bounds |
| Energy bin calc | k3_finetransport.cu | 158 | Print bin calculation |
| Weight audit | k5_audit.cu | 41-45 | Print breakdown on failure |
| Bucket overflow | k3_finetransport.cu | 302-309 | Detect and warn |
| Step decision | k3_finetransport.cu | 198-216 | Print limiting factor |
| Slot allocation | k3_finetransport.cu | 369-389 | Track attempt/fail rate |
| Source injection | k1k6_pipeline.cu | ~24 | Print source params |
| Termination | k1k6_pipeline.cu | 664-667 | Print termination reason |

---

## Existing Debug Messages

The codebase already has extensive debug printf statements. These are primarily in:
- `k3_finetransport.cu`: Lines 77-81, 119-121, 126, 191-196, 213-216, 234-243, 266-269, 275-277, 327-329, 342-345, 410-416
- `k2_coarsetransport.cu`: Lines 66-69, 86-88, 124-125, 223-226, 273-276
- `k4_transfer.cu`: Lines 56-57, 70-83, 118-122, 176-183
- `k1k6_pipeline.cu`: Lines 55-57, 71-80, 634-636, 639-649, 651-661, 685-691, 693-704, 722-733, 740-751, 772-773, 797

**Note**: These debug statements use `active_idx == 0` or similar conditions to limit output. Consider adding a configuration flag to enable/disable them.

---

## Error Pattern Analysis (from claude-mem)

This section documents recurring error patterns identified through:
- Git commit history analysis
- claude-mem database records (1,302 observations, 102 bugfix entries)
- Historical bug fixes

### Top Error Categories

| Category | Priority | Count | Description |
|----------|----------|-------|-------------|
| **Weight Doubling** | 🔴 Critical | High | Bucket clearing, psi_out accumulation, E_trigger threshold |
| **Energy Binning** | 🔴 High | 29 | Boundary detection, bin interpretation errors |
| **Coordinate System** | 🟠 Medium | 2 | Transformations, indexing issues |
| **Conservation** | 🟠 Medium | 17 | Mass/energy tracking violations |
| **Buffer Management** | 🟠 Medium | 3 | Clearing order, hardcoded sizes |
| **Escape Tracking** | 🟠 Medium | 22 | Channel tracking problems |
| **Test Failures** | 🟢 Low | 34 | Golden snapshot, API mismatches |

### Recent Bug Fixes (Git History)

```
3859085 fix(k1k6): fix bucket clearing and psi_out accumulation to reduce weight doubling
88044fb Fix weight doubling bugs: 1) Changed E_trigger from 300MeV to 50MeV t...
6345b2c Fix(k1k6): fix energy bin interpretation and boundary detection
42f8642 fix(k1k6): fix buffer clearing bug and boundary detection order
32f9b36 fix(k1k6): fix coordinate system bugs, energy deposition, remove dead code
8c67683 fix(k1k6): reduce memory usage and fix hardcoded array sizes
2114c9e Fix critical GPU Bohr straggling physics bug - add missing 1/β energy...
a74d709 fix: implement proper Fermi-Eyges lateral spread for MCS
```

### Weight Doubling Bug Pattern

**Symptoms**: Particle weight accumulates incorrectly across iterations

**Root Causes Identified**:
1. Bucket clearing not resetting psi_out accumulation
2. E_trigger threshold too high (300MeV → 50MeV fix)
3. Buffer clearing order issues

**Recommended Mitigations**:
- Add assertions for bucket state consistency
- Validate psi_out resets after each iteration
- Unit tests for boundary edge cases

### Energy Binning Bug Pattern

**Symptoms**: Particles assigned to wrong energy bins

**Root Causes Identified**:
1. Lower-edge vs center-based bin calculation confusion
2. Boundary detection order errors

**Recommended Mitigations**:
- Explicit debug output showing bin calculation method
- Unit tests for energy boundary edge cases
- Document coordinate system in code comments

### Conservation Violation Pattern

**Symptoms**: Mass/energy not conserved within tolerance

**Root Causes Identified**:
1. Escape channel tracking errors (string vs integer enum)
2. Nuclear attenuation accounting gaps
3. Bohr straggling missing 1/β factor

**Recommended Mitigations**:
- Automated conservation checks in CI
- Audit breakdown on failure (see Debug Message #3 above)

### Database Summary (claude-mem)

| Table | Records | Description |
|-------|---------|-------------|
| `observations` | 1,302 | Tool executions, discoveries |
| `session_summaries` | 136 | AI-compressed session data |
| `bugfix entries` | 102 | Recorded bug fixes |
| `sdk_sessions` | 42 | Coding sessions |

**Storage**: `~/.claude-mem/claude-mem.db` (SQLite with FTS5 full-text search)
