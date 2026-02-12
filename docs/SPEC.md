# Hierarchical Deterministic Transport Solver - Specification v1.0

Status date: 2026-02-06

This document is intentionally split into two layers:
- Part I: Physics SoT (stable requirements)
- Part II: Implementation SoT (current code + required transition contract)

If Part I and Part II conflict, Part I defines the long-term correctness target.
Part II must explicitly record any temporary deviation.

This revision adopts the low-energy policy explicitly:
- Persistent storage: coarse grid state only.
- Fine grid computation: temporary scratch only for low-energy region (`E <= 10 MeV` by default).
- Fine results must be restricted back to coarse state with strict conservation.

---

## Part I. Physics SoT

## 1. Physical Scope

### 1.1 Included physics
- Continuous slowing down via CSDA range-energy relation `R(E)`.
- Multiple Coulomb scattering (MCS) effect on lateral spread.
- Primary attenuation by nuclear interaction surrogate model.
- Energy cutoff handling with local residual deposition.
- Boundary energy/weight loss accounting.

### 1.2 Excluded physics (current scope)
- Explicit secondary particle transport.
- Delta-ray transport.
- Multi-material interface transport.

### 1.3 Accuracy targets

| Observable | Target |
|---|---|
| Bragg peak position | within +/-2% of expected range |
| Lateral sigma_x at mid-range | within +/-15% |
| Lateral sigma_x near Bragg | within +/-20% |
| Global weight conservation | relative error < 1e-6 |
| Global energy conservation | relative error < 1e-5 |

## 2. State Variables and Units

Each transport component is represented by:
`(theta, E, w, x, z, mu, eta)`

- `theta`: polar angle [rad]
- `E`: kinetic energy [MeV]
- `w`: statistical weight
- `x, z`: position [mm]
- `mu = cos(theta)`, `eta = sin(theta)`

All kernel steps must preserve physical invariants defined in Section 5.

## 3. Physics Model Requirements

### 3.1 Energy loss model (mandatory)
- Step control MUST be range-based using `R(E)` and `E(R)` LUT inversion.
- Energy update MUST satisfy monotonic decrease for surviving primaries.
- Deposited energy MUST be accounted in `Edep` (plus explicit nuclear/cutoff channels).

### 3.2 MCS model (equivalence requirement)
Accepted implementations:
1. Variance accumulation + conditional angular splitting, or
2. Deterministic lateral spread mapping that is equivalent at acceptance-test level.

Mandatory invariant:
- MCS handling MUST increase effective lateral variance with depth and must not violate Section 5 conservation equations.

### 3.3 Nuclear attenuation model (mandatory)
- Weight removal by nuclear surrogate model MUST be tracked as a separate channel.
- Associated removed energy MUST be explicitly accounted as nuclear energy term.

### 3.4 Cutoff model (mandatory)
- For `E <= E_cutoff`, residual energy MUST be deposited locally.
- Cutoff weight and energy contributions MUST be auditable.

### 3.5 Hierarchical resolution policy (mandatory)
- Fine resolution MUST be used only in low-energy region where stopping power and MCS become dominant.
- Default low-energy activation threshold MUST be `E_fine_on = 10 MeV`.
- Optional hysteresis MAY be used (`E_fine_off >= E_fine_on`, recommended `11 MeV`) to avoid path jitter.
- Particles crossing `E_fine_on` inside a coarse step MUST be split at the threshold crossing point (crossing guard).
- Fine state is temporary scratch state; persistent run state is coarse grid state.
- Fine-to-coarse restriction MUST preserve Section 5 invariants.

## 4. Kernel Physics Contract

| Kernel | Physics contract |
|---|---|
| K1 | Classify fine/coarse cells by `E_fine_on` and active-weight threshold; support optional hysteresis |
| K2 | Coarse transport for high-energy region (`E > E_fine_on`); MUST split step when crossing `E_fine_on` |
| K3 | Fine transport for low-energy region (`E <= E_fine_on`) on temporary fine scratch state with full bookkeeping |
| K4 | Inter-cell transfer and fine->coarse restriction without artificial creation/loss |
| K5 | Conservation audit (weight and energy) |
| K6 | Buffer/scratch lifecycle update only (no physics mutation) |

## 5. Conservation Invariants (Non-negotiable)

### 5.1 Weight conservation
For each audited window (iteration delta or full run):

`W_in = W_out + W_cutoff + W_nuclear + W_boundary`

### 5.2 Energy conservation
For each audited window:

`E_in = E_out + E_dep + E_nuclear + E_boundary + E_cutoff`

Notes:
- `E_cutoff` can be tracked directly, or derived if the implementation guarantees equivalence.
- Any dropped-slot/bucket mechanism MUST be represented in audit channels (or fail-fast).

### 5.3 Audit policy
- Conservation checks MUST run at least at final step.
- Validation mode SHOULD run per-iteration checks.
- Production mode MAY run periodic checks, but MUST run final check.

### 5.4 Fine/Coarse consistency invariant
For each low-energy update window:
- `C_in --P--> F_in --K3--> F_out --R--> C_out` must satisfy Section 5 equations.
- `P` (prolongation) and `R` (restriction) are part of the physics contract, not optional implementation detail.

## 6. K5 Requirements (Updated)

### 6.1 Why K5 is mandatory
K5 is not redundant even if each kernel appears correct:
- Errors can emerge at kernel boundaries (K2->K4->K6), not only inside one kernel.
- Atomic contention, slot allocation saturation, and transfer ordering are pipeline-level risks.
- Therefore an explicit post-step conservation audit is required.

### 6.2 Required K5 outputs
K5 MUST report:
- `W_error`, `W_pass`
- `E_error`, `E_pass`
- Reduced totals for all conservation terms used in Section 5 equations.

### 6.3 Required behavior on failure
- Validation mode: fail immediately when threshold exceeded.
- Production mode: configurable policy (`warn`, `abort`) with default `warn` + final hard check.

## 7. Acceptance Tests

Minimum required gates:
- Energy-loss baseline test with expected Bragg depth tolerance.
- Full-physics test with global energy/weight audit pass.
- Boundary stress test with non-zero boundary loss and finite total accounting.
- Determinism regression under fixed seed.
- Threshold-crossing test around `E = 10 MeV` (no discontinuity at fine/coarse boundary).
- Fine-scratch equivalence test (`coarse -> fine -> coarse`) under conservation checks.

---

## Part II. Implementation SoT (Current Code Contract)

## 8. Runtime Configuration Contract

Primary runtime source:
- `IncidentParticleConfig.transport` and INI `[transport]` section.

Current implemented defaults:
- `N_theta = 36`
- `E_trigger = 10.0 MeV`
- `weight_active_min = 1e-12`
- `E_coarse_max = 300.0 MeV`
- `step_coarse = 5.0 mm`
- `n_steps_per_cell = 1`
- `max_iterations = 0` (uses `grid.max_steps`)

Required transition-policy config (this spec cycle):
- `E_fine_on_MeV` default `10.0`
- `E_fine_off_MeV` default `11.0` (optional hysteresis)
- `fine_batch_max_cells` (scratch upper bound)
- `fine_halo_cells` default `1`
- `preflight_vram_margin` default `0.85`

Reference:
- `src/include/core/incident_particle_config.hpp`
- `src/include/core/config_loader.hpp`

## 9. Grid and Binning Contract (Current)

### 9.1 Energy grid
- Piecewise-uniform `energy_groups` are used at runtime.
- Current default groups:
  - `0.1-2.0 : 0.1 MeV`
  - `2.0-20.0 : 0.2 MeV`
  - `20.0-100.0 : 0.25 MeV`
  - `100.0-250.0 : 0.25 MeV`

### 9.2 Local bins (compile-time)
Current compile-time constants:
- `N_theta_local = 4`
- `N_E_local = 2`
- `N_x_sub = 8`
- `N_z_sub = 4`
- `LOCAL_BINS = 256`

4D local index encoding is used: `(theta_local, E_local, x_sub, z_sub)`.

In target architecture, these local bins are used by fine scratch only, not by full-grid persistent storage.

### 9.3 Runtime/compile-time lock
- Runtime values for `N_theta_local`, `N_E_local` are validated against compile-time constants.
- Mismatch currently throws before execution.

This is an intentional current constraint and must be documented in all operator-facing configs.

## 10. Kernel Pipeline Mapping (Current vs Target)

| Stage | Current behavior summary | Required target behavior |
|---|---|---|
| K1 | Active mask by low-energy block threshold + weight threshold | Same + hysteresis + threshold crossing guard scheduling |
| K2 | Coarse deterministic transport | Coarse-only for `E > 10 MeV` with step split at `E = 10 MeV` crossing |
| K3 | Fine deterministic transport in full-grid buffers | Fine on scratch tiles/bricks only for `E <= 10 MeV` |
| K4 | Global per-cell outflow buckets | Batch-local transfer + conservative restriction to coarse |
| K5 | Weight-only audit currently implemented | Mandatory weight+energy pass/fail semantics |
| K6 | Pointer swap | Persistent coarse buffer update + scratch recycle |

## 11. K5 Status and Required Upgrade Path

### 11.1 Current state in code
- K5 currently computes weight-only conservation reduction.
- `AuditReport` currently has only weight fields.

### 11.2 Gap to Part I
- Energy conservation is not enforced by K5 kernel path yet.
- Final host-side energy printout exists but is not equivalent to mandatory K5 pass/fail semantics.

### 11.3 Required implementation update
Upgrade K5 from `K5_WeightAudit` to `K5_ConservationAudit`:

```cpp
struct AuditReport {
    float W_error;
    int W_pass;
    float E_error;
    int E_pass;

    float W_in_total, W_out_total, W_cutoff_total, W_nuclear_total, W_boundary_total;
    double E_in_total, E_out_total, E_dep_total, E_cutoff_total, E_nuclear_total, E_boundary_total;

    int processed_cells;
};
```

Required thresholds:
- `W_error < 1e-6`
- `E_error < 1e-5`

## 12. Memory Contract and Profiles

### 12.1 Current dense behavior
Current key constants:
- `DEVICE_Kb = 32`
- `DEVICE_Kb_out = 32`
- `LOCAL_BINS = 256`

Dense full-grid design allocates:
- Full `psi_in` and `psi_out` for all cells.
- Full outflow buckets for all cells/faces.

Implication:
- Large grids exceed 8GB quickly.
- Default large-grid profile and 8GB-safe profile must be separated.

### 12.2 Target 8GB contract (`coarse persistent + fine scratch`)
Memory budget MUST follow:

`M_total = N_cells * B_coarse_persistent + N_fine_batch * B_fine_scratch + M_overhead`

where:
- `B_coarse_persistent` is compact coarse state per cell.
- `B_fine_scratch` is temporary fine-state footprint per active scratch cell.
- `N_fine_batch` is bounded by preflight and runtime scheduler.

Required policy:
- Persistent memory budget <= 55% VRAM.
- Fine scratch budget <= 30% VRAM.
- Safety/headroom >= 15% VRAM.
- Preflight MUST estimate `N_fine_batch_max` before allocation.

## 13. Active/Coarse Transition Contract

Required low-energy rule:
- Fine path is enabled only for cells satisfying `E_cell <= E_fine_on` and `W_cell > weight_active_min`.
- High-energy region (`E_cell > E_fine_on`) is processed by coarse path.

Required continuity rule:
- If coarse step crosses `E_fine_on`, split step at threshold crossing point and hand the remainder to fine path in the same iteration window.

Recommended stability rule:
- Use hysteresis (`E_fine_on=10`, `E_fine_off=11`) to avoid fine/coarse oscillation.

## 14. Overflow and Drop Handling Contract

Current behavior:
- Slot/bucket overflow is counted in debug counters.

Required behavior:
- Any non-negligible dropped weight/energy must either:
  1. be included in conservation channels, or
  2. trigger fail-fast in validation mode.

This applies equally to coarse path, fine scratch path, and restriction/prolongation operators.

## 15. Validation and CI Contract

Must-pass group for release branch:
- `EnergyLossOnlyTest.EnergyLossOnly`
- `EnergyLossOnlyTest.FullPhysics`
- `EnergyLossOnlyTest.StragglingOnly`
- `EnergyLossOnlyTest.NuclearOnly`

Additional mandatory gate for this architecture:
- Transition continuity test around `10 MeV` with conservation pass.

Any regression in Bragg depth or conservation metrics blocks release.

## 16. Known Deviations (as of 2026-02-06)

- K5 is weight-only in kernel path; energy audit extension pending.
- Energy-loss suite still has 3 failing scenarios in current build.
- Runtime transport config exists, but local-bin dimensions remain compile-time locked.
- Current runtime still allocates full-grid `psi_in/psi_out` and full-grid outflow buckets.
- Fine scratch lifetime and conservative fine->coarse restriction are not yet implemented as SoT behavior.

These are tracked implementation gaps, not accepted physics exceptions.

---

## Appendix A. Reference Paths

Core configuration:
- `src/include/core/incident_particle_config.hpp`
- `src/include/core/config_loader.hpp`

Grid/bins and device structures:
- `src/include/core/local_bins.hpp`
- `src/cuda/device/device_psic.cuh`
- `src/cuda/device/device_bucket.cuh`

Pipeline and kernels:
- `src/cuda/k1k6_pipeline.cu`
- `src/cuda/kernels/k1_activemask.cu`
- `src/cuda/kernels/k2_coarsetransport.cu`
- `src/cuda/kernels/k3_finetransport.cu`
- `src/cuda/kernels/k4_transfer.cu`
- `src/cuda/kernels/k5_audit.cu`

Validation tests:
- `tests/gpu/test_energy_loss_only_gpu.cu`
