# File Reference Guide

## Overview
This document provides a comprehensive reference to all files in the SM_2D codebase related to physics, transport, and pipeline implementation.

---

## Core Physics Files

### Header Files (CPU)

| File | Purpose | Key Contents |
|------|---------|--------------|
| `src/include/physics/physics.hpp` | Physics constants and state | m_p, E_cutoff, ComponentState |
| `src/include/physics/highland.hpp` | Multiple Coulomb Scattering | `highland_sigma()`, MCS direction update |
| `src/include/physics/energy_straggling.hpp` | Energy straggling models | Vavilov regime handling, Bohr/Landau |
| `src/include/physics/nuclear.hpp` | Nuclear interactions | ICRU 63 cross-sections |
| `src/include/physics/step_control.hpp` | Step size control | R-based step, energy update |
| `src/include/physics/fermi_eyges.hpp` | Fermi-Eyges theory | (Not directly used in transport) |

### Source Files (CPU)

Most physics functions are header-only (inline) for performance.

---

## GPU Device Physics Files

| File | Purpose | Key Contents |
|------|---------|--------------|
| `src/cuda/device/device_physics.cuh` | Device physics functions | `device_highland_sigma()`, `device_energy_straggling_sigma()` |
| `src/cuda/device/device_lut.cuh` | Device LUT structure | `DeviceRLUT`, lookup functions |
| `src/cuda/device/device_bucket.cuh` | Device bucket structure | `DeviceOutflowBucket`, bucket operations |
| `src/cuda/device/device_psic.cuh` | Device PsiC structure | Phase space storage |

---

## Lookup Tables

| File | Purpose | Key Contents |
|------|---------|--------------|
| `src/include/lut/r_lut.hpp` | Range LUT header | `RLUT` structure, lookup functions |
| `src/lut/r_lut.cpp` | Range LUT implementation | `GenerateRLUT()`, log-log interpolation |
| `src/lut/nist_loader.cpp` | NIST data loading | PSTAR data parsing |

---

## Transport Kernels (CUDA)

| File | Purpose | Key Function |
|------|---------|--------------|
| `src/cuda/kernels/k1_activemask.cuh` | Active mask header | `K1_ActiveMask()` |
| `src/cuda/kernels/k1_activemask.cu` | Active mask implementation | Cell classification |
| `src/cuda/kernels/k2_coarsetransport.cuh` | Coarse transport header | `K2_CoarseTransport()` |
| `src/cuda/kernels/k2_coarsetransport.cu` | Coarse transport implementation | High energy transport |
| `src/cuda/kernels/k3_finetransport.cuh` | Fine transport header | `K3_FineTransport()` |
| `src/cuda/kernels/k3_finetransport.cu` | Fine transport implementation | Full physics transport |
| `src/cuda/kernels/k4_transfer.cuh` | Bucket transfer header | `K4_BucketTransfer()` |
| `src/cuda/kernels/k4_transfer.cu` | Bucket transfer implementation | Boundary crossing |
| `src/cuda/kernels/k5_audit.cuh` | Weight audit header | `K5_WeightAudit()` |
| `src/cuda/kernels/k5_audit.cu` | Weight audit implementation | Conservation check |
| `src/cuda/kernels/k6_swap.cuh` | Buffer swap header | `K6_SwapBuffers()` |
| `src/cuda/kernels/k6_swap.cu` | Buffer swap implementation | Pointer swap |

---

## Pipeline Implementation

| File | Purpose | Key Contents |
|------|---------|--------------|
| `src/cuda/k1k6_pipeline.cuh` | Pipeline header | `K1K6PipelineConfig`, `K1K6PipelineState` |
| `src/cuda/k1k6_pipeline.cu` | Pipeline implementation | `run_k1k6_pipeline_transport()` |
| `src/cuda/gpu_transport_wrapper.cu` | GPU wrapper | Entry point for GPU transport |
| `src/include/gpu/gpu_transport_runner.hpp` | GPU runner header | `GPUTransportRunner::run()` |

---

## Core Data Structures

| File | Purpose | Key Contents |
|------|---------|--------------|
| `src/include/core/psi_storage.hpp` | Phase space storage | `PsiC`, `DevicePsiC` |
| `src/include/core/grids.hpp` | Grid definitions | `EnergyGrid`, `AngularGrid` |
| `src/include/core/buckets.hpp` | Bucket structures | `OutflowBucket` |
| `src/include/core/block_encoding.hpp` | Block ID encoding | Block ID bit layout |
| `src/include/core/local_bins.hpp` | Local bin definitions | `LOCAL_BINS`, encoding/decoding |
| `src/include/core/config_loader.hpp` | Configuration loading | `IncidentParticleConfig` |
| `src/include/core/incident_particle_config.hpp` | Incident particle config | Beam parameters |

---

## Validation and Testing

| File | Purpose | Key Contents |
|------|---------|--------------|
| `src/include/validation/pencil_beam.hpp` | Pencil beam validation | `run_pencil_beam()` |
| `src/include/validation/bragg_peak.hpp` | Bragg peak analysis | `find_bragg_peak_z()` |
| `src/include/validation/deterministic_beam.hpp` | Deterministic transport | CPU reference implementation |
| `tests/validation/test_pencil_beam.cpp` | Pencil beam tests | Validation tests |
| `tests/validation/test_bragg_peak.cpp` | Bragg peak tests | Peak position/width tests |

---

## Audit and Conservation

| File | Purpose | Key Contents |
|------|---------|--------------|
| `src/include/audit/conservation.hpp` | Conservation checks | Weight/energy conservation |
| `src/include/audit/reporting.hpp` | Audit reporting | `AuditReport` structure |
| `src/include/audit/global_budget.hpp` | Global budget | Total system tracking |

---

## Main Entry Point

| File | Purpose | Key Contents |
|------|---------|--------------|
| `run_simulation.cpp` | Main program | `main()`, config loading, output |
| `sim.ini` | Simulation config | Default configuration |

---

## Test Programs

| File | Purpose |
|------|---------|
| `test_boundary.cpp` / `test_boundary.cu` | Boundary crossing tests |
| `test_energy_loss.cpp` | Energy loss verification |
| `test_range_simple.cpp` | Range calculation tests |
| `test_step_size.cpp` | Step size control tests |
| `debug_k2.cpp` | K2 kernel debugging |
| `debug_range.cpp` | Range LUT debugging |

---

## Documentation

| File | Purpose |
|------|---------|
| `README.md` | Project overview |
| `docs/` | Additional documentation |

---

## Build System

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | CMake build configuration |

---

## File Dependency Graph

```
run_simulation.cpp
    ├─> core/config_loader.hpp
    ├─> validation/pencil_beam.hpp
    └─> gpu/gpu_transport_runner.hpp
            └─> cuda/gpu_transport_wrapper.cu
                    └─> cuda/k1k6_pipeline.cu
                            ├─> kernels/k1_activemask.cu
                            ├─> kernels/k2_coarsetransport.cu
                            │       ├─> device/device_physics.cuh
                            │       └─> device/device_lut.cuh
                            ├─> kernels/k3_finetransport.cu
                            │       ├─> device/device_physics.cuh
                            │       ├─> device/device_lut.cuh
                            │       └─> device/device_bucket.cuh
                            ├─> kernels/k4_transfer.cu
                            │       └─> device/device_bucket.cuh
                            ├─> kernels/k5_audit.cu
                            └─> kernels/k6_swap.cu
```

---

## Physics Functions Reference

### CPU Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `highland_sigma()` | `physics/highland.hpp` | MCS scattering angle |
| `energy_straggling_sigma()` | `physics/energy_straggling.hpp` | Energy straggling σ |
| `sample_energy_loss_with_straggling()` | `physics/energy_straggling.hpp` | Sample dE |
| `apply_nuclear_attenuation()` | `physics/nuclear.hpp` | Nuclear weight loss |
| `compute_max_step_physics()` | `physics/step_control.hpp` | Physics-limited step |
| `compute_energy_after_step()` | `physics/step_control.hpp` | E after step |
| `compute_energy_deposition()` | `physics/step_control.hpp` | dE in step |

### GPU Device Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `device_highland_sigma()` | `device/device_physics.cuh` | MCS on GPU |
| `device_energy_straggling_sigma()` | `device/device_physics.cuh` | Straggling on GPU |
| `device_sample_energy_loss()` | `device/device_physics.cuh` | Sample dE on GPU |
| `device_apply_nuclear_attenuation()` | `device/device_physics.cuh` | Nuclear on GPU |
| `device_compute_max_step()` | `device/device_lut.cuh` | Step on GPU |
| `device_lookup_R()` | `device/device_lut.cuh` | R(E) on GPU |
| `device_lookup_S()` | `device/device_lut.cuh` | S(E) on GPU |
| `device_lookup_E_inverse()` | `device/device_lut.cuh` | E(R) on GPU |

---

## Configuration Parameters

### Pipeline Configuration (`K1K6PipelineConfig`)

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `E_trigger` | float | 50 MeV | Fine transport threshold |
| `weight_active_min` | float | 1e-12 | Minimum active weight |
| `E_coarse_max` | float | 300 MeV | Max coarse energy |
| `step_coarse` | float | 1.0 mm | Coarse step size |
| `n_steps_per_cell` | int | 1 | Sub-steps per cell |
| `Nx`, `Nz` | int | - | Grid dimensions |
| `dx`, `dz` | float | - | Cell sizes |
| `N_theta`, `N_E` | int | - | Global phase space bins |
| `N_theta_local`, `N_E_local` | int | - | Local bins per block |

### Physics Constants

| Constant | Value | Unit | Location |
|----------|-------|------|----------|
| `m_p` | 938.272 | MeV/c² | `physics.hpp` |
| `E_cutoff` | 0.1 | MeV | `physics.hpp` |
| `X0_water` | 360.8 | mm | `highland.hpp` |
| `weight_epsilon` | 1e-12 | - | `physics.hpp` |

---

## Debug Context Files

| File | Purpose |
|------|---------|
| `debug_context/00_SUMMARY.md` | Review summary |
| `debug_context/01_physics_implementation_analysis.md` | Physics formulas |
| `debug_context/02_physics_usage_analysis.md` | Physics usage in code |
| `debug_context/03_pipeline_logic_analysis.md` | Pipeline flow |
| `debug_context/04_unclear_areas_and_debug.md` | Debug recommendations |
| `debug_context/05_file_reference.md` | This file |
