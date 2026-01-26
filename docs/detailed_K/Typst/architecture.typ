#set text(font: "Linux Libertine", size: 11pt)
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

#show math: set text(weight: "regular")

= Architecture Overview

== Project Summary

SM_2D is a high-performance 2D deterministic transport solver for proton therapy dose calculation using CUDA-accelerated GPU computing. The project implements a hierarchical S-matrix solver with block-sparse phase-space representation.

=== Key Statistics

#figure(
  table(
    columns: (auto, 1fr),
    inset: 8pt,
    align: left,
    table.header([*Metric*], [*Value*]),
    [Total Files], [30+ C++ source files],
    [CUDA Kernels], [6 major kernels (K1-K6)],
    [Lines of Code], [~15,000 lines],
    [Memory per Simulation], [~3 GB GPU memory],
    [Grid Size], [Up to 200 × 640 cells],
  ),
  caption: [System Metrics],
)

== System Architecture

=== Architecture Layers

The system is organized into six distinct layers:

==== Input Layer

* Configuration from `sim.ini`
* NIST PSTAR physics data
* Command-line parameters

==== Core Layer

* Energy/Angle Grids
* Block Encoding (24-bit)
* Phase-Space Storage (Hierarchical)
* Bucket Emission (Inter-cell)

==== Physics Layer

* Highland MCS
* Vavilov Straggling
* Nuclear Attenuation
* R-based Step Control
* Fermi-Eyges Lateral Spread

==== CUDA Pipeline

* K1: ActiveMask
* K2: Coarse Transport
* K3: Fine Transport (MAIN PHYSICS)
* K4: Bucket Transfer
* K5: Conservation Audit
* K6: Swap Buffers

==== Output Layer

* 2D Dose Distribution
* Depth-Dose Curve
* Conservation Report

== Module Dependency Graph

=== Foundation Layer

* LUT Module (`r_lut`, `nist_loader`)
* Config Loader
* Logger

=== Data Structures

* Grids (Energy, Angle)
* Block Encoding (24-bit ID)
* Local Bins (4D sub-cell)
* Psi Storage (Hierarchical)
* Buckets (Emission)

=== Physics Module

* Highland MCS
* Energy Straggling
* Nuclear Attenuation
* Step Control
* Fermi-Eyges Spread

=== CUDA Kernels

* K1-K6 Pipeline

=== Sources

* Pencil Source
* Gaussian Source

=== Boundaries

* Boundary Conditions
* Loss Tracking

=== Audit

* Conservation Check
* Global Budget
* Report Generation

=== Validation

* Bragg Peak Validation
* Lateral Spread Validation
* Determinism Test

== Memory Layout

=== GPU Memory Breakdown

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Buffer*], [*Size*], [*Type*]),
    [PsiC_in/out], [1.1 GB each], [float32],
    [EdepC], [0.5 GB], [float64],
    [AbsorbedWeight_cutoff], [0.25 GB], [float32],
    [AbsorbedWeight_nuclear], [0.25 GB], [float32],
    [AbsorbedEnergy_nuclear], [0.25 GB], [float64],
    [BoundaryLoss], [0.1 GB], [float32],
    [ActiveMask/List], [0.5 GB], [uint8/uint32],
  ),
  caption: [GPU Memory Layout],
)

#text(size: 10pt)[*Total: ~4.3 GB GPU memory*]

== Phase-Space Representation

=== 4D Phase Space

Particles are represented in four dimensions:

* $ theta $ (Angle): 512 bins from -90 deg to +90 deg
* $ E $ (Energy): 256 bins from 0.1 to 250 MeV (log-spaced)
* $ x_"sub" $: 4 sub-bins within each cell (transverse)
* $ z_"sub" $: 4 sub-bins within each cell (depth)

=== Block Encoding (24-bit)

```
┌─────────────────────────┬──────────────────────────┐
│     b_E (12 bits)       │    b_theta (12 bits)     │
│    Bits 12-23           │     Bits 0-11            │
│    Range: 0-4095        │     Range: 0-4095        │
└─────────────────────────┴──────────────────────────┘
                    24-bit Block ID
```

=== Encoding Details

* Bits 0-11: `b_theta` (0-4095 angular bins)
* Bits 12-23: `b_E` (0-4095 energy bins)

=== Local Index (16-bit)

Local index encoding:

$ "idx" = theta_."local" + 8 times (E_."local" + 4 times (x_"sub" + 4 times z_"sub")) $

Where:
* $ theta_."local" $: 8 values (0-7)
* $ E_."local" $: 4 values (0-3)
* $ x_"sub" $: 4 values (0-3)
* $ z_"sub" $: 4 values (0-3)

Total: $ 8 times 4 times 4 times 4 = 512 $ local bins per block

== Physics Pipeline per Step

=== Step Sequence

For each transport step, the following operations occur:

1. *Step Control*: $ dif s = min(2% times R, dif x, dif z) $
2. *Energy Loss*: $ E = E - (dif E)/(dif s) times dif s $
3. *Straggling*: $ dif E ~ "Vavilov"(kappa) $
4. *MCS*: $ theta = theta + sigma_"theta" times N(0,1) $
5. *Nuclear*: $ W = W times exp(-sigma times dif s) $
6. *Energy Deposition*: $ E_"dep" = E_"in" - E_"out" $
7. *Boundary Check*: Emit to bucket if crossing

== Directory Structure

```
SM_2D/
├── run_simulation.cpp          # Main entry point
├── sim.ini                     # Configuration file
├── visualize.py                # Python visualization
│
├── src/
│   ├── core/                   # Core data structures
│   │   ├── grids.cpp           # Energy/angle grids
│   │   ├── block_encoding.hpp  # 24-bit encoding
│   │   ├── local_bins.hpp      # 4D sub-cell partitioning
│   │   ├── psi_storage.cpp     # Hierarchical phase-space
│   │   └── buckets.cpp         # Bucket emission
│   │
│   ├── physics/                # Physics implementations
│   │   ├── highland.hpp        # Multiple Coulomb scattering
│   │   ├── energy_straggling.hpp  # Vavilov straggling
│   │   ├── nuclear.hpp         # Nuclear attenuation
│   │   ├── step_control.hpp    # R-based step control
│   │   └── fermi_eyges.hpp     # Lateral spread theory
│   │
│   ├── lut/                    # Lookup tables
│   │   ├── nist_loader.cpp     # NIST PSTAR data
│   │   └── r_lut.cpp           # Range-energy interpolation
│   │
│   ├── source/                 # Beam sources
│   │   ├── pencil_source.cpp   # Pencil beam
│   │   └── gaussian_source.cpp # Gaussian beam
│   │
│   ├── boundary/               # Boundary conditions
│   │   ├── boundaries.cpp      # Boundary types
│   │   └── loss_tracking.cpp   # Loss accounting
│   │
│   ├── audit/                  # Conservation auditing
│   │   ├── conservation.cpp    # Weight/energy checks
│   │   ├── global_budget.cpp   # Global aggregation
│   │   └── reporting.cpp       # Report generation
│   │
│   ├── cuda/kernels/           # CUDA kernels
│   │   ├── k1_activemask.cu    # Active cell detection
│   │   ├── k2_coarsetransport.cu  # High-energy transport
│   │   ├── k3_finetransport.cu # Fine transport (main)
│   │   ├── k4_transfer.cu      # Bucket transfer
│   │   ├── k5_audit.cu         # Conservation audit
│   │   └── k6_swap.cu          # Buffer swap
│   │
│   └── utils/                  # Utilities
│       ├── logger.cpp          # Logging system
│       └── memory_tracker.cpp  # GPU memory tracking
```

== Key Design Principles

=== 1. Block-Sparse Storage

Only allocate memory for active phase-space blocks, saving >70% memory vs dense storage.

=== 2. Hierarchical Refinement

Coarse transport for high-energy, fine transport for low-energy (Bragg peak region).

=== 3. GPU-First Design

All physics computation on GPU, minimal host-device transfer.

=== 4. Conservation by Design

Built-in auditing at every step for weight and energy conservation.

=== 5. Modular Physics

Each physics process in separate header for easy validation and testing.

== References

* NIST PSTAR Database: #link("https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html")[https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html]
* PDG 2024: #link("https://pdg.lbl.gov/")[https://pdg.lbl.gov/] (Highland formula)
* ICRU Report 73: Stopping powers for electrons and positrons

---
#set align(center)
*SM_2D Architecture Documentation*

#text(size: 9pt)[Version 1.0.0]
