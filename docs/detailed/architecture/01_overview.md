# SM_2D Architecture Overview

## Project Summary

**SM_2D** is a high-performance 2D deterministic transport solver for proton therapy dose calculation using CUDA-accelerated GPU computing. The project implements a hierarchical S-matrix solver with block-sparse phase-space representation.

### Key Statistics
- **Total Files**: 66 C++ source files, 9 CUDA kernels, 5 Python scripts
- **CUDA Kernels**: 9 files (K1-K6 pipeline + wrapper + support)
- **Lines of Code**: ~20,000 lines
- **Memory per Simulation**: ~4.3GB GPU memory
- **Grid Size**: Up to 200 × 640 cells
- **Test Files**: 31 test files (GoogleTest framework)

---

## System Architecture

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        INI["sim.ini<br/>(Configuration)"]
        NIST["NIST PSTAR<br/>(Physics Data)"]
    end

    subgraph Main["Main Entry Point"]
        MAIN["run_simulation.cpp"]
    end

    subgraph Core["Core Layer"]
        GRIDS["Energy/Angle Grids"]
        ENCODE["Block Encoding<br/>(24-bit)"]
        PSI["Phase-Space Storage<br/>(Hierarchical)"]
        BUCKETS["Bucket Emission<br/>(Inter-cell)"]
    end

    subgraph Physics["Physics Layer"]
        HIGHLAND["Highland MCS"]
        VAVILOV["Vavilov Straggling"]
        NUCLEAR["Nuclear Attenuation"]
        STEP["R-based Step Control"]
        FERMI["Fermi-Eyges<br/>Lateral Spread"]
    end

    subgraph CUDA["CUDA Pipeline"]
        K1["K1: ActiveMask"]
        K2["K2: Coarse Transport"]
        K3["K3: Fine Transport<br/>(MAIN PHYSICS)"]
        K4["K4: Bucket Transfer"]
        K5["K5: Conservation Audit"]
        K6["K6: Swap Buffers"]
    end

    subgraph Output["Output Layer"]
        DOSE["2D Dose Distribution"]
        PDD["Depth-Dose Curve"]
        AUDIT["Conservation Report"]
    end

    INI --> MAIN
    NIST --> MAIN
    MAIN --> GRIDS
    MAIN --> ENCODE
    MAIN --> PSI

    GRIDS --> CUDA
    ENCODE --> CUDA
    PSI --> CUDA
    BUCKETS --> CUDA

    HIGHLAND --> K3
    VAVILOV --> K3
    NUCLEAR --> K3
    STEP --> K3
    FERMI --> K3

    K1 --> K2
    K2 --> K3
    K3 --> K4
    K4 --> K5
    K5 --> K6
    K6 --> K1

    CUDA --> DOSE
    CUDA --> PDD
    CUDA --> AUDIT

    classDef inputStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef coreStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef physicsStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef cudaStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef outputStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    class INI,NIST inputStyle
    class GRIDS,ENCODE,PSI,BUCKETS coreStyle
    class HIGHLAND,VAVILOV,NUCLEAR,STEP,FERMI physicsStyle
    class K1,K2,K3,K4,K5,K6 cudaStyle
    class DOSE,PDD,AUDIT outputStyle
```

---

## Module Dependency Graph

```mermaid
flowchart LR
    subgraph Foundation["Foundation Layer"]
        LUT["LUT Module<br/>(r_lut, nist_loader)"]
        CONFIG["Config Loader"]
        LOG["Logger"]
    end

    subgraph DataStructures["Data Structures"]
        GRIDS["Grids<br/>(Energy, Angle)"]
        BLOCKS["Block Encoding<br/>(24-bit ID)"]
        LOCAL["Local Bins<br/>(4D sub-cell)"]
        PSI["Psi Storage<br/>(Hierarchical)"]
        BUCKETS["Buckets<br/>(Emission)"]
    end

    subgraph Physics["Physics Module"]
        HIGHLAND["Highland<br/>MCS"]
        STRAGGLE["Energy<br/>Straggling"]
        NUCLEAR["Nuclear<br/>Attenuation"]
        STEP["Step<br/>Control"]
        FERMI["Fermi-Eyges<br/>Spread"]
    end

    subgraph Sources["Source Module"]
        PENCIL["Pencil<br/>Source"]
        GAUSS["Gaussian<br/>Source"]
    end

    subgraph Boundaries["Boundary Module"]
        BCOND["Boundary<br/>Conditions"]
        LOSS["Loss<br/>Tracking"]
    end

    subgraph Audit["Audit Module"]
        CONS["Conservation<br/>Check"]
        GLOBAL["Global<br/>Budget"]
        REPORT["Report<br/>Generation"]
    end

    subgraph CUDA["CUDA Kernels"]
        KERNELS["K1-K6<br/>Pipeline"]
    end

    subgraph Validation["Validation Module"]
        BRAGG["Bragg Peak<br/>Validation"]
        LATERAL["Lateral Spread<br/>Validation"]
        DET["Determinism<br/>Test"]
    end

    LUT --> PHYSICS
    CONFIG --> DATASTRUCTURES
    LOG --> ALL

    GRIDS --> PSI
    BLOCKS --> PSI
    LOCAL --> PSI
    PSI --> BUCKETS

    LUT --> HIGHLAND
    LUT --> STEP
    LUT --> STRAGGLE
    LUT --> NUCLEAR

    HIGHLAND --> PHYSICS
    STRAGGLE --> PHYSICS
    NUCLEAR --> PHYSICS
    STEP --> PHYSICS
    FERMI --> PHYSICS

    DATASTRUCTURES --> SOURCES
    PHYSICS --> KERNELS
    SOURCES --> KERNELS

    BUCKETS --> KERNELS
    BCOND --> KERNELS
    LOSS --> BOUNDARIES

    KERNELS --> CONS
    LOSS --> GLOBAL
    CONS --> GLOBAL

    GLOBAL --> REPORT

    KERNELS --> VALIDATION
    LUT --> BRAGG
    FERMI --> LATERAL
    KERNELS --> DET
```

---

## CUDA Kernel Pipeline Detail

```mermaid
sequenceDiagram
    participant CPU as CPU Host
    participant K1 as K1: ActiveMask
    participant K2 as K2: Coarse Transport
    participant K3 as K3: Fine Transport
    participant K4 as K4: Bucket Transfer
    participant K5 as K5: Conservation Audit
    participant K6 as K6: Swap Buffers

    Note over CPU: Initialize Simulation
    CPU->>K1: Launch with PsiC_in

    Note over K1: Scan all cells<br/>Check E < E_trigger<br/>Set ActiveMask
    K1->>K2: Return ActiveMask

    CPU->>K2: Launch for Coarse Cells
    Note over K2: High-energy cells<br/>Fast approximate physics<br/>No straggling
    K2->>K4: Return OutflowBuckets

    CPU->>K3: Launch for Fine Cells
    Note over K3: Low-energy cells<br/>Full physics:<br/>- MCS with variance<br/>- Vavilov straggling<br/>- Nuclear attenuation<br/>- 2-bin energy discretization
    K3->>K4: Return OutflowBuckets

    CPU->>K4: Launch Bucket Transfer
    Note over K4: Transfer particles<br/>between cells<br/>Atomic slot allocation
    K4->>K5: Return PsiC_out

    CPU->>K5: Launch Conservation Audit
    Note over K5: Check W_in = W_out + W_loss<br/>Check E_in = E_out + E_dep<br/>Compute relative errors
    K5->>K6: Return AuditReport

    CPU->>K6: Swap Buffers
    Note over K6: Exchange in/out pointers<br/>No memory copy

    K6->>K1: Ready for next iteration
```

---

## Memory Layout

```mermaid
block-beta
    columns 8

    block:PSI1:3
        A["PsiC_in<br/>(1.1GB)"]
    end
    block:PSI2:3
        B["PsiC_out<br/>(1.1GB)"]
    end
    block:EDEP:2
        C["EdepC<br/>(0.5GB)<br/>float64"]
    end
    block:AB1:2
        D["AbsorbedWeight<br/>_cutoff<br/>(0.25GB)"]
    end
    block:AB2:2
        E["AbsorbedWeight<br/>_nuclear<br/>(0.25GB)"]
    end
    block:AB3:2
        F["AbsorbedEnergy<br/>_nuclear<br/>(0.25GB)"]
    end
    block:BOUND:2
        G["BoundaryLoss<br/>(0.1GB)"]
    end
    block:ACTIVE:2
        H["ActiveMask<br/>ActiveList<br/>(0.5GB)"]
    end
```

### Memory Breakdown

| Buffer | Size | Type | Purpose |
|--------|------|------|---------|
| `PsiC_in/out` | 1.1GB each | `float32` | Phase-space storage (hierarchical) |
| `EdepC` | 0.5GB | `float64` | Energy deposition grid |
| `AbsorbedWeight_cutoff` | 0.25GB | `float32` | Cutoff weight tracking |
| `AbsorbedWeight_nuclear` | 0.25GB | `float32` | Nuclear absorption tracking |
| `AbsorbedEnergy_nuclear` | 0.25GB | `float64` | Nuclear energy budget |
| `BoundaryLoss` | 0.1GB | `float32` | Boundary loss tracking |
| `ActiveMask/List` | 0.5GB | `uint8/uint32` | Active cell identification |

**Total**: ~4.3GB GPU memory

---

## Phase-Space Representation

```mermaid
graph TD
    subgraph "4D Phase Space"
        THETA["θ (Angle)<br/>512 bins"]
        ENERGY["E (Energy)<br/>256 bins"]
        X_SUB["x_sub (4 bins)"]
        Z_SUB["z_sub (4 bins)"]
    end

    subgraph "Block Encoding (24-bit)"
        B_THETA["b_theta<br/>12 bits"]
        B_E["b_E<br/>12 bits"]
    end

    subgraph "Local Bins (512)"
        L_THETA["θ_local: 8"]
        L_E["E_local: 4"]
        L_X["x_sub: 4"]
        L_Z["z_sub: 4"]
    end

    THETA --> B_THETA
    ENERGY --> B_E
    B_THETA -->|"encode_block"| BLOCK_ID["Block ID: 24-bit"]
    B_E --> BLOCK_ID

    L_THETA --> LOCAL_IDX["Local Index"]
    L_E --> LOCAL_IDX
    L_X --> LOCAL_IDX
    L_Z --> LOCAL_IDX

    BLOCK_ID --> PSI_CELL["PsiC Cell"]
    LOCAL_IDX --> PSI_CELL

    PSI_CELL --> STORAGE["Storage: [Nx×Nz][32 slots][512 bins]"]
```

### Encoding Details

**Block ID (24-bit):**
- Bits 0-11: `b_theta` (0-4095 angular bins)
- Bits 12-23: `b_E` (0-4095 energy bins)

**Local Index (16-bit):**
```cpp
idx = theta_local + 8 × (E_local + 4 × (x_sub + 4 × z_sub))
```

---

## Physics Pipeline per Step

```mermaid
flowchart TD
    START(["Input State:<br/>(θ, E, x, z, w)"])

    STEP1["Step Control:<br/>ds = min(2%×R, dx, dz)"]
    STEP2["Energy Loss:<br/>E = E - dE/ds × ds"]
    STEP3["Straggling:<br/>ΔE ~ Vavilov(κ)"]
    STEP4["MCS:<br/>θ = θ + σ_θ × N(0,1)"]
    STEP5["Nuclear:<br/>W = W × exp(-σ×ds)"]
    STEP6["Energy Deposition:<br/>Edep = E_in - E_out"]
    STEP7["Boundary Check:<br/>Emit to bucket if crossing"]

    START --> STEP1
    STEP1 --> STEP2
    STEP2 --> STEP3
    STEP3 --> STEP4
    STEP4 --> STEP5
    STEP5 --> STEP6
    STEP6 --> STEP7
    STEP7 --> END(["Output State"])

    classDef physicsStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    class STEP1,STEP2,STEP3,STEP4,STEP5,STEP6,STEP7 physicsStyle
```

---

## Directory Structure

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
│   ├── validation/             # Validation tests
│   │   ├── bragg_peak.cpp      # Bragg peak analysis
│   │   ├── lateral_spread.cpp  # Lateral validation
│   │   ├── determinism.cpp     # Reproducibility tests
│   │   └── deterministic_beam.cpp  # Analytical reference
│   │
│   ├── utils/                  # Utilities
│   │   ├── logger.cpp          # Logging system
│   │   ├── memory_tracker.cpp  # GPU memory tracking
│   │   └── cuda_pool.cpp       # Memory pool
│   │
│   └── cuda/kernels/           # CUDA kernels
│       ├── k1_activemask.cu    # Active cell detection
│       ├── k2_coarsetransport.cu  # High-energy transport
│       ├── k3_finetransport.cu # Fine transport (main)
│       ├── k4_transfer.cu      # Bucket transfer
│       ├── k5_audit.cu         # Conservation audit
│       └── k6_swap.cu          # Buffer swap
│
├── src/include/                # Header files (mirror structure)
│   ├── core/
│   ├── physics/
│   ├── lut/
│   ├── source/
│   ├── boundary/
│   ├── audit/
│   ├── validation/
│   └── utils/
│
├── tests/                      # Unit tests
│   ├── unit/                   # Core tests
│   ├── kernels/                # Kernel tests
│   ├── physics/                # Physics validation
│   └── validation/             # Integration tests
│
└── docs/                       # Documentation
    ├── detailed/               # This documentation
    ├── SPEC.md                 # Project specification
    └── DEV_PLAN.md             # Development plan
```

---

## Key Design Principles

1. **Block-Sparse Storage**: Only allocate memory for active phase-space blocks
2. **Hierarchical Refinement**: Coarse transport for high-energy, fine transport for low-energy
3. **GPU-First Design**: All physics computation on GPU, minimal host-device transfer
4. **Conservation by Design**: Built-in auditing at every step
5. **Modular Physics**: Each physics process in separate header for easy validation

---

## References

- NIST PSTAR Database: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
- PDG 2024: https://pdg.lbl.gov/ (Highland formula)
- ICRU Report 73: Stopping powers for electrons and positrons
