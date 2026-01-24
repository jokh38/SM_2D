# SM_2D Visual Diagrams Collection

This document contains all Mermaid diagrams for visualizing the SM_2D system.

---

## 1. Complete System Architecture

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

## 2. CUDA Kernel Pipeline Sequence

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

## 3. Phase-Space Encoding

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

---

## 4. Module Dependency Graph

```mermaid
flowchart LR
    subgraph Foundation["Foundation Layer"]
        LUT["LUT Module"]
        CONFIG["Config Loader"]
        LOG["Logger"]
    end

    subgraph DataStructures["Data Structures"]
        GRIDS["Grids"]
        BLOCKS["Block Encoding"]
        LOCAL["Local Bins"]
        PSI["Psi Storage"]
        BUCKETS["Buckets"]
    end

    subgraph Physics["Physics Module"]
        HIGHLAND["Highland MCS"]
        STRAGGLE["Energy Straggling"]
        NUCLEAR["Nuclear"]
        STEP["Step Control"]
        FERMI["Fermi-Eyges"]
    end

    subgraph Kernels["CUDA Kernels"]
        KERNELS["K1-K6 Pipeline"]
    end

    subgraph Sources["Sources"]
        PENCIL["Pencil Source"]
        GAUSS["Gaussian Source"]
    end

    subgraph Output["Output"]
        AUDIT["Audit"]
        VALID["Validation"]
    end

    LUT --> PHYSICS
    CONFIG --> DATASTRUCTURES

    GRIDS --> PSI
    BLOCKS --> PSI
    LOCAL --> PSI

    PSI --> BUCKETS
    DATASTRUCTURES --> SOURCES
    PHYSICS --> KERNELS
    SOURCES --> KERNELS

    KERNELS --> AUDIT
    KERNELS --> VALID
```

---

## 5. Physics Pipeline

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

## 6. Bucket Transfer Flow

```mermaid
graph TD
    subgraph Cell["Cell (i,j)"]
        P["Particles"]
    end

    subgraph Buckets["Outflow Buckets"]
        BZP["Bucket +z"]
        BZM["Bucket -z"]
        BXP["Bucket +x"]
        BXM["Bucket -x"]
    end

    subgraph Neighbors["Receiving Cells"]
        NZP["Cell (i, j+1)"]
        NZM["Cell (i, j-1)"]
        NXP["Cell (i+1, j)"]
        NXM["Cell (i-1, j)"]
    end

    P -->|"exit +z"| BZP
    P -->|"exit -z"| BZM
    P -->|"exit +x"| BXP
    P -->|"exit -x"| BXM

    BZP --> NZP
    BZM --> NZM
    BXP --> NXP
    BXM --> NXM

    classDef cellStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef bucketStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    class P,NZP,NZM,NXP,NXM cellStyle
    class BZP,BZM,BXP,BXM bucketStyle
```

---

## 7. Memory Layout

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

---

## 8. Class Hierarchy

```mermaid
classDiagram
    class EnergyGrid {
        +int N_E
        +float E_min
        +float E_max
        +vector~float~ edges
        +vector~float~ rep
        +FindBin(float E) int
        +GetRepEnergy(int bin) float
    }

    class AngularGrid {
        +int N_theta
        +float theta_min
        +float theta_max
        +vector~float~ edges
        +vector~float~ rep
        +FindBin(float theta) int
        +GetRepTheta(int bin) float
    }

    class PsiC {
        +int Nx, Nz
        +int Kb
        +vector block_id
        +vector value
        +find_or_allocate_slot() int
        +get_weight() float
        +set_weight() void
        +clear() void
    }

    class RLUT {
        +EnergyGrid grid
        +vector~float~ R
        +vector~float~ S
        +lookup_R(float E) float
        +lookup_S(float E) float
        +lookup_E_inverse(float R) float
    }

    class PencilSource {
        +float x0, z0
        +float theta0
        +float E0
        +float W_total
    }

    class GaussianSource {
        +float x0, z0
        +float sigma_x
        +float sigma_theta
        +float E0, sigma_E
        +int n_samples
    }

    class GlobalAudit {
        +double W_in_total
        +double W_out_total
        +double W_error
        +double E_in_total
        +double E_out_total
        +double E_error
        +weight_pass() bool
        +energy_pass() bool
    }

    RLUT --> EnergyGrid : uses
    PsiC --> EnergyGrid : uses
    PsiC --> AngularGrid : uses
```

---

## 9. Simulation Flow

```mermaid
flowchart TD
    START(["Start"])

    INIT["Load Config (sim.ini)"]
    LUT["Build R-LUT from NIST data"]
    GRID["Create Energy/Angle grids"]
    SRC["Initialize Source"]
    ALLOC["Allocate GPU memory"]

    STEP_Loop["Simulation Step Loop"]

    K1_CALL["K1: ActiveMask"]
    K2_CALL["K2: Coarse Transport"]
    K3_CALL["K3: Fine Transport"]
    K4_CALL["K4: Bucket Transfer"]
    K5_CALL["K5: Conservation Audit"]
    K6_CALL["K6: Swap Buffers"]

    CHECK_END{"Max steps<br/>reached?"}

    COLLECT["Collect results from GPU"]
    AUDIT_CHECK{"Audit<br/>passed?"}
    SAVE["Save dose files"]
    ERROR(["Exit with error"])
    SUCCESS(["Exit success"])

    START --> INIT
    INIT --> LUT
    LUT --> GRID
    GRID --> SRC
    SRC --> ALLOC
    ALLOC --> STEP_Loop

    STEP_Loop --> K1_CALL
    K1_CALL --> K2_CALL
    K2_CALL --> K3_CALL
    K3_CALL --> K4_CALL
    K4_CALL --> K5_CALL
    K5_CALL --> K6_CALL
    K6_CALL --> CHECK_END

    CHECK_END -->|No| STEP_Loop
    CHECK_END -->|Yes| COLLECT

    COLLECT --> AUDIT_CHECK
    AUDIT_CHECK -->|No| ERROR
    AUDIT_CHECK -->|Yes| SAVE
    SAVE --> SUCCESS
```

---

## 10. Grid Structure

```mermaid
graph TD
    subgraph World["World (200mm × 320mm)"]
        subgraph Grid["Computational Grid (200 × 640 cells)"]
            subgraph Cell["Cell (i,j) - 0.5mm × 0.5mm"]
                subgraph Subcell["Sub-cell bins (4 × 4)"]
                    S1["x=0"]
                    S2["x=1"]
                    S3["x=2"]
                    S4["x=3"]
                end
            end
        end
    end

    WORLD["x: 0-200mm, z: 0-320mm"] --> GRID
    GRID --> Cell
    Cell --> Subcell
```

---

*Diagram Collection for SM_2D Documentation*
