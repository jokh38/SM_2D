# SM_2D 아키텍처 개요 (Architecture Overview)

## 프로젝트 요약 (Project Summary)

**SM_2D**는 CUDA 가속 GPU 컴퓨팅을 사용하는 양성자 치료 선량 계산을 위한 고성능 2D 결정론적 전송 솔버입니다. 이 프로젝트는 블록-희소 위상 공간 표현을 사용하는 계층적 S-매트릭스 솔버를 구현합니다.

### 주요 통계 (Key Statistics)
- **전체 파일**: 30개 이상의 C++ 소스 파일
- **CUDA 커널**: 6개 주요 커널 (K1-K6)
- **코드 라인**: 약 15,000 라인
- **시뮬레이션당 메모리**: 약 3GB GPU 메모리
- **그리드 크기**: 최대 200 × 640 셀

---

## 시스템 아키텍처 (System Architecture)

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

## 모듈 의존성 그래프 (Module Dependency Graph)

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

## CUDA 커널 파이프라인 상세 (CUDA Kernel Pipeline Detail)

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

## 메모리 레이아웃 (Memory Layout)

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

### 메모리 분석 (Memory Breakdown)

| 버퍼 (Buffer) | 크기 (Size) | 타입 (Type) | 용도 (Purpose) |
|--------|------|------|---------|
| `PsiC_in/out` | 각 1.1GB | `float32` | 위상 공간 저장 (계층적) |
| `EdepC` | 0.5GB | `float64` | 에너지 퇴적 그리드 |
| `AbsorbedWeight_cutoff` | 0.25GB | `float32` | 컷오프 가중치 추적 |
| `AbsorbedWeight_nuclear` | 0.25GB | `float32` | 핵 흡수 추적 |
| `AbsorbedEnergy_nuclear` | 0.25GB | `float64` | 핵 에너지 예산 |
| `BoundaryLoss` | 0.1GB | `float32` | 경계 손실 추적 |
| `ActiveMask/List` | 0.5GB | `uint8/uint32` | 활성 셀 식별 |

**전체**: 약 4.3GB GPU 메모리

---

## 위상 공간 표현 (Phase-Space Representation)

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

### 인코딩 상세 (Encoding Details)

**블록 ID (24-bit):**
- 비트 0-11: `b_theta` (0-4095 각도 빈)
- 비트 12-23: `b_E` (0-4095 에너지 빈)

**로컬 인덱스 (16-bit):**
```cpp
idx = theta_local + 8 × (E_local + 4 × (x_sub + 4 × z_sub))
```

---

## 스텝별 물리 파이프라인 (Physics Pipeline per Step)

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

## 디렉토리 구조 (Directory Structure)

```
SM_2D/
├── run_simulation.cpp          # 메인 진입점 (Main entry point)
├── sim.ini                     # 설정 파일 (Configuration file)
├── visualize.py                # Python 시각화 (Python visualization)
│
├── src/
│   ├── core/                   # 핵심 데이터 구조 (Core data structures)
│   │   ├── grids.cpp           # 에너지/각도 그리드 (Energy/angle grids)
│   │   ├── block_encoding.hpp  # 24-bit 인코딩 (24-bit encoding)
│   │   ├── local_bins.hpp      # 4D 서브셀 분할 (4D sub-cell partitioning)
│   │   ├── psi_storage.cpp     # 계층적 위상 공간 (Hierarchical phase-space)
│   │   └── buckets.cpp         # 버킷 방출 (Bucket emission)
│   │
│   ├── physics/                # 물리 구현 (Physics implementations)
│   │   ├── highland.hpp        # 다중 쿨롱 산란 (Multiple Coulomb scattering)
│   │   ├── energy_straggling.hpp  # 바빌로프 스트래글링 (Vavilov straggling)
│   │   ├── nuclear.hpp         # 핵 감쇠 (Nuclear attenuation)
│   │   ├── step_control.hpp    # R 기반 스텝 제어 (R-based step control)
│   │   └── fermi_eyges.hpp     # 횡방향 확산 이론 (Lateral spread theory)
│   │
│   ├── lut/                    # 룩업 테이블 (Lookup tables)
│   │   ├── nist_loader.cpp     # NIST PSTAR 데이터 (NIST PSTAR data)
│   │   └── r_lut.cpp           # 거리-에너지 보간 (Range-energy interpolation)
│   │
│   ├── source/                 # 빔 소스 (Beam sources)
│   │   ├── pencil_source.cpp   # 펜슬 빔 (Pencil beam)
│   │   └── gaussian_source.cpp # 가우시안 빔 (Gaussian beam)
│   │
│   ├── boundary/               # 경계 조건 (Boundary conditions)
│   │   ├── boundaries.cpp      # 경계 타입 (Boundary types)
│   │   └── loss_tracking.cpp   # 손실 회계 (Loss accounting)
│   │
│   ├── audit/                  # 보존 감사 (Conservation auditing)
│   │   ├── conservation.cpp    # 가중치/에너지 검사 (Weight/energy checks)
│   │   ├── global_budget.cpp   # 전체 집계 (Global aggregation)
│   │   └── reporting.cpp       # 보고서 생성 (Report generation)
│   │
│   ├── validation/             # 검증 테스트 (Validation tests)
│   │   ├── bragg_peak.cpp      # 브래그 피크 분석 (Bragg peak analysis)
│   │   ├── lateral_spread.cpp  # 횡방향 검증 (Lateral validation)
│   │   ├── determinism.cpp     # 재현성 테스트 (Reproducibility tests)
│   │   └── deterministic_beam.cpp  # 해석적 참조 (Analytical reference)
│   │
│   ├── utils/                  # 유틸리티 (Utilities)
│   │   ├── logger.cpp          # 로깅 시스템 (Logging system)
│   │   ├── memory_tracker.cpp  # GPU 메모리 추적 (GPU memory tracking)
│   │   └── cuda_pool.cpp       # 메모리 풀 (Memory pool)
│   │
│   └── cuda/kernels/           # CUDA 커널 (CUDA kernels)
│       ├── k1_activemask.cu    # 활성 셀 감지 (Active cell detection)
│       ├── k2_coarsetransport.cu  # 고에너지 전송 (High-energy transport)
│       ├── k3_finetransport.cu # 정밀 전송 (Fine transport, main)
│       ├── k4_transfer.cu      # 버킷 전송 (Bucket transfer)
│       ├── k5_audit.cu         # 보존 감사 (Conservation audit)
│       └── k6_swap.cu          # 버퍼 스왑 (Buffer swap)
│
├── src/include/                # 헤더 파일 (Header files, mirror structure)
│   ├── core/
│   ├── physics/
│   ├── lut/
│   ├── source/
│   ├── boundary/
│   ├── audit/
│   ├── validation/
│   └── utils/
│
├── tests/                      # 단위 테스트 (Unit tests)
│   ├── unit/                   # 코어 테스트 (Core tests)
│   ├── kernels/                # 커널 테스트 (Kernel tests)
│   ├── physics/                # 물리 검증 (Physics validation)
│   └── validation/             # 통합 테스트 (Integration tests)
│
└── docs/                       # 문서 (Documentation)
    ├── detailed/               # 이 문서 (This documentation)
    ├── SPEC.md                 # 프로젝트 명세서 (Project specification)
    └── DEV_PLAN.md             # 개발 계획 (Development plan)
```

---

## 핵심 설계 원칙 (Key Design Principles)

1. **블록-희소 저장 (Block-Sparse Storage)**: 활성 위상 공간 블록에만 메모리 할당
2. **계층적 세분화 (Hierarchical Refinement)**: 고에너지용 거친 전송, 저에너지용 정밀 전송
3. **GPU 우선 설계 (GPU-First Design)**: 모든 물리 계산은 GPU에서, 최소의 호스트-디바이스 전송
4. **설계에 의한 보존 (Conservation by Design)**: 모든 단계에서 내장된 감사
5. **모듈식 물리 (Modular Physics)**: 각 물리 프로세스는 검증을 위해 별도 헤더에 구현

---

## 참고문헌 (References)

- NIST PSTAR Database: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
- PDG 2024: https://pdg.lbl.gov/ (Highland formula)
- ICRU Report 73: Stopping powers for electrons and positrons
