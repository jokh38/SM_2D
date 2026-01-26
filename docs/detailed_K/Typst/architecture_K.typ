#set text(font: "Linux Libertine", size: 11pt)
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

#show math: set text(weight: "regular")
#include "std"

= 아키텍처 개요

== 프로젝트 요약

SM_2D는 CUDA 가속 GPU 컴퓨팅을 사용하는 양성자 치료 선량 계산을 위한 고성능 2D 결정론적 수송 솔버입니다. 이 프로젝트는 블록-희소 위상 공간 표현을 사용하는 계층적 S-행렬 솔버를 구현합니다.

=== 핵심 통계

#figure(
  table(
    columns: (auto, 1fr),
    inset: 8pt,
    align: left,
    table.header([*메트릭*], [*값*]),
    [전체 파일], [30개 이상의 C++ 소스 파일],
    [CUDA 커널], [6개 주요 커널(K1-K6)],
    [코드 라인 수], [~15,000 라인],
    [시뮬레이션당 메모리], [~3 GB GPU 메모리],
    [그리드 크기], [최대 200 × 640 셀],
  ),
  caption: [시스템 메트릭],
)

== 시스템 아키텍처

=== 아키텍처 계층

시스템은 6개의 개별 계층으로 구성됩니다:

==== 입력 계층

* `sim.ini`의 구성
* NIST PSTAR 물리 데이터
* 명령줄 매개변수

==== 코어 계층

* 에너지/각도 그리드
* 블록 인코딩(24비트)
* 위상 공간 저장소(계층적)
* 버킷 방출(셀 간)

==== 물리 계층

* Highland MCS
* Vavilov 분산
* 핵 감쇠
* R 기반 단계 제어
* Fermi-Eyges 횡방향 확산

==== CUDA 파이프라인

* K1: ActiveMask
* K2: 거친 수송
* K3: 정밀 수송(주요 물리)
* K4: 버킷 전송
* K5: 보존 감사
* K6: 버퍼 교환

==== 출력 계층

* 2D 선량 분포
* 깊이-선량 곡선
* 보존 보고서

== 모듈 의존성 그래프

=== 기초 계층

* LUT 모듈(`r_lut`, `nist_loader`)
* 구성 로더
* 로거

=== 데이터 구조

* 그리드(에너지, 각도)
* 블록 인코딩(24비트 ID)
* 로컬 빈(4D 하위 셀)
* Psi 저장소(계층적)
* 버킷(방출)

=== 물리 모듈

* Highland MCS
* 에너지 분산
* 핵 감쇠
* 단계 제어
* Fermi-Eyges 확산

=== CUDA 커널

* K1-K6 파이프라인

=== 소스

* 연필 소스
* 가우시안 소스

=== 경계

* 경계 조건
* 손실 추적

=== 감사

* 보존 검사
* 전체 예산
* 보고서 생성

=== 검증

* 브래그 피크 검증
* 횡방향 확산 검증
* 결정론성 테스트

== 메모리 레이아웃

=== GPU 메모리 분해

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*버퍼*], [*크기*], [*유형*]),
    [PsiC_in/out], [각각 1.1 GB], [float32],
    [EdepC], [0.5 GB], [float64],
    [AbsorbedWeight_cutoff], [0.25 GB], [float32],
    [AbsorbedWeight_nuclear], [0.25 GB], [float32],
    [AbsorbedEnergy_nuclear], [0.25 GB], [float64],
    [BoundaryLoss], [0.1 GB], [float32],
    [ActiveMask/List], [0.5 GB], [uint8/uint32],
  ),
  caption: [GPU 메모리 레이아웃],
)

#text(size: 10pt)[*총계: ~4.3 GB GPU 메모리*]

== 위상 공간 표현

=== 4차원 위상 공간

입자는 4차원으로 표현됩니다:

* $theta$ (각도): -90°에서 +90°까지 512개 빈
* $E$ (에너지): 0.1에서 250 MeV까지 256개 빈(로그 간격)
* x_sub: 각 셀 내 4개 하위 빔(횡방향)
* z_sub: 각 셀 내 4개 하위 빔(깊이)

=== 블록 인코딩(24비트)

```
┌─────────────────────────┬──────────────────────────┐
│     b_E (12 bits)       │    b_theta (12 bits)     │
│    Bits 12-23           │     Bits 0-11            │
│    Range: 0-4095        │     Range: 0-4095        │
└─────────────────────────┴──────────────────────────┘
                    24-bit Block ID
```

=== 인코딩 세부사항

* 비트 0-11: `b_theta` (0-4095 각도 빈)
* 비트 12-23: `b_E` (0-4095 에너지 빈)

=== 로컬 인덱스(16비트)

로컬 인덱스 인코딩:

$ "idx" = theta_sub."local" + 8 times (E_sub."local" + 4 times (x_sub + 4 times z_sub)) $

여기서:
* $theta_sub."local"$: 8개 값(0-7)
* $E_sub."local"$: 4개 값(0-3)
* $x_sub$: 4개 값(0-3)
* $z_sub$: 4개 값(0-3)

전체: $8 times 4 times 4 times 4 = 512$ 블록당 로컬 빈

== 단계별 물리 파이프라인

=== 단계 시퀀스

각 수송 단계에서 다음 작업이 수행됩니다:

1. *단계 제어*: $dif s = min(2% times R, dif x, dif z)$
2. *에너지 손실*: $E = E - dif E/dif s times dif s$
3. *분산*: $dif E ~ "Vavilov"(kappa)$
4. *MCS*: $theta = theta + sigma_theta times N(0,1)$
5. *핵*: $W = W times exp(-sigma times dif s)$
6. *에너지 퇴적*: $E_sub.dep = E_sub."in" - E_sub."out"$
7. *경계 검사*: 교차 시 버킷에 방출

== 디렉토리 구조

```
SM_2D/
├── run_simulation.cpp          # 메인 진입점
├── sim.ini                     # 구성 파일
├── visualize.py                # Python 시각화
│
├── src/
│   ├── core/                   # 코어 데이터 구조
│   │   ├── grids.cpp           # 에너지/각도 그리드
│   │   ├── block_encoding.hpp  # 24비트 인코딩
│   │   ├── local_bins.hpp      # 4D 하위 셀 분할
│   │   ├── psi_storage.cpp     # 계층적 위상 공간
│   │   └── buckets.cpp         # 버킷 방출
│   │
│   ├── physics/                # 물리 구현
│   │   ├── highland.hpp        # 다중 쿨롬 산란
│   │   ├── energy_straggling.hpp  # Vavilov 분산
│   │   ├── nuclear.hpp         # 핵 감쇠
│   │   ├── step_control.hpp    # R 기반 단계 제어
│   │   └── fermi_eyges.hpp     # 횡방향 확산 이론
│   │
│   ├── lut/                    # 조회 테이블
│   │   ├── nist_loader.cpp     # NIST PSTAR 데이터
│   │   └── r_lut.cpp           # 사거리-에너지 보간
│   │
│   ├── source/                 # 빔 소스
│   │   ├── pencil_source.cpp   # 연필 빔
│   │   └── gaussian_source.cpp # 가우시안 빔
│   │
│   ├── boundary/               # 경계 조건
│   │   ├── boundaries.cpp      # 경계 유형
│   │   └── loss_tracking.cpp   # 손실 회계
│   │
│   ├── audit/                  # 보존 감사
│   │   ├── conservation.cpp    # 가중치/에너지 검사
│   │   ├── global_budget.cpp   # 전체 집계
│   │   └── reporting.cpp       # 보고서 생성
│   │
│   ├── cuda/kernels/           # CUDA 커널
│   │   ├── k1_activemask.cu    # 활성 셀 감지
│   │   ├── k2_coarsetransport.cu  # 고에너지 수송
│   │   ├── k3_finetransport.cu # 정밀 수송(메인)
│   │   ├── k4_transfer.cu      # 버킷 전송
│   │   ├── k5_audit.cu         # 보존 감사
│   │   └── k6_swap.cu          # 버퍼 교환
│   │
│   └── utils/                  # 유틸리티
│       ├── logger.cpp          # 로깅 시스템
│       └── memory_tracker.cpp  # GPU 메모리 추적
```

== 핵심 설계 원칙

=== 1. 블록-희소 저장소

활성 위상 공간 블록에만 메모리를 할당하여 조밀 저장소 대비 70% 이상 메모리 절약.

=== 2. 계층적 세분화

고에너지용 거친 수송, 저에너지용 정밀 수송(브래그 피크 영역).

=== 3. GPU 우선 설계

모든 물리 연산을 GPU에서 수행, 호스트-장치 전송 최소화.

=== 4. 설계에 의한 보존

모든 단계에서 가중치와 에너지 보존을 위한 내장 감사.

=== 5. 모듈형 물리

각 물리 과정을 별도 헤더에 구현하여 검증과 테스트 용이.

== 참고문헌

* NIST PSTAR 데이터베이스: #link("https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html")[https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html]
* PDG 2024: #link("https://pdg.lbl.gov/")[https://pdg.lbl.gov/] (Highland 공식)
* ICRU 보고서 73: 전자와 양전자에 대한 stopping power

---
#set align(center)
*SM_2D 아키텍처 문서*

#text(size: 9pt)[버전 1.0.0]
