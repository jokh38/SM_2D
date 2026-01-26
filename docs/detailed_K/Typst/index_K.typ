#set text(font: "Linux Libertine", size: 11pt)
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

#show math: set text(weight: "regular")

= SM_2D: 결정론적 양성자 수송 솔버

#set align(center)
*완전한 코드 문서*

_버전 1.0.0_

#set align(left)

== 초록

SM_2D는 방사선 치료 선량 계산을 위한 결정론적 양성자 수송 솔버입니다. 이 시스템은 GPU 가속(CUDA)을 통해 임상 속도 계산을 수행하며, Highland 다중 쿨롬 산란, Vavilov 에너지 분산, 핵 상호작용을 포함한 포괄적인 물리 모델과 계층적 S-행렬 방법을 구현합니다.

#v(1em)

=== 프로젝트 통계

#figure(
  table(
    columns: (auto, 1fr),
    inset: 8pt,
    align: left,
    table.header([*매개변수*], [*값*]),
    [언어], [C++17 with CUDA],
    [코드 라인 수], [~15,000],
    [GPU 메모리], [시뮬레이션당 ~4.3 GB],
    [정확도], [브래그 피크 <1%, 횡방향 확산 <15%],
    [컴퓨팅], [RTX 2080+ (Compute Capability 7.5+)],
  ),
  caption: [프로젝트 개요],
)

#v(1em)

=== 목차

#outline()

== 빠른 시작

=== SM_2D란 무엇인가요?

SM_2D는 다음을 구현합니다:

* 임상 속도 계산을 위한 GPU 가속(CUDA)
* 결정론적 수송을 위한 계층적 S-행렬 방법
* 포괄적인 물리 모델(Highland MCS, Vavilov 분산, 핵 상호작용)
* 수치 정확도 검증을 위한 보존 감사

=== 디렉토리 구조

#figure(
  table(
    columns: (auto, 2fr),
    inset: 6pt,
    align: (x, y) => (left, center).at(x),
    table.header([*디렉토리*], [*설명*]),
    [`run_simulation.cpp`], [메인 진입점],
    [`sim.ini`], [구성 파일],
    [`src/core/`], [데이터 구조(그리드, 저장소, 인코딩)],
    [`src/physics/`], [물리 모델(MCS, 분산, 핵)],
    [`src/cuda/kernels/`], [CUDA 커널(K1-K6 파이프라인)],
    [`src/lut/`], [NIST 데이터 및 사거리-에너지 테이블],
    [`src/source/`], [빔 소스(연필, 가우시안)],
    [`src/boundary/`], [경계 조건 및 손실 추적],
    [`src/audit/`], [보존 검사],
    [`src/validation/`], [물리 검증],
    [`src/utils/`], [로깅, 메모리 추적],
    [`tests/`], [단위 테스트(GoogleTest)],
  ),
  caption: [디렉토리 구조],
)

== 시스템 개요

=== CUDA 커널 파이프라인

시뮬레이션은 6단계 CUDA 커널 파이프라인을 구현합니다:

#figure(
  table(
    columns: (auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*커널*], [*목적*]),
    [K1], [활성 셀 찾기(E < E_sub.trigger)],
    [K2], [고에너지 입자를 위한 거친 수송],
    [K3], [완전한 물리와 함께 정밀 수송],
    [K4], [셀 간 버킷 전송],
    [K5], [보존 감사],
    [K6], [다음 반복을 위한 버퍼 교환],
  ),
  caption: [CUDA 커널 파이프라인],
)

=== 핵심 개념

==== 위상 공간 표현

입자는 4차원 위상 공간으로 표현됩니다:

* $theta$ (각도): -90°에서 +90°까지 512개 빈
* $E$ (에너지): 0.1에서 250 MeV까지 256개 빈(로그 간격)
* x_sub: 각 셀 내 4개 하위 빔(횡방향)
* z_sub: 각 셀 내 4개 하위 빔(깊이)

==== 블록-희소 저장소

```cpp
// 24비트 블록 ID = (b_E << 12) | b_theta
uint32_t block_id = encode_block(theta_bin, energy_bin);

// 블록당 512개 로컬 빔(분산 보존용)
uint16_t local_idx = encode_local_idx_4d(theta_local, E_local, x_sub, z_sub);
```

==== 계층적 수송

#figure(
  table(
    columns: (auto, auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*에너지 범위*], [*수송 방법*], [*이유*]),
    [E > 10 MeV], [거침(K2)], [빠름, 근사 물리],
    [E <= 10 MeV], [정밀(K3)], [브래그 피크를 위한 완전한 물리],
  ),
  caption: [에너지별 수송 방법],
)

== 물리 요약

=== 다중 쿨롬 산란(Highland)

$ sigma_theta = (13.6 " MeV" / (beta c p)) times sqrt(x / X_0) times [1 + 0.038 times ln(x / X_0)] / sqrt(2) $

* X_sub.0 (물): 360.8 mm
* 2D 보정: 적절한 분산을 위한 $1 / sqrt(2)$

=== 에너지 분산(Vavilov)

$kappa = xi / T_sub.max$를 기준으로 세 가지 영역:

* $kappa > 10$: Bohr(가우시안)
* $0.01 < kappa < 10$: Vavilov(보간)
* $kappa < 0.01$: Landau(비대칭)

=== 핵 감쇠

$ W times exp(-sigma(E) times dif s) $

ICRU 63의 에너지 의존적 단면적.

=== 단계 제어(R 기반)

$ dif s = min(0.02 times R, 1 " mm", cell_size) $

안정성을 위해 stopping power 대신 range-energy LUT를 사용합니다.

== 메모리 레이아웃

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*버퍼*], [*크기*], [*목적*]),
    [PsiC_in/out], [각각 1.1 GB], [위상 공간 저장소],
    [EdepC], [0.5 GB], [에너지 퇴적],
    [AbsorbedWeight_*], [0.5 GB], [차단/핵 추적],
    [AbsorbedEnergy_*], [0.25 GB], [핵 에너지 예산],
    [BoundaryLoss], [0.1 GB], [경계 손실],
    [ActiveMask/List], [0.5 GB], [활성 셀 추적],
  ),
  caption: [GPU 메모리 레이아웃],
)

#text(size: 10pt)[*총계: ~4.3 GB GPU 메모리*]

== 정확도 목표

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*관측 가능량*], [*목표*], [*상태*]),
    [브래그 피크 위치], [±2%], [✅ 통과],
    [횡방향 시그마(중간 범위)], [±15%], [✅ 통과],
    [횡방향 시그마(브래그)], [±20%], [✅ 통과],
    [가중치 보존], [<1e-6], [✅ 통과],
    [에너지 보존], [<1e-5], [✅ 통과],
  ),
  caption: [검증 결과],
)

== 주요 클래스

#figure(
  table(
    columns: (auto, auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*클래스*], [*모듈*], [*목적*]),
    [EnergyGrid], [core], [로그 간격 에너지 빈],
    [AngularGrid], [core], [균일 각도 빈],
    [PsiC], [core], [계층적 위상 공간 저장소],
    [RLUT], [lut], [사거리-에너지 보간],
    [PencilSource], [source], [결정론적 빔 소스],
    [GaussianSource], [source], [확률적 빔 소스],
    [GlobalAudit], [audit], [보존 추적],
    [BraggPeakResult], [validation], [피크 분석],
  ),
  caption: [주요 클래스],
)

== 추가 참고자료

* #link("architecture.typ")[아키텍처 개요] - 완전한 시스템 설계
* #link("physics.typ")[물리 모델] - 완전한 물리 참조
* #link("data_structures.typ")[데이터 구조] - 저장소 및 인코딩 세부사항
* #link("cuda_pipeline.typ")[CUDA 파이프라인] - 상세한 커널 문서
* #link("api.typ")[API 참조] - 함수별 문서

== 참고문헌

#figure(
  table(
    columns: (auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*출처*], [*주제*]),
    [NIST PSTAR], [Stopping power 및 사거리],
    [PDG 2024], [Highland 공식],
    [ICRU 63], [핵 단면적],
    [Vavilov 1957], [에너지 분산],
  ),
  caption: [참고문헌],
)

---
#set align(center)
*SM_2D 양성자 치료 수송 솔버용 생성*

#text(size: 9pt)[MIT 라이선스 - 버전 1.0.0]
