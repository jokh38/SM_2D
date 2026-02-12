# SPEC-구현 정합성 리뷰 보고서 (2026-02-06)

## 0) 검토 범위/방법

- 명세: `SPEC.md` (v0.8)
- 구현: `src/`, `tests/`, 실행 설정 `sim.ini`
- 검증 방법:
  - 정적 코드 대조 (명세 항목 ↔ 코드/테스트)
  - 테스트 실행: `ctest --test-dir build --output-on-failure`
  - 실제 실행 확인: `./run_simulation sim.ini`

## 1) 명세(SPEC) 구성 품질 판단

### 1.1 총평

명세는 섹션 구조(물리 범위, 격자, 커널 파이프라인, 감사, 테스트 케이스)가 잘 갖춰져 있어 "설계 의도"를 전달하는 문서로는 우수합니다. 다만, 현재 저장소의 실제 구현 기준 문서로 사용하기에는 내부 충돌과 누락된 결정이 있어 **부분적으로만 적합**합니다.

### 1.2 명세 자체의 주요 문제

| 항목 | 내용 | 근거 |
|---|---|---|
| 파라미터 충돌 | `E_trigger` 값이 20 MeV와 10 MeV로 문서 내부에서 충돌 | `SPEC.md:858`, `SPEC.md:928` |
| 활성화 규칙 모호성 | Active 조건 의사코드가 사실상 OR 구조로 읽히며(저에너지 또는 weight 조건), 구현 의도(저에너지 AND 충분 weight)와 다름 | `SPEC.md:833`, `SPEC.md:849` |
| 구현-운영 제약 미반영 | GPU 메모리/버킷 확장(현 코드)로 인한 실행 가능성 제약이 명세에 반영되지 않음 | `SPEC.md:148`, `SPEC.md:160` |

판정: **구조는 좋지만 현재 코드베이스의 단일 진실원천(SoT)으로는 부적합**.

## 2) 구현 정합성 리뷰 (명세 대비)

### 2.1 종합 판정

- 핵심 물리/알고리즘 요구(특히 K3, MCS, 2-bin, K5 energy audit)는 **중요 불일치가 다수 존재**합니다.
- 따라서 "명세 v0.8이 코드에 올바르게 구현되었다"고 보기 어렵습니다.

### 2.2 주요 불일치 (중요도 순)

| 중요도 | 명세 요구 | 구현 상태 | 근거 |
|---|---|---|---|
| Critical | K3에서 MCS 분산 누적 + RMS 임계 시 7-point split | 구현 없음. K3는 방향 유지 + 결정론적 가우시안 확산 중심 | `SPEC.md:253`, `SPEC.md:264`, `SPEC.md:565`, `src/cuda/kernels/k3_finetransport.cu:307`, `src/cuda/kernels/k3_finetransport.cu:351` |
| Critical | 2-bin 에너지 이산화(경계 기반) | 미사용. 단일 bin 기반 재배치/버킷 기록 | `SPEC.md:311`, `SPEC.md:658`, `src/cuda/kernels/k3_finetransport.cu:403`, `src/cuda/device/device_bucket.cuh:389` |
| Critical | K5에서 에너지 보존 감사 포함 | GPU K5는 weight 감사만 수행 | `SPEC.md:706`, `SPEC.md:722`, `src/cuda/kernels/k5_audit.cuh:5`, `src/cuda/kernels/k5_audit.cu:7` |
| Critical | 기본 실행 가능성(8GB target) | 기본 `sim.ini`가 OOM으로 실패 | `SPEC.md:14`, `sim.ini:68`, `src/include/core/local_bins.hpp:23`, `src/cuda/device/device_bucket.cuh:25`, 실행 로그(`./run_simulation sim.ini`) |
| High | 격자/로컬빈 상수: `N_E=256`, `N_theta=512`, `LOCAL_BINS=32` | 기본 설정/컴파일 상수 불일치 (`N_theta=36`, `LOCAL_BINS=256`, piecewise `N_E≈1029`) | `SPEC.md:63`, `SPEC.md:83`, `SPEC.md:103`, `src/include/core/incident_particle_config.hpp:178`, `src/include/core/local_bins.hpp:19`, `src/include/core/local_bins.hpp:23`, `src/gpu/gpu_transport_runner.cpp:80` |
| High | `Kb_out=64` | 디바이스는 `Kb_out=32` | `SPEC.md:164`, `SPEC.md:923`, `src/cuda/device/device_bucket.cuh:25` |
| High | Highland bracket<0.1 시 step 줄여 재시도 | bracket clamp(0.25), 재시도 경로 없음 | `SPEC.md:245`, `SPEC.md:248`, `src/include/physics/highland.hpp:75`, `src/cuda/device/device_physics.cuh:66` |
| High | Kb overflow 시 rebin/emergency fallback | 드롭 카운팅 위주, rebin fallback 미구현 | `SPEC.md:885`, `SPEC.md:889`, `src/cuda/device/device_bucket.cuh:84`, `src/cuda/device/device_bucket.cuh:449` |
| Medium | R-step 제어 파라미터 (2% 기준) | 디바이스는 1% + 에너지 그룹 최소 step 정책 | `SPEC.md:203`, `src/cuda/device/device_lut.cuh:171`, `src/cuda/device/device_lut.cuh:193` |
| Medium | 핵반응 단면 근사값(예: >100MeV 0.0050) | 다른 보정 모델 사용(`~0.0012@100MeV`) | `SPEC.md:290`, `src/include/physics/nuclear.hpp:27` |
| Medium | Gaussian source 구조에 `z0` 포함 | CPU Gaussian source는 `z0` 필드/사용 없음(`iz=0` 고정) | `SPEC.md:392`, `src/include/source/gaussian_source.hpp:6`, `src/source/gaussian_source.cpp:31` |

### 2.3 부분 일치 항목

| 항목 | 상태 | 근거 |
|---|---|---|
| 24-bit block encoding | 일치 | `SPEC.md:123`, `src/include/core/block_encoding.hpp:21` |
| 저에너지 우선 K1 활성화 방향성 | 부분 일치 | `SPEC.md:845`, `src/cuda/kernels/k1_activemask.cu:35` |
| 경계 neighbor -1 처리 | 일치 | `SPEC.md:461`, `src/cuda/device/device_bucket.cuh:707` |
| NIST 기반 R(E)/역변환 LUT | 일치(정밀도 기준은 상이) | `SPEC.md:747`, `src/lut/r_lut.cpp:131`, `src/lut/r_lut.cpp:171` |

## 3) 테스트/실행 근거

### 3.1 테스트 실행 결과

- 명령: `ctest --test-dir build --output-on-failure`
- 결과: **84개 중 81개 통과, 3개 실패**
- 실패 테스트:
  - `EnergyLossOnlyTest.EnergyLossOnly`
  - `EnergyLossOnlyTest.FullPhysics`
  - `EnergyLossOnlyTest.StragglingOnly`

핵심 실패 관측:

- 에너지 총계 오차가 허용치 초과 (`tests/gpu/test_energy_loss_only_gpu.cu:588`, `tests/gpu/test_energy_loss_only_gpu.cu:613`, `tests/gpu/test_energy_loss_only_gpu.cu:636`)
- Bragg peak 깊이가 기대(약 158mm) 대비 얕음 (`tests/gpu/test_energy_loss_only_gpu.cu:589`, `tests/gpu/test_energy_loss_only_gpu.cu:614`)

### 3.2 실제 실행 결과

- 명령: `./run_simulation sim.ini`
- 결과: `psi_out` 할당 단계에서 `out of memory`로 종료
- 관측 근거:
  - 입력 격자: `sim.ini` 기준 `Nx=200`, `Nz=640` (`sim.ini:68`, `sim.ini:70`)
  - `LOCAL_BINS=256` (`src/include/core/local_bins.hpp:23`)
  - `DEVICE_Kb=32` (`src/cuda/device/device_psic.cuh:30`)
  - 버킷 `DEVICE_Kb_out=32` (`src/cuda/device/device_bucket.cuh:25`)
  - 버킷 전체 할당(`N_cells*4*sizeof(DeviceOutflowBucket)`) (`src/cuda/k1k6_pipeline.cu:622`)

## 4) 명세 밖(혹은 미기록) 세부사항 리스크

### 4.1 실행 환경 리스크 (즉시 영향)

- 로컬빈 4D 확장(`LOCAL_BINS=256`)과 대형 버킷 구조가 결합되어 대형 격자에서 메모리 한계를 초과합니다.
- 이 제약은 SPEC의 8GB 타겟/메모리 표와 현재 구현 사이의 실질적 괴리입니다.

### 4.2 물리 결과 리스크 (정확도 영향)

- K3 재양자화 과정에서 연속 에너지 상태가 손실되는 구조(단일 bin 재기록)가 누적 오차/range 단축을 유발할 가능성이 큽니다.
- K5가 GPU 경로에서 에너지 보존을 직접 감사하지 않아, 에너지 드리프트를 조기 차단하지 못합니다.

### 4.3 유지보수 리스크 (진행 방해)

- 코드/주석/문서의 상수 값이 서로 다릅니다(예: `LOCAL_BINS` 관련 문구, Kb 관련 코멘트).
- `TransportConfig`는 런타임 설정을 받지만 로컬빈은 컴파일 상수 강제(`GPUTransportRunner`에서 mismatch 예외)라 실험/튜닝 반복을 어렵게 만듭니다.

## 5) 최종 결론

1. 명세 문서는 구조적으로 잘 짜여 있으나, 현재 상태에서는 내부 충돌과 최신 구현 반영 부족으로 **완전한 기준 문서로 보기 어렵습니다**.
2. 코드 구현은 명세 v0.8 핵심 요구사항과 **중요 항목에서 다수 불일치**합니다.
3. 특히 K3/MCS/2-bin/K5 에너지 감사/메모리 예산 문제는 실제 결과(Bragg depth, 에너지 총계, OOM)에 직접 영향을 주고 있습니다.
4. 따라서 현 시점 판정은 **"부분 구현 + 설계 드리프트 누적 상태"**가 타당합니다.

## 6) 우선 조치 권고

1. SPEC를 "현재 구현 기준"으로 재정렬(상수/정책/예외 포함)하거나, 반대로 코드를 v0.8 요구로 되돌리는 방향을 먼저 결정.
2. 최소한 다음 4개를 단기 우선순위로 고정:
   - K5 에너지 감사 통합
   - K3 에너지 이산화(2-bin 또는 등가 보존 스킴) 재도입
   - 메모리 예산 맞춤형 파라미터 세트(8GB 안전 프로파일) 확정
   - 실패 중인 `EnergyLossOnlyTest` 3건을 CI 게이트로 승격

