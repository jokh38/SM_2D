# 메모리 예산 및 운영 프로파일

작성일: 2026-02-06

## 1. 현행 dense 구조 계산식 (현재 구현 상수 기준)

기준 상수:
- `LOCAL_BINS = 256` (`src/include/core/local_bins.hpp:23`)
- `DEVICE_Kb = 32` (`src/cuda/device/device_psic.cuh:30`)
- `DEVICE_Kb_out = 32` (`src/cuda/device/device_bucket.cuh:25`)

`N_cells = Nx * Nz`일 때:

- PsiC(단일 버퍼):
  - `N_cells * DEVICE_Kb * (sizeof(block_id) + LOCAL_BINS*sizeof(float))`
  - `= N_cells * 32 * (4 + 256*4) = N_cells * 32896 bytes`
- PsiC(2버퍼):
  - `N_cells * 65792 bytes`
- OutflowBuckets:
  - `N_cells * 4 * sizeof(DeviceOutflowBucket)`
  - `sizeof(DeviceOutflowBucket) ~= 32972 bytes`
  - `= N_cells * 131888 bytes`

주요 3개 합계:
- `N_cells * (65792 + 131888) = N_cells * 197680 bytes`

추가 배열(ActiveMask/List, audit, tally 등)까지 포함한 실사용 근사치:
- `N_cells * 197785 bytes`

## 2. 현행 dense 구조 대표 케이스

### 2.1 기본 `sim.ini` (`Nx=200`, `Nz=640`)
- `N_cells = 128000`
- 주요 구조 합계: 약 `23.57 GiB`
- 결론: 8GB 환경에서 실행 불가.

### 2.2 8GB 후보 프로파일
- `160 x 240`:
  - 주요 구조 합계: 약 `7.07 GiB`
  - 기타 배열/오버헤드 고려 시 경계선.
- `140 x 220`:
  - 주요 구조 합계: 약 `5.67 GiB`
  - 상대적으로 안전.
- `120 x 200`:
  - 주요 구조 합계: 약 `4.42 GiB`
  - 디버그/추가 계측 포함 시에도 여유.

## 3. 목표 구조 계산식 (coarse persistent + fine scratch)

목표 구조에서는 메모리를 아래처럼 분리한다.

`M_total = N_cells * B_coarse + N_fine_batch * B_fine + M_overhead`

- `B_coarse`: 상시 coarse 상태 바이트/셀.
- `B_fine`: fine scratch 바이트/셀(저에너지 active 셀에만 적용).
- `N_fine_batch`: 동시에 처리하는 fine scratch 셀 수.

보수적 상한 추정(현행 dense per-cell footprint 사용):
- `B_fine ~= 197785 bytes/cell`

8GB 장치에서 안전마진 15%를 제외한 사용 가능 메모리:
- `8 GiB * 0.85 ~= 6.8 GiB`
- `N_fine_batch_max ~= floor(6.8 GiB / 197785) = 36916 cells`

참고(보수적 `B_fine` 가정 시):
- `N_fine_batch=20000` -> 약 `3.68 GiB`
- `N_fine_batch=30000` -> 약 `5.53 GiB`
- `N_fine_batch=36000` -> 약 `6.63 GiB`

핵심:
- 실제 `B_coarse`는 `B_fine`보다 훨씬 작아야 하며, 이 차이가 목표 구조의 실질 이득이다.
- fine 비율이 낮을수록(`E<=10MeV` 영역 제한) 8GB 운영 여유가 크게 증가한다.

## 4. 운영 정책 제안

1. 기본 실행 프로파일과 8GB 안전 프로파일을 분리한다.
2. 실행 전 Preflight에서 다음을 강제한다.
   - `coarse persistent`, `fine scratch`, `overhead`를 분리 출력
   - `N_fine_batch_max` 산정 및 적용
   - `예상 사용량 > (VRAM * safety_margin)`이면 실패-fast
3. `sim.ini` 기본값은 최소 8GB 안전 프로파일로 교체한다.
4. 저에너지 영역(`E<=10MeV`) active 셀이 급증할 때는 배치를 축소해 scratch를 순차 처리한다.

## 5. 실측 참고 (현행 dense 경로, RTX 2080 8GB)

- `120x200`, `max_steps=200`: `91.197 s`
- `140x220`, `max_steps=200`: `113.857 s`
- `160x240`, `max_steps=200`: `149.461 s`

시사점:
- 현행 dense 경로는 메모리뿐 아니라 시간도 목표(수십초) 대비 불리하다.
- target 구조의 핵심 성과 지표는 `N_fine_batch` 축소와 전역 초기화 비용 제거다.

## 6. 반영 우선순위
- P0(Preflight) 완료 전까지는 큰 격자 기본값을 배포 기본 설정으로 두지 않는다.
