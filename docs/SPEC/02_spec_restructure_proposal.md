# SPEC 재구성 제안

작성일: 2026-02-06

## 1. 제안 요약
- 현재 `SPEC.md`를 단일 문서로 유지하면, 물리 의도와 구현 제약이 섞여 드리프트가 반복된다.
- 따라서 SPEC를 아래 2층으로 분리한다.
- 이번 개정의 중심 정책은 `상시 coarse 저장 + E<=10MeV fine scratch + 보존형 fine->coarse 반영`이다.

## 2. 문서 계층

### 2.1 `SPEC-Physics` (불변식 계층)
- 물리 보존식, 커널별 필수 물리 동작, 수용 오차.
- 환경/메모리/디바이스 상수는 최대한 배제.
- 예시 항목:
  - K3/K2 energy bookkeeping 불변식
  - Edep/Boundary/Nuclear/Cutoff 총합 규칙
  - Bragg depth 목표 범위(시험 조건 명시)
  - `E=10MeV` 전이 구간 연속성 규칙(교차 지점 분할)

### 2.2 `SPEC-Implementation` (운영 계층)
- 현재 코드 기반 구조/상수/제약/프로파일.
- 예시 항목:
  - 4D local bins (`N_theta_local=4`, `N_E_local=2`, `N_x_sub=8`, `N_z_sub=4`)
  - `DEVICE_Kb`, `DEVICE_Kb_out`, 버킷 구조(현행)
  - 메모리 프로파일(8GB/16GB/24GB)
  - fine scratch 배치 상한(`fine_batch_max_cells`)
  - debug/diagnostic 플래그 정책

## 3. 핵심 구조 제안

### 3.1 상태 저장 전략
- Persistent SoT 상태는 coarse grid만 유지한다.
- fine grid 상태는 active tile/brick에 대해서만 scratch로 생성/파기한다.
- full-grid fine 버퍼 상시 할당은 금지한다.

### 3.2 에너지 전이 규칙
- fine 계산은 `E <= 10 MeV`에서만 수행한다(`E_fine_on=10`).
- coarse 계산 중 `E=10MeV`를 가로지르면 crossing guard로 step을 분할한다.
- 옵션으로 hysteresis를 사용한다(`E_fine_off=11` 권장).

### 3.3 보존형 반영 연산자
- `P(coarse->fine)`와 `R(fine->coarse)`를 명시 연산자로 관리한다.
- `R`은 weight/energy 보존을 수학적으로 강제해야 한다.
- drop/overflow는 감사 채널에 포함하거나 validation에서 즉시 fail한다.

## 4. 현 코드 반영 원칙

### 4.1 즉시 반영(승격)
- 런타임 transport config + `energy_groups` piecewise 그리드.
- 4D local-bin 인코딩 체계(단, target에서는 fine scratch 내부 표현으로 사용).
- K1~K6 파이프라인 state/config 분리 구조.

### 4.2 조건부 반영(개선 후 승격)
- K5 energy audit(현재는 weight-only).
- K2/K3 중복 후처리 로직(공통 헬퍼 통합 후).
- 런타임/컴파일타임 파라미터 정책(한 방향으로 정리 후).
- full-grid bucket 구조(현행)는 batch-local transfer 구조로 대체 후만 승격.

## 5. 반영하지 말아야 할 항목
- 주석/코멘트로만 존재하는 수치.
- 테스트에서 불안정한 임시 완화값.
- 디버그 목적 계측을 기본 경로로 전제하는 규칙.

## 6. 승인 게이트
- 아래 4개가 충족돼야 `SPEC-Implementation`을 SoT로 승인:
  1. `EnergyLossOnly` 계열 핵심 테스트 통과.
  2. `K5 Weight+Energy` 감사 통합.
  3. `E=10MeV` 전이 연속성 테스트 통과(교차 분할 검증 포함).
  4. 8GB 안전 프로파일과 기본 실행 프로파일 분리 + preflight batch 상한 적용.
