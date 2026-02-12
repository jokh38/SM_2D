# 코드 개선 로드맵 (우선순위)

작성일: 2026-02-06

## P0. 저에너지 전이 규칙 고정 (`E<=10MeV`)

### 목표
- fine/coarse 경계 조건을 코드/설정/문서에서 단일 의미로 고정한다.

### 작업
- `E_fine_on=10MeV`, `E_fine_off=11MeV`(옵션) 파라미터 추가.
- coarse step 중 `E=10MeV` crossing guard 구현.
- 전이 정책 검증 테스트(경계 연속성) 추가.

### 완료 기준
- 전이 경계에서 Bragg/에너지 회계 불연속이 재현되지 않는다.

## P1. coarse persistent + fine scratch 구조 전환

### 목표
- 상시 full-grid fine 메모리 할당을 제거하고, low-energy active 영역만 scratch로 계산한다.

### 작업
- persistent `CoarseState` 정의(상시 저장).
- `FineScratchManager` 도입(배치/재사용/해제).
- `P(coarse->fine)`, `R(fine->coarse)` 경로 도입.

### 완료 기준
- full-grid `psi_in/psi_out` 상시 할당 없이 저에너지 셀만 fine 계산한다.

## P2. K5를 Weight+Energy 통합 감사로 확장

### 목표
- iteration 단위로 에너지 보존을 커널 레벨에서 강제한다.

### 작업
- `AuditReport`에 energy 항목 추가.
- `K5_WeightAudit`를 `K5_ConservationAudit`로 확장:
  - `E_in`, `E_out`, `E_dep`, `E_boundary`, `E_nuclear`, `E_cutoff` 집계.
- 임계 초과 시 런타임 fail-fast 옵션 추가.

### 완료 기준
- K5 결과만으로 weight/energy 보존 판정 가능.

## P3. 메모리 Preflight + batch 스케줄러 도입

### 목표
- 8GB 기준에서 scratch 배치 상한을 자동 산정하고 OOM을 사전에 차단한다.

### 작업
- `PreflightMemoryPlanner` 추가:
  - coarse persistent
  - fine scratch batch
  - pipeline overhead
- `fine_batch_max_cells` 자동 다운시프트 또는 fail-fast.

### 완료 기준
- `sim.ini` 같은 대형 케이스에서 OOM 전에 명확한 에러 메시지로 종료.

## P4. K2/K3 공통 후처리 정리

### 목표
- 중복 코드를 단일 경로로 정리해 회귀 리스크를 줄인다.

### 작업
- 공통 device helper 추출:
  - 재양자화
  - in-cell spread
  - lateral tail 처리
  - transfer/drop accounting
- K2/K3는 물리 계산 부분만 다르게 유지.

### 완료 기준
- K2/K3 중복 블록 제거.
- 동일 시나리오에서 기존 대비 결과 편차 허용범위 내 유지.

## P5. 테스트/CI 게이트 강화

### 목표
- 현재 실패 중인 물리 회귀를 CI에서 즉시 검출한다.

### 작업
- `EnergyLossOnly` 계열을 필수 게이트로 승격.
- `10MeV transition continuity` 테스트 신설.
- 에너지 회계 출력을 표준화(`Edep+Boundary+Nuclear+Cutoff`).

### 완료 기준
- CI에서 Bragg depth 단축, 회계 오차, 전이 경계 불연속 재발 시 즉시 fail.

## P6. SPEC 동기화 자동화 + 3D 확장 준비

### 목표
- 코드 변경 후 명세 불일치를 자동 감지하고, 3D 확장에서 동일 정책을 재사용한다.

### 작업
- 주요 상수/정책 diff 체크 스크립트 추가.
- 릴리즈/PR 템플릿에 SPEC 영향 항목 포함.
- 3D 확장 시에도 `coarse persistent + low-energy fine scratch` 정책을 유지하도록 체크리스트 추가.

### 완료 기준
- SPEC와 코드 상수 드리프트를 PR 단계에서 차단.
- 3D 브랜치에서도 `E<=10MeV fine-only` 정책이 깨지지 않는다.
