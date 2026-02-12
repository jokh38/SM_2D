# SPEC 개선 문서 세트

작성일: 2026-02-06

## 목적
- `SPEC.md`와 현재 구현 간 드리프트를 줄이기 위한 기준 문서 묶음이다.
- 물리 불변식(Physics)과 구현/운영 제약(Implementation)을 분리해 관리한다.
- 본 라운드의 핵심 기준은 아래 3가지다.
  - 상시 저장 상태는 coarse grid로 제한한다.
  - fine grid 계산은 `E <= 10 MeV` 저에너지 영역에서만 임시 scratch로 수행한다.
  - fine 결과는 보존형(`weight/energy`)으로 coarse 상태에 반영한다.

## 문서 목록
- `docs/SPEC/01_assessment_2026-02-06.md`
  - 현재 코드 평가, 저에너지 fine-scratch 목표 대비 갭, 재검증 결과.
- `docs/SPEC/02_spec_restructure_proposal.md`
  - SPEC 재구성(2층 분리) + `E<=10 MeV` fine-scratch 구조 제안.
- `docs/SPEC/03_code_improvement_roadmap.md`
  - 코드 개선 우선순위(전이 규칙, scratch, 보존 감사, 8GB 운영 순).
- `docs/SPEC/04_memory_budget_profile.md`
  - 현행 dense 구조 메모리 계산식 + target scratch 구조 예산식, 8GB 운영 프로파일.
- `docs/SPEC/05_file_level_backlog.md`
  - 실제 코드 반영을 위한 파일 단위 작업 목록(저에너지 fine-scratch 기준).

## 현재 상태 요약
- 구현은 여전히 full-grid dense 경로 중심이며, 목표 구조(상시 coarse + 임시 fine scratch)와 차이가 크다.
- 정확도 게이트(`EnergyLossOnly` 계열)는 4개 중 3개 실패 상태다.
- 기본 `sim.ini` 격자(`200x640`)는 8GB 환경에서 OOM으로 실행 불가하므로 운영 프로파일 분리가 필수다.
