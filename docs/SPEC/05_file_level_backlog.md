# 파일 단위 작업 백로그

작성일: 2026-02-06

## 1. P0 전이 규칙 (`E<=10MeV`) 반영
- `src/include/core/incident_particle_config.hpp`
  - `E_fine_on`, `E_fine_off`(옵션) 필드 추가.
- `src/include/core/config_loader.hpp`
  - INI 파싱/직렬화 항목 추가.
- `src/cuda/kernels/k1_activemask.cu`
  - hysteresis 지원 포함한 fine/coarse 분류 정책 반영.
- `src/cuda/kernels/k2_coarsetransport.cu`
  - coarse step 내 `E=10MeV` crossing guard 분할 처리.
- `src/cuda/k1k6_pipeline.cu`
  - 전이 정책 파라미터 전달 경로 정리.

## 2. P1 coarse persistent + fine scratch 구조
- 신규 파일:
  - `src/include/core/coarse_state.hpp`
  - `src/cuda/device/device_fine_scratch.cuh`
  - `src/include/gpu/fine_scratch_manager.hpp`
  - `src/gpu/fine_scratch_manager.cpp`
- 수정 파일:
  - `src/cuda/gpu_transport_wrapper.cu`
    - full-grid `psi_in/psi_out` 상시 할당 제거, scratch 배치 호출로 전환.
  - `src/cuda/k1k6_pipeline.cuh`
  - `src/cuda/k1k6_pipeline.cu`
    - active tile/brick 배치 실행 경로 추가.

## 3. P2 보존형 Prolongation/Restriction 도입
- 신규 파일:
  - `src/cuda/device/device_prolong_restrict.cuh`
- 수정 파일:
  - `src/cuda/kernels/k3_finetransport.cu`
    - fine scratch 입출력 시그니처 반영.
  - `src/cuda/kernels/k4_transfer.cu`
    - batch-local transfer + coarse 반영 경로 추가.
  - `src/cuda/device/device_bucket.cuh`
    - global full-grid bucket 의존도 제거.

## 4. P3 K5 Energy Audit
- `src/cuda/kernels/k5_audit.cuh`
  - `AuditReport`에 energy 집계 필드 추가.
- `src/cuda/kernels/k5_audit.cu`
  - weight+energy 동시 감사 커널로 확장.
- `src/cuda/k1k6_pipeline.cuh`
  - K5 wrapper 시그니처 확장.
- `src/cuda/k1k6_pipeline.cu`
  - K5 호출 인자(에너지 관련 배열) 연결 및 실패 정책 반영.

## 5. P4 메모리 Preflight + batch 상한
- `src/gpu/gpu_transport_runner.cpp`
  - 실행 시작 전 메모리 추정 호출 추가.
- `src/include/gpu/gpu_transport_runner.hpp`
  - preflight 인터페이스 선언.
- 신규 파일:
  - `src/include/perf/memory_preflight.hpp`
  - `src/perf/memory_preflight.cpp`
  - `src/include/perf/fine_batch_planner.hpp`
  - `src/perf/fine_batch_planner.cpp`

## 6. P5 K2/K3 공통 후처리 통합
- 신규 파일:
  - `src/cuda/device/device_transport_poststep.cuh`
- 수정 파일:
  - `src/cuda/kernels/k2_coarsetransport.cu`
  - `src/cuda/kernels/k3_finetransport.cu`
  - `src/cuda/device/device_bucket.cuh` (헬퍼 연동)

## 7. P6 로컬빈 정책 단일화
- `src/include/core/local_bins.hpp`
  - 정책 주석/상수 정리.
- `src/include/core/incident_particle_config.hpp`
  - 런타임 입력 허용/금지 정책 확정.
- `src/gpu/gpu_transport_runner.cpp`
  - mismatch 처리 정책 일관화.

## 8. P7 테스트/CI 게이트
- `tests/gpu/test_energy_loss_only_gpu.cu`
  - 중심/경계 시나리오 분리 및 수용치 재정의.
- 신규 테스트:
  - `tests/gpu/test_transition_10mev_gpu.cu`
    - `E=10MeV` 전이 연속성, crossing guard 검증.
  - `tests/gpu/test_prolong_restrict_conservation_gpu.cu`
    - `coarse->fine->coarse` 보존성 검증.
- `tests/CMakeLists.txt`
  - 필수 게이트 그룹 정리.
- (선택) CI 스크립트:
  - `.github/workflows/*` 또는 내부 CI 설정 파일에 필수 테스트 등록.

## 9. P8 SPEC 동기화 자동화
- 신규 스크립트:
  - `scripts/check_spec_sync.py` (상수/정책 diff)
- 문서:
  - `docs/SPEC/README.md`에 체크 절차 링크 추가.
