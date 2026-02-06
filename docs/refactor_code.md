# Refactor Plan: SM_2D

코드 변경은 아직 하지 않았고, 현재 코드 기준으로 정리 우선순위를 다음과 같이 확정했다.

## 핵심 문제 영역 (우선순위 순)

1. 타입 경계가 깨져 있고 `reinterpret_cast`로 우회한다 (안전성 리스크).
   - `src/include/cuda/gpu_transport_wrapper.hpp:8`
   - `src/cuda/gpu_transport_wrapper.cu:115`
   - `src/cuda/gpu_transport_wrapper.cu:150`
   - `src/gpu/gpu_transport_runner.cpp:110`

2. 실험용 하드코딩이 기본 실행 경로에 남아 있고, 설정값 일부가 사실상 무시된다.
   - `src/cuda/gpu_transport_wrapper.cu:162`
   - `src/cuda/gpu_transport_wrapper.cu:172`
   - `src/cuda/k1k6_pipeline.cu:1019`
   - `src/include/core/config_loader.hpp:274`

3. K2/K3 실행 때 매 반복마다 같은 GPU 메모리를 할당/복사/해제한다 (성능/복잡도 저하).
   - `src/cuda/k1k6_pipeline.cu:775`
   - `src/cuda/k1k6_pipeline.cu:822`
   - `src/cuda/k1k6_pipeline.cu:844`
   - `src/cuda/k1k6_pipeline.cu:893`

4. K2/K3 커널 내부 후반 로직이 대량 복붙 상태다 (버그 수정시 이중 수정 위험).
   - `src/cuda/kernels/k2_coarsetransport.cu:355`
   - `src/cuda/kernels/k3_finetransport.cu:368`

5. 확실히 미사용인 코드 덩어리가 누적되어 있다 (유지보수 노이즈).
   - `src/include/source/source_adapter.hpp:49`
   - `src/include/core/incident_particle_config.hpp:350`
   - `src/cuda/kernels/k4_transfer.cuh:25`
   - `src/cuda/k1k6_pipeline.cu:535`
   - `src/cuda/device/device_bucket.cuh:109`
   - `src/cuda/device/device_psic.cuh:362`

6. 핫패스에 디버그성 출력/검증 복사가 남아 있어 기본 실행이 무겁다.
   - `src/gpu/gpu_transport_runner.cpp:81`
   - `src/cuda/gpu_transport_wrapper.cu:322`
   - `src/cuda/k1k6_pipeline.cu:1079`

7. 불필요 연산/주석-코드 불일치가 혼재한다.
   - `src/cuda/k1k6_pipeline.cu:468`
   - `src/cuda/device/device_psic.cuh:25`
   - `src/cuda/device/device_bucket.cuh:26`
   - `src/cuda/kernels/k5_audit.cu:26`

8. 레거시 CPU 소스 주입 로직이 격자 간격을 하드코딩한다 (재사용 시 오동작 가능).
   - `src/source/pencil_source.cpp:11`
   - `src/source/gaussian_source.cpp:19`

## 정리 실행 계획

1. 안정성 1단계: `RLUT/DeviceRLUT` 타입 네임스페이스 정리, `reinterpret_cast` 제거, 래퍼 함수 반환형을 `bool/expected`로 바꿔 실패 전파를 강제한다.
2. 설정 단일화 2단계: 에너지 그리드/`E_trigger`/`step_coarse`/`max_iter`를 `IncidentParticleConfig` 또는 별도 `TransportConfig`로 올리고, `grid.max_steps`를 실제 루프에 연결한다.
3. 파이프라인 경량화 3단계: `d_theta_edges/d_E_edges`를 `K1K6PipelineState`에 1회 할당 후 재사용하도록 변경한다.
4. 중복 제거 4단계: K2/K3 공통 후처리(경계 처리, in-cell spread, bucket emit)를 `device` 공통 헬퍼로 추출한다.
5. 죽은 코드 정리 5단계: 실제 참조 없는 API/헬퍼를 삭제하거나 `legacy/experimental`로 격리하고 빌드 타깃에서 분리한다.
6. 로깅/디버그 정책 6단계: 기본 경로에서는 최소 로그만 허용하고, 대용량 host copy 검증은 `SM2D_ENABLE_DEBUG_DUMPS` 또는 런타임 debug 레벨에서만 수행하게 한다.
7. 테스트 보강 7단계: 설정 반영(max_steps, transport params), runner/wrapper grid 일치, 오류 전파, K2/K3 공통 헬퍼 회귀 테스트를 추가한다.

## 현재 확인한 검증 상태

1. `cmake --build build -j4` 성공
2. `ctest -R EnergyGridTest` 통과
3. `ctest -R K3Test` 통과
4. `ctest -R "PencilSourceTest|GaussianSourceTest"` 통과


## 추가 점검 기록 (2026-02-06)

이전 커밋 기준 누락 수정 여부를 점검했고, 아래 항목이 추가 보완 대상으로 확인되었다.

### 확인된 미해결 항목

1. EnergyLossOnly 계열 GPU 테스트 3건이 실패한다.
   - `tests/gpu/test_energy_loss_only_gpu.cu:481`
   - `tests/gpu/test_energy_loss_only_gpu.cu:503`
   - `tests/gpu/test_energy_loss_only_gpu.cu:524`
   - 실패 테스트:
     - `EnergyLossOnlyTest.EnergyLossOnly`
     - `EnergyLossOnlyTest.FullPhysics`
     - `EnergyLossOnlyTest.StragglingOnly`

2. K2/K3 lateral tail 처리에서 격자 경계 셀(ix=0, ix=Nx-1)일 때 tail 가중치/에너지 누락 가능성이 있다.
   - `src/cuda/kernels/k3_finetransport.cu:479`
   - `src/cuda/kernels/k3_finetransport.cu:503`
   - `src/cuda/kernels/k2_coarsetransport.cu:450`
   - `src/cuda/kernels/k2_coarsetransport.cu:474`
   - 현재 로직은 neighbor가 있을 때만 bucket 전송/경계 집계를 수행하고, neighbor가 없을 때 대체 집계가 없다.

3. EnergyLossOnly 테스트 설정과 기대값 정합성이 맞지 않는다.
   - 소스 위치가 경계 셀에서 시작: `tests/gpu/test_energy_loss_only_gpu.cu:182`
   - 초기 빔 폭이 큼(`sigma_x_initial = 6.0f`): `tests/gpu/test_energy_loss_only_gpu.cu:363`
   - 해당 조건에서 `total_edep ≈ E0`를 강하게 기대: `tests/gpu/test_energy_loss_only_gpu.cu:481`, `tests/gpu/test_energy_loss_only_gpu.cu:524`

### 실행 로그 요약

1. `cmake --build build -j4` 성공
2. `ctest --output-on-failure` 실행 결과: 84개 중 81개 통과, 3개 실패(EnergyLossOnly 계열)


## 추가 점검 기록 (2026-02-06, 후속 세션 인계용)

이번 세션에서 일부 경계/좌표 전달 버그를 보완했지만, EnergyLossOnly 계열 3건 실패는 여전히 남아 있다.

### 이번 세션에서 적용한 코드 수정

1. Z-face 이웃 전달 시 x 오프셋 보존 수정
   - `src/cuda/device/device_bucket.cuh:628`
   - `device_get_neighbor_x_offset()`의 default 경로를 `x_exit` 그대로 반환하도록 수정.

2. X-face 경계 통과 시 z 오프셋 보존 수정 (K2/K3)
   - `src/cuda/kernels/k3_finetransport.cu:315`
   - `src/cuda/kernels/k2_coarsetransport.cu:309`
   - 기존 `z_new - dz * 0.5f`를 `z_new`로 수정.

3. lateral tail의 격자 경계 셀 처리 보완 (K2/K3)
   - `src/cuda/kernels/k3_finetransport.cu:501`
   - `src/cuda/kernels/k3_finetransport.cu:527`
   - `src/cuda/kernels/k2_coarsetransport.cu:472`
   - `src/cuda/kernels/k2_coarsetransport.cu:498`
   - neighbor가 없는 경우(`ix==0`, `ix==Nx-1`) tail을 `BoundaryLoss_energy`에 집계하도록 보완.

4. lateral 분배 안정화 보정 (K2/K3)
   - `src/cuda/kernels/k3_finetransport.cu:451`
   - `src/cuda/kernels/k3_finetransport.cu:487`
   - `src/cuda/kernels/k2_coarsetransport.cu:422`
   - `src/cuda/kernels/k2_coarsetransport.cu:458`
   - `w_in_cell`를 `[0,1]`로 clamp하고, sub-cell/tail 가중치 정규화를 추가.

### 현재 미해결 상태

1. EnergyLossOnly 계열 3건 실패 지속
   - 실패 테스트:
     - `EnergyLossOnlyTest.EnergyLossOnly`
     - `EnergyLossOnlyTest.FullPhysics`
     - `EnergyLossOnlyTest.StragglingOnly`
   - 최신 재현(2026-02-06):
     - EnergyLossOnly: `total_edep=2.39373 MeV`, `bragg_depth=2 mm`
     - FullPhysics: `total_edep=3.25724 MeV`, `bragg_depth=2 mm`
     - StragglingOnly: `total_edep=2.54591 MeV`, `bragg_depth=2 mm`
   - 반면 `EnergyLossOnlyTest.NuclearOnly`는 통과.

2. 실패 패턴상, tail 경계 누락 단일 이슈가 아니라 “입자/에너지 회계가 초반 반복에서 사라지는 경로”가 남아 있을 가능성이 높다.
   - 현재 증상은 Bragg peak가 깊이 158 mm 근처가 아닌 2 mm에 고정되는 형태.

### 이번 세션 검증 로그

1. `cmake --build build -j4` 성공
2. `ctest -R "EnergyLossOnlyTest\\.(EnergyLossOnly|FullPhysics|StragglingOnly|NuclearOnly)" --output-on-failure`
   - 4개 중 1개 통과(`NuclearOnly`), 3개 실패
3. `ctest -R K3Test --output-on-failure`
   - 4개 전부 통과

### 다음 세션 우선 디버깅 순서 (권장)

1. 반복별 weight/energy 회계 계측 추가
   - `psi_in` 총 weight
   - `K3` 후 `psi_out` 총 weight
   - `K4` 후 `psi_out` 총 weight
   - `BoundaryLoss_weight/energy`, `AbsorbedWeight_cutoff`, `EdepC` 누적
   - 매 iteration에서 닫힘식(보존식) 오차를 출력해 최초 붕괴 지점을 찾는다.

2. 슬롯 포화/할당 실패 경로 정량화
   - `DEVICE_Kb` / `DEVICE_Kb_out` 포화 시 drop 여부를 카운터로 계측.
   - 특히 `K3`의 `out_slot < 0` 및 bucket slot allocate 실패 횟수 추적 필요.

3. EnergyLossOnly 테스트 시나리오 정합성 재검토
   - 경계 셀 시작(`source_cell=0`) + `sigma_x_initial=6.0f` 조건에서
     `total_edep≈E0`를 강제하는 기대가 물리/경계조건과 일치하는지 재확인.
