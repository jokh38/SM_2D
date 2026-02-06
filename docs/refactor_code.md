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
2. `ctest --test-dir build --output-on-failure` 실행 결과: 84개 중 81개 통과, 3개 실패(EnergyLossOnly 계열)


## 추가 점검 기록 (2026-02-06, 후속 세션 인계용)

이번 세션은 코드 수정 없이 재현/교차검증 위주로 점검했다. 결론적으로 EnergyLossOnly 계열 3건 실패는 그대로 재현되지만, 전체 파이프라인 경로가 전면 붕괴된 상태는 아니다.

### 이번 세션에서 추가로 확인한 사실

1. 테스트 실행 경로 주의
   - 루트에서 `ctest`만 실행하면 `No tests were found`가 나온다.
   - 실제 기준은 `ctest --test-dir build ...`이다.

2. 전체 테스트 상태(재확인)
   - 실행: `ctest --test-dir build --output-on-failure`
   - 결과: 84개 중 81개 통과, 3개 실패
   - 실패 항목:
     - `EnergyLossOnlyTest.EnergyLossOnly`
     - `EnergyLossOnlyTest.FullPhysics`
     - `EnergyLossOnlyTest.StragglingOnly`

3. EnergyLossOnly 계열 최신 재현값(2026-02-06)
   - 실행: `ctest --test-dir build -R "EnergyLossOnlyTest\\.(EnergyLossOnly|FullPhysics|StragglingOnly|NuclearOnly)" --output-on-failure`
   - `EnergyLossOnly`: `total_edep=2.39373 MeV`, `bragg_depth=2 mm`
   - `FullPhysics`: `total_edep=3.25724 MeV`, `bragg_depth=2 mm`
   - `StragglingOnly`: `total_edep=2.54591 MeV`, `bragg_depth=2 mm`
   - `NuclearOnly`: 통과

4. 파이프라인 교차검증
   - 실행: `./run_simulation test_c.ini`
   - 결과: 정상 종료, `Transport complete after 125 iterations`, `Bragg Peak: 148 mm depth`
   - 즉, 현 실패는 K3/K4 전체 경로 붕괴보다는 EnergyLossOnly 테스트 시나리오/가정과 더 강하게 연관될 가능성이 있다.

### 테스트 시나리오 정합성 체크 포인트

1. 테스트 시작 셀이 격자 경계다.
   - `tests/gpu/test_energy_loss_only_gpu.cu:182` (`source_cell = 0`)

2. lateral spreading이 항상 켜져 있고 빔 폭도 크다.
   - `tests/gpu/test_energy_loss_only_gpu.cu:309`
   - `tests/gpu/test_energy_loss_only_gpu.cu:363` (`sigma_x_initial = 6.0f`)

3. 반면 기대값은 사실상 “도메인 외 손실이 거의 없는 조건”을 요구한다.
   - `tests/gpu/test_energy_loss_only_gpu.cu:481`
   - `tests/gpu/test_energy_loss_only_gpu.cu:503`
   - `tests/gpu/test_energy_loss_only_gpu.cu:524`
   - `tests/gpu/test_energy_loss_only_gpu.cu:482`
   - `tests/gpu/test_energy_loss_only_gpu.cu:505`

4. (추론) `source_cell=0` + 큰 `sigma_x_initial` + 항상-on lateral spreading 조합이 현재 기대식(`total_edep≈E0`, `bragg_depth≈158mm`)과 충돌할 가능성이 높다.

### 다음 세션 우선 작업 (권장)

1. EnergyLossOnly 테스트 기준선 분리
   - 중앙 셀 시작 케이스(`source_cell` 중앙)와 경계 셀 케이스를 분리.
   - 작은 빔 폭(예: `sigma_x_initial <= 1.0f`) 케이스를 추가해 “경계 손실 최소” 기준에서 물리 기대를 검증.

2. 폐쇄식(회계식) 출력 강화
   - `EdepC`뿐 아니라 `BoundaryLoss_energy`, `AbsorbedEnergy_nuclear`, cutoff 항목까지 합산한 총량을 테스트 로그로 출력.
   - 어느 항목에서 에너지가 빠지는지 즉시 식별 가능하도록 만든다.

3. 슬롯 포화/드롭 계측
   - `src/cuda/kernels/k3_finetransport.cu:410` (`out_slot < 0` 경로)
   - `src/cuda/kernels/k2_coarsetransport.cu:381` (`out_slot < 0` 경로)
   - `src/cuda/kernels/k4_transfer.cu:105` (bucket transfer slot allocate 실패 경로)
   - 위 경로별 카운터를 넣어 드롭 발생 여부를 정량화한다.


## 추가 점검 기록 (2026-02-06, 후속 세션 인계용 - 업데이트 2)

이번 세션에서는 실제 코드 수정과 계측 추가를 진행했다. 핵심은 “에너지 소실 경로를 계측 가능한 상태로 만들고, lateral tail 누락 경로를 보완”하는 것이다.

### 이번 세션 코드 수정

1. K3/K2/K4 드롭 계측 API 추가
   - `src/cuda/kernels/k3_finetransport.cuh:73`
   - `src/cuda/kernels/k3_finetransport.cu:57`
   - `src/cuda/kernels/k2_coarsetransport.cuh:96`
   - `src/cuda/kernels/k2_coarsetransport.cu:39`
   - `src/cuda/kernels/k4_transfer.cuh:16`
   - `src/cuda/kernels/k4_transfer.cu:22`
   - `out_slot < 0`, bucket emit drop, K4 slot allocate 실패를 host에서 읽을 수 있는 카운터로 노출.

2. bucket emit 실패 가시화
   - `src/cuda/device/device_bucket.cuh:89`
   - `device_emit_to_bucket()`를 `bool` 반환으로 변경하고,
     `device_emit_component_to_bucket_4d()`가 dropped weight를 반환하도록 변경.

3. lateral tail 보존 조건 수정 (K3/K2 공통)
   - `src/cuda/kernels/k3_finetransport.cu:521`
   - `src/cuda/kernels/k2_coarsetransport.cu:477`
   - 기존 `sigma_x > dx*0.5f` 조건을 `w_cell_fraction < 1` 기준으로 변경해
     tail 누락 가능성을 줄임.

4. 슬롯 수 확장
   - `src/cuda/device/device_psic.cuh:30` (`DEVICE_Kb: 8 -> 32`)
   - `src/cuda/device/device_bucket.cuh:25` (`DEVICE_Kb_out: 8 -> 32`)
   - 목적: slot/bucket drop 완화.

5. EnergyLossOnly 테스트 구조 보강
   - `tests/gpu/test_energy_loss_only_gpu.cu:174`
   - baseline/boundary 시나리오 분리, `Edep+Boundary+Nuclear` 회계 출력,
     K3/K4 드롭/프루닝 출력, iteration 수 출력 추가.

### 최신 검증 결과 (2026-02-06)

1. `ctest --test-dir build -R "EnergyLossOnlyTest\\.(EnergyLossOnly|FullPhysics|StragglingOnly|NuclearOnly)" --output-on-failure`
   - 4개 중 1개 통과(`NuclearOnly`), 3개 실패
   - `EnergyLossOnly`
     - `AccountedTotal=145.673 MeV` (이전 18.5 MeV 수준 대비 크게 개선)
     - `bragg_depth=90 mm`
     - `K3/K4 drop 카운터는 0`
   - `FullPhysics`
     - `AccountedTotal=155.742 MeV`
     - `bragg_depth=88 mm`
     - drop 카운터는 존재하지만 drop weight/energy 절대량은 매우 작음
   - `StragglingOnly`
     - `AccountedTotal=142.308 MeV`
     - `bragg_depth=88 mm`
     - drop 카운터는 존재하지만 drop weight/energy 절대량은 매우 작음

2. `ctest --test-dir build -R K3Test --output-on-failure`
   - 4개 전부 통과

### 해석 및 남은 이슈

1. 개선 확인
   - tail 보정/계측 추가 이후, EnergyLossOnly 회계량은
     `~18.5 MeV -> ~145.7 MeV`로 크게 개선됨.

2. 미해결
   - Bragg depth가 여전히 `~88-90 mm`로 이론치 `~158 mm` 대비 얕다.
   - FullPhysics/StragglingOnly에서는 drop count가 보이지만,
     누락 에너지의 주 원인으로 보기엔 절대량이 작다.
   - 즉, 남은 오차의 중심은 “slot drop 단일 이슈”가 아니라
     물리/스텝/경계 모델 결합 영역일 가능성이 높다.

3. 부수 영향
   - `DEVICE_Kb/DEVICE_Kb_out=32` 적용으로 테스트 런타임이 유의미하게 증가했다.
   - 성능-정확도 절충(`32` 유지 vs `16` 등) 재평가가 필요하다.


## 추가 점검 기록 (2026-02-06, 후속 세션 인계용 - 업데이트 3)

이번 세션은 `업데이트 2` 항목의 반영 여부와 재현성 확인에 집중했다. 결론적으로 소스 추가 수정 없이, 기존 기록의 핵심 현상과 수치가 동일하게 재현된다.

### 반영 상태 확인

1. `업데이트 2`에서 명시한 계측/보정 항목이 코드에 반영된 상태를 재확인했다.
   - K3/K2/K4 drop 계측 API: `src/cuda/kernels/k3_finetransport.cuh:73`, `src/cuda/kernels/k2_coarsetransport.cuh:96`, `src/cuda/kernels/k4_transfer.cuh:16`
   - bucket emit 실패 가시화: `src/cuda/device/device_bucket.cuh:90`, `src/cuda/device/device_bucket.cuh:389`
   - lateral tail 경계 보정(K3/K2): `src/cuda/kernels/k3_finetransport.cu:521`, `src/cuda/kernels/k2_coarsetransport.cu:477`
   - 슬롯 수 확장(32): `src/cuda/device/device_psic.cuh:30`, `src/cuda/device/device_bucket.cuh:25`
   - EnergyLossOnly 테스트 구조 보강: `tests/gpu/test_energy_loss_only_gpu.cu:182`, `tests/gpu/test_energy_loss_only_gpu.cu:526`

2. 이번 세션에서 소스 코드 추가 수정은 수행하지 않았다(검증만 수행).

### 재검증 결과 (2026-02-06)

1. `cmake --build build -j4`
   - 성공

2. `ctest --test-dir build -R K3Test --output-on-failure`
   - 4개 전부 통과

3. `ctest --test-dir build -R "EnergyLossOnlyTest\\.(EnergyLossOnly|FullPhysics|StragglingOnly|NuclearOnly)" --output-on-failure`
   - 4개 중 1개 통과(`NuclearOnly`), 3개 실패
   - `EnergyLossOnly`
     - `AccountedTotal=145.673 MeV`
     - `bragg_depth=90 mm`
   - `FullPhysics`
     - `AccountedTotal=155.742 MeV`
     - `bragg_depth=88 mm`
   - `StragglingOnly`
     - `AccountedTotal=142.308 MeV`
     - `bragg_depth=88 mm`

4. `ctest --test-dir build --output-on-failure`
   - 84개 중 81개 통과, 3개 실패(EnergyLossOnly 계열 3건)

5. `./run_simulation test_c.ini`
   - 정상 종료
   - `Transport complete after 155 iterations`
   - `Bragg Peak: 148 mm depth`

### 결론

1. `업데이트 2` 이후 개선 효과(회계량 대폭 개선)는 유지되며 재현성도 확인됐다.
2. 미해결 축도 동일하다: EnergyLossOnly 계열에서 Bragg depth가 `~88-90 mm`로 이론치 `~158 mm` 대비 여전히 얕다.
3. drop 카운터는 관측되더라도 절대 drop weight/energy가 작아, 잔여 오차는 물리/스텝/경계 모델 결합 영역 추가 점검이 필요하다.
