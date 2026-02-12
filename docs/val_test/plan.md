# One-Step Z 진행 검증 상세 계획 (docs/detailed 반영 + 코드 교차검증)

## 0. 목표와 분석 범위
목표는 `z` 진행을 1 iteration으로 제한한 상태에서 phase-space 상태를 저장하고, `energy loss`와 `angular scattering`을 해석 예측치와 비교하여 오차 원인을 코드 파일/함수 단위로 특정하는 것이다.

분석 경로는 다음 실행 체인을 기준으로 고정한다.
- `run_simulation.cpp`
- `src/gpu/gpu_transport_runner.cpp`
- `src/cuda/gpu_transport_wrapper.cu`
- `src/cuda/k1k6_pipeline.cu`
- `src/cuda/kernels/k2_coarsetransport.cu`, `src/cuda/kernels/k3_finetransport.cu`
- `src/cuda/device/device_lut.cuh`, `src/cuda/device/device_physics.cuh`

## 1. docs/detailed 대비 현재 코드 기준 정합성 체크
`docs/detailed`를 참고하되, 실험 파라미터는 아래 "현재 코드값"을 우선 기준으로 사용한다.

1. 로컬 bin 상수
- 현재 코드: `N_theta_local=4`, `N_E_local=2`, `N_x_sub=8`, `N_z_sub=4`, `LOCAL_BINS=256`
- 위치: `src/include/core/local_bins.hpp`
- 비고: `docs/detailed`의 일부 문서(8x4x4x4=512)와 다를 수 있음

2. 각도 격자 범위
- 현재 코드 기본 transport 각도 범위: `[-0.35, +0.35] rad`
- 위치: `src/gpu/gpu_transport_runner.cpp`

3. 에너지 격자
- 현재 코드 기본: piecewise `energy_groups` 사용
- 기본 구간: `[0.1-2:0.1], [2-20:0.2], [20-100:0.25], [100-250:0.25]`
- 위치: `src/include/core/incident_particle_config.hpp`

4. K3 물리 플래그
- 현재 코드: K3 실행 시 `enable_straggling=true`, `enable_nuclear=true` 고정
- 위치: `src/cuda/k1k6_pipeline.cu`
- 의미: 기본 실행만으로 "순수 energy-loss only" K3 분리는 불가

5. 상태 덤프 기본값
- `SM2D_ENABLE_DEBUG_DUMPS` 기본 `OFF`
- 위치: `CMakeLists.txt`

## 2. 실험 케이스 설계 (1-step 전용)
공통:
- `transport.max_iterations=1`
- `grid.max_steps=1` (보조 안전장치)
- 출력 경로 분리 (`results/one_step_case_*`)

케이스 매트릭스:
1. Case A (Energy-loss, K2 우세)
- `beam.profile=pencil`
- `sigma_x_mm=0`, `sigma_theta_rad=0`, `sigma_MeV=0`
- `sampling.n_samples=1`
- `E0=150 MeV`, `E_fine_on=10`, `E_fine_off=11`
- 기대: coarse 경로 비중이 큼

2. Case B (Energy-loss, K3 강제)
- `beam.profile=pencil`
- `sigma_x_mm=0`, `sigma_theta_rad=0`, `sigma_MeV=0`
- `sampling.n_samples=1`
- `E0=150 MeV`, `E_fine_on=260`, `E_fine_off=261`
- 기대: K3 경로를 강제로 타게 하여 fine physics 점검

3. Case C (Angular scattering 앙상블, K3 강제)
- `beam.profile=gaussian`
- `sigma_x_mm > 0` (예: 3.0), `sigma_theta_rad=0`, `sigma_MeV=0`
- `sampling.n_samples` 확대 (예: 2000+)
- `E0=150 MeV`, `E_fine_on=260`, `E_fine_off=261`
- 목적: 다수 성분에서 `dtheta` 분포를 얻어 분산 비교 가능하게 구성

4. Case D (민감도 전용, 선택)
- Case B/C 기준에서 `N_theta`와 `energy_groups`만 변경
- 목적: 이산화 오차 분리

## 3. 저장/계측 항목 (상태값 전량)
필수 저장 대상:
1. Iteration 0/1 phase-space raw dump
- 권장 컬럼:
- `iter, cell, slot, bid, lidx, weight`
- `b_theta, b_E, theta_local, E_local, x_sub, z_sub`
- `theta_bin, E_bin, theta_rep, E_rep`
- `x_offset_mm, z_offset_mm`

2. Iteration 0/1 누적 채널
- `EdepC`
- `AbsorbedWeight_cutoff`, `AbsorbedEnergy_cutoff`
- `AbsorbedWeight_nuclear`, `AbsorbedEnergy_nuclear`
- `BoundaryLoss_weight`, `BoundaryLoss_energy`
- `transport_dropped_weight`, `transport_dropped_energy`
- `transport_audit_residual_energy`

3. 소스 계정값
- `source_injected_*`, `source_out_of_grid_*`, `source_slot_dropped_*`, `source_representation_loss_energy`

4. 실행 메타데이터
- 사용 ini 전문
- 빌드 플래그
- git commit hash

## 4. 해석 예측치 정의
성분 단위 비교 기준:

1. 에너지 손실 예측
- CSDA 기준:
- `dE_pred_csda = E_in - E_inverse(R(E_in) - ds_used)`
- Stopping power 근사:
- `dE_pred_sp = S(E_in) * rho * ds_used / 10`
- K3 주의:
- K3는 `enable_straggling=true` 고정이므로 단일 성분 비교 시 `dE_sim`은 샘플값이다.
- 따라서 K3 단일 성분은 평균식 대비 상대오차만 보지 않고 `z_dE = (dE_sim - mean_dE)/sigma_dE`도 함께 확인한다.

2. 산란 각 예측
- Highland RMS:
- `sigma_theta_pred = (13.6/(beta*p)) * sqrt(ds_used/X0) * (1 + 0.038*ln(ds_used/X0))`
- 코드상 clamp 반영:
- bracket 하한 `0.25`

3. step 길이(`ds_used`) 정의
- K2:
- `coarse_step = min(step_coarse, min(dx,dz))`
- 필요 시 `E_fine_on` crossing guard로 분할
- K3:
- `step_phys = device_compute_max_step(...)`
- `step_boundary = distance_to_boundary * 1.001`
- `ds_used = min(step_phys, step_boundary)`

## 5. 비교 지표
1. 에너지 손실 오차
- `eps_dE_csda = (dE_sim - dE_pred_csda) / dE_pred_csda`
- `eps_dE_sp = (dE_sim - dE_pred_sp) / dE_pred_sp`

2. 각 산란 오차
- 앙상블 케이스(Case C):
- `R_theta = Var(dtheta_sim) / sigma_theta_pred^2`
- `Mean(dtheta_sim)`이 0에서 크게 벗어나는지도 확인
- 단일 성분 케이스(A/B):
- 분산비 대신 `|dtheta_sim| / sigma_theta_pred`를 참고 지표로만 사용
- (단일 샘플 특성상 통계적 판정 지표로 해석하지 않음)

3. 보존성 체크
- K5 audit residual
- 시스템 에너지 닫힘:
- `E_source_total - E_accounted_total`

4. bin 민감도
- `N_theta` 증가, `dE` 축소 시 `eps_dE_csda`, `R_theta` 변화율

## 6. 원인 분류 룰
1. K2만 오차가 크면
- `src/cuda/kernels/k2_coarsetransport.cu` 우선 점검
- coarse step 정의, crossing guard, rebin 처리 확인

2. K3만 오차가 크면
- `src/cuda/kernels/k3_finetransport.cu` 우선 점검
- `theta_scatter` 생성, `sigma_theta_step` 계산, boundary 전후 energy closure 확인

3. K2/K3 모두 공통 편향이면
- `src/cuda/device/device_lut.cuh` 우선 점검
- `device_compute_max_step`, `device_compute_energy_after_step`, representative energy 규약 확인

4. 오차가 bin 조정 시 크게 줄면
- 물리식 문제보다 `E/theta` 이산화 및 재bin 손실 가능성 우세

## 7. 실행 순서
1. Case A/B용 ini 생성
2. debug dump 빌드 활성화 후 재빌드
3. Case A/B 1-step 실행
4. raw dump + audit 채널 수집
5. 해석치 계산 및 오차 테이블 생성
6. bin 민감도(Case C) 실행
7. 원인 분류 및 수정 우선순위 도출

## 8. 최종 산출물
1. `results/one_step_case_*/` 원시 상태 덤프
2. 케이스별 오차표 (`energy loss`, `angular scattering`, `audit residual`)
3. 원인 분석 리포트:
- 재현 조건
- 정량 오차
- 의심 코드 위치
- 수정 우선순위
