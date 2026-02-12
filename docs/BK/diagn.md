# 현재 코드 다중 이슈 진단 (NaN, 에너지 비보존, lateral/ distal 동시 이상)

작성일: 2026-02-06

## 핵심 결론

문제는 한 군데가 아니라, **K2/K3 수송·분산·감사(audit) 경계가 동시에 어긋난 복합 이슈**입니다.

1. **lateral tail을 이웃 셀로 bucket 전송하면서도, 같은 분량의 잔여 에너지를 현 셀 `Edep`에 추가로 적산**하고 있습니다.
   - 이는 물리적으로는 “운반된 생존 에너지(E_new*w)”인데, 코드에서는 “즉시 침적”으로도 잡혀 **distal energy loss/보존오차**를 유발합니다.
2. **K2는 boundary 제한을 끈 상태로 coarse step을 강행**하고 있어(의도적으로 경계 제한 제거), step/경계/분산 로직이 서로 다른 가정을 사용합니다.
   - 이로 인해 과도한 경계 통과, bucket 의존 증가, 그리고 lateral/깊이 분포의 불안정성이 커집니다.
3. **NaN은 LUT 경계(E_max) 패치로 일부 완화되었지만, 출력단에서 NaN/Inf를 0으로 치환**하는 방식이 남아 있어, 근본 원인이 아닌 증상 은폐 경로가 존재합니다.
4. 문서상 Fermi-Eyges 누적 모멘트 기반이라고 하나 실제 K2/K3는 depth 근사식 기반 로직과 혼재되어 있어, 디버깅 시 원인 분리가 어렵습니다.

## 코드 레벨 관찰 포인트

### 1) lateral bucket 전송 + 현 셀 Edep 동시 적산 (이중 계상 위험)

- K2 lateral tail 전송 후:
  - `device_emit_component_to_bucket_4d(...)`로 이웃 전송
  - 직후 `cell_edep += E_new * w_spread_left/right` 수행
- K3도 동일 패턴

즉, **이웃으로 보낸 생존분(weight)은 이동시키면서, 그 생존에너지까지 현 셀에 침적으로 더함**.
이는 distal 쪽 에너지 부족(상류에서 미리 사라짐)과 전체 energy budget 왜곡을 동시에 만듭니다.

### 2) 경계 step 제한 정책 충돌

- K2에서 `coarse_step_limited = coarse_step`로 경계 거리 제한을 명시적으로 제거.
- 동시에 경계 crossing은 별도 `exit_face` 판정 및 bucket emission에 의존.

결과적으로 “step 제어”와 “경계 처리”가 느슨하게 결합돼, 작은 조건 변화에도 프로파일이 민감해집니다.

### 3) NaN 처리의 위치

- LUT lookup은 E_max 경계 예외처리가 들어가 있음.
- 하지만 최종 출력 저장시 NaN/Inf를 0으로 치환하고 있어, 런타임 중 발생한 수치 불안정을 사후 마스킹합니다.

## 권장 해결 순서 (우선순위)

1. **에너지 계정 단일 규칙 확정 (최우선)**
   - 규칙: `Edep`에는 오직 (a) 전자적 감쇠 `dE*weight`, (b) 핵반응 local deposition(`E_rem`)만 적산.
   - **bucket으로 보낸 생존분 `E_new * w_transfer`는 절대 `Edep`에 넣지 않음**.
   - lateral tail 전송 코드(K2/K3)에서 해당 라인 제거 후, 필요하면 `BoundaryLoss_energy` 또는 별도 transport carry 항목으로만 관리.

2. **K2 step-boundary 정책 재정렬**
   - coarse step을 완전 자유롭게 두지 말고, 최소한 경계까지의 안전 거리와 일관된 기준으로 clamp.
   - K3와 동일한 “path length vs geometric length” 정의를 문서/코드에 통일.

3. **NaN 조기 감지 모드 추가**
   - 출력단 치환 이전에 kernel 단계에서 `isfinite` 검사 카운터를 수집.
   - 어떤 stage(K2/K3/K4, LUT lookup, Gaussian CDF, nuclear attenuation)에서 처음 NaN이 생기는지 분리 추적.

4. **모델 표기 통일**
   - depth 기반 근사식 vs 누적 모멘트식 중 현재 실제 사용 경로를 명확히 하나로 정의.
   - 문서(`PLAN_MCS`, detailed physics)와 커널 주석을 동기화.

## 원인 미확정 시 단계별 분리 테스트 제안

### Test A: 최소 보존 smoke (단일 셀/무경계)

목적: 경계·bucket 영향을 제거하고 pure energy bookkeeping만 검증.

- 설정
  - `Nx=1, Nz=1`
  - lateral spreading OFF(또는 매우 작은 sigma)
  - nuclear OFF
- 검증
  - `E_in == Edep + E_out` (허용오차 1e-6~1e-5)

### Test B: 경계 수송만 검증 (분산 OFF)

목적: boundary crossing + bucket + K4 전송에서 weight/energy 손실 여부 확인.

- 설정
  - 1D에 가까운 배치 (`Nx=3, Nz=1`)
  - sigma_x 매우 작게
- 검증
  - 전송 전/후 총 weight 동일
  - `BoundaryLoss_*`와 neighbor 유입량 대응

### Test C: lateral tail 회계 검증 (핵심)

목적: 현재 의심되는 이중계상을 직접 검출.

- 설정
  - `sigma_x`를 크게 하여 tail bucket 전송 강제 발생
- 계측 항목(새 카운터)
  - `E_lateral_transferred = Σ(E_new*w_tail)`
  - `E_lateral_deposited = Σ(해당 tail 분량으로 Edep에 더한 값)`
- 기대
  - 올바른 구현이면 `E_lateral_deposited == 0`

### Test D: NaN 최초 발생 위치 추적

목적: NaN masking 전, 생성 지점을 pinpoint.

- 각 stage별 `isfinite` 카운터 추가:
  - `E`, `dE`, `E_new`, `sigma_x`, `w_new`, `weights[i]`, LUT 반환값
- 최초 non-finite 발생 cell/slot/lidx + theta/E bin 로깅

### Test E: distal energy loss 회귀 테스트

목적: 수정 후 실제 임상적으로 중요한 축(PDD distal tail) 복원 확인.

- 기준(run 전/후) PDD를 저장해 상대오차 곡선 비교
- Bragg peak 이후 distal 구간 적분값 비교

## 실무 적용 팁

- 먼저 K2/K3의 lateral tail 블록에서 `cell_edep += E_new * w_spread_*` 제거하고,
  Test C/E를 우선 돌리면 distal 손실과 보존오차 개선 여부를 가장 빠르게 확인할 수 있습니다.
- NaN은 출력 치환 대신, 내부 finite-check 카운터 기반으로 “처음 깨지는 수식”을 찾는 방식이 재발 방지에 효과적입니다.