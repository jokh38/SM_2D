# Lateral Spreading 문제 원인 분석 (2026-02-06)

## 결론 요약

현재 lateral spreading 불일치의 핵심은 **물리 모델 자체보다 구현의 수송 경로 단절**입니다.

1. K2/K3 모두에서 셀 내 가우시안 분산 중 이웃 셀로 넘어간 분량을 bucket 전송하지 않고
   `Edep + cutoff`로 강제 소멸시켜 lateral tail를 인공적으로 잘라냅니다.
2. `device_gaussian_spread_weights()`는 `dx` 간격의 "셀 단위" 분산 함수인데,
   호출부는 이를 `x_sub`(서브셀) 분산처럼 사용해 공간 스케일이 불일치합니다.
3. K2/K3에서 `sigma_x_initial = 6.0f`가 하드코딩되어 입력 beam 설정과 분리되어 있습니다.

이 3가지가 결합되어 측면 프로파일에서 과도한 중심 집중/비정상 tail 손실/깊이별 왜곡이 동시에 발생합니다.

---

## 코드 레벨 주요 징후

### 1) inter-cell 분산이 수송되지 않고 소멸됨

K2와 K3 모두 `target_cell != cell` 분기에서 실제 전송 대신 에너지 침적으로 처리합니다.

- K2: `cell_edep += E_new * w_spread; cell_w_cutoff += w_spread;`
- K3: 동일한 단순 소멸 처리

이 로직은 lateral spreading의 본질(가중치의 횡방향 이동)을 깨며,
결과적으로 profile tail를 줄이고 보존성 오차를 유발합니다.

### 2) gaussian 스케일-도메인 불일치

`device_gaussian_spread_weights()`는 `dx` 간격 N개로 분산을 구성하는데,
K2/K3 호출부는 이를 `x_sub` 업데이트에 직접 매핑합니다.

즉, 수학적으로는 "N개의 셀"에 대한 분포를 만든 뒤,
구현은 "한 셀 내부의 N개 sub-bin"에 쓰는 혼합 구조입니다.

이 구조에서는 인접 셀로 퍼져야 할 질량이 다수 발생하고,
현재 구현상 그 질량이 위 1) 로직에서 소멸됩니다.

### 3) 초기 빔폭 하드코딩

`sigma_x_initial = 6.0f`가 K2/K3에 고정되어 있어,
실제 입력(`sim.ini`/API 인자)과 무관하게 baseline 폭이 고정됩니다.

검증 실험마다 입력 빔폭을 바꿔도 lateral profile 시작 조건이 코드에서 덮어써져
피팅 및 원인 분리에 혼선을 줍니다.

---

## 권장 해결 방안

### A. inter-cell lateral 분량을 bucket으로 전송 (최우선)

- `target_cell != cell`일 때 소멸 금지
- `exit_face` 기반으로 `OutflowBuckets`에 emission
- K4에서 인접 셀로 유입되도록 경로 통일

이 한 가지로 profile tail 절단과 비물리적 cutoff 증가를 크게 줄일 수 있습니다.

### B. 분산 연산의 공간 단위를 분리

둘 중 하나로 명확히 통일:

1. **셀 단위 분산**: `dx` 기반으로 이웃 셀까지 분포 + 셀 내부는 x_sub 보정 최소화
2. **서브셀 단위 분산**: 폭을 `dx/N_x_sub`로 바꾼 전용 함수 작성

현재 혼합형(셀 분포 생성 + sub-bin 기록)은 오차와 소멸 경로를 동시에 키웁니다.

### C. `sigma_x_initial`를 입력 파라미터로 연결

- K2/K3 커널 인자로 `sigma_x0` 전달
- wrapper에서 source 설정값을 그대로 전달
- 하드코딩 상수 제거

### D. Fermi-Eyges 갱신 경로 단일화

현재는 depth 기반 근사(`theta0(z) * z / sqrt(3)`)와 누적 moment 설계가 혼재합니다.

- 한 경로(누적 moment A/B/C)로 통일
- 문서/코드/검증식 일치
- K2->K3 전환 제약은 별도 정책으로 분리

---

## 기대 효과

- lateral profile tail 복원
- boundary/cutoff 인공 손실 감소
- 입력 빔폭 변화에 대한 재현성 향상
- 문서/구현 일관성 개선으로 디버깅 비용 감소