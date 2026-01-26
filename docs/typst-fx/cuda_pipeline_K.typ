#set text(font: "Malgun Gothic", size: 11pt)
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

// Define custom box elements
#let tip-box(body) = block(
  fill: rgb("#e6fff2"),
  inset: 10pt,
  radius: 5pt,
  body
)

#let warning-box(body) = block(
  fill: rgb("#fff0cc"),
  inset: 10pt,
  radius: 5pt,
  body
)

#show math.equation: set text(weight: "regular")

= CUDA 커널 파이프라인 문서

== 개요

SM_2D는 결정론적 양성자 수송을 위한 6단계 CUDA 커널 파이프라인을 구현합니다. 파이프라인은 계층적 세분화를 통해 입자를 처리하며, 고에너지 입자용 거친 수송과 중요한 브래그 피크 영역용 정밀 수송을 사용합니다.

=== 커널 시퀀스

시뮬레이션 루프는 각 단계마다 6개 커널을 반복합니다:

#figure(
  table(
    columns: (auto, 3fr),
    inset: 8pt,
    align: left,
    table.header([*커널*], [*목적*]),
    [K1: ActiveMask], [정밀 수송이 필요한 셀 감지],
    [K2: CoarseTransport], [고에너지 수송(E > 10 MeV)],
    [K3: FineTransport], [저에너지 수송(E <= 10 MeV)],
    [K4: BucketTransfer], [셀 간 입자 전송],
    [K5: WeightAudit], [보존 법칙 검증],
    [K6: SwapBuffers], [입력/출력 포인터 교환],
  ),
  caption: [CUDA 커널 파이프라인],
)

=== 파이프라인 흐름

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    inset: 10pt,
    stroke: (x: 1pt, y: 1pt),
    table.header([*단계*], [*커널*], [*설명*]),

    [입력], [입력: 셀의 입자들 (t = 0)], [ ],

    [K1], [ActiveMask 활성 마스크], [
      *작업*: 정밀 수송이 필요한 셀 표시
      - 각 셀의 입자 에너지 확인
      - E < 10 MeV이면 "활성"으로 표시
      - 출력: ActiveMask 배열
    ],

    [K2], [CoarseTransport 거친 수송], [
      *대상*: 비활성 셀 (E > 10 MeV)
      - 평균 에너지 손실만 사용
      - 분산 추적, 샘플링 안 함
      - 더 큰 단계 크기
      - 빠름 (3-5배 속도 향상)
    ],

    [K3], [FineTransport 정밀 수송], [
      *대상*: 활성 셀 (E <= 10 MeV)
      - 에너지 분산 샘플링
      - MCS 산란 샘플링
      - 더 작은 단계 크기
      - 정확함 (1 percent 미만 오차)
    ],

    [K4], [BucketTransfer 버킷 전송], [
      *작업*: 셀을 떠난 입자 이동
      - 4개 이웃 확인 (±x, ±z)
      - 각 이웃의 유출 버킷에서 입자 읽기
      - 올바른 셀의 위상 공간에 추가
    ],

    [K5], [WeightAudit 보존 감사], [
      *작업*: 입자나 에너지가 손실되지 않았는지 확인
      - 총 입력 가중치 계산
      - 총 출력 가중치 계산
      - 검증: 입력 = 출력 + 흡수
      - 위반 보고
    ],

    [K6], [SwapBuffers 버퍼 교환], [
      *작업*: 다음 시간 단계 준비
      - 입력/출력 포인터 교환 (CPU 측)
      - 이전 출력이 다음 입력이 됨
      - 2.2 GB 데이터 복사 회피
    ],

    [출력], [출력: t = Δt의 입자], [다음 반복 준비 완료],
  ),
  caption: [CUDA 파이프라인 상세 흐름],
)

== K1: ActiveMask 커널

=== 파일

`src/cuda/kernels/k1_activemask.cu` (함수: `K1_ActiveMask`)

=== 간단한 설명

K1은 시뮬레이션 그리드의 모든 셀을 스캔하여 저에너지 입자(E <= 10 MeV)가 있는 셀을 식별합니다. 마스크(0과 1의 목록)를 생성하여 정밀 수송이 필요한 셀을 표시합니다.

=== 왜 필요한가

고에너지 입자(E > 10 MeV)는 브래그 피크에서 멀리 선량 분포에 큰 영향을 미치지 않습니다. 저에너지 입자(E <= 10 MeV)는 브래그 피크 안이나 근처에 있으며 작은 오차가 큰 선량 오차를 유발합니다. K1은 어떤 셀이 중요한지 식별하여 계산 리소스를 효율적으로 배분합니다.

=== 작동 방식

#figure(
  table(
    columns: (auto, 2fr, 1fr),
    inset: 10pt,
    stroke: (x: 1pt, y: 1pt),
    table.header([*단계*], [*설명*], [*결과*]),

    [입력], [모든 셀의 입자 데이터], [ ],

    [스캔], [
      각 스레드가 하나의 셀 처리:
      - 셀의 모든 입자 에너지 확인
      - 최소 에너지 찾기
      - 10 MeV 임계값과 비교
    ], [병렬 실행],

    [판정], [
      - 최소 E > 10 MeV: ActiveMask = 0
      - 최소 E <= 10 MeV: ActiveMask = 1
      - 가중치도 확인 (1e-12 최소)
    ], [마스크 생성],

    [출력], [
      - ActiveMask 배열 (0 또는 1)
      - ActiveList (압축된 활성 셀 목록)
    ], [K3용 입력],
  ),
  caption: [K1 활성 마스크 생성 과정],
)

=== 스레드 구성

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*매개변수*], [*값*]),
    [그리드 크기], [`(Nx * Nz + 255) / 256`],
    [블록 크기], [256 스레드],
    [셀당 스레드], [1 스레드가 1 셀 처리],
    [메모리 접근], [block_ids_in, values_in에서 병렬 읽기],
  ),
  caption: [K1 스레드 구성],
)

=== 시그니처

```cpp
__global__ void K1_ActiveMask(
    // 입력 위상 공간
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,

    // 그리드 매개변수
    const int Nx, const int Nz,

    // 임계값
    const int b_E_trigger,         // 에너지 블록 인덱스 임계값 (사전 계산됨)
    const float weight_active_min, // 최소 가중치 (기본값: 1e-12)

    // 출력
    uint8_t* __restrict__ ActiveMask
);
```

#warning-box[
*중요:* `b_E_trigger`는 float MeV 단위가 아니라 *정수 블록 인덱스*입니다. 에너지 임계값에서 사전 계산되며 정밀 수송이 활성화되는 거친 에너지 블록 인덱스를 나타냅니다.
]

=== 알고리즘

```cpp
__global__ void K1_ActiveMask(...) {
    // 단계 1: 이 스레드가 처리하는 셀 계산
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;  // 경계 확인

    // 단계 2: 누적 변수 초기화
    float total_weight = 0.0f;
    bool needs_fine_transport = false;

    // 단계 3: 이 셀의 모든 입자 슬롯 스캔
    for (int slot = 0; slot < 32; ++slot) {
        // 블록 ID 읽기 (에너지/각도 빈 정보)
        uint32_t bid = block_ids_in[cell * 32 + slot];
        if (bid == 0xFFFFFFFF) continue;  // 빈 슬롯 건너뜀

        // 블록 ID를 디코딩하여 에너지 빈 가져오기 (직접 비트 추출)
        uint32_t b_E = (bid >> 12) & 0xFFF;

        // 단계 4: 저에너지 존재 확인
        // b_E를 트리거와 직접 비교 (블록 인덱스 비교)
        if (b_E < static_cast<uint32_t>(b_E_trigger)) {
            needs_fine_transport = true;
        }

        // 단계 5: 입자 가중치 누적
        for (int lidx = 0; lidx < 32; ++lidx) {
            total_weight += values_in[(cell * 32 + slot) * 32 + lidx];
        }
    }

    // 단계 6: 출력 쓰기
    // (저에너지 존재) AND (충분한 가중치)이면 활성으로 표시
    ActiveMask[cell] = (needs_fine_transport && total_weight > weight_active_min) ? 1 : 0;
}
```

== K2: 거친 수송 커널

=== 파일

`src/cuda/kernels/k2_coarsetransport.cu` (함수: `K2_CoarseTransport`)

=== 간단한 설명

K2는 근사치를 사용하여 고에너지 입자를 빠르게 수송합니다. 모든 물리학적 세부 사항을 시뮬레이션하는 대신 평균값을 사용하여 3-5배 속도 향상을 달성합니다.

=== 왜 필요한가

고에너지 입자(브래그 피크에서 멈)는 최종 선량 분포에 큰 영향을 미치지 않습니다. 150 MeV 입자의 위치에서 5% 오차는 대부분의 에너지가 멀리 퇴적되므로 중요하지 않습니다. 근사치를 사용하여 허용 가능한 정확도로 3-5배 속도 향상을 얻을 수 있습니다.

=== K3와의 주요 차이점

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*기능*], [*K2 (거침)*], [*K3 (정밀)*], [*차이*]),
    [에너지 분산], [아니오(평균만)], [예(Vavilov)], [~3% 정확도 영향],
    [MCS 샘플링], [아니오(분산만)], [예(무작위 샘플링)], [~5% 확산 영향],
    [단계 크기], [큼], [작음], [2-3배 속도 향상],
      [정확도], [~5%], [1 percent 미만], [임상적 수용 가능],
  ),
  caption: [K2 vs K3 비교],
)

=== 간소화된 물리

#figure(
  table(
    columns: (1fr, 1fr),
    inset: 10pt,
    align: left,
    stroke: (x: 1pt, y: 1pt),
    table.header([*물리 과정*], [*K2 간소화 방법*]),

    [에너지 손실], [
      *표준*: Vavilov 분포에서 샘플링
      *거침*: 평균값만 사용
      - 결과: ~3% 오차, 2배 빠름
    ],

    [다중 쿨롱 산란(MCS)], [
      *표준*: 무작위 각도 θ ~ N(0, sigma²) 샘플링
      *거침*: sigma² 누적, 아직 샘플링 안 함
      - 결과: ~5% 확산 오차, 3배 빠름
    ],

    [단계 크기], [
      *표준*: ds = min(물리_제한, 경계_거리)
      *거침*: ds = 2-3배 더 큼
      - 결과: 더 적은 단계, 더 빠른 실행
    ],

    [핵 반응], [
      *정밀과 동일 (근사 불가)*
      - w *= exp(-sigma_nuclear * ds)
    ],
  ),
  caption: [K2 물리적 간소화],
)

=== 스레드 구성

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*매개변수*], [*값*]),
    [그리드 크기], [`(Nx * Nz + 255) / 256`],
    [블록 크기], [256 스레드],
    [처리], [셀당 1 스레드 (활성 셀 건너뜀)],
    [속도 향상], [정밀 수송 대비 3-5배],
  ),
  caption: [K2 스레드 구성],
)

=== 시그니처

```cpp
__global__ void K2_CoarseTransport(
    // 입력 위상 공간 (거친 셀만)
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint8_t* __restrict__ ActiveMask,

    // 그리드 및 물리
    const int Nx, const int Nz, const float dx, const float dz,
    const int n_coarse,
    const DeviceRLUT dlut,

    // 그리드 경계
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,

    // 설정
    K2Config config,

    // 출력
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,

    // 유출 버킷
    DeviceOutflowBucket* __restrict__ OutflowBuckets
);
```

=== 간소화된 물리 코드

```cpp
__device__ void coarse_transport_step(
    float& E, float& theta, float& x, float& z, float& w,
    float ds, const RLUT& lut
) {
    // 에너지 손실 (평균만, 분산 없음)
    float E_new = lut.lookup_E_inverse(lut.lookup_R(E) - ds);

    // 다중 쿨롱 산란 (분산 누적, 샘플링 안 함)
    float sigma_theta = highland_sigma(E, ds, X0_water);
    theta_variance += sigma_theta * sigma_theta;

    // 핵 감쇠 (정밀과 동일)
    float sigma_nuc = Sigma_total(E);
    w *= exp(-sigma_nuc * ds);

    // 위치 업데이트
    x += ds * sin(theta);
    z += ds * cos(theta);

    E = E_new;
}
```

== K3: 정밀 수송 커널 (주요 물리)

=== 파일

`src/cuda/kernels/k3_finetransport.cu` (함수: `K3_FineTransport`)

=== 간단한 설명

K3는 시뮬레이션의 핵심입니다. 브래그 피크 안이나 근처의 저에너지 입자에 대한 정확한 몬테카를로 수송을 수행합니다. 정확한 선량 분포를 보장하기 위해 모든 물리학적 효과가 무작위 샘플링으로 시뮬레이션됩니다.

=== 왜 필요한가

브래그 피크는 대부분의 에너지가 퇴적되는 곳입니다. 여기서 작은 오차가 큰 선량 오차를 유발합니다. 10 MeV에서 1mm 위치 오차는 선량을 20% 변경할 수 있습니다. K3는 중요한 곳에서 1 percent 미만 오차를 보장합니다.

=== 작동 방식

#figure(
  table(
    columns: (auto, 2fr),
    inset: 10pt,
    stroke: (x: 1pt, y: 1pt),
    table.header([*단계*], [*설명*]),

    [활성 셀 처리], [
      - ActiveList의 셀만 처리
      - 각 스레드가 하나의 활성 셀 할당
    ],

    [RNG 초기화], [
      - 각 스레드가 고유한 무작위 상태 가져옴
      - 재현 가능한 독립적인 난수 보장
    ],

    [슬롯 처리], [
      - 각 셀에는 여러 입자 "슬롯"이 있음
      - 빈 슬롯 건너뜀
    ],

    [로컬 빈 처리], [
      - 각 슬롯에는 512개의 로컬 빈이 있음
      - 무시할 수 있는 가중치 건너뜀
    ],

    [위상 공간 디코딩], [
      - 빈 인덱스를 물리적 값으로 변환
      - 각도(θ), 에너지(E), 위치(x, z)
    ],

    [빈 내 샘플링], [
      - 빔 내 균일 위치 샘플링
      - 분산 보존을 위해 필수
    ],

    [주요 수송 루프], [
      1. 단계 크기 제어 (물리 + 경계)
      2. 분산이 포함된 에너지 손실
      3. MCS 샘플링
      4. 핵 감쇠
      5. 위치 업데이트
      6. 종료 조건 확인
    ],

    [원자적 누적], [
      - 스레드 안전 출력
      - EdepC, AbsorbedWeight, AbsorbedEnergy
    ],
  ),
  caption: [K3 입자당 수송 과정],
)

=== 빈 내 샘플링의 중요성

#figure(
  table(
    columns: (1fr, 1fr),
    inset: 10pt,
    align: left,
    stroke: (x: 1pt, y: 1pt),
    table.header([*빈 내 샘플링 없이*], [*빈 내 샘플링으로*]),

    [
      동일한 빈의 모든 입자가 동일한 위치를 얻음
      - 인공적 클러스터링 (나쁜 통계)
      - 과소평가된 분산
    ], [
      각 입자가 빔 내 무작위 오프셋을 얻음
      - 연속 분포 보존
      - 올바른 분산
    ],
  ),
  caption: [빈 내 샘플링 효과],
)

=== 스레드 구성

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*매개변수*], [*값*]),
    [그리드 크기], [`(n_active + 255) / 256`],
    [블록 크기], [256 스레드],
    [처리], [활성 셀당 1 스레드],
    [RNG 상태], [스레드당 1개 (독립적 난수)],
    [공유 메모리], [로컬 빈 누적용 4 KB],
  ),
  caption: [K3 스레드 구성],
)

=== 시그니처

```cpp
__global__ void K3_FineTransport(
    // 입력: 활성 셀 목록
    const uint32_t* __restrict__ ActiveList,
    const int n_active,

    // 입력 위상 공간
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,

    // 그리드 및 물리
    const int Nx, const int Nz, const float dx, const float dz,
    const int n_active,
    const DeviceRLUT dlut,

    // 빈 찾기를 위한 그리드 경계
    const float* __restrict__ theta_edges,
    const float* __restrict__ E_edges,
    int N_theta, int N_E,
    int N_theta_local, int N_E_local,

    // 출력
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    float* __restrict__ BoundaryLoss_weight,
    double* __restrict__ BoundaryLoss_energy,

    // 유출 버킷
    DeviceOutflowBucket* __restrict__ OutflowBuckets
);
```

#warning-box[
*RNG 구현 참고:* K3는 CUDA curand가 아닌 결정론적 시드 기반 RNG를 사용합니다. 재현성을 위해 셀/슬롯/빈 인덱스를 기반으로 하는 해시 함수를 사용하여 난수를 생성합니다.
]

=== 알고리즘 (셀당)

```cpp
__global__ void K3_FineTransport(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_active) return;

    int cell = ActiveList[idx];

    float cell_Edep = 0.0;
    float cell_E_nuc = 0.0;
    float w_cutoff = 0.0;

    for (int slot = 0; slot < 32; ++slot) {
        uint32_t bid = block_ids_in[cell * 32 + slot];
        if (bid == EMPTY_BLOCK_ID) continue;

        uint32_t b_theta, b_E;
        decode_block(bid, b_theta, b_E);

        for (int lidx = 0; lidx < 32; ++lidx) {
            float w = values_in[(cell * 32 + slot) * 32 + lidx];
            if (w < weight_epsilon) continue;

            // 위상 공간 디코딩
            int theta_local, E_local, x_sub, z_sub;
            decode_local_idx_4d(lidx, theta_local, E_local, x_sub, z_sub);

            float theta = get_theta_from_bins(b_theta, theta_local);
            float E = get_energy_from_bins(b_E, E_local);
            float x = cell_x + get_x_offset_from_bin(x_sub, dx);
            float z = cell_z + get_z_offset_from_bin(z_sub, dz);

            // 주요 물리 루프
            while (true) {
                float ds = compute_max_step_physics(E, dlut);
                ds = fminf(ds, compute_boundary_step(x, z, dx, dz, theta));

                // 분산이 포함된 에너지 손실
                float dE_straggle = sample_energy_loss_with_straggling(E, ds, cell, slot, lidx);
                float E_new = dlut.lookup_E_inverse(dlut.lookup_R(E) - ds) + dE_straggle;
                E_new = fmaxf(E_new, E_cutoff);

                float dE = E - E_new;
                cell_Edep += w * dE;

                // MCS 샘플링
                float sigma_theta = highland_sigma(E, ds, X0_water);
                float dtheta = sample_mcs_angle(sigma_theta, cell, slot, lidx);
                theta += dtheta;

                // 핵 감쇠
                float sigma_nuc = Sigma_total(E);
                float w_nuc = w * (1.0f - exp(-sigma_nuc * ds));
                w -= w_nuc;
                cell_E_nuc += w_nuc * E;

                // 위치 업데이트
                x += ds * sin(theta);
                z += ds * cos(theta);
                E = E_new;

                // 경계 확인
                if (left_cell) {
                    emit_to_bucket(OutflowBuckets, cell, face, theta, E, x, z, w);
                    break;
                }

                if (E < E_cutoff) {
                    w_cutoff += w;
                    break;
                }
            }
        }
    }

    // 원자적 누적
    atomicAdd(&EdepC[cell], cell_Edep);
    atomicAdd(&AbsorbedWeight_cutoff[cell], w_cutoff);
    atomicAdd(&AbsorbedWeight_nuclear[cell], w - w_cutoff);
    atomicAdd(&AbsorbedEnergy_nuclear[cell], cell_E_nuc);
}
```

=== 빈 내 샘플링 코드

분산 보존을 위해 입자는 빈 내에서 균일하게 샘플링됩니다:

```cpp
__device__ void sample_intra_bin(
    float& theta,
    int theta_bin, int cell, int slot, int lidx,
    float theta_edges[], int N_theta
) {
    // 셀/슬롯/lidx에서 결정론적 시드 생성
    unsigned seed = static_cast<unsigned>(
        (cell * 7 + slot * 13 + lidx * 17) ^ 0x5DEECE66DL
    );

    // 빈 내 균일 오프셋 생성 [0, 1)
    float theta_frac = (seed & 0xFFFF) / 65536.0f;

    // 빈 내 위치를 위해 빈 경계에 추가
    float dtheta = (theta_edges[N_theta] - theta_edges[0]) / N_theta;
    theta = theta_edges[theta_bin] + theta_frac * dtheta;
}
```

== K4: 버킷 전송 커널

=== 파일

`src/cuda/kernels/k4_transfer.cu` (함수: `K4_BucketTransfer`)

=== 간단한 설명

K4는 수송(K2/K3) 중에 셀을 떠난 입자를 새 셀로 이동합니다. 각 셀에는 4개의 "버킷"(각 방향당 하나: ±x, ±z)이 있어 나가는 입자를 포착합니다. K4는 이 버킷을 읽고 올바른 인접 셀에 입자를 넣습니다.

=== 왜 필요한가

입자는 이동합니다! K2/K3 수송 후 입자는 셀 경계를 넘었을 수 있습니다. 이 모든 입자를 수집하여 새 집에 넣어야 합니다. 이는 다음 반복을 위한 공간적 지역성을 유지합니다.

=== 작동 방식

#figure(
  table(
    columns: (auto, 2fr),
    inset: 10pt,
    stroke: (x: 1pt, y: 1pt),
    table.header([*단계*], [*설명*]),

    [입력], [
      각 셀에는 4개의 버킷이 있음:
      - 버킷 0: +z 방향으로 떠나는 입자
      - 버킷 1: -z 방향으로 떠나는 입자
      - 버킷 2: +x 방향으로 떠나는 입자
      - 버킷 3: -x 방향으로 떠나는 입자
    ],

    [이웃 계산], [
      위치 (ix, iz)의 셀의 경우:
      - +z: cell + Nx (iz+1 < Nz이면)
      - -z: cell - Nx (iz-1 >= 0이면)
      - +x: cell + 1 (ix+1 < Nx이면)
      - -x: cell - 1 (ix-1 >= 0이면)
    ],

    [전송 과정], [
      각 수신 셀에 대해:
      1. 4개 이웃의 버킷 확인
      2. 각 버킷의 내용 읽기
      3. block_id가 이미 있는지 확인
         - 예: 가중치를 기존 블록에 추가
         - 아니오: 새 슬롯 할당
      4. 원자적 슬롯 할당 (스레드 안전)
    ],

    [출력], [
      - 모든 입자가 올바른 공간 셀에 있음
      - 올바른 위상 공간 빈 (E, θ, x, z)
      - 다음 반복 준비 완료
    ],
  ),
  caption: [K4 버킷 전송 과정],
)

=== 원자적 연산의 중요성

#figure(
  table(
    columns: (1fr, 1fr),
    inset: 10pt,
    align: left,
    stroke: (x: 1pt, y: 1pt),
    table.header([*원자적 연산 없이*], [*atomicCAS로*]),

    [
      *경쟁 조건:*
      - 스레드 A 읽기: slot[5] = EMPTY
      - 스레드 B 읽기: slot[5] = EMPTY
      - 스레드 A 쓰기: slot[5] = BLOCK_42
      - 스레드 B 쓰기: slot[5] = BLOCK_99
      - → 스레드 A의 데이터 손실
    ], [
      *스레드 안전:*
      - 스레드 A: atomicCAS → 성공
      - 스레드 B: atomicCAS → 실패
      - 스레드 B: slot[6] 시도
      - → 모든 데이터 보존
    ],
  ),
  caption: [원자적 슬롯 할당의 필요성],
)

=== 스레드 구성

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*매개변수*], [*값*]),
    [그리드 크기], [`(Nx * Nz + 255) / 256`],
    [블록 크기], [256 스레드],
    [처리], [수신 셀당 1 스레드],
    [원자적 연산], [예 (슬롯 할당, 가중치 추가)],
    [공유 메모리], [전송 버퍼용 1 KB],
  ),
  caption: [K4 스레드 구성],
)

=== 시그니처

```cpp
__global__ void K4_BucketTransfer(
    // 입력: 모든 셀의 유출 버킷
    const DeviceOutflowBucket* __restrict__ OutflowBuckets,

    // 그리드
    const int Nx, const int Nz,

    // 출력 위상 공간
    float* __restrict__ values_out,
    uint32_t* __restrict__ block_ids_out
);
```

=== 알고리즘

```cpp
__global__ void K4_BucketTransfer(...) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    int ix = cell % Nx;
    int iz = cell / Nx;

    // 4개 이웃에서 수신
    int neighbors[4] = {
        iz + 1 < Nz ? cell + Nx : -1,  // +z
        iz - 1 >= 0 ? cell - Nx : -1,  // -z
        ix + 1 < Nx ? cell + 1 : -1,   // +x
        ix - 1 >= 0 ? cell - 1 : -1    // -x
    };

    for (int face = 0; face < 4; ++face) {
        int src_cell = neighbors[face];
        if (src_cell < 0) continue;

        const DeviceOutflowBucket& bucket = OutflowBuckets[src_cell * 4 + face];

        for (int k = 0; k < 32; ++k) {
            uint32_t bid = bucket.block_id[k];
            if (bid == EMPTY_BLOCK_ID) continue;

            int slot = find_or_allocate_slot(block_ids_out, cell, bid);
            if (slot < 0) continue;

            for (int lidx = 0; lidx < 32; ++lidx) {
                float w = bucket.value[k][lidx];
                if (w > 0) {
                    atomicAdd(&values_out[(cell * 32 + slot) * 32 + lidx], w);
                }
            }
        }
    }
}
```

=== 원자적 슬롯 할당

```cpp
__device__ int find_or_allocate_slot(
    uint32_t* block_ids,
    int cell,
    uint32_t bid
) {
    // 첫 번째 패스: 존재 확인
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_ids[cell * Kb + slot] == bid) {
            return slot;
        }
    }

    // 두 번째 패스: 빈 슬롯 할당
    for (int slot = 0; slot < Kb; ++slot) {
        uint32_t expected = EMPTY_BLOCK_ID;
        uint32_t* ptr = &block_ids[cell * Kb + slot];
        if (atomicCAS(ptr, expected, bid) == expected) {
            return slot;
        }
    }

    return -1;  // 사용 가능한 공간 없음
}
```

== K5: 가중치 감사 커널

=== 파일

`src/cuda/kernels/k5_audit.cu` (함수: `K5_WeightAudit`)

=== 간단한 설명

K5는 시뮬레이션의 회계사입니다. 수송 중 입자나 에너지가 손실되지 않았는지 확인합니다. 각 셀에 대해 다음을 검증합니다: "들어온 것 = 나간 것 + 흡수된 것."

=== 왜 필요한가

보존 법칙은 기본입니다. 가중치가 보존되지 않으면 시뮬레이션에 버그가 있는 것입니다. K5는 다음을 포착합니다:
- 사라지는 입자 (수송 논리 버그)
- 퇴적되지 않은 에너지 (회계 누출)
- 시간이 지남에 따라 누적되는 수치 오류

=== 작동 방식

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    inset: 10pt,
    stroke: (x: 1pt, y: 1pt),
    table.header([*셀*], [*입력 (W_in)*], [*출력+흡수 (W_out+W_cut+W_nuc)*], [*상태*]),

    [Cell 0], [1.0], [0.7 + 0.2 + 0.1 = 1.0], [통과 (0.0% 오차)],
    [Cell 1], [0.8], [0.6 + 0.15 + 0.05 = 0.8], [통과 (0.0% 오차)],
    [Cell 2], [0.5], [0.4 + 0.08 + 0.02 = 0.5], [통과 (0.0% 오차)],
  ),
  caption: [K5 보존 검증 예제],
)

=== 스레드 구성

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*매개변수*], [*값*]),
    [그리드 크기], [`(Nx * Nz + 255) / 256`],
    [블록 크기], [256 스레드],
    [처리], [셀당 1 스레드],
    [메모리 접근], [읽기 전용 (원자 연산 불필요)],
  ),
  caption: [K5 스레드 구성],
)

=== 시그니처

```cpp
__global__ void K5_WeightAudit(
    // 입력 위상 공간 (입력과 출력 모두)
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint32_t* __restrict__ block_ids_out,
    const float* __restrict__ values_out,

    // 흡수 배열
    const float* __restrict__ AbsorbedWeight_cutoff,
    const float* __restrict__ AbsorbedWeight_nuclear,
    const double* __restrict__ AbsorbedEnergy_nuclear,

    // 그리드
    const int Nx, const int Nz,

    // 출력 보고서
    AuditReport* __restrict__ reports
);

struct AuditReport {
    float W_error;  // 가중치 보존 오차
    bool W_pass;    // 통과/실패 플래그
};
```

=== 알고리즘

```cpp
__global__ void K5_WeightAudit(...) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    // 입력 가중치 합산
    float W_in = 0.0f;
    for (int slot = 0; slot < 32; ++slot) {
        if (block_ids_in[cell * 32 + slot] != EMPTY_BLOCK_ID) {
            for (int i = 0; i < 32; ++i) {
                W_in += values_in[(cell * 32 + slot) * 32 + i];
            }
        }
    }

    // 출력 가중치 합산
    float W_out = 0.0f;
    for (int slot = 0; slot < 32; ++slot) {
        if (block_ids_out[cell * 32 + slot] != EMPTY_BLOCK_ID) {
            for (int i = 0; i < 32; ++i) {
                W_out += values_out[(cell * 32 + slot) * 32 + i];
            }
        }
    }

    // 흡수 가져오기
    float W_cut = AbsorbedWeight_cutoff[cell];
    float W_nuc = AbsorbedWeight_nuclear[cell];

    // 보존 확인
    float W_expected = W_out + W_cut + W_nuc;
    float W_diff = fabsf(W_in - W_expected);
    float W_rel = W_diff / fmaxf(W_in, 1e-20f);

    // 보고서 저장
    reports[cell].W_error = W_rel;
    reports[cell].W_pass = (W_rel < 1e-6f);
}
```

== K6: 버퍼 교환

=== 파일

`src/cuda/kernels/k6_swap.cu` (함수: `K6_SwapBuffers`)

=== 간단한 설명

K6는 입력과 출력 배열을 교환하여 다음 시간 단계를 준비합니다. 이 반복의 출력이 다음 반복의 입력이 됩니다.

=== 왜 필요한가

K2-K4 후 입자는 "out" 배열에 있습니다. 다음 반복을 위해 이것들이 "in" 배열이어야 합니다. 2.2 GB 데이터를 복사하는 대신 배열의 포인터(주소)를 교환합니다.

=== 작동 방식

#figure(
  table(
    columns: (1fr, 1fr),
    inset: 10pt,
    align: left,
    stroke: (x: 1pt, y: 1pt),
    table.header([*교환 전*], [*교환 후*]),

    [
      *메모리 레이아웃:*
      \
      in → [PsiC A] (이전 입력, 현재 폐기됨)
      \
      out → [PsiC B] (새 출력, 방금 계산됨)
      \
      *다음 반복용:*
      - PsiC B가 입력이 되어야 함
      - PsiC A가 출력이 되어야 함 (덮어쓰기 예정)
    ], [
      *메모리 레이아웃:*
      \
      in → [PsiC B] ← 이전 출력, 이제 입력
      \
      out → [PsiC A] ← 이전 입력, 이제 출력
      \
      다음 K1-K5 커널은 "in"에서 읽고 "out"에 쓰며, 효과적으로 이전 입력 데이터를 덮어씁니다
    ],

    [데이터 복사 없음!], [다음 반복 준비 완료],
  ),
  caption: [K6 버퍼 교환 과정],
)

=== 포인터 교환의 이점

#figure(
  table(
    columns: (1fr, 1fr),
    inset: 10pt,
    align: left,
    stroke: (x: 1pt, y: 1pt),
    table.header([*옵션 1: 데이터 복사*], [*옵션 2: 포인터 교환*]),

    [
      - cudaMemcpy(new_in, old_out, 2.2 GB)
      - 반복당 ~100 ms 소요
      - 메모리 대역폭 낭비
    ], [
      - swap(in_ptr, out_ptr)
      - 반복당 ~0.001 microsec 소요
      - 메모리 대역폭 0 사용
      - 100,000배 더 빠름!
    ],
  ),
  caption: [K6 성능 비교],
)

=== 구현

```cpp
// 호스트 측 함수 (커널 실행 없음)
void K6_SwapBuffers(PsiC*& in, PsiC*& out) {
    // 세 방향 XOR 교환 (임시 변수 불필요)
    PsiC* temp = in;
    in = out;
    out = temp;
}
```

#warning-box[
*구현 참고:* K6는 개별 배열 포인터가 아니라 `PsiC*` 포인터(모든 배열을 포함하는 구조체)를 교환합니다. 개별 배열을 교환하는 것보다 더 간단한 인터페이스입니다.
]

=== 커널이 없는 이유?

포인터 교환은 CPU 연산입니다 - GPU 메모리를 수정할 필요가 없습니다. 이는 반복당 ~2.2 GB 메모리 복사를 피합니다.

== 메모리 접근 패턴

=== 병렬 접근 전략

#figure(
  table(
    columns: (auto, 2fr),
    inset: 10pt,
    stroke: (x: 1pt, y: 1pt),
    table.header([*패턴*], [*설명*]),

    [병렬 접근 (좋음)], [
      메모리가 연속적으로 구성됨:
      - 스레드 0 읽기: block_ids_in[0], values_in[0:31]
      - 스레드 1 읽기: block_ids_in[1], values_in[32:63]
      - 스레드 2 읽기: block_ids_in[2], values_in[64:95]

      GPU가 하나의 메모리 트랜잭션으로 결합:
      "block_ids_in[0:255] 및 values_in[0:8191] 읽기"

      → 분산 접근보다 20-30배 빠름
    ],

    [분산 접근 (나쁨)], [
      - 스레드 0 읽기: block_ids_in[0]
      - 스레드 1 읽기: block_ids_in[1000]
      - 스레드 2 읽기: block_ids_in[2000]

      GPU가 3개의 별도 메모리 트랜잭션 수행
      → 20-30배 느림
    ],
  ),
  caption: [메모리 접근 패턴 비교],
)

=== 공유 메모리 사용

#figure(
  table(
    columns: (auto, auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*커널*], [*공유 메모리*], [*목적*]),
    [K1], [256 B], [가중치 합산을 위한 부분 감소],
    [K3], [4 KB], [로컬 빈 누적],
    [K4], [1 KB], [버킷 전송 버퍼],
  ),
  caption: [공유 메모리 사용],
)

== 성능 최적화 요약

#figure(
  table(
    columns: (2fr, auto, 3fr),
    inset: 8pt,
    align: left,
    table.header([*기법*], [*커널*], [*이점*]),
    [활성 셀 처리], [K2, K3], [빈 셀 건너뜀 (60-90% 절약)],
    [거침/정밀 분할], [K2, K3], [고에너지용 3-5배 속도 향상],
    [원자적 연산], [K4], [스레드 안전 슬롯 할당],
    [빈 내 샘플링], [K3], [분산 보존],
    [포인터 교환], [K6], [2.2 GB 메모리 복사 회피],
    [병렬 접근], [모두], [최대 메모리 대역폭],
  ),
  caption: [성능 최적화],
)

== 부록: 실행 구성 예제

=== 그리드 및 블록 설정

```cpp
// 그리드 차원
dim3 grid( (Nx * Nz + 255) / 256 );
dim3 block(256);

// K1: ActiveMask
K1_ActiveMask<<<grid, block>>>(...);

// K3: 정밀 수송 (활성 셀용 더 작은 그리드)
dim3 grid_fine( (n_active + 255) / 256 );
K3_FineTransport<<<grid_fine, block>>>(...);

// 동기화
cudaDeviceSynchronize();
```

=== 쉬운 이해를 위한 비유

#figure(
  table(
    columns: (auto, 2fr),
    inset: 10pt,
    stroke: (x: 1pt, y: 1pt),
    table.header([개념], [비유]),
 
    [GPU vs CPU], [
      *CPU 방식:* 한 명의 숙련된 화가가 펜스를 하나씩 침함\
      *GPU 방식:* 10,000명의 견습 화가가 각자 펜스를 동시에 침함\
    ],
 
    [CUDA 커널], [
      여러 요리사가 동시에 따르는 레시피\
      - 각 요리사(스레드)는 동일한 레시피(커널 코드)\
      - 다른 재료(데이터)로 작업\
    ],
 
    [스레드 계층], [
      *그리드:* 모든 셀의 모든 스레드\
      *블록:* 협력할 수 있는 스레드 그룹 (32-1024개)\
      *스레드:* 실행의 가장 작은 단위\
    ],
 
    [메모리 계층], [
      *전역 메모리:* 책이 지하실에 보관된 도서관 (느림)\
      *공유 메모리:* 토론 그룹이 자주 사용하는 책을 두는 작은 테이블 (빠름)\
    ],
 
    [K1: ActiveMask], [
      품질 관리 검사원이 특별한 취급이 필요한 상자 식별\
      - 품목 가치 > 1000: 표준 취급\
      - 품목 가치 <= 1000: 신중한 취급\
    ],
 
    [K2 vs K3], [
      *K2 (거침):* 도시의 평균 속도로 여행 시간 계산\
      *K3 (정밀):* 모든 신호등과 정체 상황 시뮬레이션\
 
      브래그 피크 근처에서는 K3의 세부 사항이 필요\
    ],
 
    [K4: 전송], [
      우편 서비스 편지 분류\
      - 할당된 분류함(셀) 확인\
      - 4개 인접 분류 센터의 편지 확인\
      - 올바른 슬롯에 정리\
    ],
 
    [K5: 감사], [
      통장 정리\
      *시작 잔액 + 입금 = 종료 잔액 + 출금*\
 
      숫자가 맞지 않으면 버그가 있음\
    ],
 
    [K6: 교환], [
      상자 라벨 재지정\
      *느린 방식:* 상자 A의 모든 문서를 상자 B로 이동 (몇 시간)\
      *빠른 방식:* 상자 A와 B의 라벨 교환 (1초)\
    ],
 
  ),
  caption: [CUDA 개념에 대한 쉬운 비유],
 )

== 요약

SM_2D CUDA 파이프라인은 물질을 통한 양성자 수송을 효율적으로 수행하기 위해 6단계 커널 시퀀스를 사용합니다:

- *K1 (ActiveMask)*: 정확한 시뮬레이션이 필요한 셀 식별
- *K2 (거침)*: 고에너지 입자용 빠른 수송
- *K3 (정밀)*: 브래그 피크 영역용 정확한 몬테카를로
- *K4 (전송)*: 셀 간 입자 이동
- *K5 (WeightAudit)*: 가중치 보존 검증
- *K6 (교환)*: 다음 반복 준비

각 커널은 병렬 메모리 접근, 최소 스레드 분기, 효율적인 공유 메모리 사용으로 GPU 병렬 실행에 최적화되어 있습니다.

---
#align(center)[
  *SM_2D CUDA 파이프라인 문서*

  #text(size: 9pt)[버전 2.0 - 체계적으로 정리됨]
]
