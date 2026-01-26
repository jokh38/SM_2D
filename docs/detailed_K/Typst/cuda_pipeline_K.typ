#set text(font: "Linux Libertine", size: 11pt)
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

#show math: set text(weight: "regular")

= CUDA 커널 파이프라인 문서

== 개요

SM_2D는 결정론적 양성자 수송을 위한 6단계 CUDA 커널 파이프라인을 구현합니다. 파이프라인은 계층적 세분화를 통해 입자를 처리하며, 고에너지 입자용 거친 수송과 중요한 브래그 피크 영역용 정밀 수송을 사용합니다.

== 파이프라인 아키텍처

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
    [K3: FineTransport], [저에너지 수송(E ≤ 10 MeV)],
    [K4: BucketTransfer], [셀 간 입자 전송],
    [K5: ConservationAudit], [보존 법칙 검증],
    [K6: SwapBuffers], [입력/출력 포인터 교환],
  ),
  caption: [CUDA 커널 파이프라인],
)

=== 데이터 흐름

```
K1 (ActiveMask) → 정밀 수송이 필요한 셀 식별
     ↓
K2 (거침) + K3 (정밀) → 입자 수송
     ↓
K4 (전송) → 셀 간 입자 이동
     ↓
K5 (감사) → 보존 확인
     ↓
K6 (교환) → 다음 단계용 버퍼 교환
```

== K1: ActiveMask 커널

=== 파일

`src/cuda/kernels/k1_activemask.cu`

=== 목적

정밀 수송이 필요한 셀 식별(브래그 피크 영역의 저에너지 입자).

=== 시그니처

```cpp
__global__ void k1_activemask(
    // 입력 위상 공간
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,

    // 그리드 매개변수
    const int Nx, const int Nz,

    // 임계값
    const float b_E_trigger,      // 에너지 임계값(기본값: 10 MeV)
    const float weight_active_min, // 최소 가중치(기본값: 1e-12)

    // 출력
    uint8_t* __restrict__ ActiveMask
);
```

=== 알고리즘

```cpp
__global__ void k1_activemask(...) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    float total_weight = 0.0f;
    bool has_low_energy = false;

    // 모든 슬롯과 로컬 빔에서 가중치 합산
    for (int slot = 0; slot < Kb; ++slot) {
        uint32_t bid = block_ids_in[cell * Kb + slot];
        if (bid == EMPTY_BLOCK_ID) continue;

        // 블록 ID 디코딩
        uint32_t b_theta, b_E;
        decode_block(bid, b_theta, b_E);

        // 대표 에너지 가져오기
        float E = get_rep_energy(b_E);

        // 저에너지 확인
        if (E < b_E_trigger) {
            has_low_energy = true;
        }

        // 가중치 누적
        for (int i = 0; i < LOCAL_BINS; ++i) {
            total_weight += values_in[flat_index(cell, slot, i)];
        }
    }

    // 저에너지 AND 충분한 가중치이면 마스크 설정
    ActiveMask[cell] = (has_low_energy && total_weight > weight_active_min) ? 1 : 0;
}
```

=== 스레드 구성

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: left,
    table.header([*매개변수*], [*값*]),
    [그리드 크기], [`(Nx * Nz + 255) / 256`],
    [블록 크기], [256 스레드],
    [메모리 접근], [block_ids_in, values_in에서 병렬 읽기],
  ),
  caption: [K1 스레드 구성],
)

== K2: 거친 수송 커널

=== 파일

`src/cuda/kernels/k2_coarsetransport.cu`

=== 목적

고에너지 입자(ActiveMask = 0)를 위한 빠른 근사 수송.

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
    [정확도], [~5%], [<1%], [임상적 수용 가능],
  ),
  caption: [K2 vs K3 비교],
)

=== 시그니처

```cpp
__global__ void k2_coarsetransport(
    // 입력 위상 공간(거친 셀만)
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,
    const uint8_t* __restrict__ ActiveMask,

    // 그리드 및 물리
    const int Nx, const int Nz, const float dx, const float dz,
    const RLUT __restrict__ lut,

    // 출력
    uint32_t* __restrict__ block_ids_out,
    float* __restrict__ values_out,
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    OutflowBucket* __restrict__ OutflowBuckets
);
```

=== 간소화된 물리

```cpp
__device__ void coarse_transport_step(
    float& E, float& theta, float& x, float& z, float& w,
    float ds, const RLUT& lut
) {
    // 에너지 손실(평균만, 분산 없음)
    float E_new = lut.lookup_E_inverse(lut.lookup_R(E) - ds);

    // MCS: 분산 누적하지만 샘플링은 안 함
    float sigma_theta = highland_sigma(E, ds, X0_water);
    theta_variance += sigma_theta * sigma_theta;  // 추적만

    // 핵 감쇠(정밀과 동일)
    float sigma_nuc = Sigma_total(E);
    w *= exp(-sigma_nuc * ds);

    // 위치 업데이트
    x += ds * sin(theta);
    z += ds * cos(theta);

    E = E_new;
}
```

== K3: 정밀 수송 커널(주요 물리)

=== 파일

`src/cuda/kernels/k3_finetransport.cu`

=== 목적

저에너지 입자(브래그 피크 영역)를 위한 고정확도 몬테카를로 수송.

=== 시그니처

```cpp
__global__ void k3_finetransport(
    // 입력: 활성 셀 목록
    const int* __restrict__ ActiveList,
    const int n_active,

    // 입력 위상 공간
    const uint32_t* __restrict__ block_ids_in,
    const float* __restrict__ values_in,

    // 그리드 및 물리
    const int Nx, const int Nz, const float dx, const float dz,
    const RLUT __restrict__ lut,
    const curandStateMRG32k3a* __restrict__ rng_states,

    // 출력
    uint32_t* __restrict__ block_ids_out,
    float* __restrict__ values_out,
    double* __restrict__ EdepC,
    float* __restrict__ AbsorbedWeight_cutoff,
    float* __restrict__ AbsorbedWeight_nuclear,
    double* __restrict__ AbsorbedEnergy_nuclear,
    OutflowBucket* __restrict__ OutflowBuckets
);
```

=== 알고리즘(셀당)

```cpp
__global__ void k3_finetransport(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_active) return;

    int cell = ActiveList[idx];
    curandStateMRG32k3a local_rng = rng_states[idx];

    // 셀의 각 슬롯 처리
    for (int slot = 0; slot < Kb; ++slot) {
        uint32_t bid = block_ids_in[cell * Kb + slot];
        if (bid == EMPTY_BLOCK_ID) continue;

        // 블록 디코딩
        uint32_t b_theta, b_E;
        decode_block(bid, b_theta, b_E);

        // 각 로컬 빈 처리
        for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
            float w = values_in[flat_index(cell, slot, lidx)];
            if (w < weight_epsilon) continue;

            // 4차원 위상 공간 좌표 디코딩
            int theta_local, E_local, x_sub, z_sub;
            decode_local_idx(lidx, theta_local, E_local, x_sub, z_sub);

            float theta = get_theta_from_bins(b_theta, theta_local);
            float E = get_energy_from_bins(b_E, E_local);
            float x = cell_x + get_x_offset_from_bin(x_sub, dx);
            float z = cell_z + get_z_offset_from_bin(z_sub, dz);

            // --- 주요 물리 루프 ---
            float cell_Edep = 0.0;
            float cell_E_nuc = 0.0;
            float w_cutoff = 0.0;

            while (true) {
                // 단계 크기 제어
                float ds = compute_max_step_physics(E, lut);
                ds = fminf(ds, compute_boundary_step(x, z, dx, dz, theta));

                // 분산이 포함된 에너지 손실
                float dE_straggle = sample_energy_loss_with_straggling(E, ds, &local_rng);
                float E_new = lut.lookup_E_inverse(lut.lookup_R(E) - ds) + dE_straggle;
                E_new = fmaxf(E_new, E_cutoff);

                // 에너지 퇴적
                float dE = E - E_new;
                cell_Edep += w * dE;

                // 샘플링이 포함된 MCS
                float sigma_theta = highland_sigma(E, ds, X0_water);
                float dtheta = sample_mcs_angle(sigma_theta, &local_rng);
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

                // 경계 탈출 확인
                int face = check_boundary_exit(x, z, dx, dz);
                if (face >= 0) {
                    emit_to_bucket(OutflowBuckets, cell, face,
                                  theta, E, x, z, w);
                    break;
                }

                // 차단 확인
                if (E < E_cutoff) {
                    w_cutoff += w;
                    break;
                }

                // 동일 셀 내 확인
                int new_cell = get_cell(x, z, dx, dz);
                if (new_cell != cell) {
                    int face = get_exit_face(x, z, dx, dz);
                    emit_to_bucket(OutflowBuckets, cell, face,
                                  theta, E, x, z, w);
                    break;
                }
            }

            // 출력 누적
            atomicAdd(&EdepC[cell], cell_Edep);
            atomicAdd(&AbsorbedWeight_cutoff[cell], w_cutoff);
            atomicAdd(&AbsorbedWeight_nuclear[cell], w - w_cutoff);
            atomicAdd(&AbsorbedEnergy_nuclear[cell], cell_E_nuc);
        }
    }

    rng_states[idx] = local_rng;
}
```

=== 빈 내 샘플링

분산 보존을 위해 입자는 빔 내에서 균일하게 샘플링됩니다:

```cpp
__device__ void sample_intra_bin(
    float& theta, float& E,
    int theta_local, int E_local,
    curandStateMRG32k3a* rng
) {
    // 빔 내 균일 오프셋 샘플링
    float u_theta = curand_uniform(rng) - 0.5f;  // [-0.5, 0.5]
    float u_E = curand_uniform(rng) - 0.5f;

    // 대표 값에 추가
    theta += u_theta * dtheta_bin;
    E *= pow(10.0f, u_E * dlogE_bin);  // E에 대한 로그 간격
}
```

== K4: 버킷 전송 커널

=== 파일

`src/cuda/kernels/k4_transfer.cu`

=== 목적

유출 버킷에서 인접 셀로 입자 가중치 전송.

=== 시그니처

```cpp
__global__ void k4_transfer(
    // 입력: 모든 셀의 유출 버킷
    const OutflowBucket* __restrict__ OutflowBuckets,

    // 그리드
    const int Nx, const int Nz,

    // 출력 위상 공간
    uint32_t* __restrict__ block_ids_out,
    float* __restrict__ values_out
);
```

=== 알고리즘

```cpp
__global__ void k4_transfer(...) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    // 셀 좌표 가져오기
    int ix = cell % Nx;
    int iz = cell / Nx;

    // 4개 이웃에서 수신
    int neighbors[4] = {
        iz + 1 < Nz ? cell + Nx : -1,  // +z
        iz - 1 >= 0 ? cell - Nx : -1,  // -z
        ix + 1 < Nx ? cell + 1 : -1,   // +x
        ix - 1 >= 0 ? cell - 1 : -1    // -x
    };

    // 각 이웃의 버킷 처리
    for (int face = 0; face < 4; ++face) {
        int src_cell = neighbors[face];
        if (src_cell < 0) continue;  // 경계

        const OutflowBucket& bucket = OutflowBuckets[src_cell * 4 + face];

        // 버킷의 각 항목 전송
        for (int k = 0; k < Kb_out; ++k) {
            uint32_t bid = bucket.block_id[k];
            if (bid == EMPTY_BLOCK_ID) continue;

            // 슬롯 찾기 또는 할당
            int slot = find_or_allocate_slot(block_ids_out, cell, bid);
            if (slot < 0) continue;  // 공간 없음

            // 가중치 추가
            for (int lidx = 0; lidx < LOCAL_BINS; ++lidx) {
                float w = bucket.value[k][lidx];
                if (w > 0) {
                    atomicAdd(&values_out[flat_index(cell, slot, lidx)], w);
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
            return slot;  // 성공적으로 할당됨
        }
    }

    return -1;  // 사용 가능한 공간 없음
}
```

== K5: 보존 감사 커널

=== 파일

`src/cuda/kernels/k5_audit.cu`

=== 목적

셀당 가중치와 에너지 보존 확인.

=== 시그니처

```cpp
__global__ void k5_audit(
    // 입력 위상 공간(입력 및 출력 모두)
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
```

=== 알고리즘

```cpp
__global__ void k5_audit(...) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= Nx * Nz) return;

    // 입력 가중치 합산
    float W_in = 0.0f;
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_ids_in[cell * Kb + slot] != EMPTY_BLOCK_ID) {
            for (int i = 0; i < LOCAL_BINS; ++i) {
                W_in += values_in[flat_index(cell, slot, i)];
            }
        }
    }

    // 출력 가중치 합산
    float W_out = 0.0f;
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_ids_out[cell * Kb + slot] != EMPTY_BLOCK_ID) {
            for (int i = 0; i < LOCAL_BINS; ++i) {
                W_out += values_out[flat_index(cell, slot, i)];
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
    reports[cell].W_in = W_in;
    reports[cell].W_out = W_out;
    reports[cell].W_error = W_rel;
    reports[cell].pass = (W_rel < 1e-6f);
}
```

== K6: 버퍼 교환 커널

=== 파일

`src/cuda/kernels/k6_swap.cu`

=== 목적

다음 반복을 위한 입력/출력 버퍼 교환(CPU 측 포인터 교환).

=== 구현

```cpp
// 호스트 측 함수(커널 실행 없음)
void k6_swap_buffers(
    uint32_t*& block_ids_in,
    uint32_t*& block_ids_out,
    float*& values_in,
    float*& values_out
) {
    // 3방향 XOR 교환(임시 변수 불필요)
    swap(block_ids_in, block_ids_out);
    swap(values_in, values_out);
}
```

=== 커널이 없는 이유?

포인터 교환은 CPU 연산입니다 - GPU 메모리를 수정할 필요가 없습니다.
이는 반복당 ~2.2 GB 메모리 복사를 피합니다.

== 메모리 접근 패턴

=== 병렬 접근 전략

```
글로벌 메모리 레이아웃:
┌─────────────────────────────────────────┐
│ Cell 0: Slot 0, Bins 0-511              │ → Thread 0-255
│ Cell 0: Slot 1, Bins 0-511              │
│ ...                                     │
└─────────────────────────────────────────┘

Thread 0 읽기:  block_ids_in[0],   values_in[0:31]
Thread 1 읽기:  block_ids_in[1],   values_in[32:63]
...
Thread 255 읽기: block_ids_in[255], values_in[8160:8191]
```

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
    [활성 셀 처리], [K2, K3], [빈 셀 건너뛰기(60-90% 절약)],
    [거침/정밀 분할], [K2, K3], [고에너지용 3-5배 속도 향상],
    [원자적 연산], [K4], [스레드 안전 슬롯 할당],
    [빈 내 샘플링], [K3], [분산 보존],
    [포인터 교환], [K6], [2.2 GB 메모리 복사 회피],
    [병렬 접근], [모두], [최대 메모리 대역폭],
  ),
  caption: [성능 최적화],
)

== 실행 구성 예제

```cpp
// 그리드 차원
dim3 grid( (Nx * Nz + 255) / 256 );
dim3 block(256);

// K1: ActiveMask
k1_activemask<<<grid, block>>>(...);

// K3: 정밀 수송(활성 셀용 더 작은 그리드)
dim3 grid_fine( (n_active + 255) / 256 );
k3_finetransport<<<grid_fine, block>>>(...);

// 동기화
cudaDeviceSynchronize();
```

---
#set align(center)
*SM_2D CUDA 파이프라인 문서*

#text(size: 9pt)[버전 1.0.0]
