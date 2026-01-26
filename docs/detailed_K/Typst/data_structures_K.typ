#set text(font: "Linux Libertine", size: 11pt)
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

#show math: set text(weight: "regular")
#include "std"

= 코어 데이터 구조 모듈

== 개요

코어 데이터 구조 모듈은 위상 공간 입자 표현을 위한 기초 저장소 및 인코딩 메커니즘을 제공합니다. 이 모듈은 GPU 연산에 최적화된 계층적 블록-희소 저장 시스템을 구현합니다.

== 1. 에너지 그리드

=== 목적

0.1에서 250 MeV까지 양성자 운동 에너지를 표현하기 위한 로그 에너지 그리드.

=== 구조

```cpp
struct EnergyGrid {
    const int N_E;                     // 에너지 빈 수(256)
    const float E_min;                 // 최소 에너지(0.1 MeV)
    const float E_max;                 // 최대 에너지(250.0 MeV)
    std::vector<float> edges;          // N_E + 1 빈 경계(로그 간격)
    std::vector<float> rep;            // N_E 대표 에너지
};
```

=== 주요 메서드

#figure(
  table(
    columns: (2fr, 3fr, auto),
    inset: 8pt,
    align: left,
    table.header([*메서드*], [*설명*], [*복잡도*]),
    [`EnergyGrid(E_min, E_max, N_E)`], [생성자 - 로그 간격 그리드 생성], [O(N)],
    [`FindBin(float E)`], [에너지 빈 이진 탐색], [O(log N)],
    [`GetRepEnergy(int bin)`], [대표(기하 평균) 에너지 가져오기], [O(1)],
  ),
  caption: [EnergyGrid 메서드],
)

=== 대표 에너지 계산

$ "rep"[i] = sqrt(edges[i] times edges[i + 1]) $

=== 사용 예제

```cpp
EnergyGrid grid(0.1f, 250.0f, 256);
int bin = grid.FindBin(150.0f);        // 150 MeV에 대한 빈 찾기
float E_rep = grid.GetRepEnergy(bin);  // 대표 에너지 가져오기
```

== 2. 각도 그리드

=== 목적

X-Z 평면에서 입자 방향을 위한 균일 각도 그리드.

=== 구조

```cpp
struct AngularGrid {
    const int N_theta;                 // 세타 빈 수(512)
    const float theta_min;             // 최소 각도(-90°)
    const float theta_max;             // 최대 각도(+90°)
    std::vector<float> edges;          // N_theta + 1 빈 경계(균일)
    std::vector<float> rep;            // N_theta 대표 각도
};
```

=== 주요 메서드

#figure(
  table(
    columns: (2fr, 3fr, auto),
    inset: 8pt,
    align: left,
    table.header([*메서드*], [*설명*], [*복잡도*]),
    [`AngularGrid(theta_min, theta_max, N_theta)`], [생성자], [O(N)],
    [`FindBin(float theta)`], [각도 빈 찾기(클램프됨)], [O(1)],
    [`GetRepTheta(int bin)`], [대표(중간점) 각도 가져오기], [O(1)],
  ),
  caption: [AngularGrid 메서드],
)

=== 대표 각도 계산

$ "rep"[i] = 0.5 times (edges[i] + edges[i + 1]) $

== 3. 블록 인코딩

=== 목적

효율적인 GPU 저장을 위한 $(theta, E)$ 위상 공간 좌표의 압축적인 24비트 인코딩.

=== 인코딩 방식

```
┌─────────────────────────┬──────────────────────────┐
│     b_E (12 bits)       │    b_theta (12 bits)     │
│    Bits 12-23           │     Bits 0-11            │
│    Range: 0-4095        │     Range: 0-4095        │
└─────────────────────────┴──────────────────────────┘
                    24-bit Block ID
```

=== 함수

```cpp
// 인코딩: (b_theta, b_E) -> block_id
__host__ __device__ inline uint32_t encode_block(uint32_t b_theta, uint32_t b_E) {
    return (b_theta & 0xFFF) | ((b_E & 0xFFF) << 12);
}

// 디코딩: block_id -> (b_theta, b_E)
__host__ __device__ inline void decode_block(uint32_t block_id,
                                               uint32_t& b_theta,
                                               uint32_t& b_E) {
    b_theta = block_id & 0xFFF;
    b_E = (block_id >> 12) & 0xFFF;
}
```

=== 특수 값

```cpp
constexpr uint32_t EMPTY_BLOCK_ID = 0xFFFFFFFF;  // 사용되지 않은 슬롯 표시
```

=== 왜 12비트씩인가요?

* 에너지 빈: 256개 필요 → 12비트는 최대 4096개 허용
* 각도 빈: 512개 필요 → 12비트는 최대 4096개 허용
* GPU 효율성을 위해 단일 32비트 정수에 맞음

== 4. 로컬 빈

=== 목적

분산 보존 셀 내 입자 추적을 위한 4D 하위 셀 분할.

=== 구조

#figure(
  table(
    columns: (auto, auto, 3fr),
    inset: 8pt,
    align: left,
    table.header([*차원*], [*빈 수*], [*설명*]),
    [$theta_sub."local"$], [8], [로컬 각도 세분화],
    [$E_sub."local"$], [4], [로컬 에너지 세분화],
    [$x_sub$], [4], [셀 내 횡방향 위치],
    [$z_sub$], [4], [셀 내 깊이 위치],
  ),
  caption: [로컬 빈 차원],
)

#text(size: 10pt)[*전체: 8 × 4 × 4 × 4 = 512 블록당 로컬 빈*]

=== 인덱스 인코딩

```cpp
constexpr int N_theta_local = 8;
constexpr int N_E_local = 4;
constexpr int N_x_sub = 4;
constexpr int N_z_sub = 4;
constexpr int LOCAL_BINS = 512;

// 4차원 좌표를 16비트 인덱스로 인코딩
__host__ __device__ inline uint16_t encode_local_idx_4d(
    int theta_local, int E_local, int x_sub, int z_sub
) {
    int inner = E_local + N_E_local * (x_sub + N_x_sub * z_sub);
    return static_cast<uint16_t>(theta_local + N_theta_local * inner);
}
```

=== 위치 변환

```cpp
// 하위 빔 중심에서 X 오프셋(범위: -0.375*dx에서 +0.375*dx)
__host__ __device__ inline float get_x_offset_from_bin(int x_sub, float dx) {
    return dx * (-0.375f + 0.25f * x_sub);
}

// 하위 빔 중심에서 Z 오프셋
__host__ __device__ inline float get_z_offset_from_bin(int z_sub, float dz) {
    return dz * (-0.375f + 0.25f * z_sub);
}
```

=== 하위 빔 중심

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: (x, y) => (center, center).at(x),
    table.header([*$x_sub$*], [*오프셋(×dif x)*]),
    [0], [-0.375],
    [1], [-0.125],
    [2], [+0.125],
    [3], [+0.375],
  ),
  caption: [하위 빔 중심 오프셋],
)

== 5. 위상 공간 저장소(PsiC)

=== 목적

4D 위상 공간에서 입자 가중치를 위한 계층적 셀 기반 저장소.

=== 데이터 구조

```cpp
struct PsiC {
    const int Nx;                     // 그리드 X 차원
    const int Nz;                     // 그리드 Z 차원
    const int Kb;                     // 셀당 최대 블록(32)

    // 저장소 레이아웃: [cell][slot][local_bin]
    std::vector<std::array<uint32_t, 32>> block_id;  // 블록 ID
    std::vector<std::array<std::array<float, LOCAL_BINS>, 32>> value;  // 가중치

private:
    int N_cells;  // Nx × Nz
};
```

=== 메모리 레이아웃

```
PsiC[cell = 0]
├── slot[0]: block_id=0x000123 -> value[0..511]
├── slot[1]: block_id=0x000456 -> value[0..511]
├── ...
└── slot[31]: block_id=EMPTY -> value[unused]

PsiC[cell = 1]
├── ...
```

=== 주요 메서드

```cpp
// 기존 블록 찾기 또는 새 슬롯 할당
int find_or_allocate_slot(int cell, uint32_t bid);

// 특정 로컬 빈에서 가중치 가져오기/설정
float get_weight(int cell, int slot, uint16_t lidx) const;
void set_weight(int cell, int slot, uint16_t lidx, float w);

// 모든 데이터 지우기
void clear();
```

=== 슬롯 할당 알고리즘

```cpp
int PsiC::find_or_allocate_slot(int cell, uint32_t bid) {
    // 첫 번째 패스: 블록이 이미 존재하는지 확인
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == bid) {
            return slot;  // 기존 슬롯 찾음
        }
    }
    // 두 번째 패스: 빈 슬롯 찾기
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == EMPTY_BLOCK_ID) {
            block_id[cell][slot] = bid;  // 새 슬롯 할당
            return slot;
        }
    }
    return -1;  // 사용 가능한 공간 없음
}
```

=== 용량 분석

200×640 그리드 및 셀당 32개 슬롯의 경우:

* 전체 셀: 128,000
* 전체 슬롯: 4,096,000
* 슬롯당 메모리: 512 빈 × 4 바이트 = 2 KB
* 전체 저장소: ~8 GB(완전히 조밀한 경우)
* 실제 사용: ~1.1 GB(희소)

== 6. 버킷 방출

=== 목적

경계 버킷 방출을 통한 효율적인 셀 간 입자 전송.

=== 구조

```cpp
struct OutflowBucket {
    static constexpr int Kb_out = 64;  // 최대 방출 버킷

    std::array<uint32_t, Kb_out> block_id;        // 블록 ID
    std::array<uint16_t, Kb_out> local_count;     // 입자 수
    std::array<std::array<float, LOCAL_BINS>, Kb_out> value;  // 가중치
};
```

=== 면 인덱싱

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*면*], [*방향*], [*인덱스*]),
    [+z], [전방], [0],
    [-z], [후방], [1],
    [+x], [우측], [2],
    [-x], [좌측], [3],
  ),
  caption: [면 인덱스 규약],
)

=== 주요 메서드

```cpp
int find_or_allocate_slot(uint32_t bid);  // 방출 슬롯 찾기/할당
void clear();                             // 모든 방출 데이터 지우기
```

=== 전송 흐름

```
Cell (i,j)
├── bucket[+z]로 방출 -> Cell (i,j+1)가 수신
├── bucket[-z]로 방출 -> Cell (i,j-1)가 수신
├── bucket[+x]로 방출 -> Cell (i+1,j)가 수신
└── bucket[-x]로 방출 -> Cell (i-1,j)가 수신
```

== 메모리 요약

#figure(
  table(
    columns: (2fr, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*구성 요소*], [*인스턴스당 크기*], [*전체 크기*]),
    [EnergyGrid], [~4 KB], [4 KB],
    [AngularGrid], [~8 KB], [8 KB],
    [PsiC], [~1.1 GB], [1.1 GB],
    [OutflowBuckets], [셀당 ~32 KB], [~4 GB(모든 셀)],
  ),
  caption: [메모리 요약],
)

#text(size: 10pt)[*활성 작업 세트: ~2.2 GB*]

== 설계 근거

=== 1. 왜 블록-희소인가요?

* 위상 공간은 대부분 비어 있음(입자는 제한된 $(theta, E)$ 영역을 차지)
* 조밀 저장소 대비 70% 이상 메모리 절약

=== 2. 왜 24비트 인코딩인가요?

* 쌍 해싱 대신 단일 정수 조회
* GPU 친화적 정수 연산
* 치료 에너지/각도 범위에 충분한 용량

=== 3. 왜 512개 로컬 빈인가요?

* 분산 보존과 메모리의 균형
* 효율적인 비트 조작을 위한 2의 거듭제곱
* 임상 정확도에 실질적으로 충분

=== 4. 왜 셀당 고정 슬롯인가요?

* GPU 병렬 처리를 위한 예측 가능한 메모리 레이아웃
* O(K_b) 검색이 빠름(K_b = 32는 작음)
* GPU에서 동적 할당 회피

---
#set align(center)
*SM_2D 데이터 구조 문서*

#text(size: 9pt)[버전 1.0.0]
