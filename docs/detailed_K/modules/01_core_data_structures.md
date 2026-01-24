# 핵심 데이터 구조 모듈

## 개요

핵심 데이터 구조 모듈은 위상 공간 입자 표현을 위한 기본 저장소 및 인코딩 메커니즘을 제공합니다. 이 모듈은 GPU 계산에 최적화된 계층형 블록-희소 저장 시스템을 구현합니다.

---

## 1. 에너지 그리드 (`grids.hpp/cpp`)

### 목적
0.1에서 250 MeV 범위의 양성자 운동 에너지를 표현하기 위한 로그 스케일 에너지 그리드입니다.

### 구조

```cpp
struct EnergyGrid {
    const int N_E;                     // 에너지 빈 수 (256)
    const float E_min;                 // 최소 에너지 (0.1 MeV)
    const float E_max;                 // 최대 에너지 (250.0 MeV)
    std::vector<float> edges;          // N_E + 1 빈 경계 (로그 간격)
    std::vector<float> rep;            // N_E 대표 에너지
};
```

### 주요 메서드

| 메서드 | 설명 | 시간 복잡도 |
|--------|-------------|------------|
| `EnergyGrid(E_min, E_max, N_E)` | 생성자 - 로그 간격 그리드 생성 | O(N) |
| `FindBin(float E)` | 에너지 빈 이진 탐색 | O(log N) |
| `GetRepEnergy(int bin)` | 대표 (기하 평균) 에너지 가져오기 | O(1) |

### 대표 에너지 계산
```cpp
rep[i] = sqrt(edges[i] * edges[i+1])  // 기하 평균
```

### 사용 예제
```cpp
EnergyGrid grid(0.1f, 250.0f, 256);
int bin = grid.FindBin(150.0f);        // 150 MeV에 대한 빈 찾기
float E_rep = grid.GetRepEnergy(bin);  // 대표 에너지 가져오기
```

---

## 2. 각도 그리드 (`grids.hpp/cpp`)

### 목적
X-Z 평면에서 입자 방향을 위한 균일 각도 그리드입니다.

### 구조

```cpp
struct AngularGrid {
    const int N_theta;                 // 세타 빈 수 (512)
    const float theta_min;             // 최소 각도 (-90°)
    const float theta_max;             // 최대 각도 (+90°)
    std::vector<float> edges;          // N_theta + 1 빈 경계 (균일)
    std::vector<float> rep;            // N_theta 대표 각도
};
```

### 주요 메서드

| 메서드 | 설명 | 시간 복잡도 |
|--------|-------------|------------|
| `AngularGrid(theta_min, theta_max, N_theta)` | 생성자 | O(N) |
| `FindBin(float theta)` | 각도 빈 찾기 (클램프) | O(1) |
| `GetRepTheta(int bin)` | 대표 (중간점) 각도 가져오기 | O(1) |

### 대표 각도 계산
```cpp
rep[i] = 0.5 * (edges[i] + edges[i+1])  // 산술 평균
```

---

## 3. 블록 인코딩 (`block_encoding.hpp`)

### 목적
효율적인 GPU 저장을 위한 (θ, E) 위상 공간 좌표의 컴팩트한 24비트 인코딩입니다.

### 인코딩 방식

```
┌─────────────────────────┬──────────────────────────┐
│     b_E (12 bits)       │    b_theta (12 bits)     │
│    Bits 12-23           │     Bits 0-11            │
│    Range: 0-4095        │     Range: 0-4095        │
└─────────────────────────┴──────────────────────────┘
                    24-bit Block ID
```

### 함수

```cpp
// Encode: (b_theta, b_E) → block_id
__host__ __device__ inline uint32_t encode_block(uint32_t b_theta, uint32_t b_E) {
    return (b_theta & 0xFFF) | ((b_E & 0xFFF) << 12);
}

// Decode: block_id → (b_theta, b_E)
__host__ __device__ inline void decode_block(uint32_t block_id,
                                               uint32_t& b_theta,
                                               uint32_t& b_E) {
    b_theta = block_id & 0xFFF;
    b_E = (block_id >> 12) & 0xFFF;
}
```

### 특수 값
```cpp
constexpr uint32_t EMPTY_BLOCK_ID = 0xFFFFFFFF;  // 사용되지 않는 슬롯 표시
```

### 왜 12비트씩인가요?
- 에너지 빈: 256개 필요 → 12비트는 최대 4096개 허용
- 각도 빈: 512개 필요 → 12비트는 최대 4096개 허용
- GPU 효율성을 위해 단일 32비트 정수에 저장

---

## 4. 로컬 빈 (`local_bins.hpp`)

### 목적
분산 보존 셀 내 입자 추적을 위한 4차원 하위 셀 분할입니다.

### 구조

| 차원 | 빈 수 | 설명 |
|-----------|------|-------------|
| θ_local | 8 | 로컬 각도 세분화 |
| E_local | 4 | 로컬 에너지 세분화 |
| x_sub | 4 | 셀 내 횡방향 위치 |
| z_sub | 4 | 셀 내 깊이 위치 |

**전체**: 블록당 `8 × 4 × 4 × 4 = 512` 로컬 빈

### 인덱스 인코딩

```cpp
constexpr int N_theta_local = 8;
constexpr int N_E_local = 4;
constexpr int N_x_sub = 4;
constexpr int N_z_sub = 4;
constexpr int LOCAL_BINS = 512;

// Encode 4D coordinates to 16-bit index
__host__ __device__ inline uint16_t encode_local_idx_4d(
    int theta_local, int E_local, int x_sub, int z_sub
) {
    int inner = E_local + N_E_local * (x_sub + N_x_sub * z_sub);
    return static_cast<uint16_t>(theta_local + N_theta_local * inner);
}
```

### 위치 변환

```cpp
// X offset from sub-bin center (range: -0.375*dx to +0.375*dx)
__host__ __device__ inline float get_x_offset_from_bin(int x_sub, float dx) {
    return dx * (-0.375f + 0.25f * x_sub);
}

// Z offset from sub-bin center
__host__ __device__ inline float get_z_offset_from_bin(int z_sub, float dz) {
    return dz * (-0.375f + 0.25f * z_sub);
}
```

### 하위 빈 중심

| x_sub | 오프셋 (×dx) |
|-------|--------------|
| 0 | -0.375 |
| 1 | -0.125 |
| 2 | +0.125 |
| 3 | +0.375 |

---

## 5. 위상 공간 저장소 (`psi_storage.hpp/cpp`)

### 목적
4차원 위상 공간에서 입자 가중치를 위한 계층형 셀 기반 저장소입니다.

### 데이터 구조

```cpp
struct PsiC {
    const int Nx;                     // 그리드 X 차원
    const int Nz;                     // 그리드 Z 차원
    const int Kb;                     // 셀당 최대 블록 수 (32)

    // Storage layout: [cell][slot][local_bin]
    std::vector<std::array<uint32_t, 32>> block_id;  // 블록 ID
    std::vector<std::array<std::array<float, LOCAL_BINS>, 32>> value;  // 가중치

private:
    int N_cells;  // Nx × Nz
};
```

### 메모리 레이아웃

```
PsiC[cell = 0]
├── slot[0]: block_id=0x000123 → value[0..511]
├── slot[1]: block_id=0x000456 → value[0..511]
├── ...
└── slot[31]: block_id=EMPTY → value[unused]

PsiC[cell = 1]
├── ...
```

### 주요 메서드

```cpp
// Find existing block or allocate new slot
int find_or_allocate_slot(int cell, uint32_t bid);

// Get/set weight in specific local bin
float get_weight(int cell, int slot, uint16_t lidx) const;
void set_weight(int cell, int slot, uint16_t lidx, float w);

// Clear all data
void clear();
```

### 슬롯 할당 알고리즘

```cpp
int PsiC::find_or_allocate_slot(int cell, uint32_t bid) {
    // First pass: check if block already exists
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == bid) {
            return slot;  // Found existing slot
        }
    }
    // Second pass: find empty slot
    for (int slot = 0; slot < Kb; ++slot) {
        if (block_id[cell][slot] == EMPTY_BLOCK_ID) {
            block_id[cell][slot] = bid;  // Allocate new slot
            return slot;
        }
    }
    return -1;  // No space available
}
```

### 용량 분석

셀당 32개 슬롯을 가진 200×640 그리드의 경우:
- 전체 셀 수: 128,000
- 전체 슬롯 수: 4,096,000
- 슬롯당 메모리: 512 빈 × 4바이트 = 2KB
- 전체 저장소: ~8GB (완전 조밀)
- 실제 사용량: ~1.1GB (희소)

---

## 6. 버킷 방출 (`buckets.hpp/cpp`)

### 목적
경계 버킷 방출을 통한 효율적인 셀 간 입자 전송입니다.

### 구조

```cpp
struct OutflowBucket {
    static constexpr int Kb_out = 64;  // 최대 방출 버킷 수

    std::array<uint32_t, Kb_out> block_id;        // 블록 ID
    std::array<uint16_t, Kb_out> local_count;     // 입자 수
    std::array<std::array<float, LOCAL_BINS>, Kb_out> value;  // 가중치
};
```

### 면 인덱싱

| 면 | 방향 | 인덱스 |
|------|-----------|-------|
| +z | 전방 | 0 |
| -z | 후방 | 1 |
| +x | 우측 | 2 |
| -x | 좌측 | 3 |

### 주요 메서드

```cpp
int find_or_allocate_slot(uint32_t bid);  // Find/allocate emission slot
void clear();                             // Clear all emission data
```

### 전송 흐름

```
Cell (i,j)
├── emits to bucket[+z] → received by Cell (i,j+1)
├── emits to bucket[-z] → received by Cell (i,j-1)
├── emits to bucket[+x] → received by Cell (i+1,j)
└── emits to bucket[-x] → received by Cell (i-1,j)
```

---

## 메모리 요약

| 구성 요소 | 인스턴스당 크기 | 전체 크기 |
|-----------|-------------------|------------|
| EnergyGrid | ~4KB | 4KB |
| AngularGrid | ~8KB | 8KB |
| PsiC | ~1.1GB | 1.1GB |
| OutflowBuckets | 셀당 ~32KB | ~4GB (모든 셀) |
| **활성 워킹 세트** | | **~2.2GB** |

---

## 설계 근거

1. **왜 블록-희소인가요?**
   - 위상 공간은 대부분 비어 있습니다 (입자들이 제한된 (θ,E) 영역을 차지)
   - 조밀 저장소 대비 70% 이상 메모리 절약

2. **왜 24비트 인코딩인가요?**
   - 쌍 해싱 대신 단일 정수 검색
   - GPU 친화적 정수 연산
   - 치료용 에너지/각도 범위에 충분한 용량

3. **왜 512개 로컬 빈인가요?**
   - 분산 보존과 메모리의 균형
   - 효율적인 비트 조작을 위한 2의 거듭제곱
   - 임상 정확도에 경험적으로 충분

4. **왜 셀당 고정 슬롯인가요?**
   - GPU 코알레싱을 위한 예측 가능한 메모리 레이아웃
   - O(Kb) 검색이 빠름 (Kb=32는 작음)
   - GPU에서 동적 할당 회피
