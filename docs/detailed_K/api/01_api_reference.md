# SM_2D API 레퍼런스

## 개요

이 문서는 SM_2D의 모든 공개 API에 대한 포괄적인 레퍼런스를 제공합니다. 모듈별로 정리되어 있습니다.

---

## 코어 API

### EnergyGrid

```cpp
struct EnergyGrid {
    const int N_E;              // 에너지 빈 개수
    const float E_min;          // 최소 에너지 [MeV]
    const float E_max;          // 최대 에너지 [MeV]
    std::vector<float> edges;   // 빈 경계 (N_E + 1)
    std::vector<float> rep;     // 대표 에너지

    // 생성자
    EnergyGrid(float E_min, float E_max, int N_E);

    // 주어진 에너지에 대한 빈 찾기 (이진 탐색)
    int FindBin(float E) const;

    // 빈의 대표 에너지 가져오기
    float GetRepEnergy(int bin) const;
};
```

#### 사용 예시

```cpp
// 0.1에서 250 MeV까지 로그 스케일 에너지 그리드 생성
EnergyGrid e_grid(0.1f, 250.0f, 256);

// 150 MeV 양성자에 대한 빈 찾기
int bin = e_grid.FindBin(150.0f);

// 대표 에너지 가져오기
float E_rep = e_grid.GetRepEnergy(bin);
```

---

### AngularGrid

```cpp
struct AngularGrid {
    const int N_theta;          // theta 빈 개수
    const float theta_min;      // 최소 각도 [degrees]
    const float theta_max;      // 최대 각도 [degrees]
    std::vector<float> edges;   // 빈 경계 (N_theta + 1)
    std::vector<float> rep;     // 대표 각도

    // 생성자
    AngularGrid(float theta_min, float theta_max, int N_theta);

    // 주어진 각도에 대한 빈 찾기 (O(1) 산술 연산)
    int FindBin(float theta) const;

    // 빈의 대표 각도 가져오기
    float GetRepTheta(int bin) const;
};
```

---

### PsiC (Phase-Space Container)

```cpp
struct PsiC {
    const int Nx;                     // 그리드 X 차원
    const int Nz;                     // 그리드 Z 차원
    const int Kb;                     // 셀당 최대 블록 수 (32)

    // 저장 배열
    std::vector<std::array<uint32_t, 32>> block_id;
    std::vector<std::array<std::array<float, LOCAL_BINS>, 32>> value;

    // 생성자
    PsiC(int Nx, int Nz, int Kb);

    // 기존 블록 찾기 또는 새 슬롯 할당
    int find_or_allocate_slot(int cell, uint32_t bid);

    // 특정 위치의 가중치 접근
    float get_weight(int cell, int slot, uint16_t lidx) const;
    void set_weight(int cell, int slot, uint16_t lidx, float w);

    // 모든 데이터 초기화
    void clear();

    // 전체 셀 수
    int N_cells;
};
```

---

## 블록 인코딩 API

```cpp
// (b_theta, b_E) → 24비트 블록 ID 인코딩
__host__ __device__ inline uint32_t encode_block(
    uint32_t b_theta,  // 각도 빈 (0-4095)
    uint32_t b_E       // 에너지 빈 (0-4095)
);

// 블록 ID → (b_theta, b_E) 디코딩
__host__ __device__ inline void decode_block(
    uint32_t block_id,
    uint32_t& b_theta,  // 출력: 각도 빈
    uint32_t& b_E       // 출력: 에너지 빈
);

// 빈 슬롯을 위한 특별 값
constexpr uint32_t EMPTY_BLOCK_ID = 0xFFFFFFFF;
```

#### 사용 예시

```cpp
// 에너지 빈 100, 각도 빈 50 인코딩
uint32_t bid = encode_block(50, 100);

// 디코딩
uint32_t theta_bin, energy_bin;
decode_block(bid, theta_bin, energy_bin);
// theta_bin = 50, energy_bin = 100
```

---

## 로컬 빈 API

```cpp
// 상수
constexpr int N_theta_local = 8;   // 로컬 각도 세분화
constexpr int N_E_local = 4;       // 로컬 에너지 세분화
constexpr int N_x_sub = 4;         // X 위치 세분화
constexpr int N_z_sub = 4;         // Z 위치 세분화
constexpr int LOCAL_BINS = 512;    // 8 × 4 × 4 × 4

// 4D 좌표를 16비트 로컬 인덱스로 인코딩
__host__ __device__ inline uint16_t encode_local_idx_4d(
    int theta_local,  // 0-7
    int E_local,      // 0-3
    int x_sub,        // 0-3
    int z_sub         // 0-3
);

// 로컬 인덱스를 4D 좌표로 디코딩
__host__ __device__ inline void decode_local_idx(
    uint16_t lidx,
    int& theta_local,
    int& E_local,
    int& x_sub,
    int& z_sub
);

// 위치 변환
__host__ __device__ inline float get_x_offset_from_bin(int x_sub, float dx);
__host__ __device__ inline float get_z_offset_from_bin(int z_sub, float dz);
```

---

## 물리 API

### Highland 공식

```cpp
// MCS 각도 시그마 계산 [rad]
__host__ __device__ float highland_sigma(
    float E_MeV,   // 양성자 에너지 [MeV]
    float ds,      // 스텝 길이 [mm]
    float X0       // 방사선 길이 [mm]
);

// 산란 각도 샘플링 (Box-Muller)
__device__ float sample_mcs_angle(
    float sigma_theta,     // RMS 산란 각도
    unsigned& seed         // RNG 상태
);

// 산란 후 방향 코사인 업데이트
__device__ void update_direction_after_mcs(
    float& mu,       // 방향 코사인 X (입력/출력)
    float& eta,      // 방향 코사인 Z (입력/출력)
    float delta_theta  // 산란 각도 [rad]
);
```

---

### 에너지 Straggling

```cpp
// Vavilov 카파 매개변수 계산
__host__ __device__ float vavilov_kappa(
    float E_MeV,   // 양성자 에너지 [MeV]
    float ds       // 스텝 길이 [mm]
);

// Bohr straggling 시그마
__host__ __device__ float bohr_straggling_sigma(
    float E_MeV,   // 양성자 에너지 [MeV]
    float ds       // 스텝 길이 [mm]
);

// Straggling을 포함한 에너지 손실 샘플링
__device__ float sample_energy_loss_with_straggling(
    float E_MeV,
    float ds,
    unsigned& seed
);
```

---

### 스텝 제어

```cpp
// 최대 스텝 크기 계산 (R 기반 방법)
__host__ __device__ float compute_max_step_physics(
    float E,          // 현재 에너지 [MeV]
    const RLUT& lut   // 사거리-에너지 룩업 테이블
);

// 스텝 후 에너지 계산 (R 기반)
__device__ float compute_energy_after_step(
    float E_in,       // 입력 에너지 [MeV]
    float ds,         // 스텝 길이 [mm]
    const RLUT& lut   // 사거리-에너지 룩업 테이블
);

// 스텝에서 에너지沉积 계산
__device__ float compute_energy_deposition(
    float E_in,
    float ds,
    const RLUT& lut
);
```

---

### 핵 상호작용

```cpp
// 전체 핵 단면적 [mm⁻¹]
__host__ __device__ float Sigma_total(
    float E_MeV   // 양성자 에너지 [MeV]
);

// 핵 감마 적용
__device__ void apply_nuclear_attenuation(
    float& weight,       // 입자 가중치 (수정됨)
    double& energy_rem,  // 에너지 누적기 (감사용)
    float E_MeV,         // 현재 에너지
    float ds             // 스텝 길이
);
```

---

## LUT API

### RLUT (Range-Energy Lookup Table)

```cpp
struct RLUT {
    EnergyGrid grid;
    std::vector<float> R;        // CSDA 사거리 [mm]
    std::vector<float> S;        // 제동 능력 [MeV cm²/g]
    std::vector<float> log_E;    // 사전 계산된 log(E)
    std::vector<float> log_R;    // 사전 계산된 log(R)
    std::vector<float> log_S;    // 사전 계산된 log(S)

    // 에너지에서 사거리 (로그-로그 보간)
    float lookup_R(float E_MeV) const;

    // 에너지에서 제동 능력
    float lookup_S(float E_MeV) const;

    // 사거리에서 에너지 (역 룩업)
    float lookup_E_inverse(float R_mm) const;
};
```

---

### NIST 데이터 로더

```cpp
struct NistDataRow {
    float energy_MeV;           // 운동 에너지 [MeV]
    float stopping_power;       // dE/dx [MeV cm²/g]
    float csda_range_g_cm2;     // CSDA 사거리 [g/cm²]
};

// 파일에서 NIST PSTAR 데이터 로드
std::vector<NistDataRow> load_nist_pstar(
    const std::string& filepath
);

// NIST 데이터에서 RLUT 생성
RLUT create_r_lut_from_nist(
    const std::vector<NistDataRow>& nist_data,
    float E_min,
    float E_max,
    int N_E
);
```

---

## 소스 API

### PencilSource

```cpp
struct PencilSource {
    float x0 = 0.0f;       // X 위치 [mm]
    float z0 = 0.0f;       // Z 위치 [mm]
    float theta0 = 0.0f;   // 초기 각도 [rad]
    float E0 = 150.0f;     // 초기 에너지 [MeV]
    float W_total = 1.0f;  // 전체 가중치
};

// 위상 공간에 펜슬 소스 주입
void inject_pencil_source(
    const PencilSource& source,
    PsiC& psi,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    int Nx, int Nz, float dx, float dz
);
```

---

### GaussianSource

```cpp
struct GaussianSource {
    float x0 = 0.0f;           // 평균 X 위치 [mm]
    float z0 = 0.0f;           // 평균 Z 위치 [mm]
    float theta0 = 0.0f;       // 평균 각도 [rad]
    float sigma_x = 5.0f;      // X 표준 편차 [mm]
    float sigma_theta = 0.01f; // 각도 표준 편차 [rad]
    float E0 = 150.0f;         // 평균 에너지 [MeV]
    float sigma_E = 1.0f;      // 에너지 표준 편차 [MeV]
    float W_total = 1.0f;      // 전체 가중치
    int n_samples = 1000;      // Monte Carlo 샘플 수
};

// 위상 공간에 가우시안 소스 주입
void inject_gaussian_source(
    const GaussianSource& source,
    PsiC& psi,
    const EnergyGrid& e_grid,
    const AngularGrid& a_grid,
    int Nx, int Nz, float dx, float dz,
    unsigned seed = 42
);
```

---

## 경계 API

### 경계 타입

```cpp
enum class BoundaryType {
    ABSORB,     // 경계에서 입자 흡수
    REFLECT,    // 입자 반사
    PERIODIC    // 반대 경계로 입자 래핑
};

struct BoundaryConfig {
    BoundaryType z_min = BoundaryType::ABSORB;
    BoundaryType z_max = BoundaryType::ABSORB;
    BoundaryType x_min = BoundaryType::ABSORB;
    BoundaryType x_max = BoundaryType::ABSORB;
};
```

---

### 손실 추적

```cpp
struct BoundaryLoss {
    float weight[4];   // 면별 손실 가중치
    double energy[4];  // 면별 손실 에너지

    // 면 인덱스: 0=+z, 1=-z, 2=+x, 3=-x
};

// 경계 손실 기록
void record_boundary_loss(
    BoundaryLoss& loss,
    int face,      // 면 인덱스 (0-3)
    float w,       // 가중치
    double E       // 에너지
);

// 전체 경계 손실 가져오기
float total_boundary_weight_loss(const BoundaryLoss& loss);
double total_boundary_energy_loss(const BoundaryLoss& loss);
```

---

## 감사 API

### 셀 수준 감사

```cpp
struct CellWeightAudit {
    float W_in;              // 입력 가중치
    float W_out;             // 출력 가중치
    float W_cutoff;          // 컷오프로 손실된 가중치
    float W_nuclear;         // 핵으로 손실된 가중치
    float W_error;           // 상대 오차

    bool check() const {
        float W_expected = W_out + W_cutoff + W_nuclear;
        float W_rel = fabs(W_in - W_expected) / fmax(W_in, 1e-20f);
        return W_rel < 1e-6f;
    }
};

struct CellEnergyAudit {
    double E_in;             // 입력 에너지
    double E_out;            // 출력 에너지
    double E_dep;            //沉积된 에너지
    double E_nuclear;        // 핵으로 손실된 에너지
    double E_error;          // 상대 오차

    bool check() const {
        double E_expected = E_out + E_dep + E_nuclear;
        double E_rel = fabs(E_in - E_expected) / fmax(E_in, 1e-20);
        return E_rel < 1e-5;
    }
};
```

---

### 전체 예산

```cpp
struct GlobalAudit {
    // 가중치
    double W_in_total;
    double W_out_total;
    double W_cutoff_total;
    double W_nuclear_total;
    double W_boundary_total;
    double W_error_relative;

    // 에너지
    double E_in_total;
    double E_out_total;
    double E_dep_total;
    double E_nuclear_total;
    double E_boundary_total;
    double E_error_relative;

    bool weight_pass() const;
    bool energy_pass() const;
    bool pass() const;
};

// 셀 감사를 전체로 집계
GlobalAudit aggregate_cell_audits(
    const std::vector<CellWeightAudit>& weight_audits,
    const std::vector<CellEnergyAudit>& energy_audits,
    const BoundaryLoss& boundary_loss
);
```

---

## 리포팅 API

```cpp
// 셀 수준 가중치 리포트 출력
void print_cell_weight_report(
    const CellWeightAudit& audit,
    int cell_id
);

// 전체 감사 리포트 출력
void print_global_report(
    const GlobalAudit& audit,
    const std::string& output_path = ""
);

// 실패한 셀 목록 출력
void print_failed_cells(
    const std::vector<CellWeightAudit>& audits,
    float threshold = 1e-6f
);

// 요약 통계 출력
void print_summary(
    const GlobalAudit& audit,
    int total_cells,
    int n_steps
);
```

---

## 검증 API

### Bragg Peak 검증

```cpp
struct BraggPeakResult {
    float position_mm;       // 피크 깊이 [mm]
    float peak_dose;         // 최대 선량
    float fwhm_mm;           // 반치폭
    float R80;               // 80% 선량 깊이
    float R20;               // 20% 선량 깊이
    float distal_falloff;    // R80 - R20
    float position_error;    // NIST 참조 대비
    bool pass;               // ±2% 기준
};

// PDD 데이터에서 Bragg peak 분석
BraggPeakResult analyze_bragg_peak(
    const std::vector<float>& z_mm,
    const std::vector<float>& dose,
    float reference_position_mm  // NIST 참조
);
```

---

### 측면 확산 검증

```cpp
struct LateralSpreadResult {
    float z_mm;               // 측정 깊이
    float sigma_sim;          // 시뮬레이션된 측면 시그마
    float sigma_fermi_eyges;  // 이론적 예측
    float relative_error;     // (시뮬레이션 - 이론) / 이론
    bool pass;                // ±15% 기준
};

// 특정 깊이에서 측면 확산 검증
LateralSpreadResult validate_lateral_spread(
    const PsiC& psi,
    float z_mm,
    const EnergyGrid& e_grid,
    float E0_MeV
);
```

---

### 결정론성 테스트

```cpp
struct DeterminismResult {
    uint32_t checksum1;       // 첫 번째 실행
    uint32_t checksum2;       // 두 번째 실행
    bool match;               // 체크섬 일치
    bool pass;                // 결정론적
};

// 시뮬레이션 두 번 실행 후 비교
DeterminismResult test_determinism(
    const std::string& config_file,
    unsigned seed1 = 42,
    unsigned seed2 = 42
);
```

---

## 유틸리티 API

### Logger

```cpp
enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERROR
};

// 싱글톤 로거 인스턴스 가져오기
Logger& get_logger();

// 로그 레벨 설정
void set_log_level(LogLevel level);

// 로깅 매크로
#define LOG_TRACE(...) get_logger().log(LogLevel::TRACE, __VA_ARGS__)
#define LOG_DEBUG(...) get_logger().log(LogLevel::DEBUG, __VA_ARGS__)
#define LOG_INFO(...)  get_logger().log(LogLevel::INFO, __VA_ARGS__)
#define LOG_WARN(...)  get_logger().log(LogLevel::WARN, __VA_ARGS__)
#define LOG_ERROR(...) get_logger().log(LogLevel::ERROR, __VA_ARGS__)
```

---

### 메모리 추적기

```cpp
struct MemoryInfo {
    size_t free_bytes;      // 여유 메모리
    size_t total_bytes;     // 전체 메모리
    size_t used_bytes;      // 사용된 메모리
    float utilization;      // 사용량 / 전체
};

// 현재 GPU 메모리 조회
MemoryInfo query_gpu_memory();

// 경고와 함께 확인
bool check_memory_warning(float threshold = 0.9f);

// 메모리 요약 출력
void print_memory_summary();
```

---

## 설정 API

### 설정 로더

```cpp
struct SimulationConfig {
    // 입자
    std::string particle_type;
    float mass_amu;
    float charge_e;

    // 빔
    std::string beam_profile;
    float beam_weight;

    // 에너지
    float E_mean_MeV;
    float E_sigma_MeV;
    float E_min_MeV;
    float E_max_MeV;

    // 공간
    float x0_mm, z0_mm;
    float sigma_x_mm, sigma_z_mm;

    // 각도
    float theta0_rad;
    float sigma_theta_rad;

    // 그리드
    int Nx, Nz;
    float dx_mm, dz_mm;
    int max_steps;

    // 출력
    std::string output_dir;
    bool normalize_dose;
    bool save_2d;
    bool save_pdd;
};

// INI 파일에서 설정 로드
SimulationConfig load_config(const std::string& ini_file);
```

---

## 메인 시뮬레이션 API

```cpp
// 메인 시뮬레이션 진입점
int run_simulation(
    const std::string& config_file = "sim.ini",
    const std::string& output_dir = "results"
);

// 시뮬레이션 결과
struct SimulationResult {
    int exit_code;                      // 0 = 성공
    int n_steps_completed;              // 실행된 스텝 수
    double wall_time_seconds;           // 실행 시간

    GlobalAudit audit;                  // 보존 감사
    BraggPeakResult bragg_peak;         // Bragg peak 분석

    std::string dose_file;              // 출력 파일들
    std::string pdd_file;
    std::string audit_file;
};

// 결과 반환과 함께 실행
SimulationResult run_simulation_detailed(
    const SimulationConfig& config
);
```
