# 물리 모델 문서 (Physics Models Documentation)

## 개요 (Overview)

SM_2D는 물 내에서의 양성자 수송을 위한 포괄적인 물리 모델을 구현하며, ICRU, PDG, NIST 표준을 따릅니다. 모든 모델은 GPU 가속을 위한 CUDA 디바이스 함수로 구현되었습니다.

---

## 1. 다중 쿨롱 산란 (Multiple Coulomb Scattering - Highland Formula)

### 참고문헌: PDG 2024 Review of Particle Physics

### 하일랜드 공식 (The Highland Formula, 2D Projection)

```
σ_θ = (13.6 MeV / βcp) × z × sqrt(x/X_0) × [1 + 0.038 × ln(x/X_0)] / √2
```

여기서:
- `βcp` = 운동량 × 속도 [Momentum × velocity, MeV/c]
- `z` = 발사체 전하 [Projectile charge, 양성자의 경우 1]
- `x` = 스텝 길이 [Step length, mm]
- `X_0` = 복사 길이 [Radiation length, 물의 경우 360.8 mm]
- `1/√2` = 3D→2D 투영 보정 [3D→2D projection correction]

### 구현 (`highland.hpp`)

```cpp
struct HighlandParams {
    float m_p_MeV = 938.272f;      // 양성자 정지 질량 [Proton rest mass]
    float X0_water = 360.8f;       // 복사 길이 [Radiation length, mm]
    float MCS_2D_CORRECTION = 0.70710678f;  // 1/√2
};

// 산란 각도 시그마 계산
__host__ __device__ float highland_sigma(float E_MeV, float ds, float X0) {
    // 상대론적 운동학 [Relativistic kinematics]
    float gamma = 1.0f + E_MeV / m_p_MeV;
    float beta = sqrt(1.0f - 1.0f / (gamma * gamma));
    float beta_cp = beta * (E_MeV + m_p_MeV);  // MeV/c

    // 하일랜드 공식 [Highland formula]
    float theta_rms = (13.6f / beta_cp) * sqrt(ds / X0);
    theta_rms *= (1.0f + 0.038f * log(ds / X0));

    // 2D 투영 보정 [2D projection correction]
    return theta_rms * MCS_2D_CORRECTION;
}

// Box-Muller 방법을 사용한 산란 각도 샘플링
__device__ float sample_mcs_angle(float sigma_theta, unsigned& seed) {
    float u1 = curand_uniform(&seed);
    float u2 = curand_uniform(&seed);
    float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
    return sigma_theta * z0;
}
```

### 분산 누적 (Variance Accumulation, v0.8)

정확한 다중 스텝 산란을 위해 분산이 누적됩니다:

```cpp
// 정확한 방법: 분산 누적 [CORRECT: Accumulate variance]
sigma_2_total += sigma_theta * sigma_theta;

// 총 분산에서 샘플링
float theta_scatter = sqrt(sigma_2_total) * sample_normal();

// 잘못된 방법: 시그마를 직접 누적하지 않음 [WRONG: Do NOT accumulate sigma directly]
// sigma_total += sigma_theta;  // This overestimates scattering!
```

### 산란 후 방향 업데이트 (Direction Update After Scattering)

```cpp
__device__ void update_direction_after_mcs(
    float& mu, float& eta,  // 방향 코사인 [Direction cosines]
    float delta_theta
) {
    // 현재 각도 [Current angle]
    float theta = atan2(eta, mu);

    // 산란 추가 [Add scattering]
    theta += delta_theta;

    // 방향 코사인 업데이트 [Update direction cosines]
    mu = cos(theta);
    eta = sin(theta);

    // 정규화 (mu² + eta² = 1 보장) [Normalize]
    float norm = sqrt(mu*mu + eta*eta);
    mu /= norm;
    eta /= norm;
}
```

---

## 2. 에너지 분산 (Energy Straggling - Vavilov Theory)

### 세 가지 영역 (Three Regimes)

| 영역 [Regime] | κ 파라미터 [κ Parameter] | 분포 [Distribution] |
|--------------|------------------------|-------------------|
| 보어 [Bohr] | κ > 10 | 가우시안 [Gaussian] |
| 바빌로프 [Vavilov] | 0.01 < κ < 10 | 바빌로프 보간 [Vavilov interpolation] |
| 란다우 [Landau] | κ < 0.01 | 란다우 (비대칭) [Landau (asymmetric)] |

### 바빌로프 파라미터 (Vavilov Parameter)

```
κ = ξ / T_max

ξ = (K/2) × (Z/A) × (z²/β²) × ρ × x
T_max = (2 m_e c² β² γ²) / (1 + 2γ m_e/m_p + (m_e/m_p)²)
```

여기서:
- `K = 0.307 MeV cm²/g`
- `Z/A = 0.555` (물의 경우)
- `m_e c² = 0.511 MeV`

### 구현 (`energy_straggling.hpp`)

```cpp
// 보어 분산 시그마 계산
__host__ __device__ float bohr_straggling_sigma(float E_MeV, float ds) {
    float gamma = 1.0f + E_MeV / m_p_MeV;
    float beta = sqrt(1.0f - 1.0f / (gamma * gamma));

    // 보어 공식 (물에 대해 단순화) [Bohr formula, simplified for water]
    float kappa_0 = 0.156f;  // 물에 대해 미리 계산됨 [Pre-computed for water]
    float sigma = kappa_0 * sqrt(ds) / beta;

    return sigma;
}

// 바빌로프 카파 파라미터 계산
__host__ __device__ float vavilov_kappa(float E_MeV, float ds) {
    float gamma = 1.0f + E_MeV / m_p_MeV;
    float beta = sqrt(1.0f - 1.0f / (gamma * gamma));

    // 최대 에너지 전달 [Maximum energy transfer]
    float m_ratio = m_ec2_MeV / m_p_MeV;
    float gamma_sq = gamma * gamma;
    float T_max = (2 * m_ec2_MeV * beta * beta * gamma_sq) /
                  (1.0f + 2.0f * gamma * m_ratio + m_ratio * m_ratio);

    // 바빌로프 파라미터 [Vavilov parameter]
    float K = 0.307f;  // MeV cm²/g
    float xi = (K / 2.0f) * 0.555f * (1.0f / (beta * beta)) * ds;  // ρ=1 for water

    return xi / T_max;
}

// 영역 의존적 분산 [Regime-dependent straggling]
__device__ float energy_straggling_sigma(float E_MeV, float ds) {
    float kappa = vavilov_kappa(E_MeV, ds);

    if (kappa > 10.0f) {
        // 보어 영역: 가우시안 [Bohr regime: Gaussian]
        return bohr_straggling_sigma(E_MeV, ds);
    } else if (kappa < 0.01f) {
        // 란다우 영역: 란다우 폭 사용 [Landau regime: Use Landau width]
        return landau_width(E_MeV, ds);
    } else {
        // 바빌로프 영역: 보간 [Vavilov regime: Interpolate]
        float sigma_bohr = bohr_straggling_sigma(E_MeV, ds);
        float sigma_landau = landau_width(E_MeV, ds);
        return interpolate_vavilov(kappa, sigma_bohr, sigma_landau);
    }
}

// 분산을 포함한 에너지 손실 샘플링
__device__ float sample_energy_loss_with_straggling(
    float E_MeV,
    float ds,
    unsigned& seed
) {
    // 평균 에너지 손실 [Mean energy loss, 베테-블로흐]
    float dE_mean = compute_mean_energy_loss(E_MeV, ds);

    // 분산 시그마 [Straggling sigma]
    float sigma = energy_straggling_sigma(E_MeV, ds);

    // 가우시안 샘플링 (보어 영역, 양성자의 경우 가장 일반적)
    float u1 = curand_uniform(&seed);
    float u2 = curand_uniform(&seed);
    float z = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);

    return dE_mean + sigma * z;
}
```

### 최빈 에너지 손실 (Most Probable Energy Loss - Landau)

```
Δp = ξ [ln(ξ/T_max) + ln(1 + β²γ²) + 0.2 - β² - δ/2]
```

여기서 `δ`는 밀도 효과 보정입니다 (물 < 250 MeV에서 무시 가능).

---

## 3. 핵 감쇠 (Nuclear Attenuation)

### 참고문헌: ICRU Report 63

### 단면 모델 (Cross-Section Model)

```cpp
// 에너지 의존적 전체 단면 (mm⁻¹)
__host__ __device__ float Sigma_total(float E_MeV) {
    // 에너지에 대한 로그 의존성 [Logarithmic dependence on energy]
    constexpr float sigma_100 = 0.0012f;  // 100 MeV에서
    constexpr float sigma_20 = 0.0016f;   // 20 MeV에서
    constexpr float E_ref = 100.0f;

    if (E_MeV < 5.0f) {
        // 5 MeV에서 0으로부터 선형 증가 [Linear ramp from 0 at 5 MeV]
        return sigma_20 * (E_MeV - 5.0f) / 15.0f;
    } else if (E_MeV < 20.0f) {
        // 5-20 MeV 보간 [Interpolate 5-20 MeV]
        float t = (E_MeV - 5.0f) / 15.0f;
        return t * sigma_20;
    } else {
        // 20 MeV 이상 로그 [Logarithmic above 20 MeV]
        float a = log(sigma_20 / sigma_100) / log(20.0f / 100.0f);
        return sigma_100 * pow(E_MeV / E_ref, a);
    }
}
```

### 생존 확률 (Survival Probability)

```cpp
// 스텝 ds 후 생존 확률
__device__ float survival_probability(float E_MeV, float ds) {
    float sigma = Sigma_total(E_MeV);
    return exp(-sigma * ds);
}
```

### 에너지 보존 (Energy Conservation)

핵 상호작용은 무게와 에너지를 모두 제거합니다:

```cpp
__device__ void apply_nuclear_attenuation(
    float& weight,      // 수정됨: weight *= survival
    double& energy_rem, // 누적기: 핵에 의해 제거된 에너지 [Accumulator]
    float E_MeV,
    float ds
) {
    float sigma = Sigma_total(E_MeV);
    float prob_interaction = 1.0f - exp(-sigma * ds);

    float weight_removed = weight * prob_interaction;
    weight -= weight_removed;

    // 보존 감사를 위한 에너지 추적 [Track energy for conservation audit]
    energy_rem += weight_removed * E_MeV;
}
```

---

## 4. R 기반 스텝 제어 (R-Based Step Control)

### 원리: dR/ds = -1 (CSDA 근사)

### 최대 스텝 크기 (Maximum Step Size)

```cpp
__host__ __device__ float compute_max_step_physics(float E, const RLUT& lut, float dx = 1.0f, float dz = 1.0f) {
    float R = lut.lookup_R(E);  // CSDA 범위 [mm]

    // 1차 제한: 남은 범위의 분율 [Primary limit: fraction of remaining range]
    float delta_R_max = 0.02f * R;  // 범위의 2%

    // 브래그 피크 근처 에너지 의존적 세밀화 계수
    float dS_factor = 1.0f;

    if (E < 5.0f) {
        // 사거리 끝 부근: 극도의 세밀화
        dS_factor = 0.2f;
        delta_R_max = fminf(delta_R_max, 0.1f);  // 최대 0.1 mm
    } else if (E < 10.0f) {
        // 브래그 피크 근처: 높은 세밀화
        dS_factor = 0.3f;
        delta_R_max = fminf(delta_R_max, 0.2f);  // 최대 0.2 mm
    } else if (E < 20.0f) {
        // 브래그 피크 영역: 중간 세밀화
        dS_factor = 0.5f;
        delta_R_max = fminf(delta_R_max, 0.5f);  // 최대 0.5 mm
    } else if (E < 50.0f) {
        // 브래그 이전: 가벼운 세밀화
        dS_factor = 0.7f;
        delta_R_max = fminf(delta_R_max, 0.7f);  // 최대 0.7 mm
    }

    // 세밀화 계수 적용
    delta_R_max = delta_R_max * dS_factor;

    // 하드 한계
    delta_R_max = fminf(delta_R_max, 1.0f);  // 최대 1 mm
    delta_R_max = fmaxf(delta_R_max, 0.05f);  // 최소 0.05 mm

    // 셀 크기 한계(셀 건너뛰기 방지)
    float cell_limit = 0.25f * fminf(dx, dz);
    delta_R_max = fminf(delta_R_max, cell_limit);

    return delta_R_max;
}
```

### R-LUT를 사용한 에너지 업데이트 (Energy Update Using R-LUT)

```cpp
// 스텝 후 에너지 계산 (R 기반 방법)
__device__ float compute_energy_after_step(float E_in, float ds, const RLUT& lut) {
    float R_in = lut.lookup_R(E_in);
    float R_out = R_in - ds;  // CSDA: dR/ds = -1
    return lut.lookup_E_inverse(R_out);  // 역검색 [Inverse lookup]
}

// 스텝에沉积된 에너지 [Energy deposited in step]
__device__ float compute_energy_deposition(float E_in, float ds, const RLUT& lut) {
    float E_out = compute_energy_after_step(E_in, ds, lut);
    return E_in - E_out;  // 모든 에너지 손실이沉积이 됨
}
```

### 왜 R 기반인가 (Why R-Based Instead of S-Based)?

| 방법 [Method] | 공식 [Formula] | 정확도 [Accuracy] | 안정성 [Stability] |
|--------------|---------------|------------------|-------------------|
| S 기반 [S-based] | E_out = E_in - S(E) × ds | S가 일정하면 좋음 | 브래그 근처에서 낮음 |
| R 기반 [R-based] | E_out = E⁻¹(R(E) - ds) | 정확함 (CSDA) | 모든 곳에서 안정적 |

**핵심 장점**: R은 단조 감소하여 고유한 역검색을 보장합니다.

---

## 5. 페르미-에이지스 횡방향 확산 (Fermi-Eyges Lateral Spread)

### 이론 (Theory)

횡방향 분산 σ²_x(z)는 산란력 적분으로부터 계산됩니다:

```
Scattering Power: T(z) = dσ_θ²/dz

모멘트 [Moments]:
A₀(z) = ∫₀ᶻ T(z') dz'        : 전체 각도 분산 [Total angular variance]
A₁(z) = ∫₀ᶻ z' × T(z') dz'    : 1차 공간 모멘트 [First spatial moment]
A₂(z) = ∫₀ᶻ z'² × T(z') dz'   : 2차 공간 모멘트 [Second spatial moment]

횡방향 분산 [Lateral variance]: σ²_x(z) = A₀×z² - 2×A₁×z + A₂
```

### 구현 (`fermi_eyges.hpp`)

```cpp
// 하일랜드 공식으로부터 산란력
__device__ float fermi_eyges_scattering_power(float E_MeV) {
    float sigma_theta = highland_sigma(E_MeV, 1.0f, X0_water);
    return sigma_theta * sigma_theta;  // T = σ² per mm
}

// 수송 중 모멘트 누적
struct FermiEygesMoments {
    double A0 = 0.0;  // 전체 각도 분산 [Total angular variance]
    double A1 = 0.0;  // 1차 공간 모멘트 [First spatial moment]
    double A2 = 0.0;  // 2차 공간 모멘트 [Second spatial moment]
};

__device__ void device_update_fermi_eyges_moments(
    FermiEygesMoments& moments,
    float z,
    float ds,
    float E_MeV
) {
    float T = fermi_eyges_scattering_power(E_MeV);

    moments.A0 += T * ds;
    moments.A1 += T * z * ds;
    moments.A2 += T * z * z * ds;
}

// 깊이 z에서 횡방향 시그마 계산
__host__ __device__ float fermi_eyges_sigma(
    const FermiEygesMoments& moments,
    float z
) {
    float variance = moments.A0 * z * z - 2.0 * moments.A1 * z + moments.A2;
    return sqrt(fmaxf(variance, 0.0f));
}
```

### 세 가지 구성 요소 횡방향 확산 (Three-Component Lateral Spread)

```cpp
// 전체 횡방향 확산 = 초기 + 기하학적 + MCS
float total_lateral_sigma_squared(
    float sigma_x0,      // 초기 빔 폭 [Initial beam width]
    float sigma_theta0,  // 초기 각도 확산 [Initial angular spread]
    float z,             // 깊이 [Depth]
    float sigma_mcs      // MCS 기여 [MCS contribution]
) {
    // 초기 빔 확산 (거리에 따라 발산) [Initial beam spread]
    float sigma_initial = sigma_x0;

    // 초기 발산으로부터 기하학적 확산 [Geometric spread from initial divergence]
    float sigma_geometric = sigma_theta0 * z;

    // MCS 기여 (페르미-에이지스로부터) [MCS contribution from Fermi-Eyges]
    float sigma_scattering = sigma_mcs;

    // 전체 분산 (직각 제곱합으로) [Total variance in quadrature]
    return sqrt(sigma_initial*sigma_initial +
                sigma_geometric*sigma_geometric +
                sigma_scattering*sigma_scattering);
}
```

---

## 6. 물리 파이프라인 통합 (Physics Pipeline Integration)

### 완전한 스텝 물리 (Complete Step Physics)

```cpp
__device__ void transport_step(
    // 입력 상태 [Input state]
    float theta, float E, float x, float z, float w,
    // 그리드 파라미터 [Grid parameters]
    float dx, float dz,
    // LUT
    const RLUT& lut,
    // 출력 [Output]
    float& E_dep, double& E_nuc_rem, float& boundary_flux[4]
) {
    // 1. 스텝 크기 제어 [Step size control]
    float ds = compute_max_step_physics(E, lut);
    ds = fminf(ds, compute_boundary_step(x, z, dx, dz, theta));

    // 2. 에너지 손실 [Energy loss]
    float E_out = compute_energy_after_step(E, ds, lut);

    // 3. 에너지 분산 [Energy straggling]
    float dE_straggle = sample_energy_loss_with_straggling(E, ds, seed);
    E_out += dE_straggle;  // 분산 보정 추가
    E_out = fmaxf(E_out, E_cutoff);

    // 4. 에너지沉积 [Energy deposition]
    E_dep = E - E_out;

    // 5. MCS (다중 쿨롱 산란)
    float sigma_theta = highland_sigma(E, ds, X0_water);
    float delta_theta = sample_mcs_angle(sigma_theta, seed);
    theta += delta_theta;

    // 6. 핵 감쇠 [Nuclear attenuation]
    float w_before = w;
    apply_nuclear_attenuation(w, E_nuc_rem, E, ds);

    // 7. 위치 업데이트 [Position update]
    x += ds * sin(theta);
    z += ds * cos(theta);

    // 8. 경계 확인 [Boundary check]
    check_boundary_emission(x, z, dx, dz, boundary_flux);
}
```

---

## 물리 상수 (Physical Constants)

| 상수 [Constant] | 값 [Value] | 단위 [Unit] | 설명 [Description] |
|----------------|-----------|------------|-------------------|
| `m_p` | 938.272 | MeV/c² | 양성자 정지 질량 [Proton rest mass] |
| `m_e c²` | 0.511 | MeV | 전자 정지 에너지 [Electron rest energy] |
| `X0_water` | 360.8 | mm | 물의 복사 길이 [Radiation length of water] |
| `E_cutoff` | 0.1 | MeV | 에너지 컷오프 [Energy cutoff] |
| `E_trigger` | 10 | MeV | 정밀 수송 트리거 [Fine transport trigger] |
| `rho_water` | 1.0 | g/cm³ | 물의 밀도 [Density of water] |

---

## 참고문헌 (References)

1. **NIST PSTAR Database** - 양성자를 위한 저지력 및 범위 [Stopping powers and ranges for protons]
2. **PDG 2024** - Particle Data Group 리뷰 (하일랜드 공식) [Highland formula]
3. **ICRU Report 63** - 양성자를 위한 핵 단면 [Nuclear cross-sections for protons]
4. **Vavilov (1957)** - 에너지 분산 이론 [Energy straggling theory]
5. **Fermi-Eyges** - 다중 산란 이론 [Multiple scattering theory]
6. **Bethe-Bloch** - 평균 에너지 손실 공식 [Mean energy loss formula]
