#set text(font: "Linux Libertine", size: 11pt)
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

#show math: set text(weight: "regular")

= 물리 모델 문서

== 개요

SM_2D는 물에서 양성자 수송을 위한 포괄적인 물리 모델을 구현하며 ICRU, PDG, NIST 표준을 따릅니다. 모든 모델은 GPU 가속을 위한 CUDA 장치 함수로 구현됩니다.

== 1. 다중 쿨롬 산란(Highland 공식)

=== 참조

PDG 2024 입자 물리학 리뷰

=== Highland 공식(2D 투영)

$ sigma_theta = (13.6 " MeV" / (beta c p)) times z times sqrt(x / X_0) times [1 + 0.038 times ln(x / X_0)] / sqrt(2) $

여기서:
* $beta c p$ = 운동량 × 속도 [MeV/c]
* $z$ = 발사체 전하(양성자의 경우 1)
* $x$ = 단계 길이 [mm]
* $X_0$ = 방사선 길이(물의 경우 360.8 mm)
* $1 / sqrt(2)$ = 3D에서 2D 투영 보정

=== 구현 매개변수

```cpp
struct HighlandParams {
    float m_p_MeV = 938.272f;      // 양성자 정지 질량
    float X0_water = 360.8f;       // 방사선 길이 [mm]
    float MCS_2D_CORRECTION = 0.70710678f;  // 1/sqrt(2)
};
```

=== 분산 누적

정확한 다중 단계 산란을 위해 분산이 누적됩니다:

```cpp
// 올바름: 분산 누적
sigma_2_total += sigma_theta * sigma_theta;

// 그런 다음 전체 분산에서 샘플링
float theta_scatter = sqrt(sigma_2_total) * sample_normal();
```

=== 방향 업데이트

산란 후 방향 코사인이 업데이트됩니다:

```cpp
// 현재 각도
float theta = atan2(eta, mu);

// 산란 추가
theta += delta_theta;

// 방향 코사인 업데이트
mu = cos(theta);
eta = sin(theta);

// 정규화(mu² + eta² = 1 보장)
float norm = sqrt(mu*mu + eta*eta);
mu /= norm;
eta /= norm;
```

== 2. 에너지 분산(Vavilov 이론)

=== 세 가지 영역

#figure(
  table(
    columns: (auto, auto, 2fr),
    inset: 8pt,
    align: left,
    table.header([*영역*], [*κ 매개변수*], [*분포*]),
    [Bohr], [κ > 10], [가우시안],
    [Vavilov], [0.01 < κ < 10], [Vavilov 보간],
    [Landau], [κ < 0.01], [Landau(비대칭)],
  ),
  caption: [에너지 분산 영역],
)

=== Vavilov 매개변수

$ kappa = xi / T_sub.max $

여기서:
* $xi = (K / 2) times (Z / A) times (z^2 / beta^2) times rho times x$
* $T_sub.max = (2 m_e c^2 beta^2 gamma^2) / (1 + 2 gamma m_e / m_p + (m_e / m_p)^2)$

상수:
* $K = 0.307 " MeV cm"² / g$
* $Z / A = 0.555$ (물의 경우)
* $m_e c^2 = 0.511 " MeV"$

=== Bohr 분산 시그마

```cpp
__host__ __device__ float bohr_straggling_sigma(float E_MeV, float ds) {
    float gamma = 1.0f + E_MeV / m_p_MeV;
    float beta = sqrt(1.0f - 1.0f / (gamma * gamma));

    // Bohr 공식(물에 대해 간소화)
    float kappa_0 = 0.156f;  // 물에 대해 미리 계산됨
    float sigma = kappa_0 * sqrt(ds) / beta;

    return sigma;
}
```

=== 최빈 에너지 손실(Landau)

$ Delta_p = xi [ln(xi / T_sub.max) + ln(1 + beta^2 gamma^2) + 0.2 - beta^2 - delta / 2]$

여기서 $delta$는 밀도 효과 보정(물 < 250 MeV에서 무시 가능).

== 3. 핵 감쇠

=== 참조

ICRU 보고서 63

=== 단면적 모델

```cpp
__host__ __device__ float Sigma_total(float E_MeV) {
    // 에너지에 대한 로그 의존성
    constexpr float sigma_100 = 0.0012f;  // 100 MeV에서
    constexpr float sigma_20 = 0.0016f;   // 20 MeV에서
    constexpr float E_ref = 100.0f;

    if (E_MeV < 5.0f) {
        // 5 MeV에서 0으로 선형 증가
        return sigma_20 * (E_MeV - 5.0f) / 15.0f;
    } else if (E_MeV < 20.0f) {
        // 5-20 MeV 보간
        float t = (E_MeV - 5.0f) / 15.0f;
        return t * sigma_20;
    } else {
        // 20 MeV 이상 로그
        float a = log(sigma_20 / sigma_100) / log(20.0f / 100.0f);
        return sigma_100 * pow(E_MeV / E_ref, a);
    }
}
```

=== 생존 확률

```cpp
__device__ float survival_probability(float E_MeV, float ds) {
    float sigma = Sigma_total(E_MeV);
    return exp(-sigma * ds);
}
```

=== 에너지 보존

핵 상호작용은 가중치와 에너지를 모두 제거합니다:

```cpp
__device__ void apply_nuclear_attenuation(
    float& weight,      // 수정됨: 가중치 *= 생존
    double& energy_rem, // 누산기: 핵에 의해 제거된 에너지
    float E_MeV,
    float ds
) {
    float sigma = Sigma_total(E_MeV);
    float prob_interaction = 1.0f - exp(-sigma * ds);

    float weight_removed = weight * prob_interaction;
    weight -= weight_removed;

    // 보존 감사를 위한 에너지 추적
    energy_rem += weight_removed * E_MeV;
}
```

== 4. R 기반 단계 제어

=== 원리

$dR / dif s = -1$ (CSDA 근사)

=== 최대 단계 크기

```cpp
__host__ __device__ float compute_max_step_physics(float E, const RLUT& lut) {
    float R = lut.lookup_R(E);  // CSDA 사거리 [mm]

    // 1차 한계: 잔여 사거리의 분수
    float delta_R_max = 0.02f * R;  // 사거리의 2%

    // 브래그 피크 근처 에너지 의존적 세분화
    if (E < 10.0f) {
        delta_R_max = fminf(delta_R_max, 0.2f);  // 브래그 근처: 최대 0.2mm
    } else if (E < 50.0f) {
        delta_R_max = fminf(delta_R_max, 0.5f);  // 중간 범위: 최대 0.5mm
    }

    // 절대 최대값
    delta_R_max = fminf(delta_R_max, 1.0f);  // 1mm 초과 안 함

    return delta_R_max;
}
```

=== R-LUT를 사용한 에너지 업데이트

```cpp
// 단계 후 에너지 계산(R 기반 방법)
__device__ float compute_energy_after_step(float E_in, float ds, const RLUT& lut) {
    float R_in = lut.lookup_R(E_in);
    float R_out = R_in - ds;  // CSDA: dR/ds = -1
    return lut.lookup_E_inverse(R_out);  // 역조회
}

// 단계에서 퇴적된 에너지
__device__ float compute_energy_deposition(float E_in, float ds, const RLUT& lut) {
    float E_out = compute_energy_after_step(E_in, ds, lut);
    return E_in - E_out;  // 모든 에너지 손실이 퇴적이 됨
}
```

=== R 기반 vs S 기반

#figure(
  table(
    columns: (auto, 2fr, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*방법*], [*공식*], [*정확도*], [*안정성*]),
    [S 기반], [E_out = E_in - S(E) × ds], [S가 일정하면 좋음], [브래그 근처에서 낮음],
    [R 기반], [E_out = E⁻¹(R(E) - ds)], [정확(CSDA)], [모든 곳에서 안정],
  ),
  caption: [단계 제어 방법 비교],
)

== 5. Fermi-Eyges 횡방향 확산

=== 이론

횡방향 분산 $sigma_x^2(z)$는 산란 power 적분에서 계산됩니다:

=== 산란 Power

$ T(z) = d sigma_theta^2 / dif z $

=== 모멘트

$ A_0(z) = integral_0^z T(z') dif z' $ \
$ A_1(z) = integral_0^z z' times T(z') dif z' $ \
$ A_2(z) = integral_0^z z'^2 times T(z') dif z' $

=== 횡방향 분산

$ sigma_x^2(z) = A_0 times z^2 - 2 times A_1 times z + A_2 $

=== 구현

```cpp
// Highland 공식에서 산란 power
__device__ float fermi_eyges_scattering_power(float E_MeV) {
    float sigma_theta = highland_sigma(E_MeV, 1.0f, X0_water);
    return sigma_theta * sigma_theta;  // T = σ²/mm
}

// 수송 중 모멘트 누적
struct FermiEygesMoments {
    double A0 = 0.0;  // 전체 각도 분산
    double A1 = 0.0;  // 1차 공간 모멘트
    double A2 = 0.0;  // 2차 공간 모멘트
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

=== 세 가지 성분 횡방향 확산

전체 횡방향 확산 = 초기 + 기하학적 + MCS

```cpp
float total_lateral_sigma_squared(
    float sigma_x0,      // 초기 빔 폭
    float sigma_theta0,  // 초기 각도 확산
    float z,             // 깊이
    float sigma_mcs      // MCS 기여
) {
    // 초기 빔 확산(거리와 함께 발산)
    float sigma_initial = sigma_x0;

    // 초기 발산에서 기하학적 확산
    float sigma_geometric = sigma_theta0 * z;

    // MCS 기여(Fermi-Eyges에서)
    float sigma_scattering = sigma_mcs;

    // 전체 분산(직교 합)
    return sqrt(sigma_initial*sigma_initial +
                sigma_geometric*sigma_geometric +
                sigma_scattering*sigma_scattering);
}
```

== 6. 물리 파이프라인 통합

=== 완전한 단계 물리

```cpp
__device__ void transport_step(
    // 입력 상태
    float theta, float E, float x, float z, float w,
    // 그리드 매개변수
    float dx, float dz,
    // LUT
    const RLUT& lut,
    // 출력
    float& E_dep, double& E_nuc_rem, float boundary_flux[4]
) {
    // 1. 단계 크기 제어
    float ds = compute_max_step_physics(E, lut);
    ds = fminf(ds, compute_boundary_step(x, z, dx, dz, theta));

    // 2. 에너지 손실
    float E_out = compute_energy_after_step(E, ds, lut);

    // 3. 에너지 분산
    float dE_straggle = sample_energy_loss_with_straggling(E, ds, seed);
    E_out += dE_straggle;
    E_out = fmaxf(E_out, E_cutoff);

    // 4. 에너지 퇴적
    E_dep = E - E_out;

    // 5. MCS
    float sigma_theta = highland_sigma(E, ds, X0_water);
    float delta_theta = sample_mcs_angle(sigma_theta, seed);
    theta += delta_theta;

    // 6. 핵 감쇠
    apply_nuclear_attenuation(w, E_nuc_rem, E, ds);

    // 7. 위치 업데이트
    x += ds * sin(theta);
    z += ds * cos(theta);

    // 8. 경계 확인
    check_boundary_emission(x, z, dx, dz, boundary_flux);
}
```

== 물리 상수

#figure(
  table(
    columns: (auto, auto, auto, 3fr),
    inset: 8pt,
    align: left,
    table.header([*상수*], [*값*], [*단위*], [*설명*]),
    [$m_p$], [938.272], [MeV/c²], [양성자 정지 질량],
    [$m_e c^2$], [0.511], [MeV], [전자 정지 에너지],
    [$X_0$" (물)$], [360.8], [mm], [물의 방사선 길이],
    [$E_sub.cutoff$], [0.1], [MeV], [에너지 차단],
    [$E_sub.trigger$], [10], [MeV], [정밀 수송 트리거],
    [$rho_sub."water"$], [1.0], [g/cm³], [물의 밀도],
  ),
  caption: [물리 상수],
)

== 참고문헌

1. NIST PSTAR 데이터베이스 - 양성자용 stopping power 및 사거리
2. PDG 2024 - 입자 데이터 그룹 리뷰(Highland 공식)
3. ICRU 보고서 63 - 양성자용 핵 단면적
4. Vavilov (1957) - 에너지 분산 이론
5. Fermi-Eyges - 다중 산란 이론
6. Bethe-Bloch - 평균 에너지 손실 공식

---
#set align(center)
*SM_2D 물리 모델 문서*

#text(size: 9pt)[버전 1.0.0]
