#set text(font: "Malgun Gothic", size: 11pt)
#set page(numbering: "1", margin: (left: 20mm, right: 20mm, top: 20mm, bottom: 20mm))
#set par(justify: true)
#set heading(numbering: "1.")

// 컨셉 박스 스타일 정의
#let concept-box(content) = block(
  fill: rgb("#e3f2fd"),
  inset: 10pt,
  radius: 5pt,
  stroke: 1pt + rgb("#2196f3"),
  content
)

#let tip-box(content) = block(
  fill: rgb("#fff3e0"),
  inset: 10pt,
  radius: 5pt,
  stroke: 1pt + rgb("#ff9800"),
  content
)

#let warning-box(content) = block(
  fill: rgb("#ffebee"),
  inset: 10pt,
  radius: 5pt,
  stroke: 1pt + rgb("#f44336"),
  content
)

#let analogy-box(content) = block(
  fill: rgb("#f3e5f5"),
  inset: 10pt,
  radius: 5pt,
  stroke: 1pt + rgb("#9c27b0"),
  content
)

#let clinical-box(content) = block(
  fill: rgb("#e8f5e9"),
  inset: 10pt,
  radius: 5pt,
  stroke: 1pt + rgb("#4caf50"),
  content
)

#show math.equation: set text(weight: "regular")

= 양성자 치료 물리학: 시각적 가이드

이 문서는 SM_2D에서 사용되는 물리 모델을 쉬운 용어로 설명합니다. 복잡한 공식을 이해하기 쉬운 개념으로 분해하고 실제 상황과 임상적 중요성을 설명합니다.

---

== 양성자 치료란 무엇인가요?

#concept-box[
*양성자 치료*는 양성자 빔을 사용하여 암 세포를 파괴하는 방사선 치료의 한 유형입니다. X선과 달리 신체를 통과하는 양성자는 특정 깊이에서 멈추고 거기서 대부분의 에너지를 방출합니다.

*중요성:* 이를 통해 의사는 종양 뒤의 건강한 조직을 보호하면서 종양에 높은 선량을 전달할 수 있습니다.
]

#analogy-box[
*실생활 비유:* 양성자를 에너지를 나르는 작은 트럭이라고 생각해보세요. X선은 몸을 통과할 때마다 여기저기에 패키지(방사선)를 떨어뜨리고 계속 달리는 자동차 같습니다. 양성자는 운전해서 도착지(종양)에 모든 패키지를 내려놓고 멈추는 트럭 같습니다.
]

---

== 물리 모델 개요

SM_2D는 물(인체 조직과 매우 유사)을 통과하는 양성자의 여정을 시뮬레이션합니다. 다음을 모델링해야 합니다:

1. *다중 쿨롬 산란* - 원자에서 튕겨나오는 양성자
2. *에너지 분산* - 에너지 손실의 변동
3. *핵 감쇠* - 핵 반응으로 인해 사라지는 양성자
4. *스텝 제어* - 단계별 여정 시뮬레이션 방법
5. *Fermi-Eyges 이론* - 양성자 빔이 옆으로 퍼지는 방식

---

== 1. 다중 쿨롬 산란 (MCS)

=== 쉬운 설명

#concept-box[
*다중 쿨롬 산란 (MCS)*은 양성자가 원자핵 근처를 지날 때 편향되는(튕겨지는) 현상을 설명합니다. 숲을 통해 작은 공을 던지는 것을 상상해보세요 - 나무에 계속 부딪히기 때문에 완전히 직선으로 가지 않습니다.

*핵심 포인트:* 각 개별 편향은 아주 작지만, 수백만 번의 상호작용 후에는 양성자의 경로가 크게 휘어집니다.
]

=== 치료에 왜 중요한가요

#clinical-box[
*임상적 중요성:*

MCS로 인해 양성자 빔이 몸 깊숙이 들어갈 때 "흐려지거나" 퍼집니다. 이것은 다음을 의미합니다:
- *장점:* 전체 종양 부피를 덮는 데 도움이 됨
- *과제:* 건강한 조직에 선량이 누출될 수 있음
- *해결책:* 빔 각도와 모양을 계획할 때 MCS를 고려해야 함

*예:* 150 MeV 양성자 빔은 깊은 곳에 있는 종양에 도달할 때까지 5mm에서 15mm로 퍼질 수 있습니다.
]

=== 시각적 다이어그램: 산란 과정

#table(
  columns: 2,
  inset: 10pt,
  align: left,
  table.header([*단계*], [*설명*]),
  [들어오는 양성자], [●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ #h(1em) ↓ #h(1em) ↗ #h(1em) ↘ #h(1em) ↗ #h(1em) ↘ #h(1em) ↗ #h(1em) ↘ #h(1em) ↗ #h(1em) ↘ #h(1em) ↗ #h(1em) ↘],
  [여러 작은 편향], [각 틱 = 한 번의 원자 만남 #h(1em) 화살표는 방향 변경 표시 (가시성을 위해 과장됨)],
  [최종 방향 변경됨], [●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ #h(1em) ↓],
)

=== Highland 공식

#tip-box[
*간단한 설명:* Highland 공식은 양성자 빔이 특정 거리를 이동한 후 얼마나 퍼질지 알려줍니다. 공기 중을 이동하는 물 스프레이가 얼마나 넓어질지 예측하는 것과 같습니다.

공식은 다음을 사용합니다:
- *양성자 에너지* - 더 빠른 양성자는 덜 산란됨
- *이동 거리* - 더 먼 거리 = 더 많은 산란
- *재료* - 일부 재료는 다른 것보다 더 많이 산란시킴
]

=== 공식 단계별 설명

$ sigma_"theta" = (13.6 " MeV" / (beta c p)) times z times sqrt(x / X_0) times [1 + 0.038 times ln(x / X_0)] / sqrt(2) $

하나씩 분석해 보겠습니다:

#table(
  columns: (auto, 4fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([*기호*], [*의미 (쉬운 용어)*], [*예시 값*]),
  [$sigma_"theta"$], [각도 확산 - 빔이 얼마나 퍼지는지], [~2-5도],
  [$13.6 " MeV"$], [입자 물리학의 보편적 상수], [고정됨],
  [$beta c p$], [양성자 운동량 × 속도 - 높을수록 덜 산란], [150 MeV 양성자의 경우 140 MeV/c],
  [$z$], [양성자 전하 수 (양성자의 경우 항상 1)], [1],
  [$x$], [재료에서 이동한 거리], [0-300 mm],
  [$X_0$], [방사선 길이 - 재료의 "산란력"], [물의 경우 360.8 mm],
  [$1 / sqrt(2)$], [2D 대 3D 보정 계수], [0.707],
)

=== 직관적 이해

#analogy-box[
*"볼링장" 비유:* 아주 긴 볼링장에서 볼링을 치는 것을 상상해보세요:
- *높은 에너지 (빠른 양성자)* = 무거운 볼링공 - 편향되기 어려움
- *낮은 에너지 (느린 양성자)* = 가벼운 탁구공 - 이리저리 튐
- *높은 Z 재료 (뼈 같은)* = 아주 가까이 배치된 핀
- *낮은 Z 재료 (물/지방 같은)* = 멀리 배치된 핀

Highland 공식은 공이 직선 경로에서 얼마나 벗어날지 예측합니다.
]

=== 에너지 의존성

#table(
  columns: 2,
  inset: 10pt,
  align: left,
  table.header([*에너지 레벨*], [*확산 정도*]),
  [높은 에너지 (200 MeV)], [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ #h(1em) (최소 확산)],
  [중간 에너지 (100 MeV)], [━━━━━━━━━━━━━━━━━━━━━━━━━→ #h(1em) (중간 확산)],
  [낮은 에너지 (50 MeV)], [━━━━━━━━━━━━━━━━━→ #h(1em) (상당한 확산)],
  [매우 낮은 에너지 (20 MeV)], [━━━━━━━━━→ #h(1em) (큰 확산 - 브래그 피크 근처)],
)

#tip-box[
*임상적 교훈:* 낮은 에너지 양성자(브래그 피크 근처)가 훨씬 더 많이 산란됩니다. 이것이 선량 분포가 양성자 사거리 끝에서 "펼쳐지는" 이유입니다.
]

=== 코드 구현 (올바른 방법)

#warning-box[
*중요한 구현 세부사항:* 정확한 다중 스텝 산란을 위해 표준 편차가 아닌 분산을 누적해야 합니다. 이것은 일반적인 실수입니다!

*틀림:*
```cpp
// 이렇게 하지 마세요!
sigma_total += sigma_theta;  // 틀림!

// 이것도 하지 마세요!
theta_total += sample_mcs(sigma_theta);  // 각 스텝마다 샘플링
```

*올바름:*
```cpp
// 이렇게 하세요: 분산 누적
sigma_2_total += sigma_theta * sigma_theta;

// 그런 다음 전체 분산에서 한 번 샘플링
float theta_scatter = sqrt(sigma_2_total) * sample_normal();
```

*왜?* 독립적인 무작위 과정의 경우 분산이 더해집니다. 표준 편차는 아니죠!
]

=== 2D 투영 보정

#concept-box[
*3D 대 2D 산란:* 실제 양성자는 3D 공간(모든 방향)으로 산란됩니다. 하지만 우리 시뮬레이션은 2D(x-z 평면)입니다. Highland 공식은 3D 산란 각도를 주므로 2D로 변환해야 합니다.

*수학:* $sigma_(2D) = sigma_(3D) / sqrt(2)$

*왜 sqrt(2)로 나누나요?* 통계 역학 - 3D 가우스 분포를 2D 평면에 투영
]

---

== 2. 에너지 분산

=== 쉬운 설명

#concept-box[
*에너지 분산*은 같은 거리를 이동한 다른 양성자가 잃는 에너지 양의 변동입니다. 모든 양성자가 같은 에너지로 시작하고 같은 경로를 이동하더라도, 끝에서 같은 에너지를 갖지는 않습니다.

*왜?* 에너지 손실은 무작위 과정입니다. 일부 양성자는 원자와 더 많은 "가까운 만남"을 하여 더 많은 에너지를 잃습니다. 다른 양성자는 더 적은 상호작용을 하여 덜 잃습니다.
]

=== 치료에 왜 중요한가요

#clinical-box[
*임상적 중요성:* 에너지 분산으로 인해 브래그 피크가 "번지거나" 넓어집니다:
- *분산 없음:* 날카롭고 좁은 브래그 피크
- *분산 있음:* 더 넓고 덜 두드러진 피크

*영향:* 종양 뒤 건강한 조직으로 선량이 누출되는 정도에 영향을 줌. 너무 많은 분산 = 종양 뒤 조직에 선량 누출

*임상적 현실:* 분산은 빔 전달 기술에 관계없이 선량 강하 날카움에 대한 근본적인 한계를 설정합니다.
]

=== 시각적 다이어그램: 분산 효과

#table(
  columns: 2,
  inset: 10pt,
  align: left,
  table.header([*단계*], [*설명*]),
  [모든 양성자가 150 MeV로 시작], [━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━● 150 MeV],
  [물에서 150mm 이동 후], [●●●●●●●●●●●●●●●●●●●●●●●●●●●●● #h(1em) 140 142 145 148 150 152 155 158 160 MeV #h(1em) ↑ #h(1em) 평균 #h(1em) ↑ #h(1em) 더 많이 손실 #h(1em) (148 MeV) #h(1em) 덜 손실],
  [결과], [범위 = 148 ± 10 MeV (분산 폭 ~7 MeV)],
)

=== 분산의 세 가지 영역

#table(
  columns: (auto, auto, 2fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([*영역*], [*κ 매개변수*], [*분포 모양*], [*임상적 맥락*]),
  [Bohr], [κ > 10], [대칭 종 모양 (가우시안)], [고에너지, 두꺼운 흡수체],
  [Vavilov], [0.01 < κ < 10], [중간 모양], [대부분의 임상 상황],
  [Landau], [κ < 0.01], [긴 꼬리의 비대칭], [저에너지, 얇은 층],
)

#tip-box[
*κ(카파) 매개변수는 우리가 어떤 영역에 있는지 알려줍니다:*

$ kappa = xi / T_("max") $

여기서:
- $xi$ = "특성 에너지 손실" - 전형적인 에너지 손실량
- $T_("max")$ = 단일 충돌에서 가능한 최대 에너지 손실

*높은 κ* = 많은 작은 에너지 전달 (Bohr/가우시안 영역)
*낮은 κ* = 가능한 몇 개의 큰 에너지 전달 (Landau 영역)
]

=== Vavilov 매개변수 공식

$ kappa = xi / T_("sub").max $

여기서:

$ xi = (K / 2) times (Z / A) times (z^2 / beta^2) times rho times x $

$ T_("sub").max = (2 m_e c^2 beta^2 gamma^2) / (1 + 2 gamma m_e / m_p + (m_e / m_p)^2) $

각 부분을 이해해 봅시다:

#table(
  columns: (auto, 4fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([*기호*], [*의미*], [*물에서의 값*]),
  [$K$], [양자 전기역학의 보편적 상수], [0.307 MeV cm²/g],
  [$Z / A$], [원자 번호와 질량 수의 비율], [물의 경우 0.555],
  [$z$], [발사체 전하 (양성자의 경우 1)], [1],
  [$beta$], [양성자 속도 / 빛의 속도], [치료용 0.1-0.6],
  [$rho$], [재료 밀도], [물의 경우 1.0 g/cm³],
  [$x$], [경로 길이], [0-300 mm],
  [$m_e c^2$], [전자 정지 에너지], [0.511 MeV],
  [$m_p$], [양성자 정지 질량], [938.27 MeV],
)

=== Bohr 분산 (간단한 경우)

#analogy-box[
*"무작위 걷기" 비유:* 에너지 분산은 무작위 걷기와 같습니다. 각 양성자는 이동하는 동안 다른 수의 "발걸음"(에너지 전달)을 취합니다. 최종 에너지의 확산은 통계적 규칙을 따릅니다.

Bohr 분산은 각 발걸음이 작고 많은 발걸음이 있을 때 적용됩니다. 이것은 가우스(정규) 분포를 제공합니다.

*공식:* $sigma = (kappa_0 / beta) times sqrt(x)$

$sqrt(x)$ 의존성을 주목하세요 - 이것은 무작위 걷기 과정의 특징입니다!
]

=== Bohr 공식 구현

```cpp
__host__ __device__ float bohr_straggling_sigma(float E_MeV, float ds) {
    float gamma = 1.0f + E_MeV / m_p_MeV;
    float beta = sqrt(1.0f - 1.0f / (gamma * gamma));

    // Bohr 공식 (물에 대해 간소화)
    float kappa_0 = 0.156f;  // 물에 대해 미리 계산됨
    float sigma = kappa_0 * sqrt(ds) / beta;

    return sigma;
}
```

#concept-box[
*주요 의존성:*
- *1/beta*: 느린 양성자(낮은 베타)가 더 많은 분산 - 원자 근처에 더 오래 머무름
- *sqrt(ds)*: 거리의 제곱근으로 퍼짐 (무작위 걷기)
- *재료*: 다른 재료는 다른 kappa_0 값을 가짐
]

=== Landau 분산 (비대칭 경우)

#tip-box[
*낮은 κ일 때 (Landau 영역):* 저에너지나 얇은 층에서 에너지 손실은 때때로 큰 에너지 전달이 지배적입니다. 이것은 다음을 가진 비대칭 분포를 만듭니다:
- *뾰족한 피크* 낮은 에너지 손실에서 (대부분의 양성자가 적게 손실)
- *긴 꼬리* 높은 에너지 손실으로 (소수의 양성자가 많이 손실)

*임상적 관련성:* 이 꼬리는 "사거리 분산"을 만듭니다 - 다른 양성자가 다른 깊이에서 멈춰서 브래그 피크를 번지게 합니다.
]

=== 최빈 에너지 손실

$ Delta_p = xi [ln(xi / T_("sub").max) + ln(1 + beta^2 gamma^2) + 0.2 - beta^2 - delta / 2]$

#warning-box[
*중요:* 최빈 에너지 손실(Landau 분포의 피크)은 평균 에너지 손실(Bethe-Bloch 공식)과 같지 않습니다!
- *최빈:* 분포가 피크하는 위치
- *평균:* 평균 에너지 손실 (꼬리 때문에 더 높음)

정확한 브래그 피크 모델링을 위해 차이가 중요합니다.
]

---

== 3. 핵 감쇠

=== 쉬운 설명

#concept-box[
*핵 감쇠*는 원자와의 핵 반응으로 인해 빔에서 양성자가 사라질 수 있음을 의미합니다. 양성자가 원자핵에 너무 가까이 가면 다음을 할 수 있습니다:
1. 원자핵에 흡수됨
2. 원자핵이 부서지게 함(핵 파편화)
3. 다른 입자를 때려 냄(2차 입자)

모든 경우에서 원래 양성자는 사라집니다 - 더 이상 치료 선량에 기여하지 않습니다.
]

=== 치료에 왜 중요한가요

#clinical-box[
*임상적 중요성:* 핵 감쇠는 종양에 도달하는 양성자 수를 줄입니다:
- *선량 감소:* 더 적은 양성자 = 예상보다 낮은 선량
- *2차 입자:* 핵 반응이 2차 방사선(중성자, 알파 입자)을 생성
- *중성자 선량:* 건강한 조직의 암 위험 증가 가능성
- *치료 계획:* 올바른 선량을 전달하기 위해 양성자 손실을 고려해야 함

*일반적인 크기:* 물에서 10cm당 약 1-3%의 양성자가 손실됨
]

=== 시각적 다이어그램: 핵 반응

#table(
  columns: 2,
  inset: 10pt,
  align: left,
  table.header([*반응 유형*], [*설명*]),
  [1. 탄성 산란 (양성자 생존)], [p + 원자핵 → p + 원자핵 (다른 방향) #h(1em) ↘ #h(1em) (양성자 방향 변경 but 빔에 유지)],
  [2. 비탄성 산란 (양성자 흡수)], [p + 원자핵 → 2차 입자들 #h(1em) ↘ #h(1em) (양성자 사라짐, 파편 생성)],
  [3. 핵 파편화], [p + 원자핵 → p' + 가벼운 원자핵 + 입자들 #h(1em) ↘ #h(1em) (원래 원자핵이 부서짐)],
)

=== 단면적 모델

#concept-box[
*단면적 (σ):* "단면적"은 핵 반응을 위한 원자핵의 유효 표적 면적입니다. 더 큰 단면적 = 상호작용할 가능성이 더 높음.

*비유:* 원자핵을 원형 표적으로 생각해보세요. 단면적은 표적의 면적입니다. 더 큰 표적은 맞히기 쉽습니다.

*단위:* cm² 또는 mm² (매우 작은 숫자!)
*전형적인 값:* 핵당 10⁻²⁶에서 10⁻²⁴ cm²
]

#tip-box[
*거시적 단면적 (Σ):* 시뮬레이션에서 부피당 단면적 Σ를 사용합니다:

$ Sigma = N times sigma $

여기서:
- $N$ = 원자핵의 수 밀도 (핵수/cm³)
- $sigma$ = 미시적 단면적 (cm²)

물의 경우: $Sigma ≈ 0.0012 " to " 0.0016 " mm"^-1$ (에너지 의존적)
]

=== 핵 단면적의 에너지 의존성

#table(
  columns: (auto, 2fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([*에너지 범위*], [*단면적 동작*], [*이유*]),
  [\< 5 MeV], [무시할 수 있음], [양성자가 핵 장벽을 극복할 수 없음],
  [5-20 MeV], [급격히 증가], [더 많은 핵 반응이 가능해짐],
  [20-100 MeV], [천천히 감소], [더 높은 에너지 = 상호작용 시간 감소],
  [\> 100 MeV], [대략 일정], [고에너지 한계 도달],
)

=== 구현: 에너지 종속적 단면적

```cpp
__host__ __device__ float Sigma_total(float E_MeV) {
    // 수소/산소에 대한 쿨롬 장벽 아래: 무시할 수 있는 핵 반응
    if (E_MeV < 5.0f) {
        return 0.0f;
    }

    // 100 MeV에서의 참조 값 (ICRU 63)
    constexpr float sigma_100 = 0.0012f;  // 100 MeV에서 mm⁻¹
    constexpr float E_ref = 100.0f;       // 참조 에너지 [MeV]

    if (E_MeV >= 20.0f) {
        // 치료용 범위 (20-250 MeV)에 대한 로그 에너지 의존성
        // σ(E) = σ_100 * [1 - 0.15 * ln(E/100)]
        // 인자 0.15는 20에서 200 MeV까지 ~30% 감소를 제공
        float log_factor = 1.0f - 0.15f * logf(E_MeV / E_ref);
        float sigma = sigma_100 * fmaxf(log_factor, 0.4f);  // 참조의 40% 최소값
        return sigma;
    } else {
        // 저에너지 (5-20 MeV): 쿨롬 장벽(5 MeV)에서 0에서 20 MeV의 sigma_20으로 선형 증가
        constexpr float sigma_20 = 0.0016f;  // 20 MeV에서 mm⁻¹
        float frac = (E_MeV - 5.0f) / 15.0f;  // 0에서 1
        return sigma_20 * frac;
    }
}
```

#tip-box[
*주요 구현 참고사항:*

1. 공식은 로그 보정을 사용합니다: $1 - 0.15 times ln(E / 100)$
2. 값은 참조의 40%에서 클램핑됩니다 (최소 단면적)
3. 20 MeV 아래에서 쿨롬 장벽 (5 MeV)에서 선형 증가를 사용합니다
]

#analogy-box[
*"필터" 비유:* 핵 감쇠를 빔이 통과하는 필터라고 생각해보세요:
- *고에너지:* 느슨한 필터 (대부분의 양성자 통과)
- *저에너지:* 더 타이트한 필터 (더 많은 양성자가 걸림)
- *두꺼운 재료:* 더 긴 필터 (잡힐 기회가 더 많음)

단면적 Σ은 주어진 에너지에서 필터가 얼마나 "타이트한지" 알려줍니다.
]

=== 생존 확률

#concept-box[
*생존 확률:* 거리 ds를 이동하는 동안 핵 반응을 하지 않는 양성자의 비율은 얼마입니까?

$ P_("survival") = e^(-Sigma times "ds") $

이것은 표준 지수 감쇠 법칙입니다 (X선 흡수에도 사용됨).
]

```cpp
__device__ float survival_probability(float E_MeV, float ds) {
    float sigma = Sigma_total(E_MeV);
    return exp(-sigma * ds);
}
```

#tip-box[
*작은 스텝의 경우 (ds ≪ 1/Sigma):* 선형 근사를 사용할 수 있습니다:

$ P_("survival") ≈ 1 - Sigma times "ds" $

이것이 몬테카를로에서 각 스텝에서 양성자가 "죽었는지" 결정하는 데 사용하는 것입니다.
]

=== 핵 반응과 에너지 보존

#warning-box[
*중요: 제거된 에너지 추적:* 양성자가 핵 상호작용으로 제거되면 에너지가 그냥 사라지지 않습니다! 에너지 보존 감사를 위해 추적해야 합니다:

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
    energy_rem += weight_removed * E_MeV;  // ← 중요!
}
```

*왜?* 총 에너지 입력 = 퇴적된 에너지 + 2차 입자가 운반한 에너지
]

=== 핵 상호작용의 임상적 영향

#clinical-box[
*2차 방사선:* 핵 반응은 2차 입자를 만듭니다:
- *중성자:* 매우 투과성이 높아 빔 경로 밖으로 멀리 이동할 수 있음
- *알파 입자:* 매우 짧은 사거리, 높은 LET (잠재적으로 더 손상적)
- *무거운 파편:* 표적 원자핵이 부서질 때

*임상적 우려:*
1. *2차 암 위험:* 중성자는 치료 영역 밖의 DNA 손상을 일으킬 수 있음
2. *영상 아티팩트:* 핵 반응으로 인한 PET 활성화
3. *선량 섭동:* 2차 입자가 다른 곳에 작은 선량을 퇴적

*현재 관행:* 현대 치료 계획 시스템은 2% 이상의 정확도를 위해 핵 상호작용 모델을 포함합니다.
]

---

== 4. R 기반 스텝 제어

=== 쉬운 설명

#concept-box[
*R 기반 스텝 제어*는 몸을 통한 양성자의 여정을 시뮬레이션하는 방법입니다. 모든 미세한 상호작용을 계산하는 대신 "스텝"을 취합니다 - 평균 물리학을 계산하는 경로의 분리된 덩어리.

"R"은 사거리(Range)를 나타냅니다 - 양성자가 멈추기 전에 얼마나 멀리 갈 수 있는지. 안정성과 정확도가 더 좋기 때문에 에너지가 아닌 사거리를 사용하여 스텝 크기를 제어합니다.
]

=== 치료에 왜 중요한가요

#clinical-box[
*임상적 중요성:* 정확한 스텝 제어는 다음에 중요합니다:
- *브래그 피크 위치:* 스텝 크기의 작은 오차 = 양성자가 멈추는 위치의 큰 오차
- *선량 계산:* 각 스텝의 에너지 퇴적을 올바르게 계산해야 함
- *시뮬레이션 속도:* 너무 많은 스텝 = 느림; 너무 적음 = 부정확

*도전:* 양성자는 사거리 끝 근처에서 급격히 느려지므로 그곳에서 매우 작은 스텝이 필요합니다. R 기반 제어는 잔여 사거리를 기준으로 스텝 크기를 자동으로 조정합니다.
]

=== 시각적 다이어그램: 적응형 스텝 크기

#table(
  columns: 2,
  inset: 10pt,
  align: left,
  table.header([*에너지 레벨*], [*스텝 크기*]),
  [높은 에너지 (150 MeV)], [━━━┃━━━┃━━━┃━━━┃━━━┃━━━┃━━━┃━━━┃ #h(1em) ↑ #h(1em) 큰 스텝 (각각 1mm) #h(1em) 양성자가 많이 느려지지 않음],
  [중간 에너지 (70 MeV)], [━┃━┃━┃━┃━┃━┃━┃━┃━┃━┃━┃━┃━┃ #h(1em) ↑ 중간 스텝 (각각 0.5mm) #h(1em) 양성자 느려짐],
  [낮은 에너지 (10 MeV) - 브래그 피크 근처], [┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃┃ #h(1em) ↑ 매우 작은 스텝 (각각 0.05mm) #h(1em) 양성자 곧 멈춤 - 높은 정밀도 필요!],
)

=== CSDA 근사

#concept-box[
*CSDA = 연속 감속 근사*

이것은 R 기반 스텝 제어 뒤에 있는 핵심 아이디어입니다:

$ "dR" / "ds" = -1 $

*의미:* 양성자가 이동하는 밀리미터마다(ds), 잔여 사거리는 정확히 1밀리미터씩 감소합니다(dR = -ds).

*작동 이유:* CSDA 근사에서 양성자가 연속적으로 에너지를 잃는다고 가정합니다(이산적 점프가 아님). 이것은 우리 목적에 대한 좋은 근사입니다.
]

#analogy-box[
*"연료 게이지" 비유:* 양성자 사거리를 차의 연료 게이지라고 생각해보세요:
- *사거리 R* = 연료 게이지의 "잔여 마일"
- *이동 거리 ds* = 실제 주행한 마일

CSDA는 다음과 같이 말합니다: 1마일을 주행할 때마다 "잔여 마일"이 정확히 1마일씩 감소합니다.

*현실 확인:* 이것은 완벽하지 않습니다 (언덕, 에어컨 사용 등), 하지만 좋은 근사입니다. 마찬가지로 CSDA는 완벽하지 않습니다(분산, 핵 반응), 하지만 몬테카를로에서 잘 작동합니다.
]

=== 최대 스텝 크기 공식

```cpp
__host__ __device__ float compute_max_step_physics(float E, const RLUT& lut, float dx = 1.0f, float dz = 1.0f) {
    float R = lut.lookup_R(E);  // CSDA 사거리 [mm]

    // 1차 한계: 잔여 사거리의 분수
    float delta_R_max = 0.02f * R;  // 사거리의 2%

    // 브래그 피크 근처 에너지 종속적 세분화 계수
    float dS_factor = 1.0f;

    if (E < 5.0f) {
        // 사거리 끝 매우 근처: 극도의 세분화
        dS_factor = 0.2f;
        delta_R_max = fminf(delta_R_max, 0.1f);  // 최대 0.1mm
    } else if (E < 10.0f) {
        // 브래그 피크 근처: 높은 세분화
        dS_factor = 0.3f;
        delta_R_max = fminf(delta_R_max, 0.2f);  // 최대 0.2mm
    } else if (E < 20.0f) {
        // 브래그 피크 영역: 중간 세분화
        dS_factor = 0.5f;
        delta_R_max = fminf(delta_R_max, 0.5f);  // 최대 0.5mm
    } else if (E < 50.0f) {
        // 브래그 이전: 가벼운 세분화
        dS_factor = 0.7f;
        delta_R_max = fminf(delta_R_max, 0.7f);  // 최대 0.7mm
    }

    // 세분화 계수 적용
    delta_R_max = delta_R_max * dS_factor;

    // 하드 한계
    delta_R_max = fminf(delta_R_max, 1.0f);  // 최대 1mm
    delta_R_max = fmaxf(delta_R_max, 0.05f);  // 최소 0.05mm

    // 셀 크기 한계 (셀 건너뛰기 방지)
    float cell_limit = 0.25f * fminf(dx, dz);
    delta_R_max = fminf(delta_R_max, cell_limit);

    return delta_R_max;
}
```

#tip-box[
*주요 한계 설명:*
1. *사거리의 2% (기본):* 스텝이 잔여 여정에 비례하는지 확인
2. *에너지 종속적 세분화:* 양성자가 느려지면 스텝이 작아짐
3. *하드 한계 (0.05-1.0 mm):* 터무니없이 크거나 작은 스텝 방지
4. *셀 크기 한계:* 전체 그리드 셀 건너뛰기 방지 (선량 퇴적을 놓침!)
]

=== R 기반 대 S 기반 에너지 업데이트

#table(
  columns: (auto, 3fr, 2fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([방법], [작동 방식], [장점], [단점]),
  [S 기반 (stopping power)], [Bethe-Bloch 사용: $E_("out") = E_("in") - S(E) times "ds"$], [간단, 직접적], [브래그 피크 근처 불안정 - stopping power가 빠르게 변함!],
  [R 기반 (사거리)], [사거리 조회 사용: $E_("out") = E^(-1)(R(E) - "ds")$], [모든 곳에서 안정, 정확 (CSDA)], [사거리 조회 테이블 필요],
)

#warning-box[
*왜 S 기반이 브래그 피크 근처에서 실패하는가:* Stopping power S(E)는 브래그 피크 근처에서 매우 빠르게 변합니다:
- 10 MeV에서: S ≈ 100 MeV/cm
- 5 MeV에서: S ≈ 200 MeV/cm
- 2 MeV에서: S ≈ 500 MeV/cm

작은 스텝 크기로도 S(E) × ds를 사용하면 스텝 동안 S가 크게 변하기 때문에 잘못된 답이 나옵니다!

R 기반은 사거리를 사용하여 이 문제를 완전히 피합니다.
]

=== 에너지 업데이트 구현

```cpp
// 스텝 후 에너지 계산 (R 기반 방법)
__device__ float compute_energy_after_step(float E_in, float ds, const RLUT& lut) {
    float R_in = lut.lookup_R(E_in);
    float R_out = R_in - ds;  // CSDA: dR/ds = -1
    return lut.lookup_E_inverse(R_out);  // 역조회
}

// 스텝에서 퇴적된 에너지
__device__ float compute_energy_deposition(float E_in, float ds, const RLUT& lut) {
    float E_out = compute_energy_after_step(E_in, ds, lut);
    return E_in - E_out;  // 모든 에너지 손실이 퇴적이 됨
}
```

#concept-box[
*조회 테이블 (LUT):* R(E)와 E(R)에 미리 계산된 테이블을 사용합니다:
- *속도:* 테이블 조회가 Bethe-Bloch 계산보다 빠름
- *정확도:* NIST PSTAR 데이터 사용 (금 표준)
- *안정성:* 공식 평가의 수치적 문제 회피

*보간:* 매끄러운 결과를 위해 테이블 포인트 사이를 보간
]

---

== 5. Fermi-Eyges 횡방향 확산 이론

=== 쉬운 설명

#concept-box[
*Fermi-Eyges 이론*은 양성자 빔이 재료를 통과할 때 옆으로(횡방향으로) 어떻게 퍼지는지 예측합니다. 이것은 MCS와 다릅니다 - 개별 양성자 편향을 추적하는 대신 Fermi-Eyges는 전체 빔의 통계적 확산을 계산합니다.

*핵심 아이디어:* 어느 깊이에서의 빔 폭은 그곳에 도달하는 길에 일어난 모든 산란에 달려 있으며, 그 지점에서의 산란만이 아닙니다.
]

=== 치료에 왜 중요한가요

#clinical-box[
*임상적 중요성:* 횡방향 빔 확산은 치료 계획에 결정적입니다:
- *필드 마진:* 빔 확산을 고려하여 종양 커버리지 보장
- *장기 보호:* 확산을 알면 중요한 구조 회피 가능
- *빔 성형:* 조리개 및 보상기는 횡방향 펜umbra 알기 필요
- *선량 페인팅:* 현대 기술은 선량 분포의 정확한 지식 필요

*임상 예:* 20cm 깊이의 3cm 반경 종양은 횡방향 확산을 고려해 4cm 반경 빔이 필요할 수 있습니다 (~1cm 확장).
]

=== 시각적 다이어그램: 빔 확산

#table(
  columns: 2,
  inset: 10pt,
  align: left,
  table.header([*깊이*], [*빔 폭*]),
  [표면에서 (z = 0)], [|← 5 mm →| #h(1em) ██████████  ← 날카롭고 좁은 빔],
  [5cm 깊이에서], [|←── 8 mm ──→| #h(1em) ████████████████  ← 퍼지기 시작],
  [10cm 깊이에서], [|←──── 12 mm ────→| #h(1em) ████████████████████████  ← 상당한 확산],
  [15cm 깊이에서 (브래그 피크)], [|←─────── 18 mm ───────→| #h(1em) ██████████████████████████████████  ← 최대 확산],
)

#tip-box[
*총 확산에 기여하는 성분:*
1. 초기 빔 크기 (항상 존재)
2. 기하학적 확산 (발산 × 거리)
3. MCS 산란 (깊이로 누적)
]

=== 산란력 (T)

#concept-box[
*산란력 T:* "산란력" T(z)는 깊이 z에서 산란이 얼마나 빠르게 일어나는지 알려줍니다:

$ T(z) = "d"sigma_theta^2 / "dz" $

*물리적 의미:* 깊이에 따른 각도 분산의 변화율
*단위:* mm당 라디안² (매우 작은 숫자!)
*비유:* T를 "산란 능력"이라고 생각해보세요 - 깊이 z의 재료가 빔을 얼마나 산란시키고 싶어하는지.
]

=== Fermi-Eyges 모멘트

빔 확산을 계산하기 위해 세 가지 "모멘트"를 추적합니다 - 경로를 따른 산란의 누적 척도:

$ A_0(z) = integral_0^z T(z') dif z' $ \
$ A_1(z) = integral_0^z z' times T(z') dif z' $ \
$ A_2(z) = integral_0^z z'^2 times T(z') dif z' $

#table(
  columns: (auto, 3fr, 2fr),
  inset: 8pt,
  align: left,
  table.header([모멘트], [물리적 의미], [공식에서의 역할]),
  [$A_0$], [누적된 총 각도 분산], [빔 방향이 얼마나 퍼졌는지],
  [$A_1$], [1차 공간 모멘트], [위치로 가중 - 초기 산란은 덜 중요],
  [$A_2$], [2차 공간 모멘트], [위치²로 가중 - 후기 산란이 더 중요],
)

#analogy-box[
*"가중된 역사" 비유:* 지출 습관을 추적하는 것을 상상해보세요:
- $A_0$ = 총 지출 (모든 거래가 동일하게 계산)
- $A_1$ = 시간으로 가중된 지출 (최근 지출이 더 중요)
- $A_2$ = 시간²로 가중된 지출 (매우 최근 지출이 훨씬 더 중요)

Fermi-Eyges는 다음과 같이 말합니다: 경로 후반부의 산란이 경로 시작부분의 산란보다 최종 빔 위치에 더 많은 영향을 줍니다.
]

=== 횡방향 분산 공식

$ sigma_x^2(z) = A_0 times z^2 - 2 times A_1 times z + A_2 $

#tip-box[
*직관적 이해:* 이 공식은 세 가지 효과를 결합합니다:
1. *$A_0 z^2$*: 거리²로 곱해진 각도 확산 - 넓은 각 × 긴 거리 = 넓은 빔
2. *$-2 A_1 z$*: 초기 산란에 대한 보정 - 초기 산란은 빔 위치에 영향을 줄 시간이 덜 있음
3. *$A_2$*: 순수 산란 기여 - 산란 사건으로 인한 직접적인 횡방향 변위

*결과:* 깊이 z에서의 순 횡방향 분산
]

=== 구현: 모멘트 누적

```cpp
// Highland 공식에서 산란력
__device__ float fermi_eyges_scattering_power(float E_MeV) {
    float sigma_theta = highland_sigma(E_MeV, 1.0f, X0_water);
    return sigma_theta * sigma_theta;  // T = σ²/mm
}

// 수송 중 모멘트 누적
struct FermiEygesMoments {
    double A0 = 0.0;  // 총 각도 분산
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
```

#concept-box[
*수치 적분:* 사각형 규칙을 사용하여 적분을 근사합니다:

$ integral_0^z T(z') dif z approx sum_i T(z_i) times Delta s $

각 스텝 i마다:
- 현재 에너지에서 산란력 T 계산
- 스텝 크기 ds를 곱함
- 적절한 모멘트(A0, A1 또는 A2)에 추가

*정확도:* 작은 스텝 크기(0.05-1mm)로 매우 정확합니다!
]

=== 세 가지 성분 횡방향 확산

#tip-box[
*총 횡방향 확산 = 세 가지 성분:* 최종 빔 폭은 세 가지 독립적인 출처에서 옵니다:
1. *초기 빔 확산* ("sigma_x0"): 빔이 어떤 폭으로 시작
2. *기하학적 확산* ("sigma_theta0" × z): 초기 발산이 거리로 증폭
3. *MCS 산란* ("sigma_mcs"): 수송 중 산란

이것들은 직교해서 더해집니다 (분산이 더해지지, 시그마가 아님):

$ sigma_("total")^2 = "sigma_x0"^2 + ("sigma_theta0" times z)^2 + "sigma_mcs"^2 $
]

```cpp
float total_lateral_sigma_squared(
    float "sigma_x0",      // 초기 빔 폭
    float "sigma_theta0",  // 초기 각도 확산
    float z,             // 깊이
    float "sigma_mcs"      // MCS 기여
) {
    // 초기 빔 확산 (거리로 발산)
    float sigma_initial = "sigma_x0";

    // 초기 발산에서 기하학적 확산
    float sigma_geometric = "sigma_theta0" * z;

    // MCS 기여 (Fermi-Eyges에서)
    float sigma_scattering = "sigma_mcs";

    // 총 분산 (직교 합)
    return sqrt(sigma_initial*sigma_initial +
                sigma_geometric*sigma_geometric +
                sigma_scattering*sigma_scattering);
}
```

#warning-box[
*중요: 분산을 더하세요, 시그마가 아님!*
*틀림:* $sigma_("total") = "sigma_x0" + "sigma_theta0" z + "sigma_mcs"$
*올바름:* $sigma_("total")^2 = "sigma_x0"^2 + ("sigma_theta0" z)^2 + "sigma_mcs"^2$

*왜?* 이것들은 독립적인 무작위 과정입니다. 그들의 분산이 더해지지, 표준 편차가 아닙니다!
]

---

== 6. 완전한 물리 파이프라인

=== 모든 것 통합하기

#concept-box[
*완전한 스텝:* 양성자 여정의 각 스텝에 대해 순서대로 모든 물리 효과를 계산합니다:
1. *스텝 크기 제어* (R 기반)
2. *에너지 손실* (R-LUT 사용)
3. *에너지 분산* (Bohr/Vavilov/Landau)
4. *에너지 퇴적* (환자에게 선량)
5. *다중 산란* (Highland 공식)
6. *핵 감쇠* (양성자 생존)
7. *위치 업데이트* (어디로 갔는가?)
8. *경계 확인* (시뮬레이션을 떠났는가?)

이 모든 것이 GPU에서 스텝당 마이크로초 안에 발생합니다!
]

=== 완전한 스텝 구현

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
    // 1. 스텝 크기 제어 (R 기반)
    float ds = compute_max_step_physics(E, lut);
    ds = fminf(ds, compute_boundary_step(x, z, dx, dz, theta));

    // 2. 에너지 손실 (결정론적)
    float E_out = compute_energy_after_step(E, ds, lut);

    // 3. 에너지 분산 (무작위)
    float dE_straggle = sample_energy_loss_with_straggling(E, ds, seed);
    E_out += dE_straggle;
    E_out = fmaxf(E_out, E_cutoff);

    // 4. 에너지 퇴적 (선량 그리드에)
    E_dep = E - E_out;

    // 5. MCS (무작위 방향 변경)
    float sigma_theta = highland_sigma(E, ds, X0_water);
    float delta_theta = sample_mcs_angle(sigma_theta, seed);
    theta += delta_theta;

    // 6. 핵 감쇠 (가중치 감소)
    apply_nuclear_attenuation(w, E_nuc_rem, E, ds);

    // 7. 위치 업데이트 (양성자 이동)
    x += ds * sin(theta);
    z += ds * cos(theta);

    // 8. 경계 확인 (탈출하는 양성자 추적)
    check_boundary_emission(x, z, dx, dz, boundary_flux);
}
```

---

== 물리 상수 참조

#figure(
  table(
    columns: (auto, auto, auto, 3fr),
    inset: 8pt,
    align: left,
    table.header([*상수*], [*값*], [*단위*], [*설명 및 임상적 관련성*]),
    [$m_p$], [938.272], [MeV/c²], [양성자 정지 질량 - beta, gamma 계산에 사용],
    [$m_e c^2$], [0.511], [MeV], [전자 정지 에너지 - 충돌당 최대 에너지 전달 설정],
    [$K$], [0.307], [MeV cm²/g], [바빌로프 상수 - 에너지 분산 계산에 사용],
    [$Z / A$ (물)], [0.555], [-], [물에 대한 원자 번호와 질량 수의 비율],
    [$X_0$ (물)], [360.8], [mm], [물의 방사선 길이 - 상당한 산란 전 양성자가 이동하는 거리],
    [$E_("cutoff")$], [0.1], [MeV], [에너지 차단 - 이 아래 양성자 종료],
    [$rho_("water")$], [1.0], [g/cm³], [물의 밀도 - 연조직과 같음 (좋은 근사!)],
  ),
  caption: [SM_2D에 사용된 물리 상수],
)

---

== 요약: 임상적 핵심 사항

=== 치료 계획을 위한 주요 물리학 개념

#tip-box[
*1. 다중 쿨롬 산란 (MCS)*
- *무엇:* 원자핵 근처를 지날 때 양성자가 옆으로 산란
- *영향:* 빔이 횡방향으로 퍼짐 - 필드 마진 필요
- *의존성:* 저에너지(브래그 피크 근처)에서 더 많은 산란

*2. 에너지 분산*
- *무엇:* 양성자 간 에너지 손실의 변동
- *영향:* 브래그 피크 번짐 - 덜 날카로운 선량 강하
- *의존성:* 경로 길이로 증가하는 분산

*3. 핵 감쇠*
- *무엇:* 핵 반응으로 일부 양성자 사라짐
- *영향:* 선량 감소 (~10cm당 1-3%) + 2차 방사선
- *의존성:* 저에너지에서 더 중요

*4. Fermi-Eyges 확산*
- *무엇:* 모든 산란에서 누적된 횡방향 빔 확산
- *영향:* 횡방향 펜umbra - 종양 커버리지를 위해 빔 확장 필요
- *의존성:* 깊이로 증가 (특히 사거리 끝 근처)
]

=== 일반적인 치료 계획 고려사항

#clinical-box[
*필드 마진:* 다음을 고려해야 함:
- *설정 불확실성:* 환자 위치 오차 (일반적으로 3-5mm)
- *사거리 불확실성:* CT 교정, stopping power 오차 (일반적으로 2-3%)
- *장기 운동:* 호흡, 방광 충전 (부위 종속적)
- *횡방향 산란:* 빔 확산 (Fermi-Eyges, 일반적으로 깊이에서 5-10mm)

*일반적인 마진 공식:* $ "Margin" = 2.5 Sigma + 0.5 " cm" $ (Sigma는 모든 불확실성 포함)

*브래그 피크 위치:* 올바르게 가져가는 것이 중요!
- *너무 얕음:* 종양 선량 부족, 이전 건강 조직 과다선량
- *너무 깊음:* 종양 선량 부족, 이후 건강 조직 과다선량
- *임상 실무:* 3-4 사거리 불확실성 마진 사용
]

---

== 참고문헌

1. *NIST PSTAR 데이터베이스* - 물에서 양성자의 stopping power 및 사거리 (금 표준)
2. *PDG 2024* - 입자 물리학 리뷰 (Highland 공식)
3. *ICRU 보고서 63* - 양성자 치료를 위한 핵 단면적
4. *Vavilov (1957)* - 에너지 분산 이론 (Bohr, Vavilov, Landau 영역)
5. *Fermi-Eyges* - 다중 산란 이론 및 횡방향 확산
6. *Bethe-Bloch* - 평균 에너지 손실 공식 (stopping power)
7. *ICRU 보고서 73* - 전자 및 양전자의 stopping power
8. *Gottschalk (2012)* - 방사선 치료 양성자의 산란력에 관하여

---

== 용어집

#table(
  columns: (auto, 4fr),
  inset: 8pt,
  align: left,
  table.header([*용어*], [*정의*]),
  [MCS], [다중 쿨롬 산란 - 원자핵에서 튕기는 양성자],
  [분산 (Straggling)], [양성자 간 에너지 손실의 통계적 변동],
  [CSDA], [연속 감속 근사 - 매끄러운 에너지 손실 가정],
  [브래그 피크], [양성자 사거리 끝의 날카로운 선량 피크 - 양성자의 주요 장점],
  [횡방향 확산], [빔 방향에 수직으로 옆으로 퍼지는 빔],
  [펜umbra], [빔 가장자리의 선량 강하 영역 - 횡방향 확산과 관련],
  [사거리], [양성자가 멈추기 전에 이동하는 거리 - 에너지 종속적],
  [Stopping Power], [단위 거리당 에너지 손실 - 재료 및 에너지 종속],
  [단면적], [핵 상호작용을 위한 유효 표적 면적],
  [분산 (Variance)], [확산 척도 (σ²) - 독립 과정의 경우 분산이 더해짐],
  [LET], [선 에너지 전달 - 단위 거리당 퇴적된 에너지],
)

---

#align(center)[
  *SM_2D 물리 모델: 향상된 문서*

  #text(size: 9pt)[버전 2.0.0 - 초보자 친화적 에디션]

  *질문이나 피드백은 메인 문서를 참조하십시오.*
]
