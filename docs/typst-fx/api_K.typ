#set text(font: "Malgun Gothic", size: 11pt)
#set page(numbering: "1")
#set par(justify: true)

= SM_2D 인터페이스 참조

이 문서는 SM_2D 양성자 수송 시뮬레이션을 실행하고 상호작용하는 인터페이스를 설명합니다.

== 개요

SM_2D는 GPU 가속을 지원하는 고성능 C++ 응용 프로그램입니다. 주요 인터페이스는 구성 파일(INI 형식)과 명령줄 실행을 통해 제공됩니다. 시각화를 위한 Python 유틸리티 스크립트가 제공됩니다.

== 시뮬레이션 실행

=== 기본 실행

시뮬레이션은 컴파일된 바이너리를 통해 실행됩니다:

```bash
# 기본 구성으로 시뮬레이션 실행 (sim.ini)
./build/run_simulation

# 사용자 정의 구성 파일로 실행
./build/run_simulation path/to/config.ini

# 상세 출력 모드로 실행
./build/run_simulation --verbose
```

=== 구성 파일 형식

시뮬레이션은 INI 형식의 구성 파일을 사용합니다. `sim.ini`에 전체 예제가 제공됩니다.

== 구성 섹션

=== [particle] - 입자 속성

```ini
[particle]
type = proton          # 입자 유형: proton, electron 등
mass_amu = 1.0         # 입자 질량 (原子質量 단위)
charge_e = 1.0         # 입자 전하 (기본 전하 단위)
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*매개변수*], [*설명*], [*기본값*]),
  [`type`], [입자 유형], [`proton`],
  [`mass_amu`], [입자 질량 (amu)], [`1.0`],
  [`charge_e`], [입자 전하 (e)], [`1.0`],
)

=== [beam] - 빔 구성

```ini
[beam]
profile = gaussian     # 빔 프로필 유형
weight = 1.0           # 빔 강도
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*매개변수*], [*설명*], [*기본값*]),
  [`profile`], [pencil, gaussian, flat, custom], [`gaussian`],
  [`weight`], [빔 가중치/강도], [`1.0`],
)

=== [energy] - 에너지 설정

```ini
[energy]
mean_MeV = 190.0      # 평균/중심 에너지 (MeV)
sigma_MeV = 1.0       # 에너지 분산 (MeV)
min_MeV = 0.0         # 최소 컷오프
max_MeV = 250.0       # 최대 컷오프
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*매개변수*], [*설명*], [*기본값*]),
  [`mean_MeV`], [중심 에너지 (MeV)], [`190.0`],
  [`sigma_MeV`], [에너지 분산 (MeV)], [`1.0`],
  [`min_MeV`], [최소 컷오프 (MeV)], [`0.0`],
  [`max_MeV`], [최대 컷오프 (MeV)], [`250.0`],
)

=== [spatial] - 공간 위치

```ini
[spatial]
x0_mm = 50.0          # 중심 X 위치 (mm)
z0_mm = 0.0           # 중심 Z 위치 (mm)
sigma_x_mm = 0.033    # X 공간 분산 (mm)
sigma_z_mm = 0.01     # Z 공간 분산 (mm)
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*매개변수*], [*설명*], [*기본값*]),
  [`x0_mm`], [중심 X 위치 (mm)], [`50.0`],
  [`z0_mm`], [중심 Z 위치 (mm)], [`0.0`],
  [`sigma_x_mm`], [X 공간 분산 (mm)], [`0.033`],
  [`sigma_z_mm`], [Z 공간 분산 (mm)], [`0.01`],
)

=== [angular] - 각도 설정

```ini
[angular]
theta0_rad = 0.0      # 중심 각도 (라디안)
sigma_theta_rad = 0.001  # 각도 발산 (라디안)
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*매개변수*], [*설명*], [*기본값*]),
  [`theta0_rad`], [중심 각도 (rad)], [`0.0`],
  [`sigma_theta_rad`], [각도 발산 (rad)], [`0.001`],
)

=== [grid] - 시뮬레이션 그리드

```ini
[grid]
Nx = 200              # 횡방향 빈
Nz = 640              # 깊이 빈
dx_mm = 0.5           # 횡방향 간격 (mm)
dz_mm = 0.5           # 깊이 간격 (mm)
max_steps = 200       # 최대 시뮬레이션 단계
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*매개변수*], [*설명*], [*기본값*]),
  [`Nx`], [횡방향 빈], [`200`],
  [`Nz`], [깊이 빈], [`640`],
  [`dx_mm`], [횡방향 간격 (mm)], [`0.5`],
  [`dz_mm`], [깊이 간격 (mm)], [`0.5`],
  [`max_steps`], [최대 단계 수], [`200`],
)

=== [output] - 출력 구성

```ini
[output]
output_dir = results  # 출력 디렉토리
dose_2d_file = dose_2d.txt
pdd_file = pdd.txt
let_file = ""         # 비어있음 = LET 출력 안 함
format = txt          # txt, csv, hdf5
normalize_dose = true
save_2d = true
save_pdd = true
save_lat_profiles = false
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*매개변수*], [*설명*], [*기본값*]),
  [`output_dir`], [출력 디렉토리 경로], [`results`],
  [`dose_2d_file`], [2D 선량 출력 파일명], [`dose_2d.txt`],
  [`pdd_file`], [PDD 출력 파일명], [`pdd.txt`],
  [`let_file`], [LET 출력 (비어있음=비활성)], [`""`],
  [`normalize_dose`], [최대값으로 정규화], [`true`],
  [`save_2d`], [2D 선량 저장], [`true`],
  [`save_pdd`], [PDD 저장], [`true`],
)

== Python 시각화 도구

=== visualize.py

시뮬레이션 결과 시각화를 위한 독립형 Python 스크립트입니다.

#block(
  fill: rgb("#e0f0ff"),
  inset: 10pt,
  radius: 5pt,
  stroke: 1pt + rgb("#0066cc"),
  [
    *사용 예시*

    ```bash
    python visualize.py
    ```

    이 스크립트는 `results/` 디렉토리에서 읽고 플롯을 생성합니다.
  ]
)

==== 사용 가능한 함수

#table(
  columns: (auto, 2fr),
  inset: 10pt,
  align: (left, center),
  stroke: 0.5pt + gray,
  table.header([*함수*], [*설명*]),
  [`load_pdd(filepath)`], [파일에서 깊이-선량 데이터 로드],
  [`load_dose_2d(filepath)`], [파일에서 2D 선량 분포 로드],
  [`plot_pdd(depths, doses, output_path)`], [깊이-선량 곡선 플롯],
  [`plot_dose_2d(x_vals, z_vals, dose_grid, output_path)`], [2D 선량 히트맵 플롯],
  [`plot_combined_panel(...)`], [모든 플롯이 포함된 3x2 패널 생성],
)

==== 출력 파일

- `pdd_plot.png` - 깊이-선량 곡선
- `dose_2d_plot.png` - 2D 선량 분포 (원본 + 정규화)
- `combined_plot.png` - PDD 및 프로필이 포함된 결합 패널

=== batch_run.py

다중 시뮬레이션 실행을 위한 매개변수 스윕 자동화입니다.

```bash
python batch_run.py
```

매개변수 스윕을 정의하기 위해 YAML 구성 파일을 사용합니다:

```yaml
template: sim.ini
sweep:
  energy:
    section: [energy]
    mean_MeV: [100, 120, 140, 160, 180]
```

=== batch_plot.py

다중 시뮬레이션 결과를 위한 배치 시각화입니다.

```bash
python batch_plot.py
```

== 출력 파일 형식

=== Dose 2D 파일

세 개 열로 구성된 텍스트 형식:

```
# x_mm z_mm dose_Gy
0.0 0.0 0.000
0.5 0.0 0.001
...
```

=== PDD 파일

두 개 열로 구성된 텍스트 형식:

```
# depth_mm dose_Gy
0.0 0.000
0.5 0.012
1.0 0.045
...
```

== 내장된 물리 모델

다음 물리 모델이 구현되어 자동으로 적용됩니다 (사용자 선택 불가):

=== 에너지 손실

- *Bethe-Bloch*: 평균 에너지 손실 (항상 사용)
- *Bohr Straggling*: 에너지 손실 변동
- *Vavilov Regime*: 카파 감지를 포함한 완전한 straggling 모델

=== 다중 쿨롱 산란

- *Highland Formula*: PDG 2024 구현
- *2D Projection*: 올바른 σ_2D = σ_3D / √2 보정
- *Variance Accumulation*: 올바른 다단계 산란

=== 핵 상호작용

- *ICRU 63 단면적*: 에너지 의존성 핵 감쇠
- *생존 확률*: exp(-Σ × ds)
- *에너지 예산 추적*: 제거된 에너지 감사

=== 횡방향 확산

- *Fermi-Eyges Theory*: A0, A1, A2 모먼트 계산
- *Scattering Power*: T(z) = dσ_θ²/dz

== 좌표계

#block(
  fill: rgb("#fff0cc"),
  inset: 10pt,
  radius: 5pt,
  stroke: 1pt + rgb("#ff9900"),
  [
    *좌표 규약*

    - X: 횡방향 (측면) 위치 [mm]
    - Z: 깊이 위치 [mm] (빔 방향)
    - θ: X-Z 평면 내 각도 [라디안]
    - 모든 위치의 단위는 **밀리미터**입니다
  ]
)

== 종료 코드

#table(
  columns: (auto, 2fr, 1fr),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*코드*], [*의미*], [*심각도*]),
  [`0`], [성공], [정보],
  [`1`], [구성 오류], [오류],
  [`2`], [파일 I/O 오류], [오류],
  [`3`], [CUDA/GPU 오류], [오류],
  [`4`], [물리 계산 오류], [오류],
)

== 참고문헌

#block(
  fill: rgb("#f0f0f0"),
  inset: 10pt,
  radius: 5pt,
  [
    1. B. Gottschalk, *On the scattering power of radiotherapy protons*, Med. Phys. 37 (2010)

    2. ICRU Report 63, *Nuclear Interactions*

    3. PDG 2024, *Highland Formula for Multiple Coulomb Scattering*

    4. Vavilov (1957), *Energy Straggling Theory*
  ]
)
