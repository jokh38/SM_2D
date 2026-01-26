#set text(font: "Times New Roman", size: 11pt)
#set page(numbering: "1")
#set par(justify: true)

= SM_2D Interface Reference

This document describes the interfaces for running and interacting with the SM_2D proton transport simulation.

== Overview

SM_2D is a high-performance C++ application with GPU acceleration. The primary interface is through configuration files (INI format) and command-line execution. Python utility scripts are provided for visualization.

== Running Simulations

=== Basic Execution

The simulation is executed via the compiled binary:

```bash
# Run simulation with default configuration (sim.ini)
./build/run_simulation

# Run with custom configuration file
./build/run_simulation path/to/config.ini

# Run with verbose output
./build/run_simulation --verbose
```

=== Configuration File Format

The simulation uses INI format configuration files. A complete example is provided in `sim.ini`.

== Configuration Sections

=== [particle] - Particle Properties

```ini
[particle]
type = proton          # Particle type: proton, electron, etc.
mass_amu = 1.0         # Particle mass in atomic mass units
charge_e = 1.0         # Particle charge in elementary charge units
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*Parameter*], [*Description*], [*Default*]),
  [`type`], [Particle type], [`proton`],
  [`mass_amu`], [Particle mass (amu)], [`1.0`],
  [`charge_e`], [Particle charge (e)], [`1.0`],
)

=== [beam] - Beam Configuration

```ini
[beam]
profile = gaussian     # Beam profile type
weight = 1.0           # Beam intensity
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*Parameter*], [*Description*], [*Default*]),
  [`profile`], [pencil, gaussian, flat, custom], [`gaussian`],
  [`weight`], [Beam weight/intensity], [`1.0`],
)

=== [energy] - Energy Settings

```ini
[energy]
mean_MeV = 190.0      # Mean/central energy (MeV)
sigma_MeV = 1.0       # Energy spread (MeV)
min_MeV = 0.0         # Minimum cutoff
max_MeV = 250.0       # Maximum cutoff
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*Parameter*], [*Description*], [*Default*]),
  [`mean_MeV`], [Central energy (MeV)], [`190.0`],
  [`sigma_MeV`], [Energy spread (MeV)], [`1.0`],
  [`min_MeV`], [Minimum cutoff (MeV)], [`0.0`],
  [`max_MeV`], [Maximum cutoff (MeV)], [`250.0`],
)

=== [spatial] - Spatial Position

```ini
[spatial]
x0_mm = 50.0          # Central X position (mm)
z0_mm = 0.0           # Central Z position (mm)
sigma_x_mm = 0.033    # Spatial spread X (mm)
sigma_z_mm = 0.01     # Spatial spread Z (mm)
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*Parameter*], [*Description*], [*Default*]),
  [`x0_mm`], [Central X position (mm)], [`50.0`],
  [`z0_mm`], [Central Z position (mm)], [`0.0`],
  [`sigma_x_mm`], [X spatial spread (mm)], [`0.033`],
  [`sigma_z_mm`], [Z spatial spread (mm)], [`0.01`],
)

=== [angular] - Angular Settings

```ini
[angular]
theta0_rad = 0.0      # Central angle (radians)
sigma_theta_rad = 0.001  # Angular divergence (radians)
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*Parameter*], [*Description*], [*Default*]),
  [`theta0_rad`], [Central angle (rad)], [`0.0`],
  [`sigma_theta_rad`], [Angular divergence (rad)], [`0.001`],
)

=== [grid] - Simulation Grid

```ini
[grid]
Nx = 200              # Transverse bins
Nz = 640              # Depth bins
dx_mm = 0.5           # Transverse spacing (mm)
dz_mm = 0.5           # Depth spacing (mm)
max_steps = 200       # Maximum simulation steps
```

#table(
  columns: (auto, 2fr, auto),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*Parameter*], [*Description*], [*Default*]),
  [`Nx`], [Transverse bins], [`200`],
  [`Nz`], [Depth bins], [`640`],
  [`dx_mm`], [Transverse spacing (mm)], [`0.5`],
  [`dz_mm`], [Depth spacing (mm)], [`0.5`],
  [`max_steps`], [Maximum steps], [`200`],
)

=== [output] - Output Configuration

```ini
[output]
output_dir = results  # Output directory
dose_2d_file = dose_2d.txt
pdd_file = pdd.txt
let_file = ""         # Empty = don't output LET
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
  table.header([*Parameter*], [*Description*], [*Default*]),
  [`output_dir`], [Output directory path], [`results`],
  [`dose_2d_file`], [2D dose output filename], [`dose_2d.txt`],
  [`pdd_file`], [PDD output filename], [`pdd.txt`],
  [`let_file`], [LET output (empty=off)], [`""`],
  [`normalize_dose`], [Normalize to maximum], [`true`],
  [`save_2d`], [Save 2D dose], [`true`],
  [`save_pdd`], [Save PDD], [`true`],
)

== Python Visualization Tools

=== visualize.py

Standalone Python script for visualizing simulation results.

#block(
  fill: rgb("#e0f0ff"),
  inset: 10pt,
  radius: 5pt,
  stroke: 1pt + rgb("#0066cc"),
  [
    *Usage Example*

    ```bash
    python visualize.py
    ```

    The script reads from `results/` directory and generates plots.
  ]
)

==== Available Functions

#table(
  columns: (auto, 2fr),
  inset: 10pt,
  align: (left, center),
  stroke: 0.5pt + gray,
  table.header([*Function*], [*Description*]),
  [`load_pdd(filepath)`], [Load depth-dose data from file],
  [`load_dose_2d(filepath)`], [Load 2D dose distribution from file],
  [`plot_pdd(depths, doses, output_path)`], [Plot depth-dose curve],
  [`plot_dose_2d(x_vals, z_vals, dose_grid, output_path)`], [Plot 2D dose heatmap],
  [`plot_combined_panel(...)`], [Generate 3x2 panel with all plots],
)

==== Output Files

- `pdd_plot.png` - Depth-dose curve
- `dose_2d_plot.png` - 2D dose distribution (raw + normalized)
- `combined_plot.png` - Combined panel with PDD and profiles

=== batch_run.py

Parameter sweep automation for running multiple simulations.

```bash
python batch_run.py
```

Uses a YAML configuration file to define parameter sweeps:

```yaml
template: sim.ini
sweep:
  energy:
    section: [energy]
    mean_MeV: [100, 120, 140, 160, 180]
```

=== batch_plot.py

Batch visualization for multiple simulation results.

```bash
python batch_plot.py
```

== Output File Formats

=== Dose 2D File

Text format with three columns:

```
# x_mm z_mm dose_Gy
0.0 0.0 0.000
0.5 0.0 0.001
...
```

=== PDD File

Text format with two columns:

```
# depth_mm dose_Gy
0.0 0.000
0.5 0.012
1.0 0.045
...
```

== Built-in Physics Models

The following physics models are implemented and automatically applied (not user-selectable):

=== Energy Loss

- *Bethe-Bloch*: Mean energy loss (always used)
- *Bohr Straggling*: Energy loss fluctuations
- *Vavilov Regime*: Full straggling model with kappa detection

=== Multiple Coulomb Scattering

- *Highland Formula*: PDG 2024 implementation
- *2D Projection*: Proper σ_2D = σ_3D / √2 correction
- *Variance Accumulation*: Correct multi-step scattering

=== Nuclear Interactions

- *ICRU 63 Cross-sections*: Energy-dependent nuclear attenuation
- *Survival Probability*: exp(-Σ × ds)
- *Energy Budget Tracking*: Removed energy is audited

=== Lateral Spread

- *Fermi-Eyges Theory*: A0, A1, A2 moment calculation
- *Scattering Power*: T(z) = dσ_θ²/dz

== Coordinate System

#block(
  fill: rgb("#fff0cc"),
  inset: 10pt,
  radius: 5pt,
  stroke: 1pt + rgb("#ff9900"),
  [
    *Coordinate Convention*

    - X: Transverse (lateral) position [mm]
    - Z: Depth position [mm] (beam direction)
    - θ: Angle in X-Z plane [radians]
    - Units are **millimeters** for all positions
  ]
)

== Exit Codes

#table(
  columns: (auto, 2fr, 1fr),
  inset: 10pt,
  align: (left, center, center),
  stroke: 0.5pt + gray,
  table.header([*Code*], [*Meaning*], [*Severity*]),
  [`0`], [Success], [Info],
  [`1`], [Configuration error], [Error],
  [`2`], [File I/O error], [Error],
  [`3`], [CUDA/GPU error], [Error],
  [`4`], [Physics computation error], [Error],
)

== Bibliography

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
