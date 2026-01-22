# SM_2D: Proton Therapy Monte Carlo Simulation

A 2D Monte Carlo simulation for proton therapy dose calculation.

## Features

- **Centralized Configuration**: Single `sim.ini` file controls all simulation parameters
- **Multiple Beam Types**: Pencil beam, Gaussian beam, scattered beam
- **Output Formats**: 2D dose distribution, depth-dose (PDD) curves
- **Visualization**: Python-based plotting tool for results

## Project Structure

```
SM_2D/
├── sim.ini                 # Main configuration file
├── run_simulation.cpp      # Main simulation entry point
├── visualize.py            # Visualization tool
├── CMakeLists.txt          # Build configuration
├── include/                # Header files
│   ├── core/              # Core data structures (grids, config, etc.)
│   ├── validation/        # Validation & analysis (pencil_beam, bragg_peak)
│   ├── source/            # Beam source definitions
│   └── ...
├── src/                   # Implementation files
├── cuda/                  # CUDA kernels for GPU acceleration
├── tests/                 # Unit tests
└── results/               # Simulation output (auto-created)
```

## Configuration (sim.ini)

The simulation is controlled by a single INI-style configuration file:

```ini
[particle]
type = proton              # Particle type
mass_amu = 1.0             # Mass in atomic mass units
charge_e = 1.0             # Charge in elementary charge units

[beam]
profile = pencil           # Beam profile: pencil, gaussian
weight = 1.0               # Total weight (normalized dose)

[energy]
mean_MeV = 150.0          # Mean energy (MeV)
sigma_MeV = 0.0           # Energy spread (0 = monoenergetic)
min_MeV = 0.0             # Minimum energy cutoff
max_MeV = 250.0           # Maximum energy cutoff

[spatial]
x0_mm = 50.0              # Central x position (mm)
z0_mm = 0.0               # Central z position (mm)
sigma_x_mm = 0.0          # Spatial spread (mm)
sigma_z_mm = 0.0          # Spatial spread (mm)

[angular]
theta0_rad = 0.0          # Central angle (radians)
sigma_theta_rad = 0.0     # Angular divergence (radians)

[grid]
Nx = 100                  # Transverse bins
Nz = 200                  # Depth bins
dx_mm = 1.0               # Transverse spacing (mm)
dz_mm = 1.0               # Depth spacing (mm)
max_steps = 100           # Maximum simulation steps

[output]
output_dir = results      # Output directory
dose_2d_file = dose_2d.txt
pdd_file = pdd.txt
normalize_dose = true     # Normalize to maximum dose
save_2d = true            # Save 2D dose distribution
save_pdd = true           # Save depth-dose curve
```

## Building

```bash
cd build
cmake ..
make run_simulation
```

## Usage

### Running Simulation

```bash
# Use default config (sim.ini)
./build/run_simulation

# Use custom config file
./build/run_simulation custom_config.ini
```

### Visualizing Results

```bash
python3 visualize.py
```

This generates plots in the `results/` directory:
- `pdd_plot.png` - Depth-dose curve
- `dose_2d_plot.png` - 2D dose heatmap
- `combined_plot.png` - Combined view

## Output Files

Results are saved to the `results/` directory:

| File | Description |
|------|-------------|
| `dose_2d.txt` | 2D dose distribution (x, z, dose, normalized) |
| `pdd.txt` | Depth-dose curve with Bragg peak info |

## Example Results

For a 150 MeV proton pencil beam:
- **Bragg Peak Depth**: ~157 mm in water
- **Range**: Accurate to NIST PSTAR data
- **Lateral Spread**: Calculated using multiple Coulomb scattering theory

## Dependencies

- C++17 compiler
- CUDA 12.4 (for GPU kernels)
- CMake 3.28+
- Python 3 + matplotlib + numpy (for visualization)

## Code Overview

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Configuration | `incident_particle_config.hpp` | Unified configuration structure |
| Grid | `grids.hpp/cpp` | Spatial grid definitions |
| Validation | `pencil_beam.hpp/cpp` | Pencil beam simulation |
| Bragg Peak | `bragg_peak.hpp/cpp` | Peak finding & analysis |
| Config Loader | `config_loader.hpp` | INI file parsing |

### Main Entry Point

`run_simulation.cpp`:
1. Loads configuration from `sim.ini`
2. Validates parameters
3. Runs Monte Carlo simulation
4. Saves results to `results/`

### Visualization Tool

`visualize.py`:
1. Loads output files from `results/`
2. Generates publication-quality plots
3. Saves PNG images

## License

MIT License
