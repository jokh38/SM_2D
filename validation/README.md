# Proton Transport Validation Scripts

This directory contains Python scripts for simulating and visualizing proton transport in water using analytical physics models.

## Scripts

### 1. `proton_transport_water.py`

Main simulation script that implements:
- **Bethe-Bloch formula** for stopping power (energy loss)
- **Highland formula** for multiple Coulomb scattering
- **Fermi-Eyges theory** for lateral spread calculation
- **CSDA range** integration for validation

#### Physics Models

**Bethe-Bloch Stopping Power:**
```
dE/dx = K * z² * (Z/A) * (1/β²) * [ln(2*m_e*c²*β²*γ²/I) - β² - δ/2]
```

Where:
- K = 0.307075 MeV·cm²/g
- z = 1 (proton charge)
- Z/A = 0.555 (for water)
- I = 75 eV (mean excitation energy)
- δ = density correction

**Highland Formula (MCS):**
```
σ_θ = (13.6 MeV)/(β*p*c) * z * sqrt(x/X0) * [1 + 0.038*ln(x/X0)]
```

Where:
- X0 = 36.08 g/cm² (radiation length for water)
- p = momentum [MeV/c]

#### Usage

```bash
# Basic usage (150 MeV, 10000 protons)
python proton_transport_water.py --energy 150 --n-protons 10000

# Custom grid resolution
python proton_transport_water.py --energy 150 --n-protons 10000 --grid-res 0.5

# Different energy
python proton_transport_water.py --energy 70 --n-protons 50000

# Debug mode (verbose output)
python proton_transport_water.py --energy 150 --n-protons 100 --debug

# Custom domain size
python proton_transport_water.py --energy 150 --n-protons 10000 \
    --grid-x 100 --grid-z 300
```

#### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--energy` | float | 150.0 | Initial proton energy [MeV] |
| `--n-protons` | int | 10000 | Number of protons to simulate |
| `--grid-res` | float | 0.5 | Grid resolution [mm] |
| `--grid-x` | float | 50.0 | Lateral grid half-width [mm] |
| `--grid-z` | float | 200.0 | Depth grid extent [mm] |
| `--debug` | flag | False | Enable debug output |
| `--output` | str | (auto) | Output CSV filename |

#### Output

The script generates:
1. **CSV file**: `proton_dose_E{energy}MeV.csv` with columns:
   - `x[mm]`: Lateral position
   - `z[mm]`: Depth position
   - `dose[Gy]`: Absorbed dose

2. **Console summary**:
   - Bragg peak position vs CSDA range
   - Maximum dose
   - Lateral scattering (σₓ)
   - Energy conservation check

#### Validation Results

For 150 MeV protons in water:
- **Bragg peak**: ~157 mm (theory: 158.3 mm, error: -0.8%)
- **Lateral σₓ** (mid-range): ~1.0 mm
- **Energy conservation**: 100% (by construction)

For 70 MeV protons in water:
- **Bragg peak**: ~41 mm (theory: 40.8 mm, error: +0.6%)
- **Lateral σₓ** (mid-range): ~1.0 mm

---

### 2. `plot_proton_dose.py`

Visualization script for dose distributions.

#### Usage

```bash
# Interactive plot (requires display)
python plot_proton_dose.py proton_dose_E150MeV.csv

# Save to PNG file
python plot_proton_dose.py proton_dose_E150MeV.csv --save
```

#### Output

Generates a three-panel figure:
1. **2D dose heatmap**: Color-coded dose distribution
2. **Bragg peak**: Depth dose profile with peak location
3. **Lateral profile**: Dose vs lateral position at Bragg peak depth

---

## Physics Validation

### CSDA Range Comparison

The simulated Bragg peak positions agree with NIST PSTAR data:

| Energy [MeV] | Simulated Range [mm] | NIST Range [mm] | Error |
|--------------|---------------------|-----------------|-------|
| 70 | 41.0 | 40.8 | +0.6% |
| 150 | 157.0 | 158.3 | -0.8% |

### Energy Conservation

The simulation conserves energy exactly:
- Energy input = N_protons × E_initial
- Energy deposited = Σ ΔE (tracked during simulation)
- Conservation: 100.0% (by construction)

### Lateral Scattering

The Highland formula provides reasonable estimates of lateral spread:
- σₓ ~ 1 mm at mid-range for 150 MeV protons
- Consistent with Fermi-Eyges theory predictions

---

## Implementation Notes

### Step Size Control

The simulation uses energy-dependent step sizing:
- E < 10 MeV: Δs = 0.05 mm (fine near Bragg peak)
- 10 ≤ E < 50 MeV: Δs = 0.1 mm
- E ≥ 50 MeV: Δs = 0.2 mm

This ensures accurate energy deposition in the high-stopping-power region.

### Edge Cases

The code handles:
- Low-energy cutoff (0.1 MeV)
- Boundary crossings
- Small step sizes (density correction)
- Non-relativistic kinematics

### Unit Conversions

All internal calculations use consistent units:
- Energy: MeV
- Distance: mm
- Mass: MeV/c²
- Stopping power: MeV/mm
- Dose: Gy (J/kg)

---

## References

1. **Bethe-Bloch**: ICRU Report 73, "Stopping Powers for Electrons and Positrons"
2. **Highland Formula**: NIM 129 (1975) 497-499
3. **NIST PSTAR**: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
4. **Fermi-Eyges**: Rev. Mod. Phys. 20, 126 (1948)

---

## Comparison with GPU Implementation

This Python script serves as a validation reference for the GPU transport solver in `src/`. Key similarities:

| Feature | Python Script | GPU Solver |
|---------|--------------|------------|
| Energy loss | Bethe-Bloch (dE/dx) | CSDA R(E) table |
| MCS | Highland formula | Highland formula |
| Step control | Energy-dependent | R-based (2% range) |
| Angular split | None (single angle) | 7-point quadrature |
| Statistics | Monte Carlo | Deterministic |

The Python script provides:
- Ground truth for Bragg peak positions
- Validation of scattering models
- Unit test reference values

---

## Future Improvements

- [ ] Add range straggling (Landau/Vavilov)
- [ ] Implement delta-ray transport
- [ ] Add nuclear interaction models
- [ ] Support multi-material phantoms
- [ ] GPU comparison plots
- [ ] Automated regression tests
