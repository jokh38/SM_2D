#!/usr/bin/env python3
"""
Quick physics validation tests for proton_transport_water.py

Tests:
1. Bethe-Bloch stopping power at reference energies
2. CSDA range vs NIST data
3. Highland scattering angle calculation
4. Energy conservation
"""

import sys
sys.path.insert(0, '/workspaces/SM_2D/validation')

import numpy as np
from proton_transport_water import BetheBloch, FermiEyges, ProtonTransport, RHO_WATER


def test_bethe_bloch():
    """Test Bethe-Bloch stopping power at reference energies."""
    print("=" * 60)
    print("BETHE-BLOCH STOPPING POWER VALIDATION")
    print("=" * 60)

    # Reference values from NIST PSTAR (approximate)
    # dE/dx [MeV·cm²/g]
    reference = {
        10: 46.7,   # MeV
        50: 6.3,
        100: 3.5,
        150: 2.7,
        200: 2.3,
    }

    print("\nEnergy [MeV] | dE/dx [MeV·cm²/g] | Ref [MeV·cm²/g] | Error [%]")
    print("-" * 65)

    for E, ref_dedx in reference.items():
        # Convert our output to same units
        dedx_mm = BetheBloch.stopping_power(E)
        dedx_cm2_g = dedx_mm * 10 / RHO_WATER  # Convert [MeV/mm] to [MeV·cm²/g]

        error = 100 * (dedx_cm2_g - ref_dedx) / ref_dedx

        print(f"{E:11.1f} | {dedx_cm2_g:15.3f} | {ref_dedx:14.1f} | {error:10.1f}")

    print()


def test_csda_range():
    """Test CSDA range calculation vs NIST data."""
    print("=" * 60)
    print("CSDA RANGE VALIDATION")
    print("=" * 60)

    # NIST PSTAR reference values
    reference = {
        50: 22.67,   # Energy [MeV] -> Range [mm]
        70: 40.75,
        100: 77.12,
        150: 158.31,
        200: 278.52,
    }

    print("\nEnergy [MeV] | Range Calc [mm] | Range NIST [mm] | Error [%]")
    print("-" * 62)

    max_error = 0
    for E, ref_range in reference.items():
        calc_range = BetheBloch.range_integral(E)
        error = 100 * (calc_range - ref_range) / ref_range
        max_error = max(max_error, abs(error))

        print(f"{E:11.1f} | {calc_range:14.2f} | {ref_range:13.2f} | {error:10.2f}")

    print(f"\nMaximum error: {max_error:.2f}%")

    if max_error < 2.0:
        print("✓ PASS: All ranges within 2% of NIST data")
    else:
        print("✗ FAIL: Some ranges exceed 2% error")

    print()


def test_highland_formula():
    """Test Highland scattering angle calculation."""
    print("=" * 60)
    print("HIGHLAND FORMULA VALIDATION")
    print("=" * 60)

    # Test cases: Energy [MeV], step [mm]
    test_cases = [
        (150, 1.0),   # High energy, 1 mm
        (150, 10.0),  # High energy, 10 mm
        (50, 1.0),    # Medium energy, 1 mm
        (10, 0.1),    # Low energy, 0.1 mm
    ]

    print("\nEnergy [MeV] | Step [mm] | σ_θ [mrad]")
    print("-" * 40)

    for E, step in test_cases:
        sigma_theta = FermiEyges.scattering_angle_sigma(E, step)
        sigma_mrad = sigma_theta * 1000  # Convert rad to mrad
        print(f"{E:11.1f} | {step:8.2f} | {sigma_mrad:10.3f}")

    print()


def test_energy_conservation():
    """Test energy conservation in simulation."""
    print("=" * 60)
    print("ENERGY CONSERVATION TEST")
    print("=" * 60)

    # Small test simulation
    n_protons = 100
    energy = 100.0  # MeV

    sim = ProtonTransport(
        energy=energy,
        grid_size_x=20.0,
        grid_size_z=100.0,
        grid_res=1.0,
        debug=False
    )

    sim.run_simulation(n_protons)

    energy_in = sim.stats['total_energy_in']
    energy_dep = sim.stats['total_energy_dep']

    error = abs(energy_in - energy_dep) / energy_in * 100

    print(f"\nInitial energy: {energy_in:.3f} MeV")
    print(f"Deposited energy: {energy_dep:.3f} MeV")
    print(f"Relative error: {error:.6f}%")

    if error < 1e-10:
        print("✓ PASS: Energy conserved to machine precision")
    else:
        print("✗ FAIL: Energy not conserved")

    print()


def test_bragg_peak_position():
    """Test Bragg peak position vs CSDA range."""
    print("=" * 60)
    print("BRAGG PEAK POSITION VALIDATION")
    print("=" * 60)

    # Test multiple energies
    energies = [50, 70, 100, 150]

    print("\nEnergy [MeV] | Bragg Peak [mm] | CSDA Range [mm] | Error [%]")
    print("-" * 60)

    for E in energies:
        # Small simulation for speed
        sim = ProtonTransport(
            energy=E,
            grid_size_x=10.0,
            grid_size_z=BetheBloch.range_integral(E) * 1.2,
            grid_res=1.0,
            debug=False
        )

        sim.run_simulation(200)  # Small number for speed

        bragg_z = sim.stats['bragg_peak_z']
        csda_range = BetheBloch.range_integral(E)
        error = 100 * (bragg_z - csda_range) / csda_range

        print(f"{E:11.1f} | {bragg_z:13.2f} | {csda_range:13.2f} | {error:10.2f}")

    print("\nNote: Bragg peak should be within ±2% of CSDA range")
    print()


def main():
    """Run all validation tests."""
    print("\n")
    print("*" * 60)
    print("* PROTON TRANSPORT PHYSICS VALIDATION")
    print("*" * 60)
    print("\n")

    # Run tests
    test_bethe_bloch()
    test_csda_range()
    test_highland_formula()
    test_energy_conservation()
    test_bragg_peak_position()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()
