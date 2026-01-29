#!/usr/bin/env python3
"""
Proton Transport in Water using Bethe-Bloch and Fermi-Eyges Theory

This script simulates proton beam transport through liquid water using:
- Bethe-Bloch formula for stopping power (energy loss)
- Fermi-Eyges theory for multiple Coulomb scattering (lateral spread)

The simulation tracks individual protons through a water phantom, computing
energy deposition and lateral scattering at each step.

Usage:
    python proton_transport_water.py --energy 150 --n-protons 10000
    python proton_transport_water.py --energy 70 --n-protons 50000 --grid-res 1.0
    python proton_transport_water.py --energy 100 --n-protons 10000 --debug

Output:
    CSV file: proton_dose_E{energy}MeV.csv with columns: x[mm], z[mm], dose[Gy]

References:
    - NIST PSTAR database for stopping power and range
    - ICRU Report 73 for stopping power constants
    - Highland formula for MCS (NIM 129 (1975) 497-499)
"""

import numpy as np
import argparse
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Fundamental constants
M_E = 0.5109989461      # Electron mass [MeV/c²]
M_P = 938.27208816      # Proton mass [MeV/c²]
C_LIGHT = 299.792458    # Speed of light [mm/ns] (not directly used, for ref)

# Bethe-Bloch constants
K_BB = 0.307075         # K constant [MeV·cm²/g]

# Water properties (H2O, liquid)
RHO_WATER = 1.0         # Density [g/cm³]
Z_WATER = 10.0          # Atomic number (2*H + O = 2*1 + 8)
A_WATER = 18.01528      # Atomic mass [g/mol]
Z_OVER_A = 0.555        # Z/A ratio for water
I_MEAN = 75e-6          # Mean excitation energy [MeV] (75 eV)

# Radiation length for water
X0_WATER_GCM2 = 36.08   # Radiation length [g/cm²]
X0_WATER_MM = X0_WATER_GCM2 * 10 / RHO_WATER  # [mm]

# Simulation constants
E_CUTOFF = 0.1          # Energy cutoff [MeV]
STEP_MAX_MM = 1.0       # Maximum step size [mm]
STEP_MIN_MM = 0.01      # Minimum step size [mm]


# ============================================================================
# BETHE-BLOCH STOPPING POWER
# ============================================================================

class BetheBloch:
    """
    Bethe-Bloch stopping power calculator for protons in water.

    The Bethe-Bloch formula describes the energy loss of charged particles
    due to ionization and atomic excitation:

    dE/dx = K * z² * (Z/A) * (1/β²) * [ln(2*m_e*c²*β²*γ²/I) - β² - δ/2]

    Where:
        K = 0.307075 MeV·cm²/g
        z = particle charge (1 for protons)
        Z/A = 0.555 for water
        I = 75 eV (mean excitation energy for water)
        δ = density correction (important at high energy)
    """

    @staticmethod
    def _compute_beta_gamma(E: float) -> Tuple[float, float]:
        """
        Compute relativistic beta and gamma for a given kinetic energy.

        Args:
            E: Kinetic energy [MeV]

        Returns:
            (beta, gamma): Relativistic parameters
        """
        if E < 1e-10:
            return 0.0, 1.0

        # Total energy = kinetic + rest mass
        E_total = E + M_P

        # beta = v/c = pc/E
        p = np.sqrt(E_total**2 - M_P**2)
        beta = p / E_total

        # gamma = 1/sqrt(1 - beta²)
        gamma = E_total / M_P

        return beta, gamma

    @staticmethod
    def _density_correction(beta: float, gamma: float) -> float:
        """
        Compute density correction δ for water.

        The density correction accounts for the polarization of the medium
        at high energies. Uses a simplified Sternheimer parameterization.

        Args:
            beta: Relativistic velocity
            gamma: Lorentz factor

        Returns:
            δ: Density correction
        """
        if beta < 0.1:
            return 0.0

        # Simplified parameterization for water
        # In practice, this uses tabulated values or detailed fits
        # Here we use an approximation valid for E < 300 MeV

        log_gamma_squared = np.log(gamma**2)

        # For water, the density effect becomes significant around 100 MeV
        if gamma < 1.1:
            return 0.0
        elif gamma < 3.0:
            # Linear transition region
            delta = 2.0 * (log_gamma_squared - np.log(1.1**2))
        else:
            # Full density effect (asymptotic)
            delta = 4.0 * np.log(10.0) * log_gamma_squared - 4.0

        return max(0.0, delta)

    @staticmethod
    def stopping_power(E: float) -> float:
        """
        Compute stopping power dE/dx using the Bethe-Bloch formula.

        Args:
            E: Proton kinetic energy [MeV]

        Returns:
            dE/dx: Stopping power [MeV/mm]

        Notes:
            - Returns 0 for energies below cutoff
            - Handles low-energy edge cases
            - Includes density correction for high energies
        """
        # Edge case: very low energy
        if E < E_CUTOFF:
            return 0.0

        beta, gamma = BetheBloch._compute_beta_gamma(E)

        # Edge case: non-relativistic
        if beta < 1e-6:
            # Use empirical low-energy behavior
            return 0.1  # Approximate [MeV/mm]

        # Main Bethe-Bloch formula
        # dE/dx = K * z² * (Z/A) * (1/β²) * [ln(...) - β² - δ/2]

        z = 1.0  # Proton charge

        # Logarithmic term
        ln_arg = 2 * M_E * beta**2 * gamma**2 / I_MEAN
        if ln_arg <= 0:
            return 0.0

        log_term = np.log(ln_arg)

        # Density correction
        delta = BetheBloch._density_correction(beta, gamma)

        # Complete stopping power
        dedx = (K_BB * z**2 * Z_OVER_A / beta**2 *
                (log_term - beta**2 - delta/2))

        # Convert from MeV·cm²/g to MeV/mm
        # dE/dx [MeV/mm] = dE/dx [MeV·cm²/g] * ρ [g/cm³] * (1 cm / 10 mm)
        dedx_mm = dedx * RHO_WATER / 10.0

        return max(0.0, dedx_mm)

    @staticmethod
    def range_integral(E_initial: float, n_steps: int = 1000) -> float:
        """
        Compute CSDA range by integrating 1/(dE/dx).

        The CSDA (Continuous Slowing Down Approximation) range is:
        R = ∫(0 to E) (dE/dx)⁻¹ dE

        Args:
            E_initial: Initial energy [MeV]
            n_steps: Number of integration steps

        Returns:
            R: CSDA range [mm]
        """
        if E_initial <= E_CUTOFF:
            return 0.0

        # Log-spaced energy grid for better low-energy resolution
        E_grid = np.logspace(np.log10(E_CUTOFF), np.log10(E_initial), n_steps)

        # Compute stopping power at each energy
        dEdx = np.array([BetheBloch.stopping_power(E) for E in E_grid])

        # Avoid division by zero
        dEdx = np.maximum(dEdx, 1e-10)

        # Integrate using trapezoidal rule
        range_mm = np.trapezoid(1.0/dEdx, E_grid)  # [MeV / (MeV/mm)] = mm

        return range_mm


# ============================================================================
# FERMI-EYGES SCATTERING THEORY
# ============================================================================

class FermiEyges:
    """
    Fermi-Eyges theory for multiple Coulomb scattering.

    The lateral spread of protons is computed using the Highland formula
    for the scattering angle distribution:

    σ_θ = (13.6 MeV)/(β*p*c) * z * sqrt(x/X0) * [1 + 0.038*ln(x/X0)]

    The spatial variance is then:
    σ_x²(z) = ∫(0 to z) (z-z')² * T(z') dz'

    where T = (σ_θ/Δs)² is the scattering power.
    """

    @staticmethod
    def scattering_angle_sigma(E: float, step_size: float) -> float:
        """
        Compute RMS scattering angle for a single step using Highland formula.

        Args:
            E: Proton energy [MeV]
            step_size: Step length [mm]

        Returns:
            sigma_theta: RMS scattering angle [rad]

        Notes:
            - Returns 0 for very small steps
            - Handles edge cases for low energy
        """
        if E < E_CUTOFF or step_size < 1e-6:
            return 0.0

        beta, gamma = BetheBloch._compute_beta_gamma(E)

        if beta < 1e-6:
            return 0.0

        # Momentum [MeV/c]
        p = np.sqrt((E + M_P)**2 - M_P**2)

        # Reduced thickness
        t = step_size / X0_WATER_MM

        # Highland formula
        # sigma_theta = (13.6 / beta*p) * sqrt(t) * [1 + 0.038*ln(t)]
        ln_term = 1.0 + 0.038 * np.log(t)

        # Safety check for very small t
        if ln_term < 0.1:
            return 0.0

        sigma_theta = (13.6 / (beta * p)) * np.sqrt(t) * ln_term

        return max(0.0, sigma_theta)

    @staticmethod
    def lateral_spread_sigma(E: float, total_distance: float,
                             n_steps: int = 100) -> float:
        """
        Compute lateral spatial spread after traveling a given distance.

        This integrates the scattering power over the path to get the
        accumulated lateral variance.

        Args:
            E: Initial proton energy [MeV]
            total_distance: Total path length [mm]
            n_steps: Number of integration steps

        Returns:
            sigma_x: RMS lateral spread [mm]
        """
        if total_distance < 1e-6:
            return 0.0

        # For accuracy, we need to account for energy loss along the path
        # Simplified approach: use energy at mid-point
        # More accurate: integrate with varying energy

        step_size = total_distance / n_steps
        variance_accumulated = 0.0

        # Current position and energy
        z = 0.0
        E_current = E

        for i in range(n_steps):
            # Scattering angle for this step
            sigma_theta = FermiEyges.scattering_angle_sigma(E_current, step_size)

            # Contribution to lateral variance
            # Remaining distance after this step
            z_remaining = total_distance - z - step_size/2

            # Lateral displacement contribution
            # This is an approximation; full theory uses convolution
            variance_contribution = (sigma_theta * z_remaining)**2
            variance_accumulated += variance_contribution

            # Update energy (lose energy in this step)
            dEdx = BetheBloch.stopping_power(E_current)
            dE = dEdx * step_size
            E_current = max(E_CUTOFF, E_current - dE)

            z += step_size

        return np.sqrt(variance_accumulated)


# ============================================================================
# PROTON TRANSPORT SIMULATION
# ============================================================================

@dataclass
class ProtonState:
    """State of a single proton during transport."""
    x: float          # Lateral position [mm]
    z: float          # Depth position [mm]
    E: float          # Kinetic energy [MeV]
    theta: float      # Angle [rad]
    alive: bool = True


class ProtonTransport:
    """
    Main proton transport simulation.

    Tracks protons through a water phantom, computing energy deposition
    and lateral scattering at each step using:
    - Bethe-Bloch for energy loss
    - Highland formula for scattering
    """

    def __init__(self, energy: float, grid_size_x: float, grid_size_z: float,
                 grid_res: float, debug: bool = False):
        """
        Initialize the simulation.

        Args:
            energy: Initial proton energy [MeV]
            grid_size_x: Lateral extent of grid [mm] (half-width, centered at 0)
            grid_size_z: Depth extent of grid [mm]
            grid_res: Grid resolution [mm]
            debug: Enable debug output
        """
        self.E_initial = energy
        self.grid_size_x = grid_size_x
        self.grid_size_z = grid_size_z
        self.grid_res = grid_res
        self.debug = debug

        # Create grid
        self.nx = int(2 * grid_size_x / grid_res)
        self.nz = int(grid_size_z / grid_res)

        # Coordinate arrays
        self.x_edges = np.linspace(-grid_size_x, grid_size_x, self.nx + 1)
        self.z_edges = np.linspace(0, grid_size_z, self.nz + 1)

        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.z_centers = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

        # Dose grid [Gy]
        self.dose_grid = np.zeros((self.nz, self.nx))

        # Track actual deposited energy [MeV]
        self._energy_deposited = 0.0

        # Statistics
        self.stats = {
            'n_protons': 0,
            'n_completed': 0,
            'n_cutoff': 0,
            'n_boundary': 0,
            'total_energy_in': 0.0,
            'total_energy_dep': 0.0,
            'bragg_peak_z': 0.0,
            'max_dose': 0.0,
            'lateral_sigma_mid': 0.0,
        }

    def _find_bin(self, x: float, z: float) -> Tuple[int, int]:
        """
        Find grid bin indices for a given position.

        Args:
            x: Lateral position [mm]
            z: Depth position [mm]

        Returns:
            (ix, iz): Bin indices (-1 if out of bounds)
        """
        ix = int((x + self.grid_size_x) / self.grid_res)
        iz = int(z / self.grid_res)

        # Check bounds
        if ix < 0 or ix >= self.nx or iz < 0 or iz >= self.nz:
            return -1, -1

        return ix, iz

    def _deposit_energy(self, x: float, z: float, dE: float):
        """
        Deposit energy in the dose grid.

        Args:
            x: Lateral position [mm]
            z: Depth position [mm]
            dE: Energy to deposit [MeV]
        """
        # Track total energy deposited
        self._energy_deposited += dE

        ix, iz = self._find_bin(x, z)

        if ix >= 0 and iz >= 0:
            # Convert energy to dose
            # Dose = energy / mass
            # Mass = ρ * volume
            # volume = grid_res * grid_res * grid_res [mm³]
            # density = 1.0 g/cm³ = 1.0e-3 g/mm³
            # mass = 1.0e-3 * grid_res³ [g] = 1.0e-6 * grid_res³ [kg]
            mass_kg = RHO_WATER * 1e-6 * self.grid_res**3  # [kg]

            # 1 MeV = 1.60218e-13 J
            # Dose [Gy] = Dose [J/kg] = (dE * 1.60218e-13) / mass_kg
            dose_gy = dE * 1.60218e-13 / mass_kg

            self.dose_grid[iz, ix] += dose_gy

    def _transport_single_proton(self) -> ProtonState:
        """
        Transport a single proton through the water phantom.

        Returns:
            Final state of the proton
        """
        # Initial state (pencil beam at origin)
        state = ProtonState(x=0.0, z=0.0, E=self.E_initial, theta=0.0)

        # Track path for variance calculation
        z_prev = state.z
        x_prev = state.x
        variance_accumulated = 0.0

        step_count = 0
        max_steps = 10000  # Safety limit

        while state.alive and step_count < max_steps:
            step_count += 1

            # Compute step size
            # Use smaller steps near Bragg peak (low energy)
            if state.E < 10.0:
                step_size = 0.05  # [mm]
            elif state.E < 50.0:
                step_size = 0.1
            else:
                step_size = 0.2

            step_size = min(step_size, STEP_MAX_MM)

            # Compute stopping power
            dEdx = BetheBloch.stopping_power(state.E)

            if dEdx < 1e-10:
                # Very low energy, deposit remaining
                self._deposit_energy(state.x, state.z, state.E)
                state.E = 0.0
                state.alive = False
                self.stats['n_cutoff'] += 1
                break

            # Energy loss in this step
            dE = dEdx * step_size

            if state.E - dE < E_CUTOFF:
                # Deposit all remaining energy
                self._deposit_energy(state.x, state.z + step_size/2, state.E)
                state.E = 0.0
                state.alive = False
                self.stats['n_cutoff'] += 1
                break

            # Deposit energy
            z_deposit = state.z + step_size / 2
            self._deposit_energy(state.x, z_deposit, dE)

            # Update energy
            state.E -= dE

            # Multiple Coulomb scattering
            sigma_theta = FermiEyges.scattering_angle_sigma(
                state.E + dE/2, step_size)  # Use mid-step energy

            if sigma_theta > 0:
                # Accumulate angular variance
                variance_accumulated += sigma_theta**2

                # Scatter: update angle (Gaussian approximation)
                d_theta = np.random.normal(0, sigma_theta)
                state.theta += d_theta

            # Move proton
            state.x += np.sin(state.theta) * step_size
            state.z += np.cos(state.theta) * step_size

            # Check boundaries
            if (state.z >= self.grid_size_z or
                abs(state.x) >= self.grid_size_x):
                state.alive = False
                self.stats['n_boundary'] += 1
                break

        if step_count >= max_steps:
            if self.debug:
                print(f"Warning: Proton reached max steps")

        return state

    def run_simulation(self, n_protons: int):
        """
        Run the simulation for multiple protons.

        Args:
            n_protons: Number of protons to simulate
        """
        self.stats['n_protons'] = n_protons
        self.stats['total_energy_in'] = n_protons * self.E_initial

        print(f"Simulating {n_protons} protons at {self.E_initial} MeV...")
        print(f"Grid: {self.nx} x {self.nz} bins ({self.grid_res} mm resolution)")
        print(f"Domain: x=[-{self.grid_size_x}, +{self.grid_size_x}] mm, z=[0, {self.grid_size_z}] mm")
        print()

        # Run protons
        for i in range(n_protons):
            if (i + 1) % 1000 == 0 or self.debug:
                print(f"  Proton {i+1}/{n_protons}", end='\r')

            state = self._transport_single_proton()

            if state.E <= E_CUTOFF:
                self.stats['n_completed'] += 1

        print()  # New line after progress indicator

        # Compute statistics
        self._compute_statistics()

    def _compute_statistics(self):
        """Compute simulation statistics."""
        # Use tracked energy instead of recalculating from dose grid
        self.stats['total_energy_dep'] = self._energy_deposited

        # Bragg peak position
        depth_dose = np.sum(self.dose_grid, axis=1)  # Integrate over x
        bragg_idx = np.argmax(depth_dose)
        self.stats['bragg_peak_z'] = self.z_centers[bragg_idx]

        # Maximum dose
        self.stats['max_dose'] = np.max(self.dose_grid)

        # Lateral spread at mid-range
        mid_idx = int(bragg_idx / 2)
        lateral_profile = self.dose_grid[mid_idx, :]

        # Compute sigma by fitting Gaussian to lateral profile
        # Simple approximation: sqrt(second moment)
        x_mesh = self.x_centers
        dose_norm = lateral_profile / np.sum(lateral_profile)
        x_mean = np.sum(x_mesh * dose_norm)
        x2_mean = np.sum(x_mesh**2 * dose_norm)
        sigma_x = np.sqrt(x2_mean - x_mean**2)
        self.stats['lateral_sigma_mid'] = sigma_x

    def print_summary(self):
        """Print simulation summary."""
        print("=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Initial Energy: {self.E_initial:.2f} MeV")
        print(f"Number of Protons: {self.stats['n_protons']}")
        print()
        print("Proton Fate:")
        print(f"  Completed (E < cutoff): {self.stats['n_completed']} "
              f"({100*self.stats['n_completed']/self.stats['n_protons']:.1f}%)")
        print(f"  Boundary exit: {self.stats['n_boundary']} "
              f"({100*self.stats['n_boundary']/self.stats['n_protons']:.1f}%)")
        print(f"  Cutoff: {self.stats['n_cutoff']} "
              f"({100*self.stats['n_cutoff']/self.stats['n_protons']:.1f}%)")
        print()
        print("Energy Conservation:")
        print(f"  Total energy in: {self.stats['total_energy_in']:.3f} MeV")
        print(f"  Total energy deposited: {self.stats['total_energy_dep']:.3f} MeV")
        eff = 100 * self.stats['total_energy_dep'] / self.stats['total_energy_in']
        print(f"  Deposition efficiency: {eff:.1f}%")
        print()
        print("Bragg Peak:")
        print(f"  Position: {self.stats['bragg_peak_z']:.2f} mm")
        print(f"  Maximum dose: {self.stats['max_dose']:.4f} Gy")
        print()
        print("Lateral Scattering (at mid-range):")
        print(f"  Sigma_x: {self.stats['lateral_sigma_mid']:.3f} mm")
        print()
        print("CSDA Range (theory):")
        range_theory = BetheBloch.range_integral(self.E_initial)
        print(f"  R({self.E_initial} MeV) = {range_theory:.2f} mm")
        print(f"  Bragg peak error: {100*(self.stats['bragg_peak_z']-range_theory)/range_theory:+.2f}%")
        print("=" * 60)

    def save_csv(self, filename: str):
        """
        Save dose distribution to CSV file.

        Args:
            filename: Output filename
        """
        filepath = Path(filename)

        print(f"Saving dose distribution to {filepath}...")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x[mm]', 'z[mm]', 'dose[Gy]'])

            for iz in range(self.nz):
                for ix in range(self.nx):
                    x = self.x_centers[ix]
                    z = self.z_centers[iz]
                    dose = self.dose_grid[iz, ix]
                    writer.writerow([f'{x:.3f}', f'{z:.3f}', f'{dose:.6e}'])

        print(f"Saved {self.nz * self.nx} data points")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Proton Transport Simulation in Water',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --energy 150 --n-protons 10000
  %(prog)s --energy 70 --n-protons 50000 --grid-res 1.0
  %(prog)s --energy 100 --n-protons 10000 --debug

Physics:
  - Bethe-Bloch stopping power for energy loss
  - Highland formula for multiple Coulomb scattering
  - CSDA range for validation
        """
    )

    parser.add_argument('--energy', type=float, default=150.0,
                        help='Initial proton energy [MeV] (default: 150)')
    parser.add_argument('--n-protons', type=int, default=10000,
                        help='Number of protons to simulate (default: 10000)')
    parser.add_argument('--grid-res', type=float, default=0.5,
                        help='Grid resolution [mm] (default: 0.5)')
    parser.add_argument('--grid-x', type=float, default=50.0,
                        help='Lateral grid half-width [mm] (default: 50)')
    parser.add_argument('--grid-z', type=float, default=200.0,
                        help='Depth grid extent [mm] (default: 200)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename (default: auto-generated)')

    args = parser.parse_args()

    # Validate arguments
    if args.energy <= 0:
        print("Error: Energy must be positive")
        return 1

    if args.n_protons <= 0:
        print("Error: Number of protons must be positive")
        return 1

    if args.grid_res <= 0:
        print("Error: Grid resolution must be positive")
        return 1

    # Create simulation
    sim = ProtonTransport(
        energy=args.energy,
        grid_size_x=args.grid_x,
        grid_size_z=args.grid_z,
        grid_res=args.grid_res,
        debug=args.debug
    )

    # Run simulation
    sim.run_simulation(args.n_protons)

    # Print summary
    sim.print_summary()

    # Save results
    if args.output is None:
        output_file = f'proton_dose_E{args.energy:.0f}MeV.csv'
    else:
        output_file = args.output

    sim.save_csv(output_file)

    return 0


if __name__ == '__main__':
    exit(main())
