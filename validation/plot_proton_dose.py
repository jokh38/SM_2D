#!/usr/bin/env python3
"""
Visualization helper for proton dose distributions.

Usage:
    python plot_proton_dose.py proton_dose_E150MeV.csv
    python plot_proton_dose.py proton_dose_E150MeV.csv --save
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_dose_distribution(csv_file: str, save: bool = False):
    """
    Plot the proton dose distribution from a CSV file.

    Args:
        csv_file: Path to CSV file with columns: x[mm], z[mm], dose[Gy]
        save: If True, save plot to PNG file
    """
    # Load data
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)

    x = data[:, 0]
    z = data[:, 1]
    dose = data[:, 2]

    # Get unique coordinates
    x_unique = np.unique(x)
    z_unique = np.unique(z)

    # Create 2D grid
    nx = len(x_unique)
    nz = len(z_unique)

    dose_grid = dose.reshape(nz, nx)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: 2D dose heatmap
    im = axes[0].pcolormesh(x_unique, z_unique, dose_grid,
                            shading='auto', cmap='hot')
    axes[0].set_xlabel('Lateral Position x [mm]')
    axes[0].set_ylabel('Depth z [mm]')
    axes[0].set_title('2D Dose Distribution')
    axes[0].set_aspect('equal')
    plt.colorbar(im, ax=axes[0], label='Dose [Gy]')

    # Plot 2: Depth dose profile (Bragg peak)
    depth_dose = np.sum(dose_grid, axis=1)
    axes[1].plot(z_unique, depth_dose, 'b-', linewidth=2)
    axes[1].set_xlabel('Depth z [mm]')
    axes[1].set_ylabel('Integrated Dose [Gy]')
    axes[1].set_title('Bragg Peak (Depth Dose Profile)')
    axes[1].grid(True, alpha=0.3)

    # Find Bragg peak
    bragg_idx = np.argmax(depth_dose)
    bragg_z = z_unique[bragg_idx]
    bragg_dose = depth_dose[bragg_idx]
    axes[1].axvline(bragg_z, color='r', linestyle='--',
                   label=f'Bragg Peak: {bragg_z:.1f} mm')
    axes[1].legend()

    # Plot 3: Lateral dose profile at Bragg peak
    lateral_dose = dose_grid[bragg_idx, :]
    axes[2].plot(x_unique, lateral_dose, 'g-', linewidth=2)
    axes[2].set_xlabel('Lateral Position x [mm]')
    axes[2].set_ylabel('Dose [Gy]')
    axes[2].set_title(f'Lateral Profile at z={bragg_z:.1f} mm')
    axes[2].grid(True, alpha=0.3)

    # Compute sigma
    dose_norm = lateral_dose / np.sum(lateral_dose)
    x_mean = np.sum(x_unique * dose_norm)
    x2_mean = np.sum(x_unique**2 * dose_norm)
    sigma_x = np.sqrt(max(0, x2_mean - x_mean**2))
    axes[2].text(0.05, 0.95, rf'$\sigma_x$ = {sigma_x:.2f} mm',
                 transform=axes[2].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save or show
    if save:
        output_file = Path(csv_file).stem + '_plot.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize proton dose distribution'
    )
    parser.add_argument('csv_file', help='CSV file from proton_transport_water.py')
    parser.add_argument('--save', action='store_true',
                        help='Save plot to PNG instead of displaying')

    args = parser.parse_args()

    if not Path(args.csv_file).exists():
        print(f"Error: File {args.csv_file} not found")
        return 1

    plot_dose_distribution(args.csv_file, args.save)

    return 0


if __name__ == '__main__':
    exit(main())
