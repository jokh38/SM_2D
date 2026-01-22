#!/usr/bin/env python3
"""
SM_2D Visualization Tool
Visualizes simulation results from the results/ directory
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import sys

def load_pdd(filepath):
    """Load depth-dose (PDD) data from file"""
    depths, doses = [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            depths.append(float(parts[0]))
            doses.append(float(parts[1]))
    return np.array(depths), np.array(doses)

def load_dose_2d(filepath):
    """Load 2D dose distribution from file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                x = float(parts[0])
                z = float(parts[1])
                dose = float(parts[2])
                data.append((x, z, dose))

    # Convert to structured grid
    data = np.array(data)
    x_vals = np.unique(data[:, 0])
    z_vals = np.unique(data[:, 1])

    nx = len(x_vals)
    nz = len(z_vals)

    dose_grid = np.zeros((nz, nx))
    for x, z, dose in data:
        ix = np.where(x_vals == x)[0][0]
        iz = np.where(z_vals == z)[0][0]
        dose_grid[iz, ix] = dose

    return x_vals, z_vals, dose_grid

def plot_pdd(depths, doses, output_path='results/pdd_plot.png'):
    """Plot depth-dose curve"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Find Bragg peak
    peak_idx = np.argmax(doses)
    peak_depth = depths[peak_idx]
    peak_dose = doses[peak_idx]

    ax.plot(depths, doses, 'b-', linewidth=2, label='Dose')
    ax.axvline(peak_depth, color='r', linestyle='--', alpha=0.7,
               label=f'Bragg Peak: {peak_depth:.1f} mm')
    ax.axhline(peak_dose, color='r', linestyle='--', alpha=0.7)

    ax.set_xlabel('Depth (mm)', fontsize=12)
    ax.set_ylabel('Dose (Gy)', fontsize=12)
    ax.set_title('Depth-Dose Distribution (PDD)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  PDD plot saved to: {output_path}")
    plt.close()

def plot_dose_2d(x_vals, z_vals, dose_grid, output_path='results/dose_2d_plot.png'):
    """Plot 2D dose distribution as heatmap"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extent for imshow
    extent = [x_vals[0], x_vals[-1], z_vals[-1], z_vals[0]]

    # 2D Heatmap
    im1 = ax1.imshow(dose_grid, extent=extent, aspect='auto', cmap='hot', origin='upper')
    ax1.set_xlabel('x (mm)', fontsize=12)
    ax1.set_ylabel('z (mm)', fontsize=12)
    ax1.set_title('2D Dose Distribution', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Dose (Gy)')

    # Normalized heatmap
    dose_norm = dose_grid / (np.max(dose_grid) + 1e-10)
    im2 = ax2.imshow(dose_norm, extent=extent, aspect='auto', cmap='hot', origin='upper',
                     vmin=0, vmax=1)
    ax2.set_xlabel('x (mm)', fontsize=12)
    ax2.set_ylabel('z (mm)', fontsize=12)
    ax2.set_title('2D Dose Distribution (Normalized)', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Normalized Dose')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  2D dose plot saved to: {output_path}")
    plt.close()

def plot_combined(depths, doses, x_vals, z_vals, dose_grid,
                  output_path='results/combined_plot.png'):
    """Plot PDD and 2D dose side by side"""
    fig = plt.figure(figsize=(16, 6))

    # PDD on left
    ax1 = fig.add_subplot(1, 2, 1)
    peak_idx = np.argmax(doses)
    peak_depth = depths[peak_idx]
    ax1.plot(depths, doses, 'b-', linewidth=2)
    ax1.axvline(peak_depth, color='r', linestyle='--', alpha=0.7,
                label=f'Bragg Peak: {peak_depth:.1f} mm')
    ax1.set_xlabel('Depth (mm)', fontsize=12)
    ax1.set_ylabel('Dose (Gy)', fontsize=12)
    ax1.set_title('Depth-Dose Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2D dose on right
    ax2 = fig.add_subplot(1, 2, 2)
    extent = [x_vals[0], x_vals[-1], z_vals[-1], z_vals[0]]
    dose_norm = dose_grid / (np.max(dose_grid) + 1e-10)
    im = ax2.imshow(dose_norm, extent=extent, aspect='auto', cmap='hot',
                    origin='upper', vmin=0, vmax=1)
    ax2.set_xlabel('x (mm)', fontsize=12)
    ax2.set_ylabel('z (mm)', fontsize=12)
    ax2.set_title('2D Dose Distribution (Normalized)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Normalized Dose')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Combined plot saved to: {output_path}")
    plt.close()

def main():
    results_dir = Path('results')

    if not results_dir.exists():
        print(f"Error: results directory not found at {results_dir}")
        print("Run simulation first: ./build/run_simulation")
        sys.exit(1)

    print(f"=== SM_2D Visualization Tool ===")
    print(f"Loading results from: {results_dir}\n")

    # Load PDD
    pdd_file = results_dir / 'pdd.txt'
    if pdd_file.exists():
        print(f"Loading PDD: {pdd_file}")
        depths, doses = load_pdd(pdd_file)
    else:
        print(f"Warning: PDD file not found: {pdd_file}")

    # Load 2D dose
    dose_2d_file = results_dir / 'dose_2d.txt'
    if dose_2d_file.exists():
        print(f"Loading 2D dose: {dose_2d_file}")
        x_vals, z_vals, dose_grid = load_dose_2d(dose_2d_file)
    else:
        print(f"Warning: 2D dose file not found: {dose_2d_file}")
        sys.exit(1)

    print("\n--- Generating Plots ---")

    # Generate plots
    if pdd_file.exists():
        plot_pdd(depths, doses)

    if dose_2d_file.exists():
        x_vals, z_vals, dose_grid = load_dose_2d(dose_2d_file)
        plot_dose_2d(x_vals, z_vals, dose_grid)

    if pdd_file.exists() and dose_2d_file.exists():
        plot_combined(depths, doses, x_vals, z_vals, dose_grid)

    print("\n=== Visualization Complete ===")

if __name__ == '__main__':
    main()
