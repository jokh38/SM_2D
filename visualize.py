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

def get_x_profile(x_vals, z_vals, dose_grid, target_depth):
    """Extract x-profile at a specific depth (interpolated if needed)"""
    if target_depth < z_vals[0] or target_depth > z_vals[-1]:
        return None, None

    # Find closest depth
    iz = np.argmin(np.abs(z_vals - target_depth))
    actual_depth = z_vals[iz]
    profile = dose_grid[iz, :]
    return x_vals, profile, actual_depth

def calculate_gaussian_sigma(x_vals, profile):
    """Calculate Gaussian sigma from profile using FWHM method"""
    # Remove background (noise) - use 1% of max as threshold
    max_dose = np.max(profile)
    threshold = 0.01 * max_dose

    # Only consider points above threshold
    mask = profile > threshold
    if not np.any(mask):
        return None

    x_above = x_vals[mask]
    dose_above = profile[mask]

    # Find peak position
    peak_idx = np.argmax(dose_above)
    peak_x = x_above[peak_idx]
    peak_dose = dose_above[peak_idx]

    # Half-maximum value
    half_max = peak_dose / 2.0

    # Find left and right points at half-maximum (interpolated)
    # Left side
    left_idx = peak_idx
    while left_idx > 0 and dose_above[left_idx] > half_max:
        left_idx -= 1
    if left_idx < peak_idx:
        # Linear interpolation for better accuracy
        x1, x2 = x_above[left_idx], x_above[left_idx + 1]
        y1, y2 = dose_above[left_idx], dose_above[left_idx + 1]
        x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
    else:
        x_left = x_above[0]

    # Right side
    right_idx = peak_idx
    while right_idx < len(dose_above) - 1 and dose_above[right_idx] > half_max:
        right_idx += 1
    if right_idx > peak_idx:
        # Linear interpolation for better accuracy
        x1, x2 = x_above[right_idx - 1], x_above[right_idx]
        y1, y2 = dose_above[right_idx - 1], dose_above[right_idx]
        x_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
    else:
        x_right = x_above[-1]

    # FWHM
    fwhm = x_right - x_left

    # Convert FWHM to sigma: sigma = FWHM / (2 * sqrt(2 * ln(2))) = FWHM / 2.355
    sigma = fwhm / 2.355

    return sigma

def plot_combined_panel(depths, doses, x_vals, z_vals, dose_grid,
                       output_path='results/combined_plot.png'):
    """Plot 3x2 panel: 2D dose plots, PDD, and x-profiles at different depths"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))

    # Find Bragg peak depth
    peak_idx = np.argmax(doses)
    peak_depth = depths[peak_idx]

    # Extent for imshow
    extent = [x_vals[0], x_vals[-1], z_vals[-1], z_vals[0]]

    # ===== First row: 2D dose distributions =====
    # Top-left: 2D Dose (raw)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(dose_grid, extent=extent, aspect='auto', cmap='hot', origin='upper')
    ax1.set_ylabel('z (mm)', fontsize=11)
    ax1.set_title('2D Dose Distribution', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Dose (Gy)')

    # Top-right: 2D Dose (normalized)
    ax2 = axes[0, 1]
    dose_norm = dose_grid / (np.max(dose_grid) + 1e-10)
    im2 = ax2.imshow(dose_norm, extent=extent, aspect='auto', cmap='hot', origin='upper', vmin=0, vmax=1)
    ax2.set_title('2D Dose Distribution (Normalized)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Normalized Dose')

    # ===== Second row: PDD and shallow x-profile =====
    # Middle-left: PDD
    ax3 = axes[0, 2]
    ax3.plot(depths, doses, 'b-', linewidth=2)
    ax3.axvline(20, color='g', linestyle='--', alpha=0.7, label='Shallow (20mm)')
    ax3.axvline(peak_depth / 2, color='orange', linestyle='--', alpha=0.7, label=f'Middle ({peak_depth/2:.1f}mm)')
    ax3.axvline(peak_depth, color='r', linestyle='--', alpha=0.7, label=f'Bragg Peak ({peak_depth:.1f}mm)')
    ax3.set_xlabel('Depth (mm)', fontsize=10)
    ax3.set_ylabel('Dose (Gy)', fontsize=10)
    ax3.set_title('PDD (Central Axis)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # Middle-right: x profile at shallow depth (20 mm)
    ax4 = axes[1, 0]
    x_prof, dose_prof, actual_depth = get_x_profile(x_vals, z_vals, dose_grid, 20.0)
    if x_prof is not None:
        sigma = calculate_gaussian_sigma(x_prof, dose_prof)
        ax4.plot(x_prof, dose_prof, 'g-', linewidth=2)
        sigma_text = f'σ = {sigma:.2f} mm' if sigma else 'σ = N/A'
        ax4.text(0.05, 0.95, sigma_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title(f'X Profile @ {actual_depth:.1f}mm', fontsize=11, fontweight='bold')
    ax4.set_xlabel('x (mm)', fontsize=10)
    ax4.set_ylabel('Dose (Gy)', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ===== Third row: x-profiles at middle and Bragg peak =====
    # Bottom-middle: x profile at middle depth (bragg peak / 2)
    ax5 = axes[1, 1]
    middle_depth = peak_depth / 2
    x_prof, dose_prof, actual_depth = get_x_profile(x_vals, z_vals, dose_grid, middle_depth)
    if x_prof is not None:
        sigma = calculate_gaussian_sigma(x_prof, dose_prof)
        ax5.plot(x_prof, dose_prof, 'orange', linewidth=2)
        sigma_text = f'σ = {sigma:.2f} mm' if sigma else 'σ = N/A'
        ax5.text(0.05, 0.95, sigma_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax5.set_title(f'X Profile @ {actual_depth:.1f}mm', fontsize=11, fontweight='bold')
    ax5.set_xlabel('x (mm)', fontsize=10)
    ax5.set_ylabel('Dose (Gy)', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # Bottom-right: x profile at bragg peak depth
    ax6 = axes[1, 2]
    x_prof, dose_prof, actual_depth = get_x_profile(x_vals, z_vals, dose_grid, peak_depth)
    if x_prof is not None:
        sigma = calculate_gaussian_sigma(x_prof, dose_prof)
        ax6.plot(x_prof, dose_prof, 'r-', linewidth=2)
        sigma_text = f'σ = {sigma:.2f} mm' if sigma else 'σ = N/A'
        ax6.text(0.05, 0.95, sigma_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.set_title(f'X Profile @ {actual_depth:.1f}mm (Bragg Peak)', fontsize=11, fontweight='bold')
    ax6.set_xlabel('x (mm)', fontsize=10)
    ax6.set_ylabel('Dose (Gy)', fontsize=10)
    ax6.grid(True, alpha=0.3)

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

    # Generate combined plot (includes 2D dose, PDD, and x-profiles)
    if pdd_file.exists() and dose_2d_file.exists():
        plot_combined_panel(depths, doses, x_vals, z_vals, dose_grid)

    print("\n=== Visualization Complete ===")

if __name__ == '__main__':
    main()
