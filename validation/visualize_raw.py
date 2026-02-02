#!/usr/bin/env python3
"""
Visualize MOQUI validation data from binary .raw files.
Fixed: Correct dimensions for all energy levels.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def get_moqui_dimensions(num_floats, energy):
    """Correct dimensions based on MOQUI config files."""
    # Dimensions from MOQUI config files (PhantomDimX, PhantomDimY, PhantomDimZ)
    dim_configs = {
        70: (100, 100, 80),     # 800000 floats
        110: (100, 100, 120),    # 1200000 floats
        150: (100, 100, 200),    # 2000000 floats
        190: (100, 100, 300),    # 3000000 floats
        230: (100, 100, 350),   # 3500000 floats
    }
    return dim_configs.get(energy, (100, 100, 80))


def load_moqui_3d_dose(filepath, energy):
    """Load MOQUI 3D dose data."""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    num_elements = len(data) // 8
    print(f"MOQUI file: {len(data)} bytes = {num_elements} float64 values")
    
    dose_flat = np.frombuffer(data, dtype=np.float64)
    dim_x, dim_y, dim_z = get_moqui_dimensions(num_elements, energy)
    
    dose_3d = dose_flat.reshape((dim_z, dim_y, dim_x), order='C')
    return dose_3d, dim_x, dim_y, dim_z


def extract_central_slice(dose_3d, dim_x, dim_y, dim_z):
    """Extract central Y slice."""
    iy_center = dim_y // 2
    dose_2d = dose_3d[:, iy_center, :]

    # Voxels are 1mm
    x_coords = np.arange(dim_x) * 1.0 + (-dim_x / 2.0)

    # Depth is reversed array index (array is stored back-to-front)
    # Index 0 = back of phantom, index dim_z-1 = surface (front)
    # So we reverse to get depth increasing from surface
    z_physical = (dim_z - 1 - np.arange(dim_z)) * 1.0

    return dose_2d, x_coords, z_physical


def calculate_gaussian_sigma(x_vals, profile):
    """Calculate Gaussian sigma from FWHM."""
    max_dose = np.max(profile)
    threshold = 0.01 * max_dose
    
    mask = profile > threshold
    if not np.any(mask):
        return None
    
    x_above = x_vals[mask]
    dose_above = profile[mask]
    
    peak_idx = np.argmax(dose_above)
    peak_dose = dose_above[peak_idx]
    half_max = peak_dose / 2.0
    
    above_half = dose_above > half_max
    if np.any(above_half):
        fwhm = x_above[above_half][-1] - x_above[above_half][0]
        sigma = fwhm / 2.355
        return sigma
    
    return None


def get_x_profile(dose_2d, x_coords, z_physical, target_depth):
    """Extract x-profile at a specific depth."""
    if target_depth < z_physical[0] or target_depth > z_physical[-1]:
        return None, None, None
    
    iz = np.argmin(np.abs(z_physical - target_depth))
    actual_depth = z_physical[iz]
    profile = dose_2d[iz, :]
    return x_coords, profile, actual_depth


def get_roi_limits(x_data, y_data, threshold_fraction=0.01):
    """Calculate axis limits."""
    y_max = np.max(y_data)
    
    if y_max < 1e-15:
        return (np.min(x_data), np.max(x_data)), (0, 1)
    
    threshold = threshold_fraction * y_max
    above_threshold = y_data > threshold
    
    if np.any(above_threshold):
        x_min = np.min(x_data[above_threshold])
        x_max = np.max(x_data[above_threshold])
        margin = 0.15 * (x_max - x_min)
        x_min = x_min - margin
        x_max = x_max + margin
    else:
        x_min, x_max = np.min(x_data), np.max(x_data)
    
    y_min = threshold
    y_max_lim = y_max * 1.05
    
    return (x_min, x_max), (y_min, y_max_lim)


def plot_combined_panel(dose_2d, x_coords, z_physical, energy_mev, output_path):
    """Plot 3x2 panel."""
    
    pdd = np.sum(dose_2d, axis=1) * (x_coords[1] - x_coords[0])
    peak_idx = np.argmax(pdd)
    peak_depth = z_physical[peak_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))

    # For imshow with origin='upper', we need extent [left, right, bottom, top]
    # z_physical is reversed: [dim_z-1, ..., 0] (deepest to surface)
    # Row 0 (TOP) shows deepest part, Row dim_z-1 (BOTTOM) shows surface
    extent = [x_coords[0], x_coords[-1], 0, z_physical[0]]
    
    # 2D Dose (raw)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(dose_2d, extent=extent, aspect='auto', cmap='hot', origin='upper')
    ax1.set_ylabel('Physical Depth (mm)', fontsize=11)
    ax1.set_title(f'MOQUI 2D Dose ({energy_mev} MeV)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Dose (a.u.)')
    ax1.axhline(peak_depth, color='r', linestyle='--', alpha=0.7, linewidth=1)
    
    # 2D Dose (normalized)
    ax2 = axes[0, 1]
    dose_norm = dose_2d / (np.max(dose_2d) + 1e-10)
    im2 = ax2.imshow(dose_norm, extent=extent, aspect='auto', cmap='hot', origin='upper', vmin=0, vmax=1)
    ax2.set_title('Normalized', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Normalized Dose')
    
    # PDD
    ax3 = axes[0, 2]
    ax3.plot(z_physical, pdd, 'r-', linewidth=2, label='MOQUI')
    ax3.axvline(20, color='g', linestyle='--', alpha=0.7, label='Shallow (20mm)')
    ax3.axvline(peak_depth / 2, color='orange', linestyle='--', alpha=0.7, label=f'Mid ({peak_depth/2:.1f}mm)')
    ax3.axvline(peak_depth, color='r', linestyle='--', alpha=0.7, label=f'Bragg Peak ({peak_depth:.1f}mm)')
    (x_min, x_max), (y_min, y_max) = get_roi_limits(z_physical, pdd)
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_xlabel('Physical Depth (mm)', fontsize=10)
    ax3.set_ylabel('Dose (a.u.)', fontsize=10)
    ax3.set_title('PDD', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # X profiles at different depths
    depths = [(20.0, 'Shallow (20mm)', 'g'), (peak_depth / 2, 'Mid-range', 'orange'), (peak_depth, 'Bragg Peak', 'r')]
    
    for i, (target_depth, label, color) in enumerate(depths):
        ax = axes[1, i]
        x_prof, dose_prof, actual_z = get_x_profile(dose_2d, x_coords, z_physical, target_depth)
        
        if x_prof is not None:
            sigma = calculate_gaussian_sigma(x_prof, dose_prof)
            dose_prof_norm = dose_prof / (np.max(dose_prof) + 1e-10)
            ax.plot(x_prof, dose_prof_norm, color=color, linewidth=2)
            (x_min, x_max), (y_min, y_max) = get_roi_limits(x_prof, dose_prof_norm)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, 1.05)
            sigma_text = f'σ = {sigma:.2f} mm' if sigma else 'σ = N/A'
            ax.text(0.05, 0.95, sigma_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title(f'{label}\n@ {actual_z:.1f}mm', fontsize=11, fontweight='bold')
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Normalized Dose', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'MOQUI Validation - {energy_mev} MeV', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    # Read from validation directory (MOQUI test data)
    validation_dir = Path('validation')

    validation_files = {
        70: '70MeV_1G210_1_Dose.raw',
        110: '110MeV_1G210_1_Dose.raw',
        150: '150MeV_1G210_1_Dose.raw',
        190: '190MeV_1G210_1_Dose.raw',
        230: '230MeV_1G210_1_Dose.raw',
    }

    print("="*60)
    print("MOQUI Test Data Visualization (Individual Energies)")
    print("="*60)

    for energy, filename in validation_files.items():
        filepath = validation_dir / filename

        if not filepath.exists():
            print(f"\nSkipping {energy} MeV - file not found: {filepath}")
            continue

        print(f"\nProcessing {energy} MeV...")

        dose_3d, dim_x, dim_y, dim_z = load_moqui_3d_dose(filepath, energy)
        print(f"  Grid: {dim_x} x {dim_y} x {dim_z}")

        dose_2d, x_coords, z_physical = extract_central_slice(dose_3d, dim_x, dim_y, dim_z)

        pdd = np.sum(dose_2d, axis=1) * (x_coords[1] - x_coords[0])
        peak_idx = np.argmax(pdd)
        peak_depth = z_physical[peak_idx]
        print(f"  Bragg Peak: {peak_depth:.2f} mm")
        print(f"  Peak Dose: {pdd[peak_idx]:.6e}")

        # Save to validation directory
        output_path = validation_dir / f'combined_plot_{energy}MeV.png'
        plot_combined_panel(dose_2d, x_coords, z_physical, energy, output_path)

    print("\n" + "="*60)
    print("Done! Plots saved to validation/")
    print("="*60)


if __name__ == '__main__':
    main()
