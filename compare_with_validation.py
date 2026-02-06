#!/usr/bin/env python3
"""
Compare SM_2D simulation results with MOQUI validation data.

The MOQUI data is 3D (100×100×80) with 1mm voxels.
We extract the central Y slice to compare with our 2D (x, z) results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct

def load_moqui_3d_dose(filepath):
    """
    Load MOQUI 3D dose data from raw binary file.

    MOQUI format: float64, 3D array with dimensions (z, y, x)
    For 70MeV: 100 (x) × 100 (y) × 80 (z) = 800000 elements

    Phantom: 100×100×80 mm³, 1mm voxels
    Position: (-50, -50, -40) mm
    """
    # Read as float64 (8 bytes per element)
    with open(filepath, 'rb') as f:
        data = f.read()

    num_elements = len(data) // 8
    print(f"MOQUI file: {len(data)} bytes = {num_elements} float64 values")

    # Convert to numpy array
    dose_flat = np.frombuffer(data, dtype=np.float64)

    # Reshape to 3D (z, y, x) based on MOQUI convention
    # Dimensions: x=100, y=100, z=80
    dim_x, dim_y, dim_z = 100, 100, 80
    dose_3d = dose_flat.reshape((dim_z, dim_y, dim_x), order='C')

    return dose_3d, dim_x, dim_y, dim_z


def load_sm2d_dose(filepath):
    """
    Load SM_2D 2D dose data from text file.

    Format: x(mm) z(mm) dose(Gy) dose_norm
    """
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

    data = np.array(data)
    x_vals = np.unique(data[:, 0])
    z_vals = np.unique(data[:, 1])

    nx = len(x_vals)
    nz = len(z_vals)

    dose_grid = np.zeros((nz, nx))
    for x, z, dose in data:
        ix = np.argmin(np.abs(x_vals - x))
        iz = np.argmin(np.abs(z_vals - z))
        dose_grid[iz, ix] = dose

    return dose_grid, x_vals, z_vals


def extract_central_slice_moqui(dose_3d, dim_x, dim_y, dim_z):
    """
    Extract central Y slice from 3D MOQUI data for 2D comparison.

    Also create coordinate arrays in mm.
    """
    # Central Y index
    iy_center = dim_y // 2

    # Extract 2D slice (z, x)
    dose_2d = dose_3d[:, iy_center, :]

    # Create coordinate arrays
    # MOQUI phantom position: (-50, -50, -40) mm
    x_coords = np.arange(dim_x) * 1.0 + (-50.0)  # -50 to +49 mm
    z_coords = np.arange(dim_z) * 1.0 + (-40.0)  # -40 to +39 mm

    return dose_2d, x_coords, z_coords


def interpolate_to_common_grid(dose_grid, x_vals, z_vals, x_target, z_target):
    """
    Interpolate dose grid to target coordinate system.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Create interpolator
    interpolator = RegularGridInterpolator(
        (z_vals, x_vals), dose_grid,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    # Create target meshgrid
    Z_target, X_target = np.meshgrid(z_target, x_target, indexing='ij')

    # Interpolate
    dose_interpolated = interpolator((Z_target, X_target))

    return dose_interpolated


def find_bragg_peak(z_coords, depth_dose):
    """Find Bragg peak position."""
    peak_idx = np.argmax(depth_dose)
    peak_z = z_coords[peak_idx]
    peak_dose = depth_dose[peak_idx]
    return peak_z, peak_dose, peak_idx


def extract_lateral_profile(dose_grid, x_coords, z_coords, target_depth):
    """Extract lateral profile at specific depth."""
    # Find closest depth index
    iz = np.argmin(np.abs(z_coords - target_depth))
    actual_depth = z_coords[iz]
    profile = dose_grid[iz, :]
    return x_coords, profile, actual_depth


def compare_profiles(sm2d_dose, sm2d_x, sm2d_z,
                     moqui_dose, moqui_x, moqui_z,
                     energy_mev):
    """
    Compare SM_2D and MOQUI results at key depths.

    Key depths:
    - Shallow: 2 cm from surface
    - Mid: Bragg peak / 2
    - Distal: Bragg peak depth

    NOTE: Coordinate systems
    - MOQUI: z_coords are -40 to +39 mm, surface at z=-40
    - SM_2D: z_coords are 0 to 200 mm, surface at z=0
    - Physical depth from surface is the common reference
    """

    # CRITICAL FIX: Convert to physical depth for interpolation
    # MOQUI: physical_depth = z + 40
    # SM_2D: physical_depth = z
    moqui_z_physical = moqui_z + 40.0  # Convert to physical depth
    sm2d_z_physical = sm2d_z.copy()  # Already physical depth

    # Interpolate SM_2D to MOQUI grid using physical depth
    sm2d_interp = interpolate_to_common_grid(sm2d_dose, sm2d_x, sm2d_z_physical, moqui_x, moqui_z_physical)

    # Compute depth-dose profiles (integrate over x)
    sm2d_pdd = np.sum(sm2d_interp, axis=1) * (moqui_x[1] - moqui_x[0])  # Integrate over x
    moqui_pdd = np.sum(moqui_dose, axis=1) * (moqui_x[1] - moqui_x[0])

    # Find Bragg peaks using physical depth
    sm2d_bp_z, sm2d_bp_dose, sm2d_bp_idx = find_bragg_peak(moqui_z_physical, sm2d_pdd)
    moqui_bp_z, moqui_bp_dose, moqui_bp_idx = find_bragg_peak(moqui_z_physical, moqui_pdd)

    # Both are now in physical depth from surface
    moqui_bp_depth = moqui_bp_z
    sm2d_bp_depth = sm2d_bp_z

    print(f"\n{'='*60}")
    print(f"BRAGG PEAK COMPARISON ({energy_mev} MeV)")
    print(f"{'='*60}")
    print(f"All values in physical depth from surface (mm)")
    print(f"\nBragg Peak positions:")
    print(f"  SM_2D:    {sm2d_bp_depth:.2f} mm")
    print(f"  MOQUI:    {moqui_bp_depth:.2f} mm")
    print(f"  DIFF:     {sm2d_bp_depth - moqui_bp_depth:+.2f} mm")
    if moqui_bp_depth > 0.1:
        print(f"            ({100*(sm2d_bp_depth - moqui_bp_depth)/moqui_bp_depth:+.2f}%)")
    print(f"\nDose at Bragg Peak:")
    print(f"  SM_2D:    {sm2d_bp_dose:.6e}")
    print(f"  MOQUI:    {moqui_bp_dose:.6e}")
    print(f"  Ratio:    {sm2d_bp_dose/moqui_bp_dose:.3f}")

    # Define comparison depths (all in physical depth from surface)
    shallow_depth = 2.0  # 2 cm from surface
    mid_depth = moqui_bp_depth / 2.0  # Mid-range to Bragg peak
    bragg_depth = moqui_bp_depth  # At Bragg peak

    print(f"\n{'='*60}")
    print(f"LATERAL PROFILE COMPARISON")
    print(f"{'='*60}")
    print(f"All depths are physical depth from surface (mm)")

    # Compare at each depth
    for name, target_depth in [("Shallow (2cm from surface)", shallow_depth),
                                ("Mid-range", mid_depth),
                                ("Bragg Peak", bragg_depth)]:
        sm2d_x_prof, sm2d_prof, actual_z = extract_lateral_profile(sm2d_interp, moqui_x, moqui_z_physical, target_depth)
        moqui_x_prof, moqui_prof, _ = extract_lateral_profile(moqui_dose, moqui_x, moqui_z_physical, target_depth)

        # Normalize to max for comparison
        sm2d_prof_norm = sm2d_prof / (np.max(sm2d_prof) + 1e-10)
        moqui_prof_norm = moqui_prof / (np.max(moqui_prof) + 1e-10)

        # Compute metrics
        max_diff = np.max(np.abs(sm2d_prof_norm - moqui_prof_norm))
        rms_diff = np.sqrt(np.mean((sm2d_prof_norm - moqui_prof_norm)**2))

        print(f"\n{name} @ z={actual_z:.1f} mm:")
        print(f"  Max difference: {max_diff:.4f} (normalized)")
        print(f"  RMS difference: {rms_diff:.4f}")

        # Calculate sigma (FWHM) for both
        def calculate_sigma(x, profile):
            max_dose = np.max(profile)
            half_max = max_dose / 2
            above_half = profile > half_max
            if np.any(above_half):
                fwhm = x[above_half][-1] - x[above_half][0]
                sigma = fwhm / 2.355
                return sigma
            return None

        sm2d_sigma = calculate_sigma(sm2d_x_prof, sm2d_prof)
        moqui_sigma = calculate_sigma(moqui_x_prof, moqui_prof)

        if sm2d_sigma and moqui_sigma:
            print(f"  SM_2D sigma:   {sm2d_sigma:.3f} mm")
            print(f"  MOQUI sigma:   {moqui_sigma:.3f} mm")
            print(f"  Sigma diff:    {sm2d_sigma - moqui_sigma:+.3f} mm ({100*(sm2d_sigma - moqui_sigma)/moqui_sigma:+.1f}%)")

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ===== Top row: PDD comparison =====
    ax1 = axes[0, 0]
    # Normalize PDD
    sm2d_pdd_norm = sm2d_pdd / (np.max(sm2d_pdd) + 1e-10) * 100
    moqui_pdd_norm = moqui_pdd / (np.max(moqui_pdd) + 1e-10) * 100
    ax1.plot(moqui_z_physical, sm2d_pdd_norm, 'b-', linewidth=2, label='SM_2D')
    ax1.plot(moqui_z_physical, moqui_pdd_norm, 'r--', linewidth=2, label='MOQUI')
    ax1.axvline(sm2d_bp_z, color='b', linestyle=':', alpha=0.5)
    ax1.axvline(moqui_bp_z, color='r', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Physical Depth from Surface (mm)')
    ax1.set_ylabel('Relative Dose (%)')
    ax1.set_title('PDD Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # 2D dose comparison (SM_2D)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(sm2d_interp, extent=[moqui_x[0], moqui_x[-1], moqui_z_physical[-1], moqui_z_physical[0]],
                     aspect='auto', cmap='hot', origin='upper')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Physical Depth (mm)')
    ax2.set_title('SM_2D Dose (interpolated)')
    plt.colorbar(im2, ax=ax2, label='Dose (Gy)')

    # 2D dose comparison (MOQUI)
    ax3 = axes[0, 2]
    im3 = ax3.imshow(moqui_dose, extent=[moqui_x[0], moqui_x[-1], moqui_z_physical[-1], moqui_z_physical[0]],
                     aspect='auto', cmap='hot', origin='upper')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Physical Depth (mm)')
    ax3.set_title('MOQUI Dose')
    plt.colorbar(im3, ax=ax3, label='Dose (Gy)')

    # ===== Bottom row: Lateral profiles =====
    depths_to_plot = [
        ("Shallow (2cm)", shallow_depth),
        ("Mid-range", mid_depth),
        ("Bragg Peak", bragg_depth)
    ]

    for i, (name, target_depth) in enumerate(depths_to_plot):
        ax = axes[1, i]
        sm2d_x_prof, sm2d_prof, actual_z = extract_lateral_profile(sm2d_interp, moqui_x, moqui_z_physical, target_depth)
        moqui_x_prof, moqui_prof, _ = extract_lateral_profile(moqui_dose, moqui_x, moqui_z_physical, target_depth)

        # Normalize
        sm2d_prof_norm = sm2d_prof / (np.max(sm2d_prof) + 1e-10) * 100
        moqui_prof_norm = moqui_prof / (np.max(moqui_prof) + 1e-10) * 100

        ax.plot(sm2d_x_prof, sm2d_prof_norm, 'b-', linewidth=2, label='SM_2D')
        ax.plot(moqui_x_prof, moqui_prof_norm, 'r--', linewidth=2, label='MOQUI')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Relative Dose (%)')
        ax.set_title(f'{name}\n@ Z={actual_z:.1f} mm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    output_file = f'comparison_{energy_mev}MeV.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nComparison plot saved to: {output_file}")
    plt.close()

    return {
        'sm2d_bragg_peak': sm2d_bp_depth,
        'moqui_bragg_peak': moqui_bp_depth,
        'bragg_peak_diff_mm': sm2d_bp_depth - moqui_bp_depth,
        'bragg_peak_diff_percent': 100 * (sm2d_bp_depth - moqui_bp_depth) / moqui_bp_depth if moqui_bp_depth > 0.1 else 0,
    }


def main():
    validation_dir = Path('validation_data')
    results_dir = Path('results')

    # Check for available validation data
    validation_files = {
        70: '70MeV_1G210_1_Dose.raw',
        110: '110MeV_1G210_1_Dose.raw',
        150: '150MeV_1G210_1_Dose.raw',
        190: '190MeV_1G210_1_Dose.raw',
        230: '230MeV_1G210_1_Dose.raw',
    }

    # Find which energy we have results for
    dose_file = results_dir / 'dose_2d.txt'
    if not dose_file.exists():
        print(f"Error: Results file not found: {dose_file}")
        print("Please run the simulation first.")
        return

    # For now, assume 70MeV (we can make this configurable later)
    energy = 70
    validation_file = validation_dir / validation_files[energy]

    if not validation_file.exists():
        print(f"Error: Validation file not found: {validation_file}")
        return

    print("="*60)
    print("SM_2D vs MOQUI Validation Comparison")
    print("="*60)
    print(f"Energy: {energy} MeV")
    print(f"SM_2D results: {dose_file}")
    print(f"MOQUI validation: {validation_file}")

    # Load data
    print("\nLoading SM_2D results...")
    sm2d_dose, sm2d_x, sm2d_z = load_sm2d_dose(dose_file)
    print(f"  SM_2D grid: {len(sm2d_x)} x {len(sm2d_z)}")
    print(f"  X range: [{sm2d_x[0]:.1f}, {sm2d_x[-1]:.1f}] mm")
    print(f"  Z range: [{sm2d_z[0]:.1f}, {sm2d_z[-1]:.1f}] mm")

    print("\nLoading MOQUI validation data...")
    moqui_3d, dim_x, dim_y, dim_z = load_moqui_3d_dose(validation_file)
    print(f"  MOQUI 3D grid: {dim_x} x {dim_y} x {dim_z}")
    print(f"  Extracting central Y slice...")

    moqui_2d, moqui_x, moqui_z = extract_central_slice_moqui(moqui_3d, dim_x, dim_y, dim_z)
    print(f"  MOQUI 2D slice: {len(moqui_x)} x {len(moqui_z)}")
    print(f"  X range: [{moqui_x[0]:.1f}, {moqui_x[-1]:.1f}] mm")
    print(f"  Z range: [{moqui_z[0]:.1f}, {moqui_z[-1]:.1f}] mm")

    # Compare
    results = compare_profiles(sm2d_dose, sm2d_x, sm2d_z,
                               moqui_2d, moqui_x, moqui_z,
                               energy)

    # Save summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Bragg Peak Difference: {results['bragg_peak_diff_mm']:+.2f} mm ({results['bragg_peak_diff_percent']:+.2f}%)")

    if abs(results['bragg_peak_diff_mm']) > 2.0:
        print("\n⚠️  WARNING: Bragg peak position difference exceeds 2 mm!")
        print("    This may indicate issues with energy loss or range calculation.")

    if abs(results['bragg_peak_diff_percent']) > 5:
        print("\n⚠️  WARNING: Bragg peak position error exceeds 5%!")
        print("    This may require physics model verification.")


if __name__ == '__main__':
    main()
