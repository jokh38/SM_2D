#!/usr/bin/env python3
"""
Compare SM_2D simulation results with MOQUI gold standard data.

Comparison items:
1. Bragg peak position
2. Lateral profiles at 2 cm, 10 cm, 14 cm depth
3. Relative doses (normalized at maximum) at 2, 10, 14 cm
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_moqui_3d_dose(filepath, energy):
    """Load MOQUI 3D dose data from binary file."""
    with open(filepath, 'rb') as f:
        data = f.read()

    num_elements = len(data) // 8
    print(f"MOQUI file: {len(data)} bytes = {num_elements} float64 values")

    dose_flat = np.frombuffer(data, dtype=np.float64)

    # Dimensions from MOQUI config files
    dim_configs = {
        70: (100, 100, 80),
        110: (100, 100, 120),
        150: (100, 100, 200),
        190: (100, 100, 300),
        230: (100, 100, 350),
    }

    dim_x, dim_y, dim_z = dim_configs.get(energy, (100, 100, 80))

    # Reshape: MOQUI stores as (z, y, x) in C order
    dose_3d = dose_flat.reshape((dim_z, dim_y, dim_x), order='C')
    return dose_3d, dim_x, dim_y, dim_z


def extract_moqui_central_slice(dose_3d, dim_x, dim_y, dim_z):
    """Extract central Y slice from MOQUI 3D data."""
    iy_center = dim_y // 2
    dose_2d = dose_3d[:, iy_center, :]

    # Voxels are 1mm
    x_coords = np.arange(dim_x) * 1.0 + (-dim_x / 2.0)

    # Depth is reversed (index 0 = back, index dim_z-1 = surface)
    z_physical = (dim_z - 1 - np.arange(dim_z)) * 1.0

    return dose_2d, x_coords, z_physical


def load_sm2d_dose(filepath):
    """Load SM_2D 2D dose distribution from text file."""
    # Parse by comment marker so header-length changes do not drop data rows.
    data = np.loadtxt(filepath, comments='#')

    x = data[:, 0]
    z = data[:, 1]
    dose = data[:, 2]
    if data.shape[1] >= 4:
        dose_norm = data[:, 3]
    else:
        peak = np.max(dose) if dose.size else 0.0
        dose_norm = dose / peak if peak > 0 else np.zeros_like(dose)

    # Get unique x and z values to reconstruct grid
    x_unique = np.unique(x)
    z_unique = np.unique(z)

    Nx = len(x_unique)
    Nz = len(z_unique)

    # Check if we have all expected data points
    expected_count = Nx * Nz
    actual_count = len(dose)

    if actual_count != expected_count:
        # Some data points may be missing - create full grid
        print(f"    Warning: Expected {expected_count} points, got {actual_count}")
        print(f"    Reconstructing full grid...")

        # Create full meshgrid
        X, Z = np.meshgrid(x_unique, z_unique, indexing='xy')

        # Create output arrays
        dose_2d = np.zeros((Nz, Nx))
        dose_norm_2d = np.zeros((Nz, Nx))

        # Find indices for each data point
        for i in range(len(dose)):
            ix = np.argmin(np.abs(x_unique - x[i]))
            iz = np.argmin(np.abs(z_unique - z[i]))
            dose_2d[iz, ix] = dose[i]
            dose_norm_2d[iz, ix] = dose_norm[i]
    else:
        # File writer loops z outer, x inner; row-major reshape is correct.
        dose_2d = dose.reshape(Nz, Nx, order='C')
        dose_norm_2d = dose_norm.reshape(Nz, Nx, order='C')

    return dose_2d, dose_norm_2d, x_unique, z_unique


def load_sm2d_pdd(filepath):
    """Load SM_2D PDD data."""
    # Parse by comment marker so header-length changes do not drop data rows.
    data = np.loadtxt(filepath, comments='#')

    z = data[:, 0]
    dose = data[:, 1]
    if data.shape[1] >= 3:
        dose_norm = data[:, 2]
    else:
        peak = np.max(dose) if dose.size else 0.0
        dose_norm = dose / peak if peak > 0 else np.zeros_like(dose)

    return z, dose, dose_norm


def get_bragg_peak_info(pdd, z_coords):
    """Get Bragg peak position and dose."""
    peak_idx = np.argmax(pdd)
    peak_depth = z_coords[peak_idx]
    peak_dose = pdd[peak_idx]
    return peak_depth, peak_dose, peak_idx


def get_lateral_profile(dose_2d, x_coords, z_coords, target_depth):
    """Extract lateral profile at specified depth."""
    iz = np.argmin(np.abs(z_coords - target_depth))
    actual_depth = z_coords[iz]
    profile = dose_2d[iz, :]
    return x_coords, profile, actual_depth


def interpolate_to_common_grid(x_orig, profile_orig, x_target):
    """Interpolate profile to common x grid."""
    from scipy.interpolate import interp1d

    # Only interpolate where we have valid data
    mask = ~np.isnan(profile_orig) & (profile_orig > 0)
    if np.sum(mask) < 2:
        return np.full_like(x_target, np.nan)

    f = interp1d(x_orig[mask], profile_orig[mask],
                 kind='linear', bounds_error=False, fill_value=0.0)
    return f(x_target)


def calculate_relative_dose_at_depth(pdd, z_coords, depth_mm, peak_dose):
    """Calculate relative dose at specific depth normalized to peak."""
    iz = np.argmin(np.abs(z_coords - depth_mm))
    actual_depth = z_coords[iz]
    dose = pdd[iz]
    rel_dose = dose / peak_dose if peak_dose > 0 else 0
    return rel_dose, dose, actual_depth


def calculate_fwhm_sigma(x_coords, profile):
    """Calculate FWHM and Gaussian sigma from profile."""
    max_dose = np.max(profile)
    if max_dose < 1e-15:
        return None, None

    half_max = max_dose / 2.0
    above_half = profile > half_max

    if not np.any(above_half):
        return None, None

    fwhm = x_coords[above_half][-1] - x_coords[above_half][0]
    sigma = fwhm / 2.355
    return fwhm, sigma


def main():
    print("="*70)
    print("SM_2D vs MOQUI Validation Comparison")
    print("="*70)

    # Paths
    project_dir = Path('/workspaces/SM_2D')
    moqui_file = project_dir / 'validation' / '150MeV.raw'
    sm2d_dose_file = project_dir / 'results' / 'dose_2d.txt'
    sm2d_pdd_file = project_dir / 'results' / 'pdd.txt'
    output_dir = project_dir / 'validation'

    energy_mev = 150

    # Load MOQUI gold standard
    print("\n[1] Loading MOQUI gold standard data...")
    dose_moqui_3d, dim_x, dim_y, dim_z = load_moqui_3d_dose(moqui_file, energy_mev)
    print(f"    Grid: {dim_x} x {dim_y} x {dim_z} (1mm voxels)")

    dose_moqui_2d, x_moqui, z_moqui = extract_moqui_central_slice(
        dose_moqui_3d, dim_x, dim_y, dim_z
    )

    # MOQUI PDD (sum over lateral direction)
    pdd_moqui = np.sum(dose_moqui_2d, axis=1) * (x_moqui[1] - x_moqui[0])

    # Load SM_2D results
    print("\n[2] Loading SM_2D simulation results...")
    dose_sm2d_2d, dose_sm2d_norm_2d, x_sm2d, z_sm2d = load_sm2d_dose(sm2d_dose_file)
    dx_sm2d = float(np.median(np.diff(x_sm2d))) if len(x_sm2d) > 1 else 0.0
    dz_sm2d = float(np.median(np.diff(z_sm2d))) if len(z_sm2d) > 1 else 0.0
    print(f"    Grid: {len(x_sm2d)} x {len(z_sm2d)} (dx={dx_sm2d:.3f}mm, dz={dz_sm2d:.3f}mm)")
    print(f"    X range: [{x_sm2d[0]:.2f}, {x_sm2d[-1]:.2f}] mm")
    print(f"    Z range: [{z_sm2d[0]:.2f}, {z_sm2d[-1]:.2f}] mm")

    z_sm2d_pdd, dose_sm2d_pdd, dose_sm2d_pdd_norm = load_sm2d_pdd(sm2d_pdd_file)

    # Get Bragg peak info
    print("\n[3] Bragg Peak Position Comparison")
    print("-" * 70)

    peak_moqui_depth, peak_moqui_dose, _ = get_bragg_peak_info(pdd_moqui, z_moqui)
    peak_sm2d_depth, peak_sm2d_dose, _ = get_bragg_peak_info(dose_sm2d_pdd, z_sm2d_pdd)

    bragg_diff_mm = peak_sm2d_depth - peak_moqui_depth
    bragg_diff_pct = (bragg_diff_mm / peak_moqui_depth) * 100

    print(f"    MOQUI Bragg Peak:     {peak_moqui_depth:.2f} mm")
    print(f"    SM_2D Bragg Peak:     {peak_sm2d_depth:.2f} mm")
    print(f"    Difference:           {bragg_diff_mm:+.2f} mm ({bragg_diff_pct:+.2f}%)")

    # NIST reference for 150 MeV protons in water
    nist_range_150mev = 158.3  # mm from PSTAR
    nist_diff = peak_sm2d_depth - nist_range_150mev
    nist_diff_pct = (nist_diff / nist_range_150mev) * 100
    print(f"    NIST Reference:       {nist_range_150mev:.2f} mm")
    print(f"    SM_2D vs NIST:        {nist_diff:+.2f} mm ({nist_diff_pct:+.2f}%)")

    # Comparison depths
    comparison_depths = [20.0, 100.0, 140.0]  # mm

    print("\n[4] Relative Dose Comparison (normalized to maximum)")
    print("-" * 70)
    print(f"    {'Depth':<10} {'MOQUI':<15} {'SM_2D':<15} {'Diff (%)':<15}")
    print("-" * 70)

    rel_dose_comparison = []
    for depth in comparison_depths:
        rel_moqui, dose_moqui, actual_moqui = calculate_relative_dose_at_depth(
            pdd_moqui, z_moqui, depth, peak_moqui_dose
        )
        rel_sm2d, dose_sm2d, actual_sm2d = calculate_relative_dose_at_depth(
            dose_sm2d_pdd, z_sm2d_pdd, depth, peak_sm2d_dose
        )

        diff_pct = ((rel_sm2d - rel_moqui) / rel_moqui * 100) if rel_moqui > 0 else 0

        print(f"    {actual_moqui:>6.1f} mm  {rel_moqui:>13.6f}  {rel_sm2d:>13.6f}  {diff_pct:>13.2f}%")
        rel_dose_comparison.append({
            'depth': actual_moqui,
            'moqui': rel_moqui,
            'sm2d': rel_sm2d,
            'diff_pct': diff_pct
        })

    # Lateral profile comparison
    print("\n[5] Lateral Profile Comparison")
    print("-" * 70)

    lateral_results = {}
    for depth in comparison_depths:
        # MOQUI profile
        x_m, prof_m, actual_m = get_lateral_profile(dose_moqui_2d, x_moqui, z_moqui, depth)

        # SM_2D profile
        x_s, prof_s, actual_s = get_lateral_profile(dose_sm2d_2d, x_sm2d, z_sm2d, depth)

        # Normalize profiles to their own maxima
        prof_m_norm = prof_m / (np.max(prof_m) + 1e-10)
        prof_s_norm = prof_s / (np.max(prof_s) + 1e-10)

        # Calculate sigma
        _, sigma_m = calculate_fwhm_sigma(x_m, prof_m_norm)
        _, sigma_s = calculate_fwhm_sigma(x_s, prof_s_norm)

        print(f"\n    Depth: {actual_m:.1f} mm")
        print(f"        MOQUI sigma:  {sigma_m:.3f} mm" if sigma_m else "        MOQUI sigma:  N/A")
        print(f"        SM_2D sigma:  {sigma_s:.3f} mm" if sigma_s else "        SM_2D sigma:  N/A")

        if sigma_m and sigma_s:
            sigma_diff = sigma_s - sigma_m
            sigma_diff_pct = (sigma_diff / sigma_m) * 100
            print(f"        Difference:   {sigma_diff:+.3f} mm ({sigma_diff_pct:+.1f}%)")

        lateral_results[depth] = {
            'x_moqui': x_m,
            'prof_moqui': prof_m_norm,
            'x_sm2d': x_s,
            'prof_sm2d': prof_s_norm,
            'sigma_moqui': sigma_m,
            'sigma_sm2d': sigma_s
        }

    # Create comparison plot
    print("\n[6] Generating comparison plots...")
    create_comparison_plots(
        x_moqui, z_moqui, dose_moqui_2d, pdd_moqui,
        x_sm2d, z_sm2d, dose_sm2d_2d, dose_sm2d_pdd,
        z_sm2d_pdd, lateral_results, comparison_depths,
        peak_moqui_depth, peak_sm2d_depth,
        output_dir
    )

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    # Assess accuracy
    bragg_pass = abs(bragg_diff_mm) < 3.0  # 3mm tolerance
    bragg_pct_pass = abs(bragg_diff_pct) < 2.0  # 2% tolerance

    print(f"\n1. Bragg Peak Position:")
    print(f"   MOQUI: {peak_moqui_depth:.2f} mm")
    print(f"   SM_2D: {peak_sm2d_depth:.2f} mm")
    print(f"   Error: {bragg_diff_mm:+.2f} mm ({bragg_diff_pct:+.2f}%)")
    print(f"   Status: {'PASS' if bragg_pass else 'FAIL'} (tolerance: ±3mm, ±2%)")

    print(f"\n2. Relative Doses (normalized to max):")
    for i, r in enumerate(rel_dose_comparison):
        status = 'PASS' if abs(r['diff_pct']) < 5.0 else 'FAIL'
        print(f"   {r['depth']:.0f} mm: MOQUI={r['moqui']:.4f}, SM_2D={r['sm2d']:.4f}, "
              f"Diff={r['diff_pct']:+.1f}% [{status}]")

    print(f"\n3. Lateral Profile Sigma (beam spread):")
    for depth in comparison_depths:
        r = lateral_results[depth]
        if r['sigma_moqui'] and r['sigma_sm2d']:
            diff = r['sigma_sm2d'] - r['sigma_moqui']
            pct = (diff / r['sigma_moqui']) * 100
            status = 'PASS' if abs(pct) < 10.0 else 'FAIL'
            print(f"   {depth:.0f} mm: MOQUI={r['sigma_moqui']:.3f}mm, SM_2D={r['sigma_sm2d']:.3f}mm, "
                  f"Diff={diff:+.3f}mm ({pct:+.1f}%) [{status}]")

    print("\n" + "="*70)
    print(f"Plots saved to: {output_dir}/comparison_*.png")
    print("="*70)

    return {
        'bragg_peak_moqui': peak_moqui_depth,
        'bragg_peak_sm2d': peak_sm2d_depth,
        'bragg_diff_mm': bragg_diff_mm,
        'bragg_diff_pct': bragg_diff_pct,
        'relative_doses': rel_dose_comparison,
        'lateral_sigma': {d: lateral_results[d]['sigma_sm2d'] for d in comparison_depths}
    }


def create_comparison_plots(x_m, z_m, dose_m_2d, pdd_m,
                            x_s, z_s, dose_s_2d, pdd_s, z_s_pdd,
                            lateral_results, depths,
                            peak_m, peak_s, output_dir):
    """Create comprehensive comparison plots."""

    fig = plt.figure(figsize=(18, 12))

    # 1. MOQUI 2D dose
    ax1 = plt.subplot(3, 4, 1)
    extent_m = [x_m[0], x_m[-1], 0, z_m[0]]
    im1 = ax1.imshow(dose_m_2d, extent=extent_m, aspect='auto', cmap='hot', origin='upper')
    ax1.set_ylabel('Depth (mm)')
    ax1.set_xlabel('X (mm)')
    ax1.set_title('MOQUI 2D Dose')
    plt.colorbar(im1, ax=ax1, label='Dose (a.u.)')

    # 2. SM_2D 2D dose
    ax2 = plt.subplot(3, 4, 2)
    # SM_2D z increases from 0 (surface) to max depth, use z_s[-1] for extent
    extent_s = [x_s[0], x_s[-1], 0, z_s[-1]]
    im2 = ax2.imshow(dose_s_2d, extent=extent_s, aspect='auto', cmap='hot', origin='upper')
    ax2.set_ylabel('Depth (mm)')
    ax2.set_xlabel('X (mm)')
    ax2.set_title('SM_2D 2D Dose')
    plt.colorbar(im2, ax=ax2, label='Dose (Gy)')

    # 3. PDD comparison
    ax3 = plt.subplot(3, 4, 3)
    pdd_m_norm = pdd_m / np.max(pdd_m)
    pdd_s_norm = pdd_s / np.max(pdd_s)
    ax3.plot(z_m, pdd_m_norm, 'b-', linewidth=2, label='MOQUI', alpha=0.7)
    ax3.plot(z_s_pdd, pdd_s_norm, 'r--', linewidth=2, label='SM_2D', alpha=0.7)
    ax3.axvline(peak_m, color='b', linestyle=':', alpha=0.5, label=f'MOQUI Peak: {peak_m:.1f}mm')
    ax3.axvline(peak_s, color='r', linestyle=':', alpha=0.5, label=f'SM_2D Peak: {peak_s:.1f}mm')
    for d in depths:
        ax3.axvline(d, color='gray', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Depth (mm)')
    ax3.set_ylabel('Normalized Dose')
    ax3.set_title('PDD Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. PDD difference
    ax4 = plt.subplot(3, 4, 4)
    # Interpolate to common grid
    from scipy.interpolate import interp1d
    f_m = interp1d(z_m, pdd_m_norm, bounds_error=False, fill_value=0)
    z_common = np.linspace(0, max(z_m[-1], z_s_pdd[-1]), 500)
    pdd_m_interp = f_m(z_common)
    f_s = interp1d(z_s_pdd, pdd_s_norm, bounds_error=False, fill_value=0)
    pdd_s_interp = f_s(z_common)
    diff = pdd_s_interp - pdd_m_interp
    ax4.plot(z_common, diff * 100, 'g-', linewidth=1)
    ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Depth (mm)')
    ax4.set_ylabel('Difference (%)')
    ax4.set_title('PDD Difference (SM_2D - MOQUI)')
    ax4.grid(True, alpha=0.3)

    # 5-7. Lateral profiles at each depth
    for i, depth in enumerate(depths):
        ax = plt.subplot(3, 4, 5 + i)
        r = lateral_results[depth]

        ax.plot(r['x_moqui'], r['prof_moqui'], 'b-', linewidth=2, label='MOQUI', alpha=0.7)
        ax.plot(r['x_sm2d'], r['prof_sm2d'], 'r--', linewidth=2, label='SM_2D', alpha=0.7)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Normalized Dose')
        ax.set_title(f'Lateral Profile @ {depth:.0f} mm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)

    # 8-10. Lateral profile differences
    for i, depth in enumerate(depths):
        ax = plt.subplot(3, 4, 9 + i)
        r = lateral_results[depth]

        # Create common x grid
        x_min = min(r['x_moqui'].min(), r['x_sm2d'].min())
        x_max = max(r['x_moqui'].max(), r['x_sm2d'].max())
        x_common = np.linspace(x_min, x_max, 200)

        from scipy.interpolate import interp1d
        prof_m_interp = interp1d(r['x_moqui'], r['prof_moqui'],
                                  bounds_error=False, fill_value=0)(x_common)
        prof_s_interp = interp1d(r['x_sm2d'], r['prof_sm2d'],
                                  bounds_error=False, fill_value=0)(x_common)

        diff = prof_s_interp - prof_m_interp
        ax.plot(x_common, diff * 100, 'g-', linewidth=1)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Difference (%)')
        ax.set_title(f'Profile Diff @ {depth:.0f} mm')
        ax.grid(True, alpha=0.3)

    # Summary text box
    ax_summary = plt.subplot(3, 4, 12)
    ax_summary.axis('off')

    summary_text = f"""
VALIDATION SUMMARY
{'='*30}

Bragg Peak:
  MOQUI: {peak_m:.1f} mm
  SM_2D: {peak_s:.1f} mm
  Diff: {peak_s - peak_m:+.1f} mm

Lateral Sigma:
"""

    for depth in depths:
        r = lateral_results[depth]
        if r['sigma_moqui'] and r['sigma_sm2d']:
            diff_pct = ((r['sigma_sm2d'] - r['sigma_moqui']) / r['sigma_moqui'] * 100)
            summary_text += f"  {depth:.0f} mm: {r['sigma_moqui']:.2f} vs {r['sigma_sm2d']:.2f} mm ({diff_pct:+.1f}%)\n"

    ax_summary.text(0.1, 0.5, summary_text, fontsize=10,
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('SM_2D vs MOQUI Validation Comparison (150 MeV Protons)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = output_dir / 'comparison_150MeV.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
