#!/usr/bin/env python3
"""
Analyze MOQUI gold standard data to extract beam parameters.
Focus on shallow depths where beam spreading is minimal.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit


def load_moqui_3d_dose(filepath, energy):
    """Load MOQUI 3D dose data."""
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
    dose_3d = dose_flat.reshape((dim_z, dim_y, dim_x), order='C')
    return dose_3d, dim_x, dim_y, dim_z


def extract_central_slice(dose_3d, dim_x, dim_y, dim_z):
    """Extract central Y slice."""
    iy_center = dim_y // 2
    dose_2d = dose_3d[:, iy_center, :]

    # Voxels are 1mm
    x_coords = np.arange(dim_x) * 1.0 + (-dim_x / 2.0)
    z_physical = (dim_z - 1 - np.arange(dim_z)) * 1.0

    return dose_2d, x_coords, z_physical


def gaussian(x, amplitude, sigma, center):
    """Gaussian function for fitting."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)


def fit_gaussian_sigma(x_coords, profile):
    """Fit Gaussian to profile and return sigma."""
    # Remove background (take min value as baseline)
    baseline = np.min(profile)
    profile_sub = profile - baseline

    # Initial guess
    amplitude = np.max(profile_sub)
    center = x_coords[np.argmax(profile_sub)]
    sigma_guess = 2.0  # initial guess

    try:
        # Fit Gaussian
        popt, pcov = curve_fit(gaussian, x_coords, profile_sub,
                               p0=[amplitude, sigma_guess, center],
                               maxfev=5000)
        amplitude_fit, sigma_fit, center_fit = popt
        sigma_error = np.sqrt(np.diag(pcov))[1]

        # Calculate R^2
        y_fit = gaussian(x_coords, *popt)
        ss_res = np.sum((profile_sub - y_fit)**2)
        ss_tot = np.sum((profile_sub - np.mean(profile_sub))**2)
        r_squared = 1 - (ss_res / ss_tot)

        return sigma_fit, sigma_error, r_squared, center_fit, amplitude_fit
    except Exception as e:
        print(f"    Fit failed: {e}")
        return None, None, None, None, None


def calculate_fwhm(x_coords, profile):
    """Calculate FWHM from profile."""
    max_dose = np.max(profile)
    baseline = np.min(profile)
    half_max = baseline + (max_dose - baseline) / 2.0

    above_half = profile >= half_max
    if not np.any(above_half):
        return None

    fwhm = x_coords[above_half][-1] - x_coords[above_half][0]
    sigma = fwhm / 2.355
    return sigma


def main():
    print("="*70)
    print("MOQUI Beam Parameter Analysis")
    print("="*70)

    # Load data
    filepath = Path('/workspaces/SM_2D/validation/150MeV.raw')
    energy = 150

    print("\n[1] Loading MOQUI data...")
    dose_3d, dim_x, dim_y, dim_z = load_moqui_3d_dose(filepath, energy)
    print(f"    Grid: {dim_x} x {dim_y} x {dim_z} (1mm voxels)")

    dose_2d, x_coords, z_physical = extract_central_slice(dose_3d, dim_x, dim_y, dim_z)

    # Analyze at multiple shallow depths to find intrinsic beam size
    print("\n[2] Analyzing lateral profiles at shallow depths...")
    print("-" * 70)

    shallow_depths = [0, 5, 10, 15, 20, 30, 40]  # mm

    results = []

    for depth in shallow_depths:
        iz = np.argmin(np.abs(z_physical - depth))
        actual_depth = z_physical[iz]
        profile = dose_2d[iz, :]

        # Only analyze if there's dose
        if np.max(profile) < 1e-10:
            continue

        # Method 1: Gaussian fit
        sigma_fit, sigma_err, r2, center, amp = fit_gaussian_sigma(x_coords, profile)

        # Method 2: FWHM
        sigma_fwhm = calculate_fwhm(x_coords, profile)

        print(f"\n    Depth: {actual_depth:.1f} mm")
        if sigma_fit is not None:
            print(f"        Gaussian fit: σ = {sigma_fit:.4f} ± {sigma_err:.4f} mm (R² = {r2:.4f})")
            results.append({
                'depth': actual_depth,
                'sigma_fit': sigma_fit,
                'sigma_err': sigma_err,
                'r2': r2
            })
        if sigma_fwhm is not None:
            print(f"        FWHM method:  σ = {sigma_fwhm:.4f} mm")

    # Extrapolate to surface (depth = 0) to find intrinsic beam size
    print("\n[3] Extrapolating to surface for intrinsic beam size...")
    print("-" * 70)

    if len(results) >= 2:
        # Fit sigma vs depth to find sigma at z=0
        # The beam sigma grows with depth due to scattering: sigma(z) = sqrt(sigma_0^2 + (theta_MCS * z)^2)
        # For small depths, we can use linear approximation: sigma(z) ≈ sigma_0 + scattering_rate * z

        depths = np.array([r['depth'] for r in results])
        sigmas = np.array([r['sigma_fit'] for r in results])

        # Linear fit to extrapolate to z=0
        coeffs = np.polyfit(depths, sigmas, 1)
        sigma_0 = coeffs[1]  # intercept at z=0
        scattering_rate = coeffs[0]

        print(f"    Linear fit: sigma(depth) = {scattering_rate:.4f} * depth + {sigma_0:.4f}")
        print(f"    Estimated intrinsic beam sigma at surface: σ₀ = {sigma_0:.4f} mm")

        # Also use only very shallow depths (0-10mm) for better estimate
        shallow_mask = depths <= 20
        if np.sum(shallow_mask) >= 2:
            coeffs_shallow = np.polyfit(depths[shallow_mask], sigmas[shallow_mask], 1)
            sigma_0_shallow = coeffs_shallow[1]
            print(f"    Using 0-20mm only: σ₀ = {sigma_0_shallow:.4f} mm")

    # Plot profiles and fits
    print("\n[4] Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Lateral profiles at shallow depths
    ax1 = axes[0, 0]
    for depth in [0, 10, 20, 40]:
        iz = np.argmin(np.abs(z_physical - depth))
        actual_depth = z_physical[iz]
        profile = dose_2d[iz, :]
        profile_norm = profile / (np.max(profile) + 1e-10)

        ax1.plot(x_coords, profile_norm, label=f'{actual_depth:.0f} mm')

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Normalized Dose')
    ax1.set_title('Lateral Profiles at Shallow Depths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Sigma vs depth
    ax2 = axes[0, 1]
    if len(results) >= 2:
        depths = np.array([r['depth'] for r in results])
        sigmas = np.array([r['sigma_fit'] for r in results])
        sigmas_err = np.array([r['sigma_err'] for r in results])

        ax2.errorbar(depths, sigmas, yerr=sigmas_err, fmt='o-', label='Data', capsize=3)

        # Plot linear fit
        z_fit = np.linspace(0, max(depths), 50)
        sigma_fit_line = coeffs[0] * z_fit + coeffs[1]
        ax2.plot(z_fit, sigma_fit_line, '--', label=f'Linear fit: σ(z) = {coeffs[0]:.4f}z + {coeffs[1]:.4f}')

        ax2.axhline(sigma_0, color='r', linestyle=':', label=f'σ₀ = {sigma_0:.4f} mm')

    ax2.set_xlabel('Depth (mm)')
    ax2.set_ylabel('Beam Sigma (mm)')
    ax2.set_title('Beam Spread vs Depth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: 2D dose map
    ax3 = axes[1, 0]
    extent = [x_coords[0], x_coords[-1], 0, z_physical[0]]
    im = ax3.imshow(dose_2d, extent=extent, aspect='auto', cmap='hot', origin='upper')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Depth (mm)')
    ax3.set_title('2D Dose Distribution')
    plt.colorbar(im, ax=ax3, label='Dose (a.u.)')

    # Plot 4: PDD with Bragg peak
    ax4 = axes[1, 1]
    pdd = np.sum(dose_2d, axis=1) * (x_coords[1] - x_coords[0])
    peak_idx = np.argmax(pdd)
    peak_depth = z_physical[peak_idx]

    ax4.plot(z_physical, pdd / np.max(pdd), 'b-', linewidth=2)
    ax4.axvline(peak_depth, color='r', linestyle='--', label=f'Bragg Peak: {peak_depth:.1f} mm')
    ax4.axvline(20, color='g', linestyle=':', alpha=0.5, label='20 mm')
    ax4.set_xlabel('Depth (mm)')
    ax4.set_ylabel('Normalized Dose')
    ax4.set_title('PDD (Depth Dose)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('MOQUI Beam Analysis - 150 MeV Protons', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = Path('/workspaces/SM_2D/validation/beam_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()

    # Summary and recommendation
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATION")
    print("="*70)
    print(f"\nEstimated intrinsic beam sigma: {sigma_0_shallow:.4f} mm")
    print(f"This is the beam size at the surface (depth = 0)")
    print(f"\nFor sim.ini, set:")
    print(f"    sigma_x_mm = {sigma_0_shallow:.4f}")
    print("="*70)

    return sigma_0_shallow


if __name__ == '__main__':
    import sys
    try:
        sigma = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
