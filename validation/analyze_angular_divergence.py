#!/usr/bin/env python3
"""
Analyze MOQUI reference data to extract initial angular divergence.

The beam width evolution follows:
sigma²(z) = sigma₀² + (theta_divergence * z)² + (theta_MCS * z)²

Where:
- sigma₀ is the initial beam size at surface
- theta_divergence is the initial angular divergence
- theta_MCS is the multiple Coulomb scattering contribution

For very shallow depths, theta_MCS is negligible, so we can estimate
theta_divergence from the slope of sigma vs z near the surface.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


def load_moqui_data(filepath, energy):
    """Load MOQUI 3D dose data."""
    with open(filepath, 'rb') as f:
        data = f.read()

    num_elements = len(data) // 8
    dose_flat = np.frombuffer(data, dtype=np.float64)

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
    x_coords = np.arange(dim_x) * 1.0 + (-dim_x / 2.0)
    z_physical = (dim_z - 1 - np.arange(dim_z)) * 1.0
    return dose_2d, x_coords, z_physical


def gaussian(x, amplitude, sigma, center):
    """Gaussian function for fitting."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)


def fit_sigma(x_coords, profile):
    """Fit Gaussian and return sigma with error."""
    baseline = np.min(profile)
    profile_sub = profile - baseline

    amplitude = np.max(profile_sub)
    center = x_coords[np.argmax(profile_sub)]
    sigma_guess = 5.0

    try:
        popt, pcov = curve_fit(gaussian, x_coords, profile_sub,
                               p0=[amplitude, sigma_guess, center],
                               maxfev=5000)
        sigma_fit = popt[1]
        sigma_err = np.sqrt(np.diag(pcov))[1]

        # R² calculation
        y_fit = gaussian(x_coords, *popt)
        ss_res = np.sum((profile_sub - y_fit)**2)
        ss_tot = np.sum((profile_sub - np.mean(profile_sub))**2)
        r_squared = 1 - (ss_res / ss_tot)

        return sigma_fit, sigma_err, r_squared
    except:
        return None, None, None


def sigma_evolution_model(z, sigma_0, theta_div):
    """
    Model: sigma²(z) = sigma₀² + (theta_div * z)²
    Taking sqrt: sigma(z) = sqrt(sigma₀² + (theta_div * z)²)
    """
    return np.sqrt(sigma_0**2 + (theta_div * z)**2)


def main():
    print("="*70)
    print("MOQUI Angular Divergence Analysis")
    print("="*70)

    # Load data
    filepath = Path('/workspaces/SM_2D/validation/150MeV.raw')
    energy = 150

    print("\n[1] Loading MOQUI data...")
    dose_3d, dim_x, dim_y, dim_z = load_moqui_data(filepath, energy)
    print(f"    Grid: {dim_x} x {dim_y} x {dim_z} (1mm voxels)")

    dose_2d, x_coords, z_physical = extract_central_slice(dose_3d, dim_x, dim_y, dim_z)

    # Analyze sigma at all depths
    print("\n[2] Fitting Gaussian profiles at all depths...")
    sigmas = []
    sigma_errs = []
    depths_valid = []

    for iz, z in enumerate(z_physical):
        profile = dose_2d[iz, :]

        # Skip if no significant dose
        if np.max(profile) < 1e-10 * np.max(dose_2d):
            continue

        sigma_fit, sigma_err, r2 = fit_sigma(x_coords, profile)

        if sigma_fit is not None and r2 > 0.99:
            sigmas.append(sigma_fit)
            sigma_errs.append(sigma_err)
            depths_valid.append(z)

    sigmas = np.array(sigmas)
    sigma_errs = np.array(sigma_errs)
    depths_valid = np.array(depths_valid)

    print(f"    Valid fits: {len(sigmas)} out of {len(z_physical)} depths")

    # Focus on shallow depths for divergence estimation
    # (before significant multiple scattering accumulates)
    shallow_mask = depths_valid <= 50  # First 50mm

    depths_shallow = depths_valid[shallow_mask]
    sigmas_shallow = sigmas[shallow_mask]
    errs_shallow = sigma_errs[shallow_mask]

    print(f"\n[3] Analyzing shallow depths (0-50mm) for divergence...")

    # Method 1: Linear fit to sigma vs z (for small theta)
    # If theta is small: sigma(z) ≈ sigma_0 + theta * z
    coeffs_linear = np.polyfit(depths_shallow, sigmas_shallow, 1,
                               w=1/errs_shallow)  # Weighted by error
    theta_linear = coeffs_linear[0]  # Slope ≈ divergence
    sigma_0_linear = coeffs_linear[1]

    print(f"    Linear fit: sigma(z) = {theta_linear:.6f} * z + {sigma_0_linear:.4f}")
    print(f"    Divergence (linear): θ = {theta_linear:.6f} rad = {np.rad2deg(theta_linear):.4f}°")

    # Method 2: Proper fit using sigma(z) = sqrt(sigma_0² + (theta*z)²)
    try:
        popt, pcov = curve_fit(
            sigma_evolution_model,
            depths_shallow,
            sigmas_shallow,
            p0=[6.0, 0.001],  # Initial guess: sigma_0=6mm, theta=0.001rad
            sigma=errs_shallow,
            maxfev=5000
        )

        sigma_0_fit = popt[0]
        theta_div_fit = popt[1]
        param_errors = np.sqrt(np.diag(pcov))

        print(f"\n    Quadratic fit: sigma(z) = sqrt({sigma_0_fit:.4f}² + ({theta_div_fit:.6f}*z)²)")
        print(f"    sigma_0 = {sigma_0_fit:.4f} ± {param_errors[0]:.4f} mm")
        print(f"    divergence θ = {theta_div_fit:.6f} ± {param_errors[1]:.6f} rad")
        print(f"    divergence θ = {np.rad2deg(theta_div_fit):.4f}° ± {np.rad2deg(param_errors[1]):.4f}°")

    except Exception as e:
        print(f"    Quadratic fit failed: {e}")
        sigma_0_fit = sigma_0_linear
        theta_div_fit = theta_linear

    # Method 3: Check if sigma is constant (no divergence)
    # If there's no divergence, sigma should remain constant
    sigma_mean = np.mean(sigmas_shallow)
    sigma_std = np.std(sigmas_shallow)
    print(f"\n    Mean sigma (0-50mm): {sigma_mean:.4f} ± {sigma_std:.4f} mm")

    # Calculate expected scattering from Highland formula
    # For 150 MeV protons in water
    print("\n[4] Theoretical MCS estimates for comparison...")
    print(f"    (Highland formula for 150 MeV protons)")

    # Highland formula approximate: theta_MCS ≈ (13.6 MeV / beta*c*p) * sqrt(x/X0) * [1 + 0.038*ln(x/X0)]
    # For water, X0 ≈ 36.08 g/cm² = 360.8 mm
    X0_water = 360.8  # mm

    for z_check in [10, 20, 50]:
        x_over_X0 = z_check / X0_water
        # Approximate for 150 MeV protons (p ≈ 150 MeV/c, beta ≈ 0.55)
        theta_MCS_rad = (13.6 / 150) * np.sqrt(x_over_X0) * (1 + 0.038 * np.log(x_over_X0))
        sigma_MCS = theta_MCS_rad * z_check
        print(f"    At z={z_check}mm: theta_MCS ≈ {np.rad2deg(theta_MCS_rad):.4f}°, sigma_MCS ≈ {sigma_MCS:.4f} mm")

    # Create visualization
    print("\n[5] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Sigma vs depth (shallow region)
    ax1 = axes[0, 0]
    ax1.errorbar(depths_valid, sigmas, yerr=sigma_errs, fmt='o', markersize=3,
                 alpha=0.5, label='Data')

    # Zoom on shallow region
    ax1.errorbar(depths_shallow, sigmas_shallow, yerr=errs_shallow, fmt='o',
                 label='Shallow (0-50mm)', color='red', alpha=0.7)

    # Linear fit
    z_fit = np.linspace(0, 50, 100)
    sigma_linear = coeffs_linear[0] * z_fit + coeffs_linear[1]
    ax1.plot(z_fit, sigma_linear, '--', label=f'Linear: θ={coeffs_linear[0]:.6f} rad', color='blue')

    # Quadratic fit
    if 'sigma_0_fit' in locals():
        sigma_quad = sigma_evolution_model(z_fit, sigma_0_fit, theta_div_fit)
        ax1.plot(z_fit, sigma_quad, '-.', label=f'Quadratic: θ={theta_div_fit:.6f} rad', color='green', linewidth=2)

    ax1.axhline(sigma_mean, color='gray', linestyle=':', label=f'Mean: {sigma_mean:.4f} mm')
    ax1.set_xlabel('Depth (mm)')
    ax1.set_ylabel('Beam Sigma (mm)')
    ax1.set_title('Beam Width Evolution (Shallow Region)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Sigma vs depth (full range)
    ax2 = axes[0, 1]
    ax2.errorbar(depths_valid, sigmas, yerr=sigma_errs, fmt='o', markersize=2, alpha=0.3)

    # Interpolate for cleaner curve
    if len(depths_valid) > 10:
        sort_idx = np.argsort(depths_valid)
        depths_sorted = depths_valid[sort_idx]
        sigmas_sorted = sigmas[sort_idx]

        # Use interpolation
        f = interp1d(depths_sorted, sigmas_sorted, kind='cubic')
        z_smooth = np.linspace(depths_sorted[0], depths_sorted[-1], 500)
        sigma_smooth = f(z_smooth)
        ax2.plot(z_smooth, sigma_smooth, '-', linewidth=2, color='blue', alpha=0.7, label='Smoothed')

    ax2.set_xlabel('Depth (mm)')
    ax2.set_ylabel('Beam Sigma (mm)')
    ax2.set_title('Beam Width Evolution (Full Range)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals from linear fit (shallow)
    ax3 = axes[1, 0]
    residuals_shallow = sigmas_shallow - (coeffs_linear[0] * depths_shallow + coeffs_linear[1])
    ax3.plot(depths_shallow, residuals_shallow, 'o-')
    ax3.axhline(0, color='r', linestyle='--')
    ax3.set_xlabel('Depth (mm)')
    ax3.set_ylabel('Residual (mm)')
    ax3.set_title('Linear Fit Residuals')
    ax3.grid(True, alpha=0.3)

    # Plot 4: 2D dose map with beam width annotation
    ax4 = axes[1, 1]
    extent = [x_coords[0], x_coords[-1], 0, z_physical[0]]
    im = ax4.imshow(dose_2d, extent=extent, aspect='auto', cmap='hot', origin='upper')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Depth (mm)')
    ax4.set_title('2D Dose Distribution')
    plt.colorbar(im, ax=ax4, label='Dose (a.u.)')

    # Add beam width annotation at different depths
    for z_annot in [10, 50, 100]:
        iz_annot = np.argmin(np.abs(z_physical - z_annot))
        z_actual = z_physical[iz_annot]
        sigma_at_z = sigmas[np.argmin(np.abs(depths_valid - z_actual))]
        ax4.axhline(z_actual, color='cyan', linestyle='--', alpha=0.5,
                   xmin=0.3, xmax=0.7)
        ax4.text(0.75, z_actual/200, f'σ={sigma_at_z:.2f}mm',
                transform=ax4.transAxes, color='cyan', fontsize=8)

    plt.suptitle('Angular Divergence Analysis - 150 MeV Protons',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = Path('/workspaces/SM_2D/validation/angular_divergence_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {output_path}")
    plt.close()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATION")
    print("="*70)

    if abs(theta_div_fit) < 0.001:  # Less than 0.001 rad ≈ 0.057°
        print(f"\nAngular divergence is VERY SMALL: {theta_div_fit:.6f} rad")
        print(f"This indicates essentially a PENCIL BEAM geometry.")
        print(f"\nFor sim.ini, recommend:")
        print(f"    sigma_theta_rad = 0.0  # or very small value like 0.0001")
    else:
        print(f"\nAngular divergence: {theta_div_fit:.6f} rad = {np.rad2deg(theta_div_fit):.4f}°")
        print(f"\nFor sim.ini, set:")
        print(f"    sigma_theta_rad = {theta_div_fit:.6f}")

    print("\n" + "="*70)

    return theta_div_fit


if __name__ == '__main__':
    import sys
    try:
        theta = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
