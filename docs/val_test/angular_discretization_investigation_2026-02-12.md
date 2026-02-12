# Angular Discretization Investigation (2026-02-12)

## Scope
- Checked the K3 angular update/re-binning path and K4 transfer path:
  - `src/cuda/kernels/k3_finetransport.cu`
  - `src/include/core/local_bins.hpp`
  - `src/cuda/device/device_bucket.cuh`
  - `src/cuda/kernels/k4_transfer.cu`
- Re-ran one-step sensitivity cases:
  - Case C (`N_theta=36`)
  - Case D (`N_theta=360`)

## Reproduction
```bash
./run_simulation docs/val_test/cases/case_c.ini
./run_simulation docs/val_test/cases/case_d.ini
python3 docs/val_test/analyze_one_step.py
python3 docs/val_test/check_theta_resolution_regression.py
```

## Key Observations
- Case C (`N_theta=36`) remains variance-collapsed:
  - `R_theta = 0.000000`
  - output theta bin count after K4: only one bin (`theta_bin=18`)
- Case D (`N_theta=360`) keeps expected spread:
  - `R_theta = 1.083255`
  - output theta bins after K4: multiple bins (e.g. 179, 180, 181, 182)

## Root-Cause Interpretation
- At `N_theta=36` with transport range `[-0.35, +0.35]`, angular bin width is:
  - `dtheta = 0.7 / 36 = 0.01944 rad`
- One-step predicted scattering width from current metrics is:
  - `sigma_theta_pred ~= 0.00126 rad`
- Ratio is small:
  - `sigma_theta_pred / dtheta ~= 0.065`
- This means one-step angular updates are much smaller than one coarse angular bin.
  - Result: phase-space remains in one angular bin after re-binning/K4 transfer.
- With `N_theta=360`:
  - `dtheta = 0.001944 rad`, so `sigma_theta_pred / dtheta ~= 0.65`
  - bin crossing occurs and angular variance is recovered.

## Outcome
- The behavior is confirmed as a discretization-resolution limit at default `N_theta=36`, not a primary Highland formula defect.
- Added a regression check script to guard this sensitivity:
  - `docs/val_test/check_theta_resolution_regression.py`
