# Angular Discretization Mitigation (2026-02-12)

## Mitigation Implemented
- File: `src/cuda/device/device_bucket.cuh`
- Path: `device_emit_component_to_bucket_4d(...)`
- Change:
  - Replaced nearest-bin theta rebinning with a sub-bin, center-based theta split.
  - Neighbor emission weight now uses squared normalized offset:
    - `w_neighbor = (|delta_theta| / dtheta)^2` (clamped to `[0, 1]`)
    - `w_center = 1 - w_neighbor`
  - Energy rebinning remains unchanged (single-bin), to avoid introducing extra energy-space diffusion.

## Why This Shape
- Linear theta interpolation at coarse `N_theta` strongly over-spread variance because one-bin spacing is large.
- The squared-offset split is a second-moment-preserving compromise for unresolved sub-bin angles:
  - tiny sub-bin offsets produce tiny cross-bin transfer,
  - coarse-grid angular variance no longer collapses,
  - overshoot is avoided.

## Regression Harness Updates
- File: `docs/val_test/check_theta_resolution_regression.py`
- Updated to mitigation-era pass criteria:
  - Case C (`N_theta=36`) must be in near-unity `R_theta` band (not collapsed),
  - Case D (`N_theta=360`) must remain near-unity,
  - `|R_theta(D)-R_theta(C)|` must stay bounded.
- `--run` now archives fresh debug exports into:
  - `results/one_step_case_C/`
  - `results/one_step_case_D/`
  before recomputing `results/one_step_summary/metrics.csv`.

## CTest Integration

The regression checker is now integrated into CTest as a validation gate:

```bash
# Configure with validation tests enabled
cd build
cmake .. -DSM2D_ENABLE_VALIDATION_TESTS=ON

# Run all validation tests
ctest -L validation

# Run only angular regression test
ctest -R angular_resolution_regression

# Run with verbose output
ctest -R angular_resolution_regression -V
```

Test properties:
- Name: `angular_resolution_regression`
- Timeout: 600 seconds
- Labels: `validation`, `angular`
- Only runs when `SM2D_ENABLE_VALIDATION_TESTS=ON`

## Diagnostic Ratio: sigma_theta_pred / dtheta_bin

A new diagnostic metric has been added to flag under-resolved angular settings:

- **Formula**: `sigma_theta_pred / dtheta_bin`
- **Where**:
  - `sigma_theta_pred` = Highland-predicted angular scattering (radians)
  - `dtheta_bin` = angular bin width = 2π / N_theta (radians)
- **Interpretation**:
  - `<< 1`: Well-resolved (bin width much larger than scattering)
  - `≈ 1`: Marginally resolved (potential discretization effects)
  - `> 1`: Under-resolved (bin width smaller than scattering, expect variance collapse)

This ratio appears in:
- `results/one_step_summary/metrics.csv` as `sigma_over_dtheta_bin`
- Angular Metrics table in `results/one_step_summary/report.md`

## Threshold Guidelines

Current regression thresholds (configurable via command-line args):

| Parameter | Default | Purpose |
|-----------|----------|----------|
| `--min-r-c` | 0.70 | Minimum R_theta for Case C (coarse, N_theta=36) |
| `--max-r-c` | 1.60 | Maximum R_theta for Case C |
| `--min-r-d` | 0.80 | Minimum R_theta for Case D (fine, N_theta=360) |
| `--max-r-d` | 1.60 | Maximum R_theta for Case D |
| `--max-abs-delta` | 0.50 | Maximum |R_theta(D)-R_theta(C)| |

Re-tune these thresholds only if physics/model changes intentionally shift one-step R_theta bands.

## Validation Run
```bash
python3 docs/val_test/check_theta_resolution_regression.py --run
python3 docs/val_test/check_theta_resolution_regression.py
```

Observed (post-mitigation):
- Case C: R_theta = 1.084906, sigma/dtheta_bin = 0.007234
- Case D: R_theta = 1.194607, sigma/dtheta_bin = 0.072338
- Delta: R_theta(D)-R_theta(C) = 0.109701
- Regression status: PASS
- Interpretation: Both cases show well-resolved angular resolution (ratio << 1), with mitigation preventing variance collapse at coarse N_theta=36.
