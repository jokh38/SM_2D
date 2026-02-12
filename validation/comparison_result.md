# SM_2D vs MOQUI Comparison Result (GPU Verified)

Date: 2026-02-07

## Summary

A fresh GPU run and fresh comparison were completed successfully.

- GPU execution: **successful**
- Comparison pipeline: **updated and rerun**
- Result vs MOQUI: **still mismatched** (Bragg depth and lateral spread)

## GPU Environment

- Host GPU visibility confirmed by `nvidia-smi`
- GPUs detected: `4 x NVIDIA GeForce RTX 2080`
- Driver: `575.64.03`
- CUDA runtime shown by NVIDIA-SMI: `12.9`

## Why previous "cannot use GPU" happened

Two issues were involved:

1. Prior shell/session had GPU visibility failure (`cudaGetDeviceCount`/NVML unavailable).
2. Current default `sim.ini` over-allocates VRAM on 8 GB cards:
   - grid `Nx=200`, `Nz=640`
   - each `DevicePsiC` ~4.0 GB
   - second `DevicePsiC` allocation fails (`out of memory`)

Evidence: `validation/latest_gpu_run_oom.log`

## Validation Script Fixes Applied

File: `validation/compare_sm2d_moqui.py`

- Replaced fixed-row skipping with comment-aware parsing:
  - `np.loadtxt(..., comments='#')`
- Fixed SM_2D 2D reshape order to match file write order:
  - `order='C'` (x varies fastest inside each z)
- Added fallback normalization if normalized columns are absent
- Fixed printed spacing to report measured `dx`, `dz` from loaded data

## Fresh GPU Run Used For Comparison

Config: `validation/gpu_compare.ini`

- `Nx=100`, `Nz=320`
- `dx_mm=1.0`, `dz_mm=1.0`
- `output_dir=/workspaces/SM_2D/results`
- `normalize_dose=true`

Run evidence: `validation/latest_gpu_run.log`

Key log lines:

- `Using GPU transport (Vavilov energy straggling)`
- `GPU: NVIDIA GeForce RTX 2080`
- `K1-K6 pipeline: completed 200 iterations`
- `GPU transport complete.`

Fresh output files:

- `/workspaces/SM_2D/results/dose_2d.txt`
- `/workspaces/SM_2D/results/pdd.txt`

## Comparison Results (Fresh GPU Outputs)

Source log: `validation/latest_comparison.log`

### Bragg peak

- MOQUI: `154.00 mm`
- SM_2D: `159.00 mm`
- Difference: `+5.00 mm` (`+3.25%`)

### Relative dose (normalized to peak)

- 20 mm: MOQUI `0.276557`, SM_2D `0.290709`, diff `+5.12%`
- 100 mm: MOQUI `0.353125`, SM_2D `0.330796`, diff `-6.32%`
- 140 mm: MOQUI `0.508965`, SM_2D `0.457522`, diff `-10.11%`

### Lateral sigma

- 20 mm: MOQUI `5.520 mm`, SM_2D `7.643 mm`, diff `+38.5%`
- 100 mm: MOQUI `5.520 mm`, SM_2D `7.643 mm`, diff `+38.5%`
- 140 mm: MOQUI `6.369 mm`, SM_2D `7.643 mm`, diff `+20.0%`

Generated figure: `validation/comparison_150MeV.png`

## Conclusion

The GPU comparison now runs correctly with fresh, traceable artifacts. Remaining discrepancies with MOQUI are reproducible after parser fixes and are likely model/transport behavior differences, not stale-file or header-parsing artifacts.
