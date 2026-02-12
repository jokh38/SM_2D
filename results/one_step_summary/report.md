# One-Step Validation Report

## Cases

| Case | Mode | N_theta | rows(in/out) | ds_used_mm |
|---|---:|---:|---:|---:|
| A | K2 | 36 | 1/1 | 0.500000 |
| B | K3 | 36 | 1/1 | 0.437958 |
| C | K3 | 36 | 264/117 | 0.437958 |
| D | K3 | 360 | 264/147 | 0.437938 |

## Energy Loss Metrics

| Case | dE_sim | dE_pred_csda | eps_dE_csda | dE_pred_sp | eps_dE_sp | sigma_dE_k3 | z_dE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.250000 | 0.272174 | -0.081470 | 0.271999 | -0.080877 | 0.000000 | nan |
| B | 0.250000 | 0.240423 | 0.039835 | 0.238248 | 0.049327 | 0.031755 | 0.301597 |
| C | 0.250000 | 0.240423 | 0.039835 | 0.238248 | 0.049327 | 0.031755 | 0.301597 |
| D | 0.250000 | 0.240411 | 0.039884 | 0.238237 | 0.049376 | 0.031754 | 0.301967 |

## Angular Metrics

| Case | dtheta_mean | sigma_theta_pred | dtheta_bin | sigma/dtheta_bin | |dtheta|/sigma | var(dtheta) | R_theta |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.000000 | 0.001358 | 0.174533 | 0.007782 | 0.000147 | 0.000000 | 0.000000 |
| B | 0.000000 | 0.001263 | 0.174533 | 0.007234 | 0.000158 | 0.000000 | 0.000000 |
| C | 0.000011 | 0.001263 | 0.174533 | 0.007234 | 0.008869 | 0.000002 | 1.084906 |
| D | 0.000120 | 0.001263 | 0.017453 | 0.072338 | 0.095137 | 0.000002 | 1.194607 |

## Energy Closure

| Case | source_total | E_out_total | Edep | E_nuclear | E_audit_residual | source_rep_loss | closure_with_eout | closure_no_eout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 150.000000 | 149.790581 | 0.272186 | 0.084560 | -0.022327 | -0.125000 | -0.000000 | 149.790581 |
| B | 150.000000 | 149.801050 | 0.250002 | 0.074073 | -0.000125 | -0.125000 | 0.000000 | 149.801050 |
| C | 150.000000 | 149.801040 | 0.249995 | 0.074072 | -0.000119 | -0.124994 | 0.000006 | 149.801045 |
| D | 150.000000 | 149.801053 | 0.249995 | 0.074072 | -0.000120 | -0.124994 | -0.000007 | 149.801047 |

## Interpretation

- Case C (N_theta=36) gives R_theta=1.084906, indicating coarse-grid angular variance is now preserved (no collapse).
- Case D (N_theta=360) gives R_theta=1.194607, recovering near-unity variance ratio.
- Coarse/fine agreement improved (|R_theta(D)-R_theta(C)|=0.109701), consistent with an effective angular rebin mitigation.
- Energy closure with E_out included is near zero in all cases, while closure without E_out is large by design in one-step runs.

## Code Paths Checked

- `run_simulation.cpp`
- `src/gpu/gpu_transport_runner.cpp`
- `src/cuda/gpu_transport_wrapper.cu`
- `src/cuda/k1k6_pipeline.cu`
- `src/cuda/kernels/k2_coarsetransport.cu`
- `src/cuda/kernels/k3_finetransport.cu`
- `src/cuda/device/device_lut.cuh`
- `src/cuda/device/device_physics.cuh`
