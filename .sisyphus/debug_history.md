# SM_2D Debug History

## Run 2 - 2025-01-27 (Root Cause Investigation)

### Configuration
- Particle: Proton
- Energy: 150 MeV
- Grid: 200 x 640 cells (0.5mm spacing)
- GPU: NVIDIA GeForce RTX 2080

### Issues Identified and Fixed

1. **N_E Too Small** (CRITICAL)
   - Original: N_E = 32 energy bins
   - Problem: Bin width at 150 MeV = 40.26 MeV
   - Fix: Increased to N_E = 256 (bin width = 4.64 MeV)
   - Location: `src/gpu/gpu_transport_runner.cpp:89`

2. **Energy Increase Bug** (CRITICAL)
   - Problem: Geometric mean caused energy to increase when reading from bins
   - Fix: Changed to lower edge: `E = expf(log_E_min + E_bin * dlog)`
   - Location: `src/cuda/kernels/k2_coarsetransport.cu:135`, `k3_finetransport.cu:157`

3. **max_iter Too Small** (CRITICAL)
   - Original: max_iter = 100
   - Problem: With 0.3mm steps, max travel = 30mm (need 158mm)
   - Fix: Increased to max_iter = 600
   - Location: `src/cuda/k1k6_pipeline.cu:604`

4. **ActiveMask Triggering** (FIXED)
   - Problem: ActiveMask was not triggering for E < 10 MeV
   - Root cause: N_E=32 gave wrong b_E values
   - Fix: After N_E=256, ActiveMask correctly triggers at iteration 89

### Current Status (After Fixes)
- Total energy deposited: 16.9 MeV (expected ~150 MeV)
- Bragg Peak: 0.5 mm depth (expected ~158 mm)
- Active cells: Trigger correctly starting iteration 89

### Remaining Issues
1. Coarse step size (0.3mm) may be too small for efficient high-energy transport
2. Particles terminating at ~3mm despite having energy to travel further
3. Only 11% of expected energy being deposited

### Git Commits
- N_E=256, geometric mean fix, max_iter=600 changes pending commit

### Next Steps
1. Investigate why particles stop at ~3mm despite correct energy loss rate
2. Consider using larger step sizes for high-energy particles (E > 50 MeV)
3. Verify energy deposition calculation in K2 coarse transport
