# SM_2D Debug History

## 2026-01-27: Workflow Verification and Bug Analysis

### Summary
Verified the code workflow implementation against SPEC.md v0.8 and identified critical issues.

### Changes Made
1. **Fixed E_max value**: Changed from 300 MeV to 250 MeV (SPEC v0.8 requirement)
   - File: `src/gpu/gpu_transport_runner.cpp:98`
   - R(300 MeV) was returning NaN due to NIST data range limitation

2. **LOCAL_BINS configuration** (kept existing due to memory constraints):
   - Current: N_theta_local=4, N_E_local=2, LOCAL_BINS=128 (with x_sub, z_sub)
   - SPEC requires: N_theta_local=8, N_E_local=4, LOCAL_BINS=32 (without x_sub, z_sub)
   - Note: Code uses extended 4D encoding with sub-cell position tracking
   - Using SPEC values would require 2GB per buffer (exceeds 8GB VRAM)

### Issues Found

#### 1. CRITICAL: Particles Not Propagating to Full Range
- **Expected**: Bragg peak at ~158mm depth for 150 MeV protons
- **Actual**: Bragg peak at 1mm depth, only 16.965 MeV deposited (11% of expected)
- **Root Cause**: Particles get stuck at low energy/weight gap

#### 2. Weight/Energy Gap Issue
At iteration 109, particles are in cell 1700 (z=8, depth=4mm) with:
- Energy: 0.901, 2.205, 3.088 MeV (all below 10 MeV fine transport threshold)
- Weight: ~1e-11 (below 1e-6 active threshold)

These particles cannot:
- Activate fine transport (weight too low)
- Be absorbed (energy above 0.1 MeV cutoff)
- Progress through coarse transport (stuck in same cell)

#### 3. Missing SPEC Implementations
- **Variance-based MCS accumulation**: Not implemented (uses single scatter per step)
- **Nuclear cross-section**: Code uses 0.0012 mm⁻¹, SPEC wants 0.0050 mm⁻¹

### Debug Output Added
- Added K2 STAY/CROSS debug messages to track particle movement
- Shows particles do progress through cells (100 → 300 → 500 → ... → 1700)
- But eventually get stuck at low energy/weight combination

### Next Steps Needed
1. Fix the weight/energy gap issue (particles below E_trigger but with insufficient weight)
2. Implement variance-based MCS accumulation per SPEC v0.8
3. Verify nuclear cross-section values
4. Consider memory optimization to enable SPEC-compliant LOCAL_BINS values

