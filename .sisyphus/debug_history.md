# SM_2D Debug History

## Run 1 - 2025-01-27

### Configuration
- Particle: Proton
- Energy: 150 MeV
- Grid: 200 x 640 cells (0.5mm spacing)
- GPU: NVIDIA GeForce RTX 2080

### Critical Physics Deviation Found

**Issue**: Bragg Peak at 1.5mm instead of expected ~158mm

**Details**:
- LUT Verification shows: `R(150 MeV) = 157.667 mm (expected ~158)` ✓ LUT is correct
- But actual Bragg Peak output: `Bragg Peak: 1.5 mm depth, 2.53308 Gy`
- Total energy deposited: `8.52941 MeV (expected ~150 MeV)` - only ~5.7% of expected!

**Analysis**:
1. The LUT (lookup table) for range calculation is correct (157.667 mm for 150 MeV)
2. The actual simulation stops depositing energy at ~2.5mm depth
3. Weight audits pass throughout all iterations (weight conservation is OK)
4. The K1-K6 pipeline completes 26 iterations successfully

**Possible Root Causes**:
1. Energy threshold issue - particles may be terminated too early
2. Range calculation in transport step may not be using LUT correctly
3. Step size calculation may be wrong
4. CSDA range vs actual transport range mismatch

**Next Steps**:
- Investigate why particles stop at ~2.5mm despite having correct LUT values
- Check energy threshold (currently: `Energy threshold: 10 MeV (b_E_trigger=9)`)
- Verify step size calculation in K2/K3 kernels

### Output Files
- `output_message.txt` - Full simulation output (overwritten on each run)
- `results/pdd.txt` - Depth-dose curve showing shallow Bragg peak
- `results/dose_2d.txt` - 2D dose distribution
