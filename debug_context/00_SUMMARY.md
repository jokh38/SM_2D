# SM_2D Code Review Summary

## Review Purpose
To verify that:
1. Physics is implemented correctly
2. Physics operations are used correctly in the code
3. The pipeline contains all steps with proper logic

## Overall Assessment: ⚠️ CRITICAL BUG DETECTED

The codebase implements a scientifically correct Monte Carlo proton transport simulation with full K1-K6 pipeline, **BUT the current HEAD produces incorrect physics results.**

### ⚠️ CRITICAL BUG WARNING (2026-01-27)

**The current HEAD (commit 3859085) produces INCORRECT physics results:**

| Metric | Expected (150 MeV) | Actual | Status |
|--------|-------------------|--------|--------|
| Bragg Peak Position | ~157 mm | **2 mm** | ❌ FAIL |
| Energy Deposited | ~150 MeV | **6.9 MeV** | ❌ FAIL |

The batch results in `results/batch/batch_mean150/` (from Jan 26 08:14) show correct results with Bragg peak at 153.5mm. This indicates a regression was introduced in one of the late-evening commits on Jan 26 (after 23:52).

**Until this bug is fixed, the simulation results should NOT be trusted for physics validation.**

---

## Document Structure

| Document | Contents |
|----------|----------|
| `01_physics_implementation_analysis.md` | Physics formulas and their correctness |
| `02_physics_usage_analysis.md` | How physics functions are called in kernels |
| `03_pipeline_logic_analysis.md` | K1-K6 pipeline flow and completeness |
| `04_unclear_areas_and_debug.md` | Areas needing clarification/debug messages |
| `05_file_reference.md` | Key files and their purposes |

---

## Key Findings

### ✓ Correct Implementations

1. **Multiple Coulomb Scattering (MCS)**
   - Highland formula (PDG 2024)
   - 2D projection correction (1/√2)
   - Proper relativistic kinematics

2. **Energy Straggling**
   - Full Vavilov regime handling
   - Bohr, Landau, and Vavilov interpolation
   - 1/β energy dependence in Bohr regime

3. **Nuclear Interactions**
   - ICRU 63 cross-sections
   - Exponential attenuation
   - Known limitation: local energy deposition

4. **Energy Loss**
   - R-based step control
   - NIST PSTAR data
   - Log-log interpolation

5. **Pipeline Logic**
   - All 6 kernels (K1-K6) implemented
   - Proper ordering and synchronization
   - Weight conservation auditing

### ✓ Previously Fixed Bugs

1. **Energy bin edge interpretation** (K3:158)
   - Was: Used bin center (E_bin + 0.5) * dlog
   - Now: Uses bin lower edge (E_bin) * dlog
   - Impact: Fixed 150→160 MeV error

2. **Particle loss in output** (K3:332-404)
   - Was: Particles remaining in cell were lost
   - Now: Properly write to psi_out
   - Impact: Fixed weight conservation

3. **Bucket clearing timing** (Pipeline:673-679)
   - Was: Buckets cleared after K2/K3
   - Now: Buckets cleared before K2/K3
   - Impact: Prevents stale data transfer

4. **K2 output writing** (K2:262-332)
   - Was: Particles not written to psi_out
   - Now: Proper slot allocation and writing
   - Impact: Fixed high-energy particle loss

### ⚠ Known Limitations

1. **Nuclear energy localization**
   - All nuclear-removed energy deposited locally
   - Reality: 70-80% transported by secondaries
   - Impact: ~1-2% local dose overestimate
   - Acceptable for: Validation purposes

2. **Landau straggling approximation**
   - Uses FWHM/2.355 for effective sigma
   - Therapeutic protons in Vavilov regime (0.01 < κ < 10)
   - Impact: Limited (most protons not in Landau regime)

3. **Coarse transport MCS**
   - No random scattering (θ_new = θ_old)
   - Trade-off: Speed vs statistical accuracy
   - Acceptable for: High energy where scattering is small

---

## Areas Identified for Debug Messages

See `04_unclear_areas_and_debug.md` for specific recommendations.

---

## Code Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Physics Accuracy | Excellent | PDG/ICRU compliant, well-documented |
| Code Organization | Good | Clear separation of CPU/GPU code |
| Documentation | Good | Inline comments explain physics |
| Debug Infrastructure | Fair | Many printf statements, could use logging |
| Test Coverage | Unknown | Needs verification |

---

## Recommendations

### High Priority
1. **🔴 CRITICAL: Fix physics bug causing particles to stop at 2mm instead of ~157mm**
2. Add comprehensive debug logging (see `04_unclear_areas_and_debug.md`)
3. Verify K5 audit failures are properly reported
4. Add unit tests for physics functions

### Medium Priority
1. Consider structured logging instead of printf
2. Add configuration file validation
3. Document the coordinate system (centered vs corner-based)

### Low Priority
1. Performance profiling
2. GPU memory usage optimization
3. Alternative nuclear models (if needed)

---

## Files Modified During Review

### Documentation Updates (2026-01-27)
1. `00_SUMMARY.md` - Added critical bug warning
2. `01_physics_implementation_analysis.md` - Added GPU comment issue note and critical bug warning
3. `02_physics_usage_analysis.md` - Added missing K2 energy bin fix documentation (section 2.1)

### Code Changes
No code files were modified - this is a review-only analysis.

---

## References

| Standard | Usage |
|----------|-------|
| PDG 2024 | Highland formula, radiation length |
| ICRU 63 | Nuclear cross-sections |
| NIST PSTAR | Stopping power, CSDA range |

---

## Review Date
- Original Review: 2026-01-27
- Documentation Updates: 2026-01-27 (Added critical bug warning and missing K2 documentation)
