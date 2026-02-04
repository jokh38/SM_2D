# Project-Specific Instructions

## Build & Execution

1. **Build with GPU**: Always build the code with GPU support enabled.

2. **Use GPU by default**: GPU execution is the default. Do not use CPU unless explicitly requested.

3. **Record Output Messages**:
   - Record the output messages of the code running in `output_message.txt`
   - Do not create another `output_message.txt` for each run. Overwrite it
   - Review the result after each run
   - If there is any progress or something that needs to be recorded, it should be in the history file (dbg/debug_history.md)

## Debugging Workflow

4. **Issue-Based Debugging**:
   - Analyze debug messages from the code to identify issues
   - If root cause is unclear, add debug messages to the **previous step** of the code logic
   - Iterate backward through the code flow to trace the source of the problem

5. **Track Debugging History**:
   - Create a todo list for debugging tasks
   - **Update todo list and history for each trial**
   - Record debugging history with **commit hashes** to distinguish between attempts
   - Document what was tried and what the outcome was
   - This accumulated history will help identify the real problem pattern

6. **Avoid Infinite Loops**:
   - If you repeat the same step more than **three times** without progress, **try another method**
   - Do not get stuck in the same debugging cycle

## Code Verification

7. **Use AST for Code Structure**:
   - Use AST (Abstract Syntax Tree) to identify the correct calling structure of the codebase
   - **Do not use non-existing module names or file names**
   - Before revising code, **verify the file/function/module actually exists**

8. **Check SPEC.md**:
   - **CRITICAL: Always verify SPEC.md BEFORE implementing changes**
   - Most bugs (77%) come from SPEC-deviation, not logic errors
   - Check energy binning formula, step size limits, MCS implementation against SPEC
   - The spec should be reflected in the code

## Known Recurring Patterns (from claude-mem analysis)

### Pattern A: Energy Grid/Binning Issues (Most Frequent)
- **Problem**: Mixing log-spaced vs piecewise-uniform grid formulas
- **Files Affected**: `k2_coarsetransport.cu`, `k3_finetransport.cu`, `grids.cpp`, `gpu_transport_wrapper.cu`
- **Fix Applied**: Use consistent `E = 0.5 * (E_edges[E_bin] + E_edges[E_bin + 1])` for piecewise-uniform
- **Key Lesson**: ALL files must use same energy grid definition

### Pattern B: Boundary/Threshold Issues
- **Problem**: `* 0.999f` limit preventing boundary crossing, missing epsilon tolerance
- **Fix Applied**: Remove artificial limits, add `BOUNDARY_EPSILON = 0.001f`
- **Key Lesson**: Don't adjust thresholds repeatedly - check for artificial limits first

### Pattern C: Step Size Multiple Limits
- **Problem**: cell_limit, 1mm cap, 0.999f limit compounding
- **Fix Applied**: Remove ALL limits, not just one
- **Key Lesson**: Search for ALL step size restrictions before fixing

### Pattern D: Double Operations/Unit Errors
- **Problem**: Double division by mu_init, inconsistent energy bin reference (center vs lower edge)
- **Fix Applied**: Trace variable origins, ensure write/read consistency
- **Key Lesson**: Verify unit conversions at system boundaries

### Pattern E: MCS (Multiple Coulomb Scattering)
- **Problem**: Random per-step scattering instead of variance-based accumulation
- **Status**: Partially resolved (88% match rate as of mcs2-phase-b)
- **Key Lesson**: SPEC requires variance accumulation with RMS threshold splitting

### Debugging Workflow Improvement (Learned from Pattern Analysis)
```
CORRECT Order:
1. Check SPEC.md requirements → 2. Verify code matches SPEC → 3. Adjust thresholds

WRONG Order (historically attempted):
1. Adjust thresholds → 2. Check if fixed → 3. Read SPEC.md (too late!)
```

### Energy Grid Consistency Rules
- **NIST Data Range**: E_max must be ≤ 250 MeV (PSTAR data limitation)
- **Grid Type**: Piecewise-uniform (Option D2) - NOT log-spaced
- **Bin Resolution**: 0.25 MeV/bin for [100-250] MeV range (1029 bins total)
- **Consistency Check**: Verify `gpu_transport_wrapper.cu`, `gpu_transport_runner.cpp`, and kernels all use same grid

## Safety Rules

9. **Never Remove Whole Codebase**: Do not delete the entire codebase at any time.

10. **Clean Up Temporary Files**:
    - Any test files, debug files, or temporary documents must be **removed after use**
    - Do **not** include temporary files in git commits
