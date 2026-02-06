# Project-Specific Instructions

## Build & Execution

1. **Build with GPU**: Always build the code with GPU support enabled.

2. **Use GPU by default**: GPU execution is the default. Do not use CPU unless explicitly requested.

3. **Record Output Messages**:
   - Record the output messages of the code running in `output_message.txt`
   - Do not create another `output_message.txt` for each run. Overwrite it
   - Review the result after each run
   - If there is any progress or something that needs to be recorded, it should be in the history file (dbg/debug_history.md)

4. **Compare with Validation Data**:
   - After the code running, **compare the result with validation_data/**
   - Use the reference data in `validation_data/` to verify correctness

## Debugging Workflow

5. **Issue-Based Debugging**:
   - Analyze debug messages from the code to identify issues
   - If root cause is unclear, add debug messages to the **previous step** of the code logic
   - Iterate backward through the code flow to trace the source of the problem

6. **Track Debugging History**:
   - Create a todo list for debugging tasks
   - **Update todo list and history for each trial**
   - Record debugging history with **commit hashes** to distinguish between attempts
   - Document what was tried and what the outcome was
   - This accumulated history will help identify the real problem pattern

7. **Avoid Infinite Loops**:
   - If you repeat the same step more than **three times** without progress, **try another method**
   - Do not get stuck in the same debugging cycle

8. **Isolate and Test One-by-One**:
   - If you find any issue, **consolidate the reason and test it by creating a test file to point out one reason only**
   - For example, if debug message shows energy deposition is matched for energy loss of traveling particle, create a test code to check **one CUDA kernel only**
   - If there are suspicious CUDA kernels, **check them one-by-one**
   - After identifying issues, **fix them one-by-one** to see the result
   - This isolates variables and helps pinpoint the exact source of the problem

## Testing

9. **Use ctest for Unit Testing**:
    - **Preferred method**: Use `ctest` for all CUDA kernel and unit testing
    - **Basic commands** (run from `build/` directory):
      ```bash
      ctest                          # Run all tests
      ctest --verbose                # Detailed output
      ctest --output-on-failure      # Show output only on failure
      ctest -j8                      # Parallel run with 8 threads
      ctest -R "K3"                  # Run tests matching pattern
      ctest --rerun-failed           # Re-run failed tests only
      ```
    - **Workflow**:
      1. Write test in `tests/kernels/test_kX_<name>.cpp`
      2. Rebuild: `cd build && cmake .. && make -j8`
      3. Run: `ctest -R <test_name> --verbose`
      4. Check results and fix issues
    - Test files use Google Test framework: `TEST(TestSuite, TestCase) { ... }`

## Code Verification

10. **Use AST for Code Structure**:
   - Use AST (Abstract Syntax Tree) to identify the correct calling structure of the codebase
   - **Do not use non-existing module names or file names**
   - Before revising code, **verify the file/function/module actually exists**

11. **Check SPEC.md**:
   - Check SPEC.md and compare the code
   - The spec should be reflected in the code

## Validation

12. **Comparison with MOQUI Validation Data**:
    - Run `python3 compare_with_validation.py` after simulation to compare with MOQUI Monte Carlo
    - Validation data is in `validation_data/` directory (70, 110, 150, 190, 230 MeV)
    - Key metrics: Bragg peak position, lateral spread (sigma), PDD shape

13. **Recommended Configuration for 70 MeV**:
    - Use `sigma_x_mm = 3.8` to match clinical beam profiles (MOQUI validation)
    - Consider `sigma_theta_rad > 0` for more realistic angular divergence
    - Scattering reduction factors have been removed (set to 1.0) for accurate physics

## Safety Rules

14. **Never Remove Whole Codebase**: Do not delete the entire codebase at any time.

15. **Clean Up Temporary Files**:
   - Any test files, debug files, or temporary documents must be **removed after use**
   - Do **not** include temporary files in git commits
