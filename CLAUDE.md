# Project-Specific Instructions

## Build & Execution

1. **Build with GPU**: Always build the code with GPU support enabled.

2. **Use GPU by default**: GPU execution is the default. Do not use CPU unless explicitly requested.

3. **Record Output Messages**:
   - Record the output messages of the code running in `output_message.txt`
   - Do not create another `output_message.txt` for each run. Overwrite it
   - Review the result after each run
   - If there is any progress or something that needs to be recorded, it should be in the history file (.sisyphus/debug_history.md)

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

## Safety Rules

8. **Never Remove Whole Codebase**: Do not delete the entire codebase at any time.

9. **Clean Up Temporary Files**:
   - Any test files, debug files, or temporary documents must be **removed after use**
   - Do **not** include temporary files in git commits
