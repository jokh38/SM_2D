# Local Rules: SM_2D Project

## Task Completion Tracking Rule

**CRITICAL**: When any task is completed, you MUST update `todo.md` at the project root.

### When to Update

Update `todo.md` when:
- A coding task is finished and verified working
- A test is implemented and passing
- A documentation section is completed
- A bug is fixed
- Any concrete deliverable is done

### Update Format

```markdown
## [Date] - Task Category

- [x] Brief description of completed task (file:line reference)
- [ ] Pending task (if applicable)
```

### Template

Keep todos organized by phase (matching docs/phases/):

```markdown
# SM_2D Todo List

## Phase 0: Setup
- [x] Project structure created
- [ ] Build system configured

## Phase 1: LUT Generation
- [ ] NIST data download script
- [ ] R(E) interpolation implementation

## Phase 2: Data Structures
- [ ] PsiC buffer layout
- [ ] OutflowBucket structure

[... etc ...]
```

### Verification

Before declaring ANY work session complete:
1. Mark the task as `[x]` in todo.md
2. Add file references (e.g., `src/lut.cpp:42`)
3. Note any remaining work in `[ ]` items

**The todo.md is the source of truth for project progress.**
