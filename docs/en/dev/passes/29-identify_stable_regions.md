# IdentifyStableRegions Pass

Marks stable orchestration call regions that can be lowered to PTO2 manual scopes.

## Overview

`IdentifyStableRegions` runs on orchestration functions after call directions are derived and the final `Simplify` cleanup has run. The built-in registry currently recognizes two conservative shapes:

```text
straight-line: qk -> softmax -> pv -> update
loop-body:     qk -> softmax -> pv -> if_flag -> if_flag -> update
```

The pass matches by kernel-name semantic tokens and call order. It does not require or validate a `func_id` sequence. The loop-body variant is restricted to innermost `ForStmt` bodies with template-declared flag `IfStmt`s and loop-carried state arity checks.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::IdentifyStableRegions()` | `passes.identify_stable_regions()` | Program-level |

## Output

Matched calls receive these attrs:

| Attr | Meaning |
| ---- | ------- |
| `stable_region_template_key` | Matched template key |
| `manual_task_index` | Task index inside the manual region |
| `manual_dep_indices` | Prior task indices this task must depend on |

For the current templates, deps are linear inside one iteration: task `1` depends on `0`, task `2` on `1`, and task `3` on `2`.

## Boundary Policy

The straight-line matcher only accepts closed straight-line regions:

- Allows inputs from function parameters, `tensor.create`, and earlier matched tasks.
- Rejects inputs that depend on an outside task.
- Rejects matched outputs consumed by an outside task.
- Rejects windows containing `for`, `while`, `if`, or existing scope statements.

The loop-body matcher is separate and more restrictive:

- Only considers innermost `ForStmt` bodies.
- Rejects nested loops, undeclared control flow, and task calls inside allowed flag branches.
- Checks template-declared loop-carried state arity before annotating calls.

Rejected regions remain unchanged and use the existing AUTO codegen path.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`
**Implementation**: `src/ir/transforms/identify_stable_regions_pass.cpp`
**Template registry**: `include/pypto/codegen/orchestration/template_registry.h`, `src/codegen/orchestration/template_registry.cpp`

## Tests

**Tests**: `tests/ut/ir/transforms/test_stable_region_manual_scope.py`
