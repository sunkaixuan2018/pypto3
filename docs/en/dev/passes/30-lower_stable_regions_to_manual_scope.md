# LowerStableRegionsToManualScope Pass

Wraps calls marked by `IdentifyStableRegions` in structured manual-scope IR.

## Overview

`LowerStableRegionsToManualScope` consumes the attrs produced by `IdentifyStableRegions` and creates a `ManualScopeStmt` around the matched straight-line region. The per-call attrs remain on the calls so orchestration codegen can emit explicit dependencies.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::LowerStableRegionsToManualScope()` | `passes.lower_stable_regions_to_manual_scope()` | Program-level |

## Input

Calls must carry:

- `stable_region_template_key`
- `manual_task_index`
- `manual_dep_indices`

## Output

The matched statements are wrapped as:

```text
ManualScopeStmt(template_key=..., body=SeqStmts([...]))
```

The pass preserves unmarked statements before and after the region, including trailing `ReturnStmt`s. It rejects unsafe wrapping if the candidate range contains control flow or an existing scope.

## Codegen Contract

Orchestration codegen lowers `ManualScopeStmt` to:

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
  ...
}
```

Inside the scope, submit results are retained as `task_result_N`, and `manual_dep_indices` emit calls such as:

```cpp
params_t1.add_dep(task_result_0.task_id());
```

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`
**Implementation**: `src/ir/transforms/lower_stable_regions_to_manual_scope_pass.cpp`

## Tests

**Tests**: `tests/ut/ir/transforms/test_stable_region_manual_scope.py`, `tests/ut/codegen/test_orchestration_codegen.py`
