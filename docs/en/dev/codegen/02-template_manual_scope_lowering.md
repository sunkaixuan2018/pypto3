# Template Manual Scope Lowering

This design identifies stable orchestration call regions at compile time and lowers them to PTO2 manual scopes with explicit task dependencies. It does not add a device-side learning cache, and `simpler` does not need to understand templates.

## Goal

The first implementation is intentionally conservative:

- Match only straight-line call sequences in the same lexical scope.
- Do not match across `for`, `while`, `if`, or existing scope statements.
- Require closed manual regions; open manual/AUTO dependency boundaries fall back to AUTO.
- Do not enforce a `func_id` sequence. `func_id` may be recorded for debugging later, but it is not a phase-one match constraint.
- Keep unmatched code on the existing AUTO path.

For PagedAttention-style code, the initial built-in template is:

```text
qk -> softmax -> pv -> update
```

The matcher uses kernel-name semantic tokens and call order. Concrete tensor argument names may vary.

## Architecture

The implementation is split into two passes plus codegen support:

1. `IdentifyStableRegions`
   Finds candidate orchestration regions, checks the built-in template registry, rejects unsafe boundaries, and annotates matched calls.
2. `LowerStableRegionsToManualScope`
   Wraps marked regions in a structured `ManualScopeStmt`.
3. Orchestration codegen
   Consumes `ManualScopeStmt` and per-call dependency attrs to emit `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` and `Arg.add_dep(...)`.

This keeps template analysis out of codegen, so orchestration codegen remains a structured IR lowering step.

## IR Representation

Manual lowering adds a scope node:

```cpp
ScopeKind::Manual
ManualScopeStmt
```

`ManualScopeStmt` carries:

- `template_key`
- optional `template_version`
- `body`

Each task call inside the region carries lightweight attrs:

- `stable_region_template_key`
- `manual_task_index`
- `manual_dep_indices`

For the initial PagedAttention template, dependencies are linear:

```text
qk      deps []
softmax deps [0]
pv      deps [1]
update  deps [2]
```

## Boundary Rule

Manual scopes can be mixed with AUTO scopes at runtime, but manual-scope tasks do not get TensorMap automatic dependencies. A manual task that needs a dependency must receive it explicitly through `Arg.add_dep(...)`.

Phase one therefore accepts only closed regions:

- Inputs may come from function parameters, `tensor.create`, or earlier tasks inside the same matched region.
- Outputs may flow to `return`, `Out`/`InOut` parameters, or later tasks inside the same matched region.
- If a matched task input depends on an outside task, reject the match.
- If a matched task output is consumed by an outside task, reject the match.

This may miss some real PagedAttention shapes, but it avoids generating a manual scope with a broken cross-boundary dependency. Later phases can store AUTO task handles and add explicit cross-boundary deps if needed.

## Pass Placement

The default pipeline places the new passes after call directions and the final cleanup simplification:

```text
DeriveCallDirections
Simplify
IdentifyStableRegions
LowerStableRegionsToManualScope
```

This gives the matcher stable call directions and a normalized call sequence immediately before codegen.

## simpler Integration

`simpler` continues to consume the normal backend output:

- `orchestration/<orch>.cpp`
- `kernel_config.py`

The generated orchestration C++ contains normal PTO2 runtime constructs, including manual scopes and explicit deps. A future debug-only manifest may record template hits, kernel-name tokens, optional `func_id`/core-type information, region ranges, and rejection reasons, but that manifest must not become a `simpler` input requirement.

## Tests

Coverage should include:

- Template hit for a straight-line `qk -> softmax -> pv -> update` region.
- Rejection of open input boundaries.
- Rejection of open output boundaries.
- Rejection across `for`, `while`, `if`, and existing scope boundaries.
- Codegen emission of `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`, retained task handles, and `params_tN.add_dep(task_result_M.task_id())`.
- Fallback to existing AUTO codegen when no template matches.
