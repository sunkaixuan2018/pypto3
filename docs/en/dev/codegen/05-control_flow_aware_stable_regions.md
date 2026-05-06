# Control-Flow-Aware Stable Regions

This document extends the manual-scope template design from straight-line orchestration regions to control-flow-aware loop bodies. It targets workloads such as `paged_attention`, where the stable task pattern lives inside an innermost `for` loop and includes limited flag-producing `if` statements plus loop-carried state.

## Goal

The goal is to let PyPTO lower an innermost orchestration loop body to `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` without moving template analysis into codegen and without requiring `simpler` to understand template metadata.

The first implementation remains conservative:

- Match only the body of the innermost `ForStmt`.
- Allow only template-declared `IfStmt` forms between task stages.
- Represent loop-carried state explicitly through template declarations and IR validation.
- Keep fallback behavior unchanged: any mismatch stays on the existing AUTO path.

## Problem Shape

The current straight-line matcher expects a region like:

```text
qk -> softmax -> pv -> update
```

The real `paged_attention` loop body is closer to:

```text
qk
softmax
pv
if bn == 0: yield 1 else yield 0
if bn == bn_this_batch - 1: yield 1 else yield 0
update
yield next_iter(mi, li, oi)
```

This means a successful match must understand:

- task calls inside a loop body
- limited control-flow nodes between stages
- loop-carried state (`mi`, `li`, `oi`)

Simply "allowing `IfStmt`" is not enough because the current matcher does not treat loop bodies as first-class candidate regions.

## Architecture

The design keeps the existing split:

1. `IdentifyStableRegions`
   - Descends into candidate innermost `ForStmt` bodies.
   - Checks a control-flow-aware template contract.
   - Annotates matched task calls.
2. `LowerStableRegionsToManualScope`
   - Wraps the matched loop-body region in `ManualScopeStmt`.
   - Preserves the `ForStmt` structure inside the scope body.
3. Orchestration codegen
   - Consumes `ManualScopeStmt` plus task attrs and loop metadata.
   - Emits manual-scope submission code and explicit dependencies.

This preserves the rule that passes do analysis and codegen only lowers structured IR.

## Template Contract

Each control-flow-aware template is declarative. The first implementation should support the following fields:

```yaml
template_key: paged_attention_loop_body_v1

match:
  region_kind: loop_body
  loop_nest: innermost_for
  tokens: [qk, softmax, pv, update]

  stage_shape:
    kind: linear
    counts:
      qk: 1
      softmax: 1
      pv: 1
      update: 1

  arg_directions:
    qk: [Input, Input, OutputExisting]
    softmax: [Input, Scalar, OutputExisting, OutputExisting, OutputExisting]
    pv: [Input, Input, OutputExisting]
    update: [Input, Input, Input, InOut, InOut, InOut, OutputExisting, Scalar, Scalar]

  allowed_between_stages:
    - between: [pv, update]
      kind: if_flag
      count: 2
      rules:
        branch_yield_kind: scalar_only
        task_calls_allowed: false
        return_vars_kind: scalar_only

  loop_shape:
    iter_arg_count: 3
    iter_arg_roles: [mi_state, li_state, oi_state]
    return_var_count: 3

recipe:
  intra_iteration_deps:
    - { task: softmax, deps: [qk] }
    - { task: pv, deps: [softmax] }
    - { task: update, deps: [pv] }

  loop_carried_reads:
    - { task: update, read_from_prev_iter: mi_state }
    - { task: update, read_from_prev_iter: li_state }
    - { task: update, read_from_prev_iter: oi_state }

  loop_carried_writes:
    - { produced_by: update, write_to_next_iter: mi_state }
    - { produced_by: update, write_to_next_iter: li_state }
    - { produced_by: update, write_to_next_iter: oi_state }
```

Field meanings:

- `region_kind`
  - The shape of the matched candidate. Phase one uses `loop_body`.
- `loop_nest`
  - Restricts matching to the innermost `ForStmt`.
- `tokens`
  - The logical task sequence.
- `stage_shape`
  - Declares how many tasks of each stage must appear.
- `arg_directions`
  - Reuses `DeriveCallDirections` output for structural validation.
- `allowed_between_stages`
  - Declares which non-task control-flow nodes may appear and where.
- `loop_shape`
  - Declares the loop-carried state contract.
- `recipe.intra_iteration_deps`
  - Static task deps within one iteration.
- `recipe.loop_carried_reads`
  - Which task reads state produced by the previous iteration.
- `recipe.loop_carried_writes`
  - Which task updates the state for the next iteration.

The contract is intentionally declarative. Phase one should not infer loop-carried state automatically from def-use chains.

## IR Representation

The design reuses the existing `ManualScopeStmt` rather than introducing a brand-new control-flow template node.

The manual region is represented by:

- `ManualScopeStmt`
  - the structured region boundary
- per-task attrs
  - `stable_region_template_key`
  - `manual_task_index`
  - `manual_dep_indices`

Loop-carried structure remains in normal control-flow IR:

- `ForStmt.iter_args`
- `ForStmt.return_vars`
- `IfStmt.return_vars`
- `YieldStmt`

If needed, later phases may add region-level attrs to the manual scope or loop node, but phase one can begin with task attrs plus template validation.

## Responsibilities

### `IdentifyStableRegions`

Owns:

- candidate region discovery
- innermost-loop detection
- token matching
- stage-shape validation
- `arg_directions` validation
- validation of allowed control-flow nodes between stages
- validation of loop-carried shape
- expansion of intra-iteration manual dependency indices

Does not own:

- code emission
- task-handle lifetime management
- runtime dependency inference

### `LowerStableRegionsToManualScope`

Owns:

- converting a matched loop-body candidate into a structured `ManualScopeStmt`
- preserving the `ForStmt` body inside the manual scope
- keeping unmatched regions untouched

Does not own:

- template matching
- dependency recomputation

### Orchestration codegen

Owns:

- lowering `ManualScopeStmt` to `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- emitting task-handle captures
- emitting `Arg.add_dep(...)`
- translating loop-carried template metadata into cross-iteration dependency code

Does not own:

- template recognition
- control-flow safety analysis
- general def-use reasoning

## Matching Strategy

The matcher should:

1. Visit orchestration functions as before.
2. Descend into `ForStmt`.
3. Treat a `ForStmt` as a candidate only when it is the innermost loop.
4. Scan its `SeqStmts` body for task calls and template-declared control-flow nodes.
5. Reject any body that contains:
   - nested loops
   - undeclared `IfStmt`
   - task calls inside an allowed flag `IfStmt`
   - mismatched loop-carried shape
6. Write manual attrs only after the full template contract validates.

This avoids a dangerous partial match where `pv -> update` appears to fit but loop-carried semantics are wrong.

## Codegen Semantics

When `ManualScopeStmt.body` contains an allowed `ForStmt`, codegen should:

- emit the outer manual scope once
- keep the loop structure in generated orchestration C++
- emit intra-iteration `add_dep(...)` from `manual_dep_indices`
- emit cross-iteration deps from the template's loop-carried recipe
- guard the first iteration so it does not depend on a nonexistent previous task handle

Phase one should use explicit state/task-handle variables rather than trying to infer the minimal previous-task set on the fly.

## Phased Rollout

### Phase 0: Recognition only

- Add innermost-`ForStmt` descent.
- Add tests proving the loop body can be identified.
- Do not lower or codegen a manual scope yet.

### Phase 1: Loop-body manual scope

- Add `allowed_between_stages` support for flag-only `IfStmt`.
- Lower the matched loop-body region into `ManualScopeStmt`.
- Support intra-iteration manual deps in codegen.

### Phase 2: Loop-carried deps

- Validate `loop_shape`.
- Emit cross-iteration deps from the declared recipe.
- Cover the full `paged_attention` loop-body path.

### Phase 3: Variants and regressions

- Extend to nearby templates such as the unaligned softmax variant.
- Add focused performance and no-regression checks.

## Risks

1. `simpler` manual-scope semantics with loops must be confirmed before the cross-iteration codegen shape is treated as stable.
2. Automatic loop-carry inference is too risky for the first version and should be avoided.
3. Loop descent must stop at the innermost candidate, or outer-loop false positives will produce invalid orchestration code.

## Testing

The minimum coverage should include:

- pass-level hit on a loop-body `qk -> softmax -> pv -> update` pattern
- rejection of undeclared control flow
- rejection when loop-carried arity does not match the template contract
- lowering of a matched innermost `ForStmt` into `ManualScopeStmt`
- manual-scope codegen for a loop body with flag `IfStmt`
- fallback to AUTO when the template contract is not satisfied
