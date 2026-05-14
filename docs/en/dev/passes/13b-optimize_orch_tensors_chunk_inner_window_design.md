# OptimizeOrchTensors Chunk-Inner Window Externalization Design

## Purpose

This document is the design-level reference for the first-stage `OptimizeOrchTensors`
extension that makes a kernel-internal window write set visible to orchestration and
runtime for a narrow q_proj-like `ChunkInner` case.

Use this file for the overall design. Keep
`13a-optimize_orch_tensors_chunk_inner_window_plan.md` as the implementation / execution
plan.

## Related Docs

- Pass overview: [`13-optimize_orch_tensors.md`](13-optimize_orch_tensors.md)
- Implementation plan: [`13a-optimize_orch_tensors_chunk_inner_window_plan.md`](13a-optimize_orch_tensors_chunk_inner_window_plan.md)
- Outline invariant: [`10-outline_incore_scopes.md`](10-outline_incore_scopes.md)
- Precondition pass: [`12-convert_tensor_to_tile_ops.md`](12-convert_tensor_to_tile_ops.md)

## Problem Statement

After `ConvertTensorToTileOps`, some outlined InCore kernels still expose a full parent
tensor in their `Out` signature even though the kernel only writes one logical window.

For the target q_proj-like shape:

- a single outlined InCore kernel owns the whole computation body
- a top-level `ChunkInner` loop carries the full `Out` tensor
- each loop iteration writes a disjoint sub-window through `tile.store(..., offsets, iter_arg)`

This is functionally correct, but orchestration and runtime only see the whole parent
tensor. As a result:

- dependency tracking is too coarse
- `manual_scope` / runtime scheduling cannot reason at the true written-window granularity
- codegen cannot express the actual write region as an orchestration-visible tensor view

## Non-Goal

This design does **not** change the earlier outlining conclusion:

- one `pl.at(...)` outlines to one InCore kernel
- `parallel` / `range` inside that scope remain lowered inside the outlined kernel
- no automatic split into per-iteration sibling kernels is implied

The goal here is narrower: preserve the single outlined kernel model, but externalize
its effective write window at the orchestration boundary.

## Target Shape

Conceptual source shape:

```python
@pl.function(type=pl.FunctionType.InCore)
def q_proj_chunk_group(self, x, group_base, out):
    for ob_ci, (out_iter,) in pl.parallel(
        4, init_values=(out,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
    ):
        q0 = group_base + ob_ci * 64
        tile = pl.load(x, [0, q0], [16, 64])
        out_next = pl.store(tile, [0, q0], out_iter)
        q_rv = pl.yield_(out_next)
    return q_rv
```

Desired orchestration-level form:

```python
out__window = pl.tensor.slice(out, [16, 256], [0, group_base])
out_next__windowed = self.q_proj_chunk_group__windowed(x, group_base, out__window)
out_next = pl.tensor.assemble(out, out_next__windowed, [0, group_base])
```

Desired cloned kernel form:

```python
@pl.function(type=pl.FunctionType.InCore)
def q_proj_chunk_group__windowed(self, x, group_base, out_window):
    for ob_ci, (out_iter,) in pl.parallel(
        4, init_values=(out_window,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
    ):
        q0 = group_base + ob_ci * 64
        tile = pl.load(x, [0, q0], [16, 64])
        out_next = pl.store(tile, [0, q0 - group_base], out_iter)
        q_rv = pl.yield_(out_next)
    return q_rv
```

## Design Principles

### 1. Keep Parent-Tensor SSA In Orchestration

The orchestration-side `tensor.assemble(...)` must stay.

Reason:

- the parent tensor remains the SSA value seen by downstream users
- later passes and orchestration codegen already understand assemble-as-alias semantics
- dependency chains after the rewritten call still attach to the parent value correctly

### 2. Expose Real Access Range At The Boundary

The runtime-relevant visibility comes from the sliced `Out` window passed into the cloned
kernel, not from splitting the kernel into smaller siblings.

### 3. Stay Narrow In Phase 1

The first implementation deliberately handles only a q_proj-like shape with a single,
provable window mapping. This keeps the analysis auditable and reduces risk of rewriting
incorrect shapes.

## Matching Constraints

The phase-1 matcher only accepts:

- exactly one `Out` param in the InCore callee
- a function body shaped as top-level `ForStmt` plus trailing `ReturnStmt`
- top-level loop tagged with `loop_origin = ChunkInner`
- exactly one loop-carried tensor iter-arg
- that iter-arg is initialized from the `Out` param
- exactly one yielded `tile.store(..., offsets, iter_arg)` in the loop
- window shape can be derived from the stored tile shape and loop trip structure
- orchestration-visible base offsets can be expressed from callee arguments

Rejected in phase 1:

- multiple stores to the same carried `Out`
- nested or chained `ChunkInner` loop trees
- overlapping-window proofs
- dynamic or ambiguous window shape inference
- non-`ChunkInner` loop-carried store patterns

## Rewrite Strategy

## Step 1: Analyze The Original InCore Kernel

The pass analyzes post-`ConvertTensorToTileOps` IR, not source DSL sugar.

It extracts:

- the full parent `Out` param
- the loop-carried iter-arg / return-var pair
- the unique `tile.store`
- the global store offsets
- the per-call base window offsets visible at the orchestration call site

## Step 2: Clone A Windowed Kernel

The pass clones the original InCore function into a `__windowed` variant and narrows:

- the `Out` param tensor type
- the loop iter-arg tensor type
- the loop return-var tensor type
- the function return type

The `tile.store` offsets are rewritten from global coordinates to local window coordinates.

Example:

- original store: `[0, q0]`
- call-site base window: `[0, group_base]`
- cloned local store: `[0, q0 - group_base]`

## Step 3: Rewrite The Orchestration Call Site

The original orchestration call:

```python
out_next = self.q_proj_chunk_group(x, group_base, out)
```

becomes:

```python
out__window = pl.tensor.slice(out, [16, 256], [0, group_base])
out_next__windowed = self.q_proj_chunk_group__windowed(x, group_base, out__window)
out_next = pl.tensor.assemble(out, out_next__windowed, [0, group_base])
```

This is the only place where the runtime needs to "see" the window.

## Why This Improves Dependency Granularity

After rewrite, later lowering/codegen materializes the sliced tensor as an orchestration
view of the parent tensor.

That means:

- the cloned kernel no longer appears to write the whole parent tensor
- runtime-visible dependency carriers can attach to the sliced window
- disjoint windows can be scheduled independently without manual extra dependency edges

The design therefore upgrades dependency visibility from:

- whole parent tensor

to:

- exact written sub-window at the orchestration boundary

## Interaction With Codegen

The intended witnesses at later stages are:

- pass-level IR:
  - `pl.tensor.slice(...)`
  - `self.kernel__windowed(...)`
  - `pl.tensor.assemble(...)`
- orchestration C++:
  - `Tensor out__window = ext_out.view(...)`
  - task submit against `out__window`
  - no direct full-parent output argument on that rewritten call
- kernel PTO:
  - the original internal `parallel` / `range` remain lowered inside the cloned kernel
  - only the store coordinates become local-to-window

## Correctness Invariants

The rewrite must preserve:

- functional equivalence of written values
- parent tensor SSA identity at orchestration level
- downstream dependency behavior on the parent tensor result
- original single-kernel outlining semantics
- the existing `enable_out_window_rewrite` switch

It must never:

- silently drop `tensor.assemble(...)`
- rewrite ambiguous offset decompositions
- rewrite overlapping or multi-store cases without proof
- expand scope beyond the audited narrow matcher

## Implementation Touchpoints

- C++ pass:
  - `src/ir/transforms/optimize_orch_tensors_pass.cpp`
- Pass overview doc:
  - `docs/en/dev/passes/13-optimize_orch_tensors.md`
- Pass/unit tests:
  - `tests/ut/ir/transforms/test_optimize_orch_tensors.py`
  - `tests/ut/codegen/test_orchestration_codegen.py`
  - `tests/ut/jit/test_qwen3_decode.py`
- Runtime/ST witness:
  - `tests/st/runtime/test_manual_scope_pipeline.py`

## Verification Matrix

Minimum verification for this design:

1. Pass UT proves the rewrite exists in IR.
2. Codegen UT proves final orchestration C++ contains `view + __windowed`.
3. JIT/q_proj regression proves the original "single outlined kernel" invariant still holds.
4. Runtime/ST proves numerical correctness and can optionally inspect saved artifacts.

## Future Extensions

Possible follow-up work, intentionally deferred from phase 1:

- multiple independent stores in one carried `Out`
- nested `ChunkInner` groups
- generalized affine offset decomposition
- overlap detection / proof
- extending the same externalization idea beyond the q_proj-like shape
