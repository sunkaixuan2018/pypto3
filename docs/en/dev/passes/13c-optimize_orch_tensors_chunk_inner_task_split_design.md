# OptimizeOrchTensors Chunk-Inner Task-Split Design

## Purpose

This document defines the second-stage `OptimizeOrchTensors` extension for the
q_proj-like `ChunkInner` case: instead of merely externalizing one aggregate
window, it hoists the top-level loop to orchestration so each iteration becomes
its own runtime-visible kernel task.

## Problem

The first-stage `__windowed` rewrite makes the written window visible at the
orchestration boundary, but it still submits one cloned kernel for the whole
top-level `ChunkInner` loop body. Runtime sees one task writing one larger
window, not one task per loop iteration.

For the target shape:

- one `pl.at(...)` has already outlined to one InCore kernel
- the outlined kernel body starts with one top-level `ChunkInner` loop
- each iteration writes exactly one disjoint sub-window through one yielded
  `tile.store(..., offsets, iter_arg)`

We want the runtime to see those per-iteration writes as separate tasks.

## Goal

Keep the original outlining invariant, but add an optional second rewrite mode
that:

- preserves the original outlined "big kernel" function
- clones a per-iteration `__iter_windowed` kernel
- rewrites the orchestration call site into an orch-visible loop
- submits one kernel task per top-level `ChunkInner` iteration

## Non-Goals

This design does not:

- split one `pl.at(...)` during `OutlineIncoreScopes`
- hoist nested inner reductions such as the `kb` accumulation range inside one
  `ob` iteration
- prove or rewrite multi-store / overlapping-window cases
- enable the rewrite by default

The first implementation only hoists the top-level loop whose iterations are
already known to be independent window writes.

## Switches

This mode is controlled independently from the existing out-window rewrite:

- `enable_out_window_rewrite`: existing Pattern 5 switch, default `True`
- `enable_out_window_task_split`: new task-split switch, default `False`

The new switch lives in `PassContext` and is threaded through `ir.compile()`,
runtime `RunConfig`, and the ST harness.

## Target Shape

Source InCore kernel after `ConvertTensorToTileOps`:

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

Rewritten orchestration shape:

```python
for ob_ci, (out_iter,) in pl.parallel(4, init_values=(out,)):
    out_iter__window = pl.tensor.slice(out_iter, [16, 64], [0, group_base + ob_ci * 64])
    out_next__iter_windowed = self.q_proj_chunk_group__iter_windowed(
        x, group_base, out_iter__window, ob_ci
    )
    out_next = pl.tensor.assemble(
        out_iter, out_next__iter_windowed, [0, group_base + ob_ci * 64]
    )
    out_rv = pl.yield_(out_next)
```

Per-iteration cloned kernel:

```python
@pl.function(type=pl.FunctionType.InCore)
def q_proj_chunk_group__iter_windowed(self, x, group_base, out_window, ob_ci):
    q0 = group_base + ob_ci * 64
    tile = pl.load(x, [0, q0], [16, 64])
    out_next = pl.store(tile, [0, 0], out_window)
    return out_next
```

## Matching Constraints

The matcher stays narrow:

- exactly one `Out` param
- function body is top-level `ForStmt` plus trailing `ReturnStmt`
- top-level loop carries exactly one tensor iter-arg initialized from the `Out`
- loop has `loop_origin = ChunkInner`
- loop kind is `Parallel` or `Sequential`
- loop body contains exactly one yielded `tile.store(..., offsets, iter_arg)`
- per-iteration offsets can be expressed from params plus the top-level loop var
- the stored tile shape is the per-task window shape

Rejected in this phase:

- nested or chained `ChunkInner` loop trees
- multiple stores to the carried `Out`
- non-window writes
- ambiguous offsets
- inner reduction hoisting

## Rewrite Strategy

### Step 1: Analyze The Top-Level Loop

Extract:

- the top-level loop metadata (`kind`, `start`, `stop`, `step`, loop var)
- the unique `tile.store`
- the expanded per-iteration global offsets
- the stored tile shape

### Step 2: Clone A Per-Iteration Kernel

Build `kernel__iter_windowed` by:

- narrowing the `Out` param and return type to one iteration's tile shape
- appending the original loop var as a scalar parameter
- cloning the original loop body without the top-level loop shell
- rewriting the single `tile.store` offset to all-zero local coordinates
- replacing the trailing `yield_` with `return`

### Step 3: Hoist The Loop To Orchestration

Replace:

```python
out_next = self.kernel(..., out)
```

with an orch-visible loop that:

- carries the parent tensor via loop iter-arg
- slices one iteration window from the carried parent tensor
- submits one `kernel__iter_windowed` task for that slice
- rebinds parent SSA via `tensor.assemble(...)`

## Why This Changes Runtime Task Granularity

Unlike the first-stage `__windowed` mode, this rewrite changes the number of
runtime-visible kernel submissions:

- before: one kernel task for the whole top-level `ChunkInner` group
- after: one kernel task per top-level `ChunkInner` iteration

That gives orchestration and runtime a real per-iteration task graph. For
parallel loops, runtime can schedule disjoint windows independently. For
sequential loops, the task chain remains explicit at the orch level.

## Correctness Invariants

The rewrite must preserve:

- original numerical results
- parent tensor SSA identity after each orch iteration
- loop kind (`Parallel` stays parallel, `Sequential` stays sequential)
- the original "one `pl.at(...)` outlines to one InCore kernel" invariant
- default behaviour when `enable_out_window_task_split` is `False`

It must never:

- silently change the default pipeline
- drop the orchestration-side `tensor.assemble(...)`
- hoist unsupported nested reduction structure

## Verification

Minimum verification for this mode:

1. Pass UT proves the orch-visible loop and `__iter_windowed` clone exist.
2. Codegen UT proves final orchestration C++ contains the hoisted loop, the
   narrower `view(...)`, and the per-iteration kernel submit.
3. Runtime/ST proves numerical correctness and can inspect saved artifacts.

