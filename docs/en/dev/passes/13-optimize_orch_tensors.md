# OptimizeOrchTensors Pass

Optimizes tensor buffer usage across orchestration and InCore functions by eliminating redundant allocations and improving data flow.

## Overview

After `ConvertTensorToTileOps`, orchestration functions may still carry redundant output buffers, lose parent-stride information, or submit kernels against full parent tensors when only a window is actually written. This pass applies six optimization patterns to reduce allocations, improve buffer/layout information, and optionally expose finer-grained ChunkInner runtime tasks.

**Requirements**:

- Input IR must have InCore scopes outlined with tile ops (run `ConvertTensorToTileOps` first)

**When to use**: Run immediately after `ConvertTensorToTileOps` and before `FlattenTileNdTo2D`.

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::OptimizeOrchTensors()` | `passes.optimize_orch_tensors()` | Program-level |

## Related Docs

- Design: [`13b-optimize_orch_tensors_chunk_inner_window_design.md`](13b-optimize_orch_tensors_chunk_inner_window_design.md)
- Task-split design: [`13c-optimize_orch_tensors_chunk_inner_task_split_design.md`](13c-optimize_orch_tensors_chunk_inner_task_split_design.md)
- Change summary: [`13d-optimize_orch_tensors_chunk_inner_task_split_change_summary.md`](13d-optimize_orch_tensors_chunk_inner_task_split_change_summary.md)
- Implementation plan: [`13a-optimize_orch_tensors_chunk_inner_window_plan.md`](13a-optimize_orch_tensors_chunk_inner_window_plan.md)

**Python usage**:

```python
from pypto.pypto_core import passes

opt_pass = passes.optimize_orch_tensors()
program_opt = opt_pass(program)
```

## Patterns

The pass applies six patterns in sequence. Each pattern sees the results of the previous one.

### Pattern 1: Iter-Arg Reuse (IterArgReuseOptimizer)

**Problem**: Inside a `for`/`while` loop, each iteration allocates a new output tensor via `tensor.create`, even though the InCore result feeds back as an iter-arg to the next iteration.

**Solution**: Merge the `Out` param into the corresponding `In` param (promoted to `InOut`), remove the `tensor.create`, and redirect `tile.store` to write into the reused buffer.

**Before**:

```python
for i in pl.range(N, init_values=[init_buf]):
    out: pl.Tensor = pl.tensor.create(shape, dtype=pl.FP32)  # redundant alloc
    result: pl.Tensor = self.incore_fn(iter_arg, out)          # In + Out params
    pl.yield_(result)
```

**After**:

```python
for i in pl.range(N, init_values=[init_buf]):
    result: pl.Tensor = self.incore_fn(iter_arg)  # InOut param (reuses iter-arg buffer)
    pl.yield_(result)
```

### Pattern 2: Assemble Parent Strides (AssembleParentStridesOptimizer)

**Problem**: When orchestration scatters InCore results into a larger tensor via `tensor.assemble`, the InCore function's `tile.store` doesn't know the parent tensor's strides, which can lead to suboptimal memory layout.

**Solution**: Analyze `tensor.assemble(parent, incore_result, offset)` patterns in orchestration. Attach the parent tensor's shape as `TensorView` strides on the InCore function's `Out` param type, so `tile.store` can use the correct memory layout.

### Pattern 4: Slice Input Strides (SliceInputStridesOptimizer)

**Problem**: When orchestration passes a sliced tensor (`tensor.slice`) as an `In` argument to an InCore function, the InCore function's parameter has contiguous strides (computed from its own shape), not the parent tensor's strides. This causes incorrect memory access when the slice is a non-contiguous view of the parent.

**Solution**: Analyze `tensor.slice(parent, size, offset)` patterns in orchestration. When a slice result is passed as an `In` argument to an InCore call, attach the parent tensor's shape-derived strides via `TensorView` on the InCore function's `In` param type, so `tile.load` uses the correct memory layout.

### Pattern 3: Assemble-Loop Rewrite (AssembleLoopRewriter)

**Problem**: An InCore function contains a `for` loop that accumulates results via `tile.assemble` into an iter-arg, then stores the final result. The `tile.assemble` creates intermediate tile copies each iteration.

**Solution**: Rewrite the loop body to use `tile.store` directly (writing into the `Out` param), initializing the iter-arg from the `Out` param instead of a `tile.create`.

### Pattern 5: Out Window Externalization (OutWindowExternalizer)

**Problem**: Some InCore kernels keep a full parent tensor in their `Out` signature, but only write a smaller sub-window. Two narrow shapes matter here:

- a direct final `tile.store` writes a smaller window at non-zero offsets
- a top-level `ChunkInner` loop carries the full `Out` tensor, while each iteration writes one disjoint window through a loop-carried `tile.store`

In orchestration and `manual_scope`, those shapes cause runtime dependencies to be built on the whole parent tensor rather than the written subview.

**Solution**: Clone the kernel into a `__windowed` variant whose write becomes local to a narrowed `Out` window. Rewrite eligible orchestration call sites from `self.kernel(..., out)` to `slice(out, window_shape, offsets) -> self.kernel__windowed(..., out_window) -> tensor.assemble(out, result, offsets)`, so later passes and runtime dependency tracking can operate on the sliced window instead of the whole tensor.

For the first-stage `ChunkInner` loop case, the pass intentionally stays narrow:

- exactly one `Out` param
- top-level loop marked with `loop_origin = ChunkInner`
- one loop-carried tensor iter-arg initialized from that `Out`
- one yielded `tile.store(..., offsets, iter_arg)`
- window shape and base offsets must be derivable from the loop trip range plus the store tile shape

The rewrite still preserves orchestration-side `tensor.assemble(...)` so the full parent tensor SSA value and downstream dependencies remain explicit.

### Pattern 6: ChunkInner Task Split (OutWindowExternalizer, optional)

**Problem**: Pattern 5 makes the written window visible at the orchestration boundary, but it still submits one cloned kernel task for the whole top-level `ChunkInner` loop body. Runtime still sees one task, not one task per eligible outer iteration.

**Solution**: When `PassContext(enable_out_window_task_split=True)` is enabled, the same narrow q_proj-like `ChunkInner` shape may be hoisted one step further:

- the original outlined InCore kernel is kept
- a per-iteration `__iter_windowed` clone is created
- the original top-level `ChunkInner` `parallel`/`range` loop is rewritten into an orchestration-visible loop
- each orchestration iteration slices one local output window, submits one cloned kernel task, and rebinds the parent SSA via `tensor.assemble(...)`

This changes runtime task granularity:

- default / Pattern 5 only: one runtime task for the whole outlined kernel call
- Pattern 6 enabled: one runtime task per top-level eligible `ChunkInner` iteration

The first implementation is intentionally narrow:

- exactly one `Out` param
- top-level loop marked with `loop_origin = ChunkInner`
- one loop-carried tensor iter-arg initialized from that `Out`
- one yielded `tile.store(..., offsets, iter_arg)`
- nested inner reductions such as `kb` remain inside the per-iteration kernel

This mode is default-off and independent from `enable_out_window_rewrite`.

## Example (Pattern 1)

**Before**:

```python
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def compute(self, x: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        x_tile = pl.load(x, (0,), (64,))
        y_tile = pl.tile.add(x_tile, x_tile)
        ret = pl.store(y_tile, (0,), out_0)
        return ret

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, buf: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.range(10, init_values=[buf]):
            out_0 = pl.tensor.create((64,), dtype=pl.FP32)
            result = self.compute(iter_arg, out_0)
            pl.yield_(result)
        return loop_result
```

**After** (Pattern 1 merges Out into In as InOut):

```python
@pl.program
class After:
    @pl.function(type=pl.FunctionType.InCore)
    def compute(self, x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        x_tile = pl.load(x, (0,), (64,))
        y_tile = pl.tile.add(x_tile, x_tile)
        ret = pl.store(y_tile, (0,), x)
        return ret

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, buf: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i in pl.range(10, init_values=[buf]):
            result = self.compute(iter_arg)
            pl.yield_(result)
        return loop_result
```

The `tensor.create` is eliminated; the iter-arg buffer is reused across iterations.

## Implementation

**Header**: `include/pypto/ir/transforms/passes.h`

**Implementation**: `src/ir/transforms/optimize_orch_tensors_pass.cpp`

**Python binding**: `python/bindings/modules/passes.cpp`

**Tests**: `tests/ut/ir/transforms/test_optimize_orch_tensors.py`

## Pass Properties

| Property | Value |
| -------- | ----- |
| Required | SplitIncoreOrch, IncoreTileOps |
| Produced | SplitIncoreOrch, IncoreTileOps |
| Invalidated | — |

## Key Components

| Component | Role |
| --------- | ---- |
| `IterArgReuseOptimizer` | Pattern 1 — merges Out params into In params for loop-carried buffers |
| `AssembleParentStridesOptimizer` | Pattern 2 — attaches parent strides via TensorView |
| `SliceInputStridesOptimizer` | Pattern 4 – attaches parent strides to In params via TensorView for slice patterns |
| `OutWindowExternalizer` | Pattern 5 – clones full-Out kernels into windowed variants and rewrites direct Out call sites through `slice + cloned call + assemble`, including the first-stage narrow `ChunkInner` loop case |
| `OutWindowExternalizer` | Pattern 6 – optionally hoists eligible top-level `ChunkInner` `parallel`/`range` iterations into orch-visible per-iteration runtime tasks via `__iter_windowed` clones |
| `AssembleLoopRewriter` | Pattern 3 – rewrites tile.assemble loops to tile.store loops |
| `BuildOutParamReturnMappings` | Shared helper — maps Out params to return indices via tile.store |
| `ComputeRowMajorStrides` | Shared helper — computes row-major strides from a shape |

## Scope

| Function type | Action |
| ------------- | ------ |
| InCore | Params/body rewritten (Patterns 1, 3, 4) |
| Orchestration / Opaque | Call sites rewritten (Patterns 1, 2; Patterns 5 and 6 apply to orchestration only) |
| Group | Unchanged |
