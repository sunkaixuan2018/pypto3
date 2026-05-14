# OptimizeOrchTensors Chunk-Inner Task-Split Change Summary

## Purpose

This note is the operator-facing summary for the current ChunkInner task-split
implementation. It complements the design documents by focusing on:

- what changed in code
- how the new switch behaves
- what artifacts to inspect during debug
- which tests prove the intended behavior

## What Changed

The new implementation adds an optional second-stage rewrite on top of the
existing ChunkInner out-window externalization.

Before this change:

- one `pl.at(...)` outlines to one InCore kernel
- the top-level `ChunkInner` `parallel` / `range` loop remains inside that
  kernel
- runtime sees one kernel submit for the whole loop body

With `enable_out_window_task_split=True`:

- the same top-level eligible `ChunkInner` loop is hoisted to orchestration
- each hoisted iteration slices one output window
- orchestration submits one `__iter_windowed` kernel per iteration
- runtime / AICore now see multiple tasks instead of one large task

The original outlining conclusion is intentionally preserved: the split happens
after outlining, inside `OptimizeOrchTensors`, not inside
`OutlineIncoreScopes`.

## Switches

- `enable_out_window_rewrite`
  - existing Pattern 5 switch
  - default: `True`
  - behavior: externalizes a narrowed output window but still keeps one kernel
    submit
- `enable_out_window_task_split`
  - new Pattern 6 switch
  - default: `False`
  - behavior: hoists the eligible top-level ChunkInner loop so each iteration
    becomes its own orchestration-visible runtime task

The switches are independent. Keeping task split default-off is part of the
compatibility contract.

## Supported Shape In This Version

This first version only rewrites a narrow post-`ConvertTensorToTileOps` shape:

- exactly one `Out` param
- a top-level `ChunkInner` `parallel` or `range`
- exactly one carried tensor iter-arg initialized from that `Out`
- exactly one yielded `tile.store(..., offsets, iter_arg)`
- offsets depend only on params and the top-level loop var

Nested inner reductions, such as a `kb` accumulation loop inside one `ob`
iteration, remain inside each cloned `__iter_windowed` kernel.

## Key Implementation Files

- Pass logic:
  - `src/ir/transforms/optimize_orch_tensors_pass.cpp`
- Switch plumbing:
  - `include/pypto/ir/transforms/pass_context.h`
  - `src/ir/transforms/pass_context.cpp`
  - `python/bindings/modules/passes.cpp`
  - `python/pypto/pypto_core/passes.pyi`
  - `python/pypto/ir/compile.py`
  - `python/pypto/ir/pass_manager.py`
  - `python/pypto/runtime/runner.py`
- ST harness propagation:
  - `tests/st/harness/core/harness.py`
  - `tests/st/harness/core/test_runner.py`

## Expected IR / Codegen Witnesses

### Pass-level IR

When task split is enabled, the orchestration function should contain:

- a `for ... in pl.parallel(...)` or `for ... in pl.range(...)`
- `pl.tensor.slice(out_iter, ...)`
- `self.kernel__iter_windowed(...)`
- `pl.tensor.assemble(out_iter, iter_result, iter_offsets)`

The cloned kernel should:

- no longer contain the top-level outer loop shell
- keep the inner reduction loops, if any
- store to local `[0, 0, ...]` offsets

### Orchestration C++

The generated orchestrator should show:

- an explicit C++ `for (...)` loop
- `Tensor ...window... = ext_out.view(...)`
- one `rt_submit_*_task(...)` per loop iteration
- the callee name `__iter_windowed`

### Kernel PTO

The per-iteration kernel PTO should still show any nested inner reduction loop
as kernel-internal control flow. Only the top-level matched ChunkInner loop is
hoisted.

## Tests Added / Updated

### Unit tests

- `tests/ut/ir/transforms/test_optimize_orch_tensors.py`
  - `test_chunk_inner_parallel_loop_task_split_hoists_iters_to_orchestration`
  - `test_chunk_inner_range_loop_task_split_hoists_iters_to_orchestration`
  - `test_chunk_inner_parallel_loop_task_split_respects_switch`

### Codegen tests

- `tests/ut/codegen/test_orchestration_codegen.py`
  - `test_chunk_inner_parallel_loop_task_split_visible_in_orchestration_codegen`

### Runtime / ST

- `tests/st/runtime/test_manual_scope_pipeline.py`
  - `_ChunkInnerTaskSplitPTO`
  - `TestChunkInnerTaskSplitRuntime`

### Regression to keep

- `tests/ut/jit/test_qwen3_decode.py -k q_projection_parallel_does_not_split_into_per_iter_kernels`

This regression matters because the default path must still preserve the old
"one outlined kernel call" behavior when task split is not enabled.

## Recommended Debug Flow

1. Run the pass UT with task split enabled.
2. Run the orchestration codegen UT and inspect `main.cpp`.
3. Run the runtime/ST case with `--save-kernels`.
4. Compare:
   - orchestration `main.cpp`
   - cloned kernel names
   - per-iteration `view(...)`
   - number of runtime submits

## Remote Commands

```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py -k task_split -v
python -m pytest tests/ut/codegen/test_orchestration_codegen.py -k task_split -v
python -m pytest tests/st/runtime/test_manual_scope_pipeline.py::TestChunkInnerTaskSplitRuntime -v --platform=a2a3 --save-kernels

# compatibility regression
python -m pytest tests/ut/jit/test_qwen3_decode.py -k q_projection_parallel_does_not_split_into_per_iter_kernels -v
```

## Current Limits

- no nested top-level tree splitting
- no multi-store carried-out rewrite
- no overlap proof for complex windows
- no default enablement
- no local Windows full verification in this worktree because `pytest` is not
  installed and the current `build/` directory is incomplete
