# OptimizeOrchTensors Chunk-Inner Window Externalization Plan

## Goal

Add a first-stage `OptimizeOrchTensors` extension that externalizes a narrow q_proj-like case where a single outlined InCore kernel keeps a full parent `Out` tensor, but its top-level `ChunkInner` parallel loop writes disjoint sub-windows via loop-carried `tile.store`.

The rewrite must:

- preserve full-tensor SSA in orchestration via `tensor.assemble`
- expose the written window to orchestration/runtime through `tensor.slice`
- stay behind the existing `enable_out_window_rewrite` switch
- remain intentionally narrow for the first implementation

## Scope

Supported in this phase:

- InCore function has exactly one `Out` param
- function body is a top-level `ForStmt` plus trailing `ReturnStmt`
- the top-level loop is `loop_origin = ChunkInner`
- loop has exactly one iter-arg, initialized from that `Out` param
- loop body has exactly one `tile.store(..., offsets, iter_arg)` that is yielded
- store window shape is constant from the stored tile shape
- base window offsets used at orchestration call site are expressible from callee params

Explicitly out of scope:

- multiple stores to the same loop-carried tensor
- nested/chained chunk-inner loops
- dynamic window shape inference
- overlapping window proofs
- non-ChunkInner parallel/range cases beyond the q_proj-like pattern

## Test-First Target

Primary failing UT:

- `tests/ut/ir/transforms/test_optimize_orch_tensors.py`
  - `TestOutWindowExternalizer.test_chunk_inner_parallel_loop_rewrites_to_windowed_clone`

This test encodes the intended first-stage behavior:

- source InCore kernel contains `pl.parallel(..., attrs={"loop_origin": pl.LoopOrigin.ChunkInner})`
- the kernel currently writes a full parent `out`
- after rewrite, orchestration becomes:
  - `out__window = pl.tensor.slice(out, [16, 256], [0, group_base])`
  - `out_next__windowed = self.q_proj_chunk_group__windowed(..., out__window)`
  - `out_next = pl.tensor.assemble(out, out_next__windowed, [0, group_base])`

## Implementation Shape

Files:

- Modify: `src/ir/transforms/optimize_orch_tensors_pass.cpp`
- Modify: `tests/ut/ir/transforms/test_optimize_orch_tensors.py`
- Modify: `docs/en/dev/passes/13-optimize_orch_tensors.md`
- Modify: `docs/zh-cn/dev/passes/13-optimize_orch_tensors.md`

Implementation sketch:

1. Extend Pattern 5 analysis with a second narrow matcher for:
   - top-level `ChunkInner` loop
   - loop-carried `Out`
   - single yielded `tile.store`
2. Produce a dedicated analysis record for this loop-window case:
   - out param index
   - orchestration-visible window shape
   - orchestration-visible base offsets
   - clone-time loop-local offset rewrite rule
3. Reuse existing orchestration call-site rewrite:
   - `tensor.slice(parent_out, shape, base_offsets)`
   - call cloned `__windowed` kernel with narrowed `Out`
   - `tensor.assemble(parent_out, result_window, base_offsets)`
4. Add a dedicated kernel clone/body rebuild path for this case:
   - narrow the `Out` param type
   - narrow loop iter-arg/return-var tensor types to the same window type
   - rewrite the `tile.store` offsets from global-window coordinates to local-window coordinates
5. Keep the existing direct-final-store Pattern 5 path unchanged

## Correctness Constraints

- Do not remove orchestration-side `tensor.assemble`; it preserves the full parent tensor SSA value for later users.
- Do not rewrite when the loop writes the `Out` tensor more than once.
- Do not rewrite when offset expressions cannot be safely separated into:
  - orchestrator-visible base window offsets
  - kernel-local offsets
- Do not bypass `enable_out_window_rewrite`.

## Verification Plan

Local Windows note:

- current worktree does not have an importable `pypto_core`, so execution verification is expected to happen on the Linux/server environment

Server commands:

```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py -k chunk_inner_parallel_loop_rewrites_to_windowed_clone -v
python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py -k TestOutWindowExternalizer -v
python -m pytest tests/ut/jit/test_qwen3_decode.py -k q_projection_parallel_does_not_split_into_per_iter_kernels -v
```
