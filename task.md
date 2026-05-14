# Task Record

## Current Task
- Summary: Implement a configurable ChunkInner task-split mode so one outlined `pl.at(...)` "big kernel" can be rewritten into multiple orchestrator-visible runtime tasks, one per eligible top-level `parallel` / `range` iteration.
- Status: in_progress
- Local Branch: out_slice_B
- Last Updated: 2026-05-14 11:02:02 Asia/Shanghai

## Phase 2 Goal
- Keep the established invariant that one `pl.at(...)` still outlines to one InCore kernel first.
- Add an optional second-stage pass rewrite that hoists the top-level eligible `ChunkInner` loop to orchestration.
- Make each hoisted iteration submit one `__iter_windowed` runtime task so AICore can reschedule at that finer granularity.
- Keep the feature behind `enable_out_window_task_split`, default `False`, independent from `enable_out_window_rewrite`.

## Phase 2 Design Snapshot
1. Match only a narrow post-`ConvertTensorToTileOps` shape:
   - one `Out`
   - one top-level `ChunkInner` `parallel` or `range`
   - one carried tensor iter-arg initialized from that `Out`
   - one yielded `tile.store(..., offsets, iter_arg)`
2. Clone a per-iteration `__iter_windowed` kernel:
   - narrow the `Out` / return type to the single-iteration window
   - append the loop var as a scalar parameter
   - localize the store offsets to `[0, 0, ...]`
3. Rewrite the orchestration call site into an orch-visible loop:
   - `tensor.slice(parent_iter, window_shape, iter_offsets)`
   - `self.kernel__iter_windowed(..., slice, loop_var)`
   - `tensor.assemble(parent_iter, iter_result, iter_offsets)`
4. Keep nested inner reductions such as `kb` inside the cloned per-iteration kernel in this first version.

## Phase 2 Files
- `src/ir/transforms/optimize_orch_tensors_pass.cpp`
- `include/pypto/ir/transforms/pass_context.h`
- `src/ir/transforms/pass_context.cpp`
- `python/bindings/modules/passes.cpp`
- `python/pypto/pypto_core/passes.pyi`
- `python/pypto/ir/compile.py`
- `python/pypto/ir/pass_manager.py`
- `python/pypto/runtime/runner.py`
- `tests/ut/ir/transforms/test_optimize_orch_tensors.py`
- `tests/ut/codegen/test_orchestration_codegen.py`
- `tests/st/runtime/test_manual_scope_pipeline.py`
- `docs/en/dev/passes/13-optimize_orch_tensors.md`
- `docs/en/dev/passes/13c-optimize_orch_tensors_chunk_inner_task_split_design.md`
- `docs/en/dev/passes/13d-optimize_orch_tensors_chunk_inner_task_split_change_summary.md`
- `docs/zh-cn/dev/passes/13c-optimize_orch_tensors_chunk_inner_task_split_design.md`
- `docs/zh-cn/dev/passes/13d-optimize_orch_tensors_chunk_inner_task_split_change_summary.md`

Phase 2 quick reference: if you need a concise “what changed / what to inspect / what to run” guide during remote execution or debugging, read `docs/en/dev/passes/13d-optimize_orch_tensors_chunk_inner_task_split_change_summary.md` first.

## Phase 2 Test Coverage
- New pass UT:
  - `test_chunk_inner_parallel_loop_task_split_hoists_iters_to_orchestration`
  - `test_chunk_inner_range_loop_task_split_hoists_iters_to_orchestration`
  - `test_chunk_inner_parallel_loop_task_split_respects_switch`
- New codegen UT:
  - `test_chunk_inner_parallel_loop_task_split_visible_in_orchestration_codegen`
- New runtime/ST case:
  - `tests/st/runtime/test_manual_scope_pipeline.py::TestChunkInnerTaskSplitRuntime`
- Existing regression to keep:
  - `tests/ut/jit/test_qwen3_decode.py -k q_projection_parallel_does_not_split_into_per_iter_kernels`

## Phase 2 Remote Validation Commands
```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py -k task_split -v
python -m pytest tests/ut/codegen/test_orchestration_codegen.py -k task_split -v
python -m pytest tests/st/runtime/test_manual_scope_pipeline.py::TestChunkInnerTaskSplitRuntime -v --platform=a2a3 --save-kernels

# keep old behavior stable when the new switch is off
python -m pytest tests/ut/jit/test_qwen3_decode.py -k q_projection_parallel_does_not_split_into_per_iter_kernels -v
```

## Notes For Next Debug Session
- `enable_out_window_task_split=False` must keep the old "one big outlined kernel call" behavior.
- `enable_out_window_task_split=True` should create multiple runtime submits only for the matched top-level `ChunkInner` loop.
- This version does not split nested inner reductions; if you inspect generated kernel PTO, inner `range` such as `kb` should still remain inside each `__iter_windowed` kernel.

## Historical Record: Phase 1 Snapshot
- Summary: Reuse the proven runtime profiling flow from `pypto3.0` to collect and return a Perfetto-importable `merged_swimlane_*.json` for the queued `TestChunkInnerOutWindowRuntime` case on `myserver`.
- Status: completed
- Local Branch: out_slice_B
- Base Merge State: `out_slice` has already been merged into this branch
- Remote Host: myserver
- Remote Repo Path: `/data/sunkaixuan/pypto3`
- Latest Tested Commit: `51c2a4de44266cc933076d73220531606d2f904a`
- Last Updated: 2026-05-14 09:25 Asia/Shanghai

## Target Behavior
- Validate and implement the conclusion that a single `pl.at(...)` still outlines to one InCore kernel, while its internal `parallel/range` stay lowered inside the kernel.
- Add a narrow pass-level optimization that makes the kernel-internal window write set visible to orchestration/runtime for the first-stage q_proj-like case.
- Reuse the existing `enable_out_window_rewrite` switch so the behavior can be enabled/disabled consistently with current Pattern 5.

## Approved First-Stage Design
1. Keep orchestration-side `tensor.assemble(...)`.
2. Preserve full-tensor SSA in orchestration.
3. Only support a narrow q_proj-like shape in phase 1.
4. Match post-`ConvertTensorToTileOps` IR, not source DSL directly.
5. Treat `ChunkInner` loop-carried `tile.store` as the target shape to externalize.

Conceptual target:

```python
out__window = pl.tensor.slice(out, window_shape, base_offsets)
out_next__windowed = self.kernel__windowed(..., out__window)
out_next = pl.tensor.assemble(out, out_next__windowed, base_offsets)
```

## Work Completed
- Verified the main worktree stayed on `out_slice_B`; local `HEAD` and `origin/out_slice_B` both point at pushed commit `51c2a4de`.
- Confirmed local-only dirt still does not participate in remote validation: `runtime` submodule state, session files, bundles, scratch artifacts, and `task.md`; remote results below target the latest pushed commit, not the full local workspace.
- Root-caused the first false-negative in `AnalyzeChunkInnerLoop`: `CountVarRefsInStmt` overrode `VisitExpr_(Var/IterArg)` without delegating to `IRVisitor`, so it failed to recurse into `IterArg.initValue_` and nested type dims.
- Fixed and pushed that helper bug as `18152de9` (`fix(ir): count iter init refs for out windows`) together with the focused codegen/runtime witness tests for the ChunkInner out-window path; latest pushed branch tip remains `51c2a4de`.
- Synced `myserver:/data/sunkaixuan/pypto3` to `51c2a4de`, rebuilt remote `pypto_core`, and revalidated the non-NPU chain successfully:
  - `tests/ut/ir/transforms/test_optimize_orch_tensors.py -k chunk_inner_parallel_loop_rewrites_to_windowed_clone -vv`
  - `tests/ut/ir/transforms/test_optimize_orch_tensors.py -k TestOutWindowExternalizer -vv`
  - `tests/ut/codegen/test_orchestration_codegen.py -k chunk_inner_parallel_loop_out_window_rewrite_visible_in_orchestration_codegen -vv`
  - `tests/ut/jit/test_qwen3_decode.py -k q_projection_parallel_does_not_split_into_per_iter_kernels -vv`
  - `python tests/ut/ir/transforms/dump_parallel_inside_pl_at_codegen.py --output-dir /tmp/parallel_inside_pl_at`
- Confirmed the correct ST selector is `tests/st/runtime/test_manual_scope_pipeline.py::TestChunkInnerOutWindowRuntime`; the older `-k chunk_inner_out_window` command was stale.
- Isolated the `myserver` queue launcher semantics:
  - simple `task-submit --run ...` commands get `--device N` auto-appended
  - compound inline commands are easy to mangle through SSH/PowerShell quoting and should use `{}` or `$TASK_DEVICE`
- Proved a reliable queue entry path by creating `/tmp/chunk_inner_out_window_taskq.sh` over direct SSH and launching it with `task-submit --device auto --run '/tmp/chunk_inner_out_window_taskq.sh {}'`; the run now reaches the wrapper and passes the allocated device id.
- Fixed the queued wrapper environment on `myserver` by using a literal heredoc push plus `/usr/bin/python` and `PYTHONPATH=/data/sunkaixuan/pypto3/python:${PYTHONPATH:-}`.
- Re-ran `task-submit --device auto --max-time 0 --run '/tmp/chunk_inner_out_window_taskq.sh {}'`; `tests/st/runtime/test_manual_scope_pipeline.py::TestChunkInnerOutWindowRuntime` passed (`2 passed, 6 deselected`) and saved the witness under `/tmp/chunk_inner_out_window_51c2a4de_taskq/chunk_inner_out_window_16x512`.
- Reused the `pypto3.0` runtime profiling pattern on `myserver` for commit `51c2a4de`:
  - first tried profiling the full `TestChunkInnerOutWindowRuntime` selector and isolated a non-product blocker: the second `test_runner.run(...)` in the same pytest process hit `init_l2_perf failed: -1` after the first profiling run had already passed.
  - reran only `tests/st/runtime/test_manual_scope_pipeline.py::TestChunkInnerOutWindowRuntime::test_correctness` with `--runtime-profiling --save-kernels --kernels-dir /tmp/chunk_inner_out_window_51c2a4de_swimlane`; that single profiling run passed and generated swimlane artifacts successfully.
- Collected and copied back the runtime profiling artifacts:
  - Remote raw trace: `/tmp/chunk_inner_out_window_51c2a4de_swimlane/chunk_inner_out_window_16x512/swimlane_data/l2_perf_records.json`
  - Remote Perfetto file: `/tmp/chunk_inner_out_window_51c2a4de_swimlane/chunk_inner_out_window_16x512/swimlane_data/merged_swimlane_20260514_092404.json`
  - Remote archival copies:
    - `/data/sunkaixuan/pypto3/artifacts/swimlane/l2_perf_records_chunk_inner_out_window_51c2a4de.json`
    - `/data/sunkaixuan/pypto3/artifacts/swimlane/merged_swimlane_chunk_inner_out_window_51c2a4de.json`
  - Local copies:
    - `D:\PTO\code\pypto对比\out_slice_B\artifacts\swimlane\l2_perf_records_chunk_inner_out_window_51c2a4de.json`
    - `D:\PTO\code\pypto对比\out_slice_B\artifacts\swimlane\merged_swimlane_chunk_inner_out_window_51c2a4de.json`

## Active Files
- `src/ir/transforms/optimize_orch_tensors_pass.cpp`
- `tests/ut/ir/transforms/test_optimize_orch_tensors.py`
- `tests/ut/codegen/test_orchestration_codegen.py`
- `tests/st/runtime/test_manual_scope_pipeline.py`
- `docs/en/dev/passes/13-optimize_orch_tensors.md`
- `docs/zh-cn/dev/passes/13-optimize_orch_tensors.md`
- `docs/en/dev/passes/13a-optimize_orch_tensors_chunk_inner_window_plan.md`

## Current Problem
- None; the swimlane profiling artifacts for `51c2a4de` have been captured and copied back locally.

## Next Suggested Action
1. Open `D:\PTO\code\pypto对比\out_slice_B\artifacts\swimlane\merged_swimlane_chunk_inner_out_window_51c2a4de.json` in Perfetto and compare it with any prior baseline or follow-up branch.
2. If you need the artifact-inspection test (`test_saved_artifact_contains_windowed_orchestration`) under profiling as well, run it in a separate queued pytest invocation from `test_correctness` to avoid the observed L2 perf collector re-init failure.

## Scheduled Task Description
- Task purpose: Build this branch on the Linux/NPU server and verify the first-stage `ChunkInner` out-window rewrite behaves as designed, without regressing the "one `pl.at(...)` outlines to one InCore kernel" conclusion.
- Working directory: repository root of the server checkout for this branch.
- Environment requirements:
  - built/importable `pypto_core`
  - `PYTHONPATH=$(pwd)/python:$PYTHONPATH`
  - if codegen tests need it, a valid `ptoas` environment such as `PTOAS_ROOT`
  - if you want the runtime artifact witness too, add `--save-kernels` and optionally `--kernels-dir <dir>`
- Commands to run:

```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py -k chunk_inner_parallel_loop_rewrites_to_windowed_clone -v
python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py -k chunk_inner_parallel_loop_respects_out_window_switch -v
python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py -k TestOutWindowExternalizer -v
python -m pytest tests/ut/codegen/test_orchestration_codegen.py -k chunk_inner_parallel_loop_out_window_rewrite_visible_in_orchestration_codegen -v
python -m pytest tests/ut/jit/test_qwen3_decode.py -k q_projection_parallel_does_not_split_into_per_iter_kernels -v
python tests/ut/ir/transforms/dump_parallel_inside_pl_at_codegen.py --output-dir /tmp/parallel_inside_pl_at
python -m pytest tests/st/runtime/test_manual_scope_pipeline.py::TestChunkInnerOutWindowRuntime -v --platform=a2a3 --save-kernels
```

## Suggested Verification Commands
```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py -k chunk_inner_parallel_loop_rewrites_to_windowed_clone -v
python -m pytest tests/ut/codegen/test_orchestration_codegen.py -k chunk_inner_parallel_loop_out_window_rewrite_visible_in_orchestration_codegen -v
python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py -k TestOutWindowExternalizer -v
python -m pytest tests/ut/jit/test_qwen3_decode.py -k q_projection_parallel_does_not_split_into_per_iter_kernels -v
python tests/ut/ir/transforms/dump_parallel_inside_pl_at_codegen.py --output-dir /tmp/parallel_inside_pl_at
python -m pytest tests/st/runtime/test_manual_scope_pipeline.py::TestChunkInnerOutWindowRuntime -v --platform=a2a3 --save-kernels
```

## Notes
- Local Windows verification is currently blocked because this worktree does not have an importable `pypto_core`.
- Do not remove orchestration-side `tensor.assemble`; it is required to preserve full-parent SSA and downstream dependencies.

## Conclusions
1. Commit `18152de9` fixed one real false-negative in `AnalyzeChunkInnerLoop` by restoring recursive Var/IterArg reference counting, but the positive ChunkInner path still fails a later analysis guard and does not yet emit a `__windowed` clone.
2. Remote direct validation on pushed commit `51c2a4de` is green for the focused UT/codegen/JIT chain, so the remaining blocker has moved from pass logic to queued-NPU launch plumbing.
3. The older `pypto3.0` task record confirms two reusable queue pitfalls for `myserver`: queued runs need the rebuilt local `pypto_core` plus explicit `PTOAS_ROOT=/usr/local/bin/ptoas-bin`, and NPU queue jobs may still fail if the allocated `TASK_DEVICE` does not propagate into the `tests/st` runtime path.
4. On `myserver`, `task-submit` auto-appends `--device N` to simple `--run` commands, so reliable queued ST reruns should go through a wrapper script with `{}` rather than inline `cd`/`python` quoting; the first wrapper-based rerun already reached the script and reduced the remaining failure to `${PYTHONPATH:-}` handling.
5. Commit `51c2a4de` now passes the full intended remote chain on `myserver`, including queued NPU ST `TestChunkInnerOutWindowRuntime`; the last blocker was only wrapper-script environment/quoting, not product logic.
6. For `TestChunkInnerOutWindowRuntime`, runtime profiling artifacts are available on `51c2a4de`, but the full two-test selector should not be profiled in one pytest process because the second `test_runner.run(...)` can fail at `init_l2_perf`; use a single-run selector such as `test_correctness` when collecting swimlane data.

## Consolidated Summaries
### Imported Pitfalls
- Reusable remote pitfalls from `pypto3.0/task.md`: direct SSH should remain the default for normal build/test work, queued `task-submit` runs need explicit runtime/toolchain environment (`pypto_core`, `PTOAS_ROOT`, and likely Ascend vars), and device-id propagation must be checked whenever `--device auto` allocates a nonzero NPU.
