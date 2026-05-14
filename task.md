# Task Record

## Current Task
- Summary: Run both original Scenario B runtime cases on `myserver` and return their execution artifacts.
- Cases:
  - `OriginalQProjChunkedLoopProgram`
  - `OriginalKVProjOuterParallelProgram`
- Required return artifacts:
  - all generated swimlane files
  - all files under each run's `orchestration/` directory
- Status: pending
- Local Branch: out_slice_B
- Remote Host: myserver
- Remote Repo Path: `/data/sunkaixuan/pypto3`
- Latest Tested Commit: pending
- Last Updated: 2026-05-14 Asia/Shanghai

## Local Test Cases
The two local runtime vehicles are in `tests/st/runtime/test_manual_scope_pipeline.py`.

### OriginalQProjChunkedLoopProgram
Runtime test class:

```text
TestOriginalQProjChunkedLoopRuntime
```

Program builder:

```text
_build_original_q_proj_chunked_loop_program()
```

Target DSL shape:

```python
with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
    for ob in pl.parallel(q_out_blocks, chunk=4):
        q0 = ob * Q_OUT_CHUNK
        q_acc = ...
        q_proj = pl.assemble(q_proj, q_acc, [b0, q0])
```

### OriginalKVProjOuterParallelProgram
Runtime test classes:

```text
TestOriginalKVProjOuterParallelRuntime
TestOriginalKVProjOuterParallelSwimlane
```

Program builder:

```text
_build_original_kv_proj_outer_parallel_program()
```

Target DSL shape:

```python
for ob_chunk in pl.parallel(0, kv_out_blocks, 4):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj"):
        for ob in pl.range(ob_chunk, ob_chunk + 4):
            kv0 = ob * KV_OUT_CHUNK
            k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])
            v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])
```

## Required Remote Runs
Run from the remote repo root:

```bash
cd /data/sunkaixuan/pypto3
```

### 1. QProj correctness + orchestration artifacts

```bash
python3 -m pytest \
  tests/st/runtime/test_manual_scope_pipeline.py::TestOriginalQProjChunkedLoopRuntime \
  -vv --maxfail=1 --platform=a2a3 --device <TASK_DEVICE> --save-kernels
```

### 2. QProj profiling + swimlane artifacts

```bash
python3 -m pytest \
  tests/st/runtime/test_manual_scope_pipeline.py::TestOriginalQProjChunkedLoopRuntime \
  -vv --maxfail=1 --platform=a2a3 --device <TASK_DEVICE> --runtime-profiling
```

### 3. KVProj correctness + orchestration artifacts

```bash
python3 -m pytest \
  tests/st/runtime/test_manual_scope_pipeline.py::TestOriginalKVProjOuterParallelRuntime \
  -vv --maxfail=1 --platform=a2a3 --device <TASK_DEVICE> --save-kernels
```

### 4. KVProj profiling + swimlane artifacts

```bash
python3 -m pytest \
  tests/st/runtime/test_manual_scope_pipeline.py::TestOriginalKVProjOuterParallelSwimlane \
  -vv --maxfail=1 --platform=a2a3 --device <TASK_DEVICE> --runtime-profiling
```

## Expected Remote Artifacts
For QProj, collect from every matching latest run directory:

```text
build_output/original_q_proj_chunked_loop_*/swimlane_data/*
build_output/original_q_proj_chunked_loop_*/orchestration/*
build_output/original_q_proj_chunked_loop_task_split_*/swimlane_data/*
build_output/original_q_proj_chunked_loop_task_split_*/orchestration/*
```

For KVProj, collect from every matching latest run directory:

```text
build_output/original_kv_proj_outer_parallel_*/swimlane_data/*
build_output/original_kv_proj_outer_parallel_*/orchestration/*
```

At minimum, return these if present:

```text
swimlane_data/l2_perf_records.json
swimlane_data/merged_swimlane_*.json
orchestration/*
```

Copy local artifacts under:

```text
artifacts/swimlane/
```

Use names that include:

```text
original_q_proj_chunked_loop
original_q_proj_chunked_loop_task_split
original_kv_proj_outer_parallel
timestamp
```

## Validation Expectations
- Each pytest command should pass before copying artifacts.
- Each returned `swimlane_data/` directory should contain a non-empty `l2_perf_records.json`.
- If `merged_swimlane_*.json` exists, copy it too because it is Perfetto-importable.
- Copy the full `orchestration/` directory, not just `main.cpp`.
- QProj default run should contain `q_proj__windowed` in generated orchestration.
- QProj task-split run should contain `q_proj__iter_windowed` in generated orchestration.
- KVProj baseline should contain `kv_proj` and should not contain `kv_proj__windowed`.
- KVProj baseline should not contain `kv_proj__iter_windowed`.
- Inspect task count, core assignment, overlap, `fanout`, and `fanout_count` from raw swimlane JSON.

## Server Pitfalls From `pypto3.0/task.md`
- Prefer direct SSH for normal remote build and test work.
- Avoid `task-submit` for ordinary build/test commands unless queue execution is specifically required.
- If queued execution is required, redirect pytest output to a remote log file so failures are recoverable.
- If `myserver:/data/sunkaixuan/pypto3` is stale or remote `origin` is unreliable, use bundle fetch/bootstrap.
- Always confirm the remote checkout is on the intended commit before running tests.
- Rebuild `pypto_core` on `myserver` after PassContext, bindings, or C++ changes.
- Stale Python bindings can look like pass/runtime failures.
- Ensure the runtime imports the rebuilt local `pypto_core`, not an old installed package.
- Export `PTOAS_ROOT=/usr/local/bin/ptoas-bin` in queued wrappers when needed.
- If onboard runtime binaries are missing, run `runtime/simpler_setup/build_runtimes.py --platforms a2a3`.
- If using `task-submit --device auto`, propagate the allocated device as `<TASK_DEVICE>`.
- The tests may otherwise default to device 0 and fail on a queue-assigned nonzero NPU.
- Keep `--platform=a2a3` explicit for these runs.
- Do not treat missing `ptoas`, stale bindings, missing runtime binaries, or wrong device id as pass failures.

## Completion Criteria
- Report the exact remote commit tested.
- Report every pytest command and result.
- Return local paths for all copied `swimlane_data/*` files.
- Return local paths for all copied `orchestration/*` files.
- Summarize whether QProj default, QProj task-split, and KVProj baseline
  generated the expected orchestration names.
