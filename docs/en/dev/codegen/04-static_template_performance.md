# Static Template Performance Profiling

Use this benchmark to compare the static-template manual-scope path with the
traditional AUTO-scope path on the same program and runtime.

| Variant | Strategy | Expected codegen |
| ------- | -------- | ---------------- |
| `static_hit` | `Default` | `PTO2ScopeMode::MANUAL` plus explicit `add_dep(...)` |
| `traditional_baseline` | `DefaultWithoutStableRegionTemplates` | AUTO scope, runtime TensorMap dependency discovery |

## Run

The recommended workload is the full paged attention graph. It exercises:

```text
kernel_init_inplace -> kernel_qk_matmul -> kernel_softmax_prepare
  -> kernel_pv_matmul -> kernel_online_update
```

```bash
export PYTHONPATH=$(pwd)/python:$(pwd)/tests/st:${PYTHONPATH}
export PTOAS_ROOT=/path/to/ptoas
export PTO2_PROFILING=1
export PTO2_ORCH_PROFILING=1
export PTO2_TENSORMAP_PROFILING=1

python3 tests/st/runtime/manual_scope_template_perf.py \
  --workload paged_attention \
  --platform a2a3 \
  --device 0 \
  --rounds 100 \
  --profiling-detail \
  --output build_output/static_template_perf

cat build_output/static_template_perf/*/summary.md
```

Paged attention shape knobs:

```bash
--pa-batch 1 \
--pa-num-heads 16 \
--pa-head-dim 128 \
--pa-block-size 128 \
--pa-context-len 1024 \
--pa-max-model-len 4096 \
--pa-scale 1.0
```

Use `--workload bgemm_manual_scope` for the small BGEMM smoke workload.

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `Runtime rounds` | Parsed device-log rounds; should be close to the requested `--rounds` for each variant |
| `Compile frontend sched (us)` | `IdentifyStableRegions + LowerStableRegionsToManualScope + orchestration_codegen` |
| `Orch trimmed avg (us)` | Device orchestrator time after dropping fixed low/high outliers |
| `Sched trimmed avg (us)` | Device scheduler loop time, useful as a secondary signal |
| `Device orch saved per round (us)` | Baseline `Orch trimmed avg` minus static-hit `Orch trimmed avg` |
| `Net gain after measured rounds (us)` | `rounds * saved_per_round - compile_delta` |
| `Break-even runs` | `ceil(compile_delta / saved_per_round)` when saving is positive |

The main expected benefit is lower `Orch trimmed avg (us)` for `static_hit`,
because manual scope skips TensorMap lookup/insert work and uses generated
explicit dependencies instead.
