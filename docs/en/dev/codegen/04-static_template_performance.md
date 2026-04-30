# Static Template Performance Profiling

Use this benchmark to compare the static-template manual-scope path with the
traditional AUTO-scope path on the same program and runtime.

| Variant | Strategy | Expected codegen |
| ------- | -------- | ---------------- |
| `static_hit` | `Default` | `PTO2ScopeMode::MANUAL` plus explicit `add_dep(...)` |
| `traditional_baseline` | `DefaultWithoutStableRegionTemplates` | AUTO scope, runtime TensorMap dependency discovery |

## Run

```bash
export PYTHONPATH=$(pwd)/python:$(pwd)/tests/st:${PYTHONPATH}
export PTOAS_ROOT=/path/to/ptoas
export PTO2_PROFILING=1
export PTO2_ORCH_PROFILING=1
export PTO2_TENSORMAP_PROFILING=1

python3 tests/st/runtime/manual_scope_template_perf.py \
  --platform a2a3 \
  --device 0 \
  --rounds 100 \
  --profiling-detail \
  --output build_output/static_template_perf

cat build_output/static_template_perf/*/summary.md
```

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `Compile frontend sched (us)` | `IdentifyStableRegions + LowerStableRegionsToManualScope + orchestration_codegen` |
| `Orch trimmed avg (us)` | Device orchestrator time after dropping fixed low/high outliers |
| `Sched trimmed avg (us)` | Device scheduler loop time, useful as a secondary signal |
| `Device orch saved per round (us)` | Baseline `Orch trimmed avg` minus static-hit `Orch trimmed avg` |
| `Net gain after measured rounds (us)` | `rounds * saved_per_round - compile_delta` |
| `Break-even runs` | `ceil(compile_delta / saved_per_round)` when saving is positive |

The main expected benefit is lower `Orch trimmed avg (us)` for `static_hit`,
because manual scope skips TensorMap lookup/insert work and uses generated
explicit dependencies instead.
