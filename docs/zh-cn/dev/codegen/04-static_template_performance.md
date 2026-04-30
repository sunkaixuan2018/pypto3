# 静态模板性能打点

使用这个 benchmark 对比同一个 program、同一套 runtime 下的静态模板
manual-scope 路径和传统 AUTO-scope 路径。

| 方案 | Strategy | 预期 codegen |
| ---- | -------- | ------------ |
| `static_hit` | `Default` | `PTO2ScopeMode::MANUAL` 和显式 `add_dep(...)` |
| `traditional_baseline` | `DefaultWithoutStableRegionTemplates` | AUTO scope，由 runtime TensorMap 动态发现依赖 |

## 执行

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

## 指标

| 指标 | 含义 |
| ---- | ---- |
| `Compile frontend sched (us)` | `IdentifyStableRegions + LowerStableRegionsToManualScope + orchestration_codegen` |
| `Orch trimmed avg (us)` | device orchestrator 耗时，固定丢弃低值/高值离群点后取平均 |
| `Sched trimmed avg (us)` | device scheduler loop 耗时，作为辅助观察项 |
| `Device orch saved per round (us)` | baseline 的 `Orch trimmed avg` 减去 static-hit 的 `Orch trimmed avg` |
| `Net gain after measured rounds (us)` | `rounds * saved_per_round - compile_delta` |
| `Break-even runs` | 当每轮节省为正时，`ceil(compile_delta / saved_per_round)` |

核心预期收益体现在 `static_hit` 的 `Orch trimmed avg (us)` 更低，因为 manual
scope 跳过 TensorMap lookup/insert，用 codegen 生成的显式依赖直接下发。
