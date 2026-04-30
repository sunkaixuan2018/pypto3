# Static Template Performance Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立静态模板命中方案的可重复性能打点，对比 `static_hit` 与传统 `AUTO` baseline 的编译期调度开销和 device 调度开销。

**Architecture:** 在主仓库新增一个只移除稳定模板识别/降低 pass 的 baseline strategy，保证同一个 program、同一组 kernel、同一套 runtime，只改变 manual-scope 命中路径。新增 PyPTO benchmark CLI，读取 `CompileProfiler` JSON，并按 `runtime/tools/benchmark_rounds.sh` 的 round 解析模型读取 device log，输出 `summary.json` 与 `summary.md`。

**Tech Stack:** Python 3.10+, pytest, PyPTO `CompileProfiler`, CANN device logs, Simpler runtime submodule commit `9887095`.

---

## Files

- Modify `python/pypto/ir/pass_manager.py`: add `DefaultWithoutStableRegionTemplates`.
- Create `python/pypto/perf/__init__.py`: export performance helpers.
- Create `python/pypto/perf/static_template.py`: compile/device parsing and comparison math.
- Modify `tests/st/runtime/test_manual_scope_template.py`: reusable BGEMM inputs and baseline AUTO assertion.
- Create `tests/st/runtime/manual_scope_template_perf.py`: benchmark CLI for two variants.
- Modify `tests/ut/ir/transforms/test_pass_manager.py`: strategy pass-list tests.
- Create `tests/ut/runtime/test_static_template_perf.py`: helper unit tests.
- Modify pass-manager docs and create codegen performance docs under both `docs/en/dev/` and `docs/zh-cn/dev/`.
- Stage `runtime` gitlink only if `git -C runtime rev-parse --short HEAD` is `9887095` and the superproject still records the old commit.

Do not edit `runtime/tools/benchmark_rounds.sh`; it belongs to the submodule. The PyPTO benchmark mirrors its parsing model so the PyPTO branch can be committed independently.

## Task 1: Baseline Strategy

**Files:** `python/pypto/ir/pass_manager.py`, `tests/ut/ir/transforms/test_pass_manager.py`, pass-manager docs.

- [ ] **Step 1: Add failing tests**

In `tests/ut/ir/transforms/test_pass_manager.py`, add:

```python
STABLE_TEMPLATE_PASSES = [
    "IdentifyStableRegions",
    "LowerStableRegionsToManualScope",
]

TENSOR_OPTIMIZATION_WITHOUT_STABLE_TEMPLATE_PASSES = [
    name for name in TENSOR_OPTIMIZATION_PASSES if name not in STABLE_TEMPLATE_PASSES
]
```

Update enum tests:

```python
def test_optimization_strategy_values(self):
    assert ir.OptimizationStrategy.Default is not None
    assert ir.OptimizationStrategy.DefaultWithoutStableRegionTemplates is not None
    assert ir.OptimizationStrategy.DebugTileOptimization is not None

def test_pass_manager_get_strategy_default_without_stable_templates(self):
    pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.DefaultWithoutStableRegionTemplates)
    assert pm.strategy == ir.OptimizationStrategy.DefaultWithoutStableRegionTemplates
    assert pm.pass_names == TENSOR_OPTIMIZATION_WITHOUT_STABLE_TEMPLATE_PASSES
    assert "DeriveCallDirections" in pm.pass_names
    assert "IdentifyStableRegions" not in pm.pass_names
    assert "LowerStableRegionsToManualScope" not in pm.pass_names
```

- [ ] **Step 2: Verify the test fails**

```bash
PYPTO_VERIFY_LEVEL=basic python -m pytest tests/ut/ir/transforms/test_pass_manager.py -v
```

Expected before implementation: `AttributeError: DefaultWithoutStableRegionTemplates`.

- [ ] **Step 3: Implement the strategy**

In `python/pypto/ir/pass_manager.py`:

```python
class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""

    Default = "Default"
    DefaultWithoutStableRegionTemplates = "DefaultWithoutStableRegionTemplates"
    DebugTileOptimization = "DebugTileOptimization"
```

Keep the existing `tile_pto_passes` list, then derive the baseline list directly from it:

```python
stable_template_pass_names = {
    "IdentifyStableRegions",
    "LowerStableRegionsToManualScope",
}
tile_pto_passes_without_stable_templates = [
    spec for spec in tile_pto_passes if spec[0] not in stable_template_pass_names
]
cls._strategy_passes = {
    OptimizationStrategy.Default: tensor_prefix_passes + tensor_only_passes + tile_pto_passes,
    OptimizationStrategy.DefaultWithoutStableRegionTemplates: (
        tensor_prefix_passes + tensor_only_passes + tile_pto_passes_without_stable_templates
    ),
    OptimizationStrategy.DebugTileOptimization: tensor_prefix_passes + tile_pto_passes,
}
```

- [ ] **Step 4: Document the strategy**

Add this meaning to both pass-manager docs:

```markdown
- `Default`: full tensor-oriented PTO pipeline, including stable-template detection and manual-scope lowering.
- `DefaultWithoutStableRegionTemplates`: same as `Default` except it omits `IdentifyStableRegions` and `LowerStableRegionsToManualScope`; use it as the traditional AUTO-scope baseline for static-template performance measurements.
- `DebugTileOptimization`: debug-only PTO tile pipeline, including stable-template detection.
```

- [ ] **Step 5: Verify and commit**

```bash
PYPTO_VERIFY_LEVEL=basic python -m pytest tests/ut/ir/transforms/test_pass_manager.py -v
git add python/pypto/ir/pass_manager.py tests/ut/ir/transforms/test_pass_manager.py docs/en/dev/passes/00-pass_manager.md docs/zh-cn/dev/passes/00-pass_manager.md
git commit -m "perf: add stable-template baseline strategy"
```

## Task 2: Metrics Helper Library

**Files:** `python/pypto/perf/__init__.py`, `python/pypto/perf/static_template.py`, `tests/ut/runtime/test_static_template_perf.py`.

- [ ] **Step 1: Add failing helper tests**

Create `tests/ut/runtime/test_static_template_perf.py`:

```python
from pypto.perf.static_template import (
    compare_static_template_variants,
    compile_metrics_from_profile,
    parse_device_timing_log,
    trimmed_mean,
)

def test_trimmed_mean_drops_fixed_tails_when_enough_rounds():
    values = [1000.0] + [10.0] * 30 + [2000.0]
    assert trimmed_mean(values, trim=10) == 10.0

def test_compile_metrics_extracts_static_template_stages():
    profile = {"stages": [
        {"name": "passes", "seconds": 0.02, "children": [
            {"name": "IdentifyStableRegions", "seconds": 0.003, "children": []},
            {"name": "LowerStableRegionsToManualScope", "seconds": 0.002, "children": []}]},
        {"name": "codegen", "seconds": 0.03, "children": [
            {"name": "orchestration_codegen", "seconds": 0.004, "children": []}]}]}
    metrics = compile_metrics_from_profile(profile)
    assert metrics["identify_stable_regions_us"] == 3000.0
    assert metrics["lower_stable_regions_us"] == 2000.0
    assert metrics["orchestration_codegen_us"] == 4000.0
    assert metrics["compile_frontend_sched_us"] == 9000.0

def test_parse_device_timing_log_uses_repeated_start_as_round_boundary():
    log = "\n".join([
        "Thread 2: orch_start=1000 orch_end=2000 orch_cost=20.000us",
        "Thread 0: sched_start=900 sched_end=2100 sched_cost=24.000us",
        "Thread 2: orch_start=3000 orch_end=3900 orch_cost=18.000us",
        "Thread 0: sched_start=2900 sched_end=4000 sched_cost=22.000us",
    ])
    metrics = parse_device_timing_log(log, freq_mhz=50)
    assert metrics.rounds == 2
    assert metrics.elapsed_avg_us == 24.0
    assert metrics.orch_avg_us == 19.0
    assert metrics.sched_avg_us == 23.0

def test_compare_static_template_variants_computes_break_even():
    result = compare_static_template_variants(
        static_hit={"compile": {"compile_frontend_sched_us": 30.0}, "runtime": {"orch_trimmed_avg_us": 80.0}},
        baseline={"compile": {"compile_frontend_sched_us": 10.0}, "runtime": {"orch_trimmed_avg_us": 100.0}},
        rounds=100,
    )
    assert result["compile_delta_us"] == 20.0
    assert result["device_orch_saved_us_per_round"] == 20.0
    assert result["break_even_runs"] == 1
    assert result["net_gain_after_rounds_us"] == 1980.0
```

Run:

```bash
python -m pytest tests/ut/runtime/test_static_template_perf.py -v
```

Expected before implementation: `ModuleNotFoundError: No module named 'pypto.perf'`.

- [ ] **Step 2: Implement helper API**

Create `python/pypto/perf/static_template.py` with:

```python
@dataclass(frozen=True)
class RuntimeTimingMetrics:
    rounds: int
    elapsed_avg_us: float
    elapsed_trimmed_avg_us: float
    sched_avg_us: float | None
    sched_trimmed_avg_us: float | None
    orch_avg_us: float | None
    orch_trimmed_avg_us: float | None

def trimmed_mean(values: list[float], *, trim: int = 10) -> float:
    ordered = sorted(values)
    if len(ordered) > 2 * trim:
        ordered = ordered[trim:-trim]
    return sum(ordered) / len(ordered)
```

Use these formulas:

```python
compile_frontend_sched_us = identify_stable_regions_us + lower_stable_regions_us + orchestration_codegen_us
elapsed_us = (max_end - min_start) / freq_mhz
sched_us = (max_sched_end - min_sched_start) / freq_mhz
orch_us = (max_orch_end - min_orch_start) / freq_mhz
compile_delta_us = static_hit_compile_frontend_sched_us - baseline_compile_frontend_sched_us
device_orch_saved_us_per_round = baseline_orch_trimmed_avg_us - static_hit_orch_trimmed_avg_us
```

`parse_device_timing_log()` must recognize the same timing lines as `runtime/tools/benchmark_rounds.sh`: `sched_start`, `sched_end`, `orch_start`, `orch_end`, and `orch_stage_end`. Start a new round when the same thread reports another `sched_start` or `orch_start`.

- [ ] **Step 3: Export and verify**

Create `python/pypto/perf/__init__.py`:

```python
"""Performance analysis helpers for PyPTO benchmarks."""

from .static_template import (
    RuntimeTimingMetrics,
    compare_static_template_variants,
    compile_metrics_from_profile,
    parse_device_timing_log,
    trimmed_mean,
)

__all__ = [
    "RuntimeTimingMetrics",
    "compare_static_template_variants",
    "compile_metrics_from_profile",
    "parse_device_timing_log",
    "trimmed_mean",
]
```

Run and commit:

```bash
python -m pytest tests/ut/runtime/test_static_template_perf.py -v
git add python/pypto/perf tests/ut/runtime/test_static_template_perf.py
git commit -m "perf: add static template metrics helpers"
```

## Task 3: Benchmark CLI and Runtime Assertions

**Files:** `tests/st/runtime/test_manual_scope_template.py`, `tests/st/runtime/manual_scope_template_perf.py`.

- [ ] **Step 1: Share BGEMM input construction**

Add to `tests/st/runtime/test_manual_scope_template.py`:

```python
def make_bgemm_manual_scope_inputs() -> tuple[torch.Tensor, ...]:
    lhs = torch.eye(_SIZE, dtype=torch.float32)
    rhs0 = torch.full((_SIZE, _SIZE), 1.25, dtype=torch.float32)
    bias0 = torch.full((_SIZE, _SIZE), 2.0, dtype=torch.float32)
    rhs1 = torch.eye(_SIZE, dtype=torch.float32)
    bias1 = torch.full((_SIZE, _SIZE), 3.0, dtype=torch.float32)
    dst = torch.zeros((_SIZE, _SIZE), dtype=torch.float32)
    return lhs, rhs0, bias0, rhs1, bias1, dst
```

Use it in the existing e2e test:

```python
lhs, rhs0, bias0, rhs1, bias1, dst = make_bgemm_manual_scope_inputs()
```

- [ ] **Step 2: Assert the baseline emits AUTO behavior**

Add:

```python
def test_bgemm_template_baseline_strategy_emits_auto_scope(output_root, test_config):
    _require_ptoas()
    compiled = ir.compile(
        BgemmManualScopeProgram,
        output_dir=str(output_root / "bgemm_auto_scope_baseline"),
        strategy=ir.OptimizationStrategy.DefaultWithoutStableRegionTemplates,
        backend_type=platform_to_backend(test_config.platform),
        platform=test_config.platform,
    )
    orchestration_cpp = _read_orchestration_cpp(compiled.output_dir)
    assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" not in orchestration_cpp
    assert ".add_dep(task_result_" not in orchestration_cpp
```

- [ ] **Step 3: Create benchmark CLI**

Create `tests/st/runtime/manual_scope_template_perf.py`. It must accept:

```bash
python3 tests/st/runtime/manual_scope_template_perf.py --platform a2a3 --device 0 --rounds 100 --profiling-detail --output build_output/static_template_perf
```

Variants:

```python
variants = {
    "static_hit": ir.OptimizationStrategy.Default,
    "traditional_baseline": ir.OptimizationStrategy.DefaultWithoutStableRegionTemplates,
}
```

For each variant:

```python
compiled = ir.compile(
    BgemmManualScopeProgram,
    output_dir=str(variant_dir),
    strategy=strategy,
    backend_type=platform_to_backend(args.platform),
    platform=args.platform,
    profiling=True,
)
for _ in range(args.rounds):
    tensors = make_bgemm_manual_scope_inputs()
    compiled(*tensors, config=RunConfig(platform=args.platform, device_id=args.device))
```

When `--profiling-detail` is set, apply:

```python
PTO2_PROFILING=1
PTO2_ORCH_PROFILING=1
PTO2_TENSORMAP_PROFILING=1
```

The CLI must write:

```text
build_output/static_template_perf/<timestamp>/summary.json
build_output/static_template_perf/<timestamp>/summary.md
```

Markdown must include these columns:

```markdown
| Variant | Manual scope | Compile frontend sched (us) | Orch trimmed avg (us) | Sched trimmed avg (us) | Elapsed trimmed avg (us) |
```

- [ ] **Step 4: Smoke-test and commit**

```bash
python3 tests/st/runtime/manual_scope_template_perf.py --platform a2a3 --device 0 --rounds 1 --output build_output/static_template_perf_smoke
python3 -m pytest tests/st/runtime/test_manual_scope_template.py -v --platform=a2a3 --device=0
git add tests/st/runtime/test_manual_scope_template.py tests/st/runtime/manual_scope_template_perf.py
git commit -m "perf: add manual scope template benchmark"
```

## Task 4: Docs and User Commands

**Files:** `docs/en/dev/codegen/04-static_template_performance.md`, `docs/zh-cn/dev/codegen/04-static_template_performance.md`.

- [ ] **Step 1: Add benchmark guide**

Both docs must include this command unchanged:

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

Metric definitions:

```text
Compile frontend sched (us) = IdentifyStableRegions + LowerStableRegionsToManualScope + orchestration_codegen
Device orch saved per round (us) = baseline Orch trimmed avg - static_hit Orch trimmed avg
Net gain after measured rounds (us) = rounds * saved_per_round - compile_delta
Break-even runs = ceil(compile_delta / saved_per_round)
```

- [ ] **Step 2: Commit docs**

```bash
git add docs/en/dev/codegen/04-static_template_performance.md docs/zh-cn/dev/codegen/04-static_template_performance.md
git commit -m "docs: explain static template performance profiling"
```

## Task 5: Full Verification, Runtime Pin, Commit, Push

**Files:** all changed files; possibly `runtime` gitlink.

- [ ] **Step 1: Build and unit-test**

```bash
[ ! -f build/CMakeCache.txt ] && cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel
export PYTHONPATH=$(pwd)/python:${PYTHONPATH}
PYPTO_VERIFY_LEVEL=basic python -m pytest \
  tests/ut/ir/transforms/test_pass_manager.py \
  tests/ut/runtime/test_static_template_perf.py \
  -v
```

- [ ] **Step 2: Run runtime e2e**

```bash
export PYTHONPATH=$(pwd)/python:$(pwd)/tests/st:${PYTHONPATH}
python3 -m pytest tests/st/runtime/test_manual_scope_template.py -v --platform=a2a3 --device=0
```

- [ ] **Step 3: Run the comparison**

```bash
export PYTHONPATH=$(pwd)/python:$(pwd)/tests/st:${PYTHONPATH}
export PTO2_PROFILING=1
export PTO2_ORCH_PROFILING=1
export PTO2_TENSORMAP_PROFILING=1
python3 tests/st/runtime/manual_scope_template_perf.py --platform a2a3 --device 0 --rounds 100 --profiling-detail --output build_output/static_template_perf
cat build_output/static_template_perf/*/summary.md
```

Expected qualitative result:

```text
static_hit Manual scope = yes
traditional_baseline Manual scope = no
traditional_baseline Orch trimmed avg > static_hit Orch trimmed avg
Break-even runs is a positive integer when Device orch saved per round is positive
```

- [ ] **Step 4: Pin runtime gitlink when needed**

```bash
git -C runtime rev-parse --short HEAD
git submodule status runtime
git add runtime
```

Stage `runtime` only when the first command prints `9887095` and `git status --short runtime` shows the superproject gitlink is modified. No `.gitmodules` change is needed.

- [ ] **Step 5: Final commit and push**

```bash
git status --short
git switch -c codex/static-template-perf-profiling
git add \
  python/pypto/ir/pass_manager.py \
  python/pypto/perf \
  tests/ut/ir/transforms/test_pass_manager.py \
  tests/ut/runtime/test_static_template_perf.py \
  tests/st/runtime/test_manual_scope_template.py \
  tests/st/runtime/manual_scope_template_perf.py \
  docs/en/dev/passes/00-pass_manager.md \
  docs/zh-cn/dev/passes/00-pass_manager.md \
  docs/en/dev/codegen/04-static_template_performance.md \
  docs/zh-cn/dev/codegen/04-static_template_performance.md \
  runtime
git commit -m "perf: add static template benchmark comparison"
git push -u origin codex/static-template-perf-profiling
```

If the branch exists, use:

```bash
git switch codex/static-template-perf-profiling
```

## How to See the Two-Scheme Effect

Run:

```bash
cat build_output/static_template_perf/*/summary.md
```

Read these rows:

- `static_hit`: default strategy; should emit `PTO2ScopeMode::MANUAL` and explicit `add_dep`.
- `traditional_baseline`: new baseline strategy; should emit AUTO scope and rely on runtime TensorMap dependency discovery.
- `Compile delta (us)`: extra front-end/codegen overhead of static matching and manual-scope emission.
- `Device orch saved per round (us)`: runtime orchestrator saving from skipping TensorMap lookup/insert work.
- `Break-even runs`: repeated executions needed to amortize the compile delta.

## Self-Review

Coverage includes compile/pass/codegen timing, runtime device timing, trimmed averages, two-scheme comparison, report output, runtime submodule pin, verification, commit, and push. Names are consistent: strategy `DefaultWithoutStableRegionTemplates`, variants `static_hit` and `traditional_baseline`, metric `compile_frontend_sched_us`.
