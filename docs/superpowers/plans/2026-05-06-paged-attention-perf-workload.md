# Paged Attention Performance Workload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full paged attention workload to the static-template performance benchmark.

**Architecture:** Keep one benchmark runner and add a small workload abstraction. The runner compiles and executes either the existing BGEMM smoke graph or the full paged attention graph under both optimization strategies, then writes the same summary format with workload metadata.

**Tech Stack:** Python 3.10, PyPTO DSL, existing `examples.models.paged_attention` helpers, existing static-template metric parser.

---

### Task 1: Workload Selection

**Files:**
- Modify: `tests/st/runtime/manual_scope_template_perf.py`

- [ ] Add CLI options: `--workload`, `--pa-batch`, `--pa-num-heads`, `--pa-head-dim`, `--pa-block-size`, `--pa-context-len`, `--pa-max-model-len`, `--pa-scale`.
- [ ] Add a `BenchmarkWorkload` dataclass with `name`, `program`, `make_inputs`, and `metadata`.
- [ ] Implement `_build_workload(args)` for `bgemm_manual_scope` and `paged_attention`.
- [ ] Pass the selected workload into `_run_variant`.

### Task 2: Summary Metadata

**Files:**
- Modify: `tests/st/runtime/manual_scope_template_perf.py`

- [ ] Add `workload` and `workload_config` to `summary.json`.
- [ ] Add workload fields and parsed rounds to `summary.md`.

### Task 3: Documentation

**Files:**
- Modify: `docs/en/dev/codegen/04-static_template_performance.md`
- Modify: `docs/zh-cn/dev/codegen/04-static_template_performance.md`

- [ ] Update the recommended command to run `--workload paged_attention`.
- [ ] Document that BGEMM remains available as a smoke workload.
- [ ] Mention the paged attention shape knobs and the parsed-rounds sanity check.

### Task 4: Verification and Commit

**Files:**
- Check staged diff only.

- [ ] Run `git diff --check` and `git diff --cached --check`.
- [ ] Skip local pytest because this benchmark requires the target hardware/runtime environment.
- [ ] Commit and push the branch.
