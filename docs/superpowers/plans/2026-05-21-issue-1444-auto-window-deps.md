# Issue 1444 Auto Window Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve out-window externalization while emitting explicit PTO task dependencies from windowed writers to later parent/root readers.

**Architecture:** Add a compiler-internal call attr that records producer call result Vars requiring explicit dependencies. `OptimizeOrchTensors` attaches that attr when a windowed output writes a buffer root that a later task reads, and orchestration codegen captures producer task ids only for referenced producer Vars.

**Tech Stack:** C++17 IR attrs and orchestration codegen, Python pytest regression coverage.

---

## Task 1: Regression Test

**Files:**

- Modify: `tests/ut/codegen/test_orchestration_codegen.py`

- [ ] **Step 1: Replace the #1444 conservative assertion**

Change the issue #1444 test named `test_windowed_writer_before_full_parent_reader_stays_unwindowed` so it expects windowing plus an explicit dependency:

```python
    def test_windowed_writer_before_full_parent_reader_gets_explicit_dep(self):
        """Issue #1444: a window writer followed by a parent reader keeps windowing and adds an explicit edge."""
```

The core assertions should require:

```python
        assert "produce__windowed" in code, code
        assert re.search(r"params_t0\.add_output\(score_flat__iter_v\\d+__window\);", code), code
        assert "params_t1.add_input(score_flat);" in code, code
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(0, params_t0);" in code, code
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        assert "params_t1_deps[params_t1_deps_count++] = task_0_outs.task_id();" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code
```

- [ ] **Step 2: Run the targeted test and confirm it fails**

Run:

```bash
python -m pytest tests/ut/codegen/test_orchestration_codegen.py::TestOptimizeOrchTensors::test_windowed_writer_before_full_parent_reader_gets_explicit_dep -q
```

Expected: FAIL before implementation because code either keeps the writer unwindowed or does not emit `set_dependencies`.

## Task 2: Internal IR Attr

**Files:**

- Modify: `include/pypto/ir/expr.h`

- [ ] **Step 1: Add a compiler-internal attr constant and helper**

Add near `kAttrManualDepEdges`:

```cpp
/**
 * @brief Compiler-internal explicit producer edges for auto-generated deps.
 *
 * Value type: ``std::vector<VarPtr>`` where each Var is the LHS of a prior
 * kernel Call. Unlike ``manual_dep_edges``, these entries are not user-visible
 * TaskId Vars; orchestration codegen captures the referenced producer task id
 * directly and emits ``Arg::set_dependencies`` for the consumer Call.
 */
inline constexpr const char* kAttrAutoDepProducerVars = "auto_dep_producer_vars";
```

Add helper:

```cpp
inline std::vector<std::pair<std::string, std::any>> WithAutoDepProducerVarsAttr(
    std::vector<std::pair<std::string, std::any>> attrs, std::vector<VarPtr> vars) {
  for (auto& [k, v] : attrs) {
    if (k == kAttrAutoDepProducerVars) {
      v = std::move(vars);
      return attrs;
    }
  }
  attrs.emplace_back(kAttrAutoDepProducerVars, std::move(vars));
  return attrs;
}
```

## Task 3: OptimizeOrchTensors Auto Edges

**Files:**

- Modify: `src/ir/transforms/optimize_orch_tensors_pass.cpp`

- [ ] **Step 1: Remove the conservative skip gate**

In `TryRewriteCall`, remove:

```cpp
      if (HasLaterFullParentReadOfRewrittenOutput(call, analysis)) return std::nullopt;
```

Do not remove root-tracking helpers yet; they are reused to identify producer and consumer roots.

- [ ] **Step 2: Track live windowed producers by buffer root**

Add a member:

```cpp
    std::unordered_map<const Var*, std::vector<VarPtr>> windowed_producers_by_root_;
```

After a call is successfully windowized, record the temporary windowed call result Var for each rewritten output root:

```cpp
      RegisterWindowedProducer(tmp_result_var, call, analysis);
```

Implement:

```cpp
    void RegisterWindowedProducer(const VarPtr& producer_var, const CallPtr& call,
                                  const CalleeRewriteAnalysis& analysis) {
      if (!producer_var) return;
      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) continue;
        const Var* root = ResolveBufferRoot(call->args_[output.out_param_index]);
        if (!root) continue;
        auto& producers = windowed_producers_by_root_[root];
        if (std::find(producers.begin(), producers.end(), producer_var) == producers.end()) {
          producers.push_back(producer_var);
        }
      }
    }
```

- [ ] **Step 3: Attach deps to later reader calls**

When visiting ordinary `AssignStmt` call statements that were not rewritten, inspect tensor args with read directions. If an arg resolves to a root in `windowed_producers_by_root_`, attach those producer Vars to the call with `WithAutoDepProducerVarsAttr`.

Use the call's `ArgDirection` vector when available. Treat `Input` and `InOut` as reads. For safety, fall back to callee `ParamDirection::In/InOut` only when arg directions are absent.

## Task 4: Orchestration Codegen

**Files:**

- Modify: `src/codegen/orchestration/orchestration_codegen.cpp`

- [ ] **Step 1: Collect producer Vars that must capture outputs**

Before codegen emits an orchestration body, traverse it to collect all Vars referenced by `kAttrAutoDepProducerVars`.

Store them in:

```cpp
  std::unordered_set<const Var*> auto_dep_capture_vars_;
  std::unordered_map<const Var*, std::string> auto_dep_task_id_exprs_;
```

- [ ] **Step 2: Capture referenced ordinary producer calls**

In the non-builtin call emission path, compute:

```cpp
const bool capture_auto_dep = auto_dep_capture_vars_.count(assign->var_.get()) > 0;
```

Pass `capture_auto_dep || IsSubmitCall(call)` into `EmitTaskSubmitAndBind`.

After emitting the call, if `capture_auto_dep`, bind:

```cpp
auto_dep_task_id_exprs_[assign->var_.get()] = "task_" + std::to_string(task_idx_before) + "_outs.task_id()";
```

- [ ] **Step 3: Emit internal auto deps on consumers**

Add `EmitAutoDeps(call, task_var)` next to `EmitManualDeps(call, task_var)`. It reads `kAttrAutoDepProducerVars`, resolves each producer Var through `auto_dep_task_id_exprs_`, emits a stack `PTO2TaskId` array, and calls `set_dependencies`.

For issue #1444 the emitted shape should be:

```cpp
PTO2TaskId params_t1_deps[1];
uint32_t params_t1_deps_count = 0;
params_t1_deps[params_t1_deps_count++] = task_0_outs.task_id();
params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);
```

## Task 5: Verification

**Files:**

- Test: `tests/ut/codegen/test_orchestration_codegen.py`

- [ ] **Step 1: Run targeted regression**

Run:

```bash
python -m pytest tests/ut/codegen/test_orchestration_codegen.py::TestOptimizeOrchTensors::test_windowed_writer_before_full_parent_reader_gets_explicit_dep -q
```

Expected: PASS when local environment has the built extension available.

- [ ] **Step 2: Run focused orchestration codegen tests**

Run:

```bash
python -m pytest tests/ut/codegen/test_orchestration_codegen.py -q
```

Expected: PASS when local environment has the built extension available.

- [ ] **Step 3: If local test environment is missing**

Record the exact import/build error in the final response and recommend remote validation using the existing server workflow.
