# Issue 1444 Windowed Dependency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent `OutWindowExternalizer` from generating windowed output writes that can race with later full-parent tensor consumers.

**Architecture:** Keep orchestration codegen as a strict IR-to-C++ lowering layer and enforce this safety rule in `OptimizeOrchTensors` Pattern 5. The pass will skip call-site window externalization when the call writes a window of a tensor root that a later task in the enclosing scope reads as the full parent/root tensor; this removes the unsafe `parent.view(...)` producer plus `parent` consumer shape reproduced in issue #1444.

**Tech Stack:** C++17 IR transform pass, Python pytest codegen regression, PyPTO developer documentation.

---

## File Structure

- Modify `src/ir/transforms/optimize_orch_tensors_pass.cpp`
  - Add a small root-use analyzer local to `OutWindowExternalizer::OrchRewriter`.
  - Track later full-parent reads while scanning `SeqStmts` from right to left.
  - Thread enclosing later-read roots into loop bodies so a writer inside a loop sees full-parent consumers after the loop.
  - Skip only the unsafe Pattern 5 call-site rewrite; leave existing safe windowed rewrites intact.
- Modify `tests/ut/codegen/test_orchestration_codegen.py`
  - Add a regression in `TestTensorReadWriteOffsetCodegen` that builds the minimal issue #1444 shape and asserts no `produce__windowed` view write is generated before the full-parent consumer.
  - Keep the existing #1420 tuple/windowed codegen test unchanged so it guards against redeclaration regressions.
- Modify `docs/en/dev/passes/13-optimize_orch_tensors.md`
  - Document the new Pattern 5 safety rule.
- Modify `docs/zh-cn/dev/passes/13-optimize_orch_tensors.md`
  - Mirror the English safety rule.
- Optional verification only, no planned code changes:
  - Run focused local tests if the local Python extension is usable.
  - If local tests cannot run, prepare results from available failure output and rely on remote validation in a follow-up task.

## Design Decision

Do not add automatic dependency bridging in orchestration codegen for this fix.

Before:

```cpp
Tensor score_flat__iter_v1__window = score_flat.view(...);
params_t0.add_output(score_flat__iter_v1__window);
rt_submit_aiv_task(0, params_t0);

params_t1.add_input(score_flat);
rt_submit_aiv_task(1, params_t1);
```

After:

```cpp
params_t0.add_inout(score_flat);
rt_submit_aiv_task(0, params_t0);

params_t1.add_input(score_flat);
rt_submit_aiv_task(1, params_t1);
```

This after shape comes from leaving the original call un-windowed. `DeriveCallDirections` already promotes sequential-loop `Out` writes to `InOut`, so the runtime sees the same `Tensor` object for producer and consumer and can infer the dependency. A more generic codegen bridge would need ordinary-call task-id capture, parallel array dependency aggregation, and root alias tracking; that is broader than the issue and contradicts the orchestration codegen document's strict 1-to-1 lowering rule.

### Task 1: Add the Regression Test

**Files:**

- Modify: `tests/ut/codegen/test_orchestration_codegen.py`

- [x] **Step 1: Add a focused #1444 codegen regression**

Insert this method in `class TestTensorReadWriteOffsetCodegen`, after `test_windowed_tuple_outputs_rebind_loop_carried_tensor_without_redeclaration`:

```python
    def test_windowed_writer_before_full_parent_reader_stays_unwindowed(self):
        """Issue #1444: window writes followed by full-parent reads must not be externalized.

        The unsafe codegen shape is:
            producer writes score_flat.view(...) with add_output/add_inout
            later consumer reads score_flat with add_input
            no explicit set_dependencies edge bridges view -> parent

        Until runtime/codegen has a generic root-aware dependency bridge,
        OutWindowExternalizer must keep this producer unwindowed so auto deps
        operate on the same parent Tensor object.
        """

        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, M, W = 64, 2048, 8

        @pl.program
        class WindowedWriteFullParentReadProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def produce(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                score: pl.Out[pl.Tensor[[N, M], pl.FP32]],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                tile: pl.Tile[[N, W], pl.FP32] = pl.tile.load(x, [0, col], [N, W], [N, W])
                ret: pl.Tensor[[N, M], pl.FP32] = pl.tile.store(tile, [0, col], score)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                score: pl.Tensor[[N, M], pl.FP32],
                probe: pl.Out[pl.Tensor[[N, M], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                tile: pl.Tile[[1, M], pl.FP32] = pl.tile.load(score, [row, 0], [1, M], [1, M])
                ret: pl.Tensor[[N, M], pl.FP32] = pl.tile.store(tile, [row, 0], probe)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[N, M], pl.FP32],
                score: pl.Out[pl.Tensor[[N, M], pl.FP32]],
                probe: pl.Out[pl.Tensor[[N, M], pl.FP32]],
            ) -> pl.Tensor[[N, M], pl.FP32]:
                score_flat: pl.Tensor[[N, M], pl.FP32] = pl.reshape(score, [N, M])
                for c0, (score_iter,) in pl.range(0, M, W, init_values=(score_flat,)):
                    score_next: pl.Tensor[[N, M], pl.FP32] = self.produce(x, score_iter, c0)
                    score_rv = pl.yield_(score_next)
                for r, (probe_iter,) in pl.range(N, init_values=(probe,)):
                    probe_next: pl.Tensor[[N, M], pl.FP32] = self.consume(score_rv, probe_iter, r)
                    probe_rv = pl.yield_(probe_next)
                return probe_rv

        transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            WindowedWriteFullParentReadProgram
        )
        code = _generate_orch_code(transformed)

        assert "produce__windowed" not in code, code
        assert "params_t0.add_inout(score_flat)" in code, code
        assert "params_t1.add_input(score_flat)" in code, code
        assert "score_flat.view(" not in code, code
```

- [x] **Step 2: Run the new test and verify it fails before implementation**

Run:

```bash
python -m pytest tests/ut/codegen/test_orchestration_codegen.py::TestTensorReadWriteOffsetCodegen::test_windowed_writer_before_full_parent_reader_stays_unwindowed -q
```

Expected before implementation: FAIL because generated code still contains `produce__windowed` and `score_flat.view(`.

### Task 2: Implement the Pattern 5 Safety Rule

**Files:**

- Modify: `src/ir/transforms/optimize_orch_tensors_pass.cpp`

- [x] **Step 1: Add root resolution and read-footprint helpers inside `OrchRewriter`**

Add these helpers near `LoopDisjointnessCandidate` and before `TryRewriteCall`:

```cpp
    using RootSet = std::unordered_set<const Var*>;

    const Var* ResolveBufferRoot(const ExprPtr& expr) const {
      auto current = ResolveLoopInitExpr(expr);
      auto var = AsVarLike(current);
      if (!var) return nullptr;
      return ResolveBufferRoot(var.get());
    }

    const Var* ResolveBufferRoot(const Var* var) const {
      const Var* current = var;
      std::unordered_set<const Var*> seen;
      while (current && seen.insert(current).second) {
        auto it = buffer_roots_.find(current);
        if (it == buffer_roots_.end()) break;
        current = it->second;
      }
      return current;
    }

    bool IsFullRootExpr(const ExprPtr& expr) const {
      auto current = ResolveLoopInitExpr(expr);
      auto var = AsVarLike(current);
      return var && ResolveBufferRoot(var.get()) == var.get();
    }

    bool IsReadDirection(ParamDirection direction) const {
      return direction == ParamDirection::In || direction == ParamDirection::InOut;
    }

    void AddFullRootReadsFromCall(const CallPtr& call, RootSet& reads) const {
      if (!call || !program_ || codegen::IsBuiltinOp(call->op_->name_)) return;
      auto callee = program_->GetFunction(call->op_->name_);
      if (!callee) return;
      for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
        if (!IsReadDirection(callee->param_directions_[i])) continue;
        if (!IsFullRootExpr(call->args_[i])) continue;
        if (const Var* root = ResolveBufferRoot(call->args_[i])) {
          reads.insert(root);
        }
      }
    }

    void AddFullRootReadsFromStmt(const StmtPtr& stmt, RootSet& reads) const {
      if (!stmt) return;
      if (auto assign = As<AssignStmt>(stmt)) {
        AddFullRootReadsFromCall(As<Call>(assign->value_), reads);
      } else if (auto eval = As<EvalStmt>(stmt)) {
        AddFullRootReadsFromCall(As<Call>(eval->expr_), reads);
      } else if (auto seq = As<SeqStmts>(stmt)) {
        for (auto it = seq->stmts_.rbegin(); it != seq->stmts_.rend(); ++it) {
          AddFullRootReadsFromStmt(*it, reads);
        }
      } else if (auto for_stmt = As<ForStmt>(stmt)) {
        AddFullRootReadsFromStmt(for_stmt->body_, reads);
      } else if (auto while_stmt = As<WhileStmt>(stmt)) {
        AddFullRootReadsFromStmt(while_stmt->body_, reads);
      } else if (auto if_stmt = As<IfStmt>(stmt)) {
        AddFullRootReadsFromStmt(if_stmt->then_body_, reads);
        if (if_stmt->else_body_.has_value()) {
          AddFullRootReadsFromStmt(if_stmt->else_body_.value(), reads);
        }
      } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
        AddFullRootReadsFromStmt(scope->body_, reads);
      }
    }
```

- [x] **Step 2: Thread later full-parent reads through `SeqStmts` and loops**

Change `VisitStmt_(const SeqStmtsPtr& op)` to scan from right to left. The key shape should be:

```cpp
      RootSet later_reads = enclosing_later_full_parent_reads_;

      for (auto it = op->stmts_.rbegin(); it != op->stmts_.rend(); ++it) {
        const auto& stmt = *it;
        auto saved_enclosing_reads = enclosing_later_full_parent_reads_;
        enclosing_later_full_parent_reads_ = later_reads;

        auto call_assign = As<AssignStmt>(stmt);
        auto bundle = call_assign ? TryRewriteCall(call_assign) : std::nullopt;
        ...
        enclosing_later_full_parent_reads_ = std::move(saved_enclosing_reads);
        AddFullRootReadsFromStmt(visited_or_original_stmt, later_reads);
      }
```

Because this reverses the scan direction, build `new_stmts_reversed` and `std::reverse` before returning the flattened sequence.

When visiting a `ForStmt` or `WhileStmt`, leave `enclosing_later_full_parent_reads_` populated while visiting the body. This is what makes a producer inside the loop see a full-parent reader after the loop.

- [x] **Step 3: Add the unsafe-call-site guard**

Add this helper:

```cpp
    bool HasLaterFullParentReadOfRewrittenOutput(const CallPtr& call,
                                                 const CalleeRewriteAnalysis& analysis) const {
      for (const auto& output : analysis.outputs) {
        if (output.out_param_index >= call->args_.size()) return true;
        const Var* root = ResolveBufferRoot(call->args_[output.out_param_index]);
        if (root && enclosing_later_full_parent_reads_.count(root) > 0) {
          return true;
        }
      }
      return false;
    }
```

Then in `TryRewriteCall`, after the disjointness check and before creating slices, add:

```cpp
      if (HasLaterFullParentReadOfRewrittenOutput(call, analysis)) return std::nullopt;
```

- [x] **Step 4: Add storage for the helper state**

Add these members near existing `scalar_defs_` and `tuple_result_subst_`:

```cpp
    std::unordered_map<const Var*, const Var*> buffer_roots_;
    RootSet enclosing_later_full_parent_reads_;
```

Initialize `buffer_roots_` in the `OrchRewriter` constructor using `codegen::BufferRootCollector`:

```cpp
      codegen::BufferRootCollector root_collector(program_);
      // initialize per function in Run/constructor is not available here, so pass
      // roots from the outer Run when constructing the rewriter.
```

If constructor access to the current function params/body is simpler, change the constructor call in `Run` to:

```cpp
      OrchRewriter rewriter(program, analyses, cloned_funcs, func);
```

and implement:

```cpp
    OrchRewriter(ProgramPtr program, const AnalysisMap& analyses,
                 const std::unordered_map<std::string, FunctionPtr>& cloned_funcs,
                 const FunctionPtr& current_func)
        : program_(std::move(program)), analyses_(analyses), cloned_funcs_(cloned_funcs) {
      codegen::BufferRootCollector root_collector(program_);
      if (current_func) {
        root_collector.Initialize(current_func->params_);
        root_collector.VisitStmt(current_func->body_);
      }
      buffer_roots_ = std::move(root_collector.buffer_roots);
    }
```

### Task 3: Update Documentation

**Files:**

- Modify: `docs/en/dev/passes/13-optimize_orch_tensors.md`
- Modify: `docs/zh-cn/dev/passes/13-optimize_orch_tensors.md`

- [x] **Step 1: Update English Pattern 5 safety rules**

Add this bullet under `Safety rules`:

```markdown
- call-site externalization is skipped when a rewritten window output's buffer root is later read as a full-parent tensor in the enclosing orchestration scope; until the runtime has root-aware view/parent dependency tracking, this preserves auto dependency tracking on the same parent `Tensor`
```

- [x] **Step 2: Update Chinese Pattern 5 safety rules**

Add the equivalent translated bullet to the zh-CN pass document. It must mirror
the English rule: skip call-site externalization when the rewritten window
output's buffer root is later read as a full-parent tensor in the enclosing
orchestration scope, preserving auto dependency tracking on the same parent
`Tensor` until runtime has root-aware view/parent dependency tracking.

### Task 4: Verify

**Files:**

- No additional edits.

- [x] **Step 1: Run focused codegen regression**

Run:

```bash
python -m pytest tests/ut/codegen/test_orchestration_codegen.py::TestTensorReadWriteOffsetCodegen::test_windowed_writer_before_full_parent_reader_stays_unwindowed -q
```

Expected after implementation: PASS.

- [x] **Step 2: Run focused existing codegen tests for #1420**

Run:

```bash
python -m pytest tests/ut/codegen/test_orchestration_codegen.py::TestTensorReadWriteOffsetCodegen -q
```

Expected after implementation: PASS, including `test_windowed_tuple_outputs_rebind_loop_carried_tensor_without_redeclaration`.

- [x] **Step 3: Run Pattern 5 transform tests**

Run:

```bash
python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py::TestOutWindowExternalizer -q
```

Expected after implementation: PASS.

- [x] **Step 4: Record any local environment blocker**

If local tests cannot import/build `pypto_core`, record the exact failing command and error in the final response and recommend the existing remote validation loop for final runtime confirmation.

## Self-Review

- Spec coverage: The plan covers issue #1444's deterministic defect by preventing `view` output writes before later full-parent readers, keeps #1420 guarded by existing tests, and updates English/Chinese pass docs.
- Placeholder scan: No placeholders remain; every code/doc change has exact file paths and snippets.
- Type consistency: The C++ helper names use local `RootSet`, `Var*`, `ExprPtr`, `CallPtr`, and existing `ParamDirection`/`BufferRootCollector` types already included in the file.
