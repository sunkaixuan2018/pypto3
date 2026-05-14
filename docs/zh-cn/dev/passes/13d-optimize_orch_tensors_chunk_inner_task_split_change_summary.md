# OptimizeOrchTensors Chunk-Inner Task-Split 修改说明

## 目的

这份文档是面向执行和调试的摘要说明，重点回答下面几个问题：

- 这次代码到底改了什么
- 新开关怎么控制行为
- 调试时应该看哪些生成物
- 哪些测试能证明行为符合预期

它和设计文档的关系是：

- 设计文档讲“为什么这样设计”
- 本文档讲“现在这版实现是什么样，怎么验证它”

## 本次修改的核心变化

这次实现是在已有的 ChunkInner out-window externalization 之上，再加了一层可选的
task split。

修改前：

- 一个 `pl.at(...)` 先 outline 成一个 InCore kernel
- top-level `ChunkInner` `parallel` / `range` 还在 kernel 内部
- runtime 只能看到一次大的 kernel submit

开启 `enable_out_window_task_split=True` 后：

- 这层 top-level `ChunkInner` 循环会被提升到 orchestration
- orchestration 每次迭代先切一个输出窗口
- 然后调用一次 `__iter_windowed` kernel
- runtime / AICore 会看到多个更小的任务，而不是一个大任务

注意：这并没有改变“一个 `pl.at(...)` 先 outline 成一个 InCore kernel”这个结论。
拆分发生在 outline 之后，由 `OptimizeOrchTensors` 完成。

## 开关说明

- `enable_out_window_rewrite`
  - 现有 Pattern 5 开关
  - 默认值：`True`
  - 作用：把输出窗口外显出来，但仍然只保留一次大 kernel submit

- `enable_out_window_task_split`
  - 新增 Pattern 6 开关
  - 默认值：`False`
  - 作用：把匹配到的 top-level ChunkInner 循环提升到 orchestration，让每个迭代都成为独立 runtime task

两个开关相互独立。默认关闭 task split，是这次实现的重要兼容性约束。

## 当前版本支持的范围

第一版只处理一个收紧后的 post-`ConvertTensorToTileOps` 形态：

- 只有一个 `Out`
- 有一层 top-level `ChunkInner` `parallel` 或 `range`
- 只有一个 carried tensor iter-arg，并且由这个 `Out` 初始化
- 只有一个 `tile.store(..., offsets, iter_arg)` 被 `yield_` 返回
- offsets 只依赖参数和最外层 loop var

像 `kb` 这种内层 reduction 循环，仍然保留在每个 `__iter_windowed` kernel 内部，不在本次拆分范围里。

## 关键实现文件

- Pass 主逻辑：
  - `src/ir/transforms/optimize_orch_tensors_pass.cpp`
- 开关贯通：
  - `include/pypto/ir/transforms/pass_context.h`
  - `src/ir/transforms/pass_context.cpp`
  - `python/bindings/modules/passes.cpp`
  - `python/pypto/pypto_core/passes.pyi`
  - `python/pypto/ir/compile.py`
  - `python/pypto/ir/pass_manager.py`
  - `python/pypto/runtime/runner.py`
- ST harness 透传：
  - `tests/st/harness/core/harness.py`
  - `tests/st/harness/core/test_runner.py`

## 期望看到的 IR / Codegen 证据

### Pass 级 IR

开启 task split 后，orchestration 里应该出现：

- `for ... in pl.parallel(...)` 或 `for ... in pl.range(...)`
- `pl.tensor.slice(out_iter, ...)`
- `self.kernel__iter_windowed(...)`
- `pl.tensor.assemble(out_iter, iter_result, iter_offsets)`

克隆出来的 kernel 应该满足：

- 不再保留最外层那层循环壳
- 内层 reduction 仍然还在
- store offsets 已经变成局部 `[0, 0, ...]`

### Orchestration C++

最终生成的 orchestrator 里，应该能看到：

- 显式的 C++ `for (...)`
- `Tensor ...window... = ext_out.view(...)`
- 每次循环都有一次 `rt_submit_*_task(...)`
- callee 名字带 `__iter_windowed`

### Kernel PTO

按单次迭代克隆出来的 kernel PTO 里，如果原始逻辑内部还有 nested reduction loop，
它们仍然应该保留在 kernel 内部。只有最外层被匹配到的 ChunkInner 循环会被 hoist。

## 本次新增或更新的测试

### 单元测试

- `tests/ut/ir/transforms/test_optimize_orch_tensors.py`
  - `test_chunk_inner_parallel_loop_task_split_hoists_iters_to_orchestration`
  - `test_chunk_inner_range_loop_task_split_hoists_iters_to_orchestration`
  - `test_chunk_inner_parallel_loop_task_split_respects_switch`

### Codegen 测试

- `tests/ut/codegen/test_orchestration_codegen.py`
  - `test_chunk_inner_parallel_loop_task_split_visible_in_orchestration_codegen`

### Runtime / ST

- `tests/st/runtime/test_manual_scope_pipeline.py`
  - `_ChunkInnerTaskSplitPTO`
  - `TestChunkInnerTaskSplitRuntime`

### 必须保留的回归测试

- `tests/ut/jit/test_qwen3_decode.py -k q_projection_parallel_does_not_split_into_per_iter_kernels`

这个回归测试很重要，因为它验证了默认不开 task split 时，仍然保持原来的
“一个 outlined kernel 调用”路径。

## 推荐调试顺序

1. 先跑 pass UT，确认 task split 已经触发
2. 再跑 orchestration codegen UT，检查 `main.cpp`
3. 再跑 runtime/ST，并带上 `--save-kernels`
4. 重点看：
   - orchestration `main.cpp`
   - 是否出现 `__iter_windowed`
   - 是否出现 per-iteration `view(...)`
   - runtime submit 的数量是否增加

## 远端建议命令

```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

python -m pytest tests/ut/ir/transforms/test_optimize_orch_tensors.py -k task_split -v
python -m pytest tests/ut/codegen/test_orchestration_codegen.py -k task_split -v
python -m pytest tests/st/runtime/test_manual_scope_pipeline.py::TestChunkInnerTaskSplitRuntime -v --platform=a2a3 --save-kernels

# 兼容性回归
python -m pytest tests/ut/jit/test_qwen3_decode.py -k q_projection_parallel_does_not_split_into_per_iter_kernels -v
```

## 当前已知限制

- 还不支持更复杂的 top-level 嵌套树拆分
- 还不支持 multi-store carried-out 重写
- 还没有做复杂窗口 overlap proof
- 默认不会开启
- 这个本地 Windows worktree 里没有 `pytest`，而且当前 `build/` 目录也不完整，所以完整验证仍然需要放到远端环境执行
