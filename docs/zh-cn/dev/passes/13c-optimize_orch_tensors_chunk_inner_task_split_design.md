# OptimizeOrchTensors Chunk-Inner Task-Split 设计

## 目的

本文档描述 `OptimizeOrchTensors` 的第二阶段扩展：对 q_proj 风格的
`ChunkInner` 场景，不再只把一个“大 kernel”的整体写窗口外显到
orchestration，而是把最外层 `ChunkInner` `parallel` / `range` 循环提升到
orchestration，使每个迭代都变成 runtime 可见的独立任务。

## 背景问题

第一阶段 `__windowed` 重写已经能把写窗口暴露到 orchestration 边界，但仍然只会
提交一次 kernel 调用。runtime 看到的是：

- 一个 outlined InCore kernel
- 一个较大的输出窗口
- 一次 runtime task submit

看不到 top-level `ChunkInner` 循环里“每个迭代各写一个局部窗口”的事实，因此无法按
每个迭代的真实粒度重新调度。

## 目标

在不改变“一个 `pl.at(...)` 先 outline 成一个 InCore kernel”这个结论的前提下，
增加一个默认关闭的可选模式：

- 保留原始 outlined kernel
- 克隆一个按单次迭代执行的 `__iter_windowed` kernel
- 把原 top-level `ChunkInner` 循环提升到 orchestration
- 让 orchestration 每次迭代都提交一个独立 runtime task

## 非目标

本阶段不做以下事情：

- 不在 `OutlineIncoreScopes` 阶段直接把一个 `pl.at(...)` 拆成多个 sibling kernel
- 不提升内层 reduction 循环，比如 `kb` 这种 `matmul_acc` 累加循环
- 不处理多次写回、重叠窗口、或更泛化的复杂 store 形态
- 不默认开启该优化

## 开关

该模式通过新的 `PassContext` 开关独立控制：

- `enable_out_window_rewrite`: 第一阶段窗口外显，默认 `True`
- `enable_out_window_task_split`: 第二阶段 task split，默认 `False`

二者相互独立：

- 只开 Pattern 5：仍然是一条大 kernel 调用，只是窗口变窄
- 开启 Pattern 6：会把 top-level `ChunkInner` 迭代变成多个 runtime task

## 目标 IR 形态

原始 InCore kernel：

```python
@pl.function(type=pl.FunctionType.InCore)
def q_proj_chunk_group(self, x, group_base, out):
    for ob_ci, (out_iter,) in pl.parallel(
        4, init_values=(out,), attrs={"loop_origin": pl.LoopOrigin.ChunkInner}
    ):
        q0 = group_base + ob_ci * 64
        tile = pl.load(x, [0, q0], [16, 64])
        out_next = pl.store(tile, [0, q0], out_iter)
        q_rv = pl.yield_(out_next)
    return q_rv
```

改写后的 orchestration：

```python
for ob_ci, (out_iter,) in pl.parallel(4, init_values=(out,)):
    out_iter__window = pl.tensor.slice(out_iter, [16, 64], [0, group_base + ob_ci * 64])
    out_next__iter_windowed = self.q_proj_chunk_group__iter_windowed(
        x, group_base, out_iter__window, ob_ci
    )
    out_next = pl.tensor.assemble(
        out_iter, out_next__iter_windowed, [0, group_base + ob_ci * 64]
    )
    out_rv = pl.yield_(out_next)
```

按单次迭代克隆的 kernel：

```python
@pl.function(type=pl.FunctionType.InCore)
def q_proj_chunk_group__iter_windowed(self, x, group_base, out_window, ob_ci):
    q0 = group_base + ob_ci * 64
    tile = pl.load(x, [0, q0], [16, 64])
    out_next = pl.store(tile, [0, 0], out_window)
    return out_next
```

## 匹配约束

第一版只匹配非常收紧的 shape：

- 只有一个 `Out` 参数
- 函数体是“顶层 `ForStmt` + 末尾 `ReturnStmt`”
- 顶层循环有且只有一个 tensor iter-arg，且由该 `Out` 初始化
- 循环标记了 `loop_origin = ChunkInner`
- 顶层循环可以是 `Parallel` 或 `Sequential`
- 循环体里只有一个被 `yield_` 送出的 `tile.store(..., offsets, iter_arg)`
- offsets 只能依赖参数和顶层 loop var
- store tile 的 shape 就是单次任务的输出窗口 shape

## 改写策略

### 1. 分析 top-level ChunkInner 循环

抽取：

- 顶层循环的 `start / stop / step / kind`
- 唯一的 `tile.store`
- 单次迭代的全局 offsets
- 单次迭代写回窗口 shape

### 2. 克隆单次迭代 kernel

生成 `__iter_windowed`：

- 把 `Out` 参数和返回类型收窄到单次迭代窗口
- 追加原 loop var 作为新的 scalar 参数
- 复制原循环体，但去掉最外层循环壳
- 把唯一一次 `tile.store` 的 offsets 改写成局部 `[0, 0, ...]`
- 把尾部 `yield_` 改写成 `return`

### 3. 把循环提升到 orchestration

把原来的：

```python
out_next = self.kernel(..., out)
```

改写成：

- orchestration 自己持有 loop iter-arg
- 每次迭代先 `tensor.slice(...)`
- 然后调用 `self.kernel__iter_windowed(...)`
- 最后用 `tensor.assemble(...)` 回绑父 tensor 的 SSA

## 为什么这会改变 runtime task 粒度

这次改写和第一阶段的最大差别是：kernel submit 的数量真的变了。

- 改写前：一次大 kernel submit
- 改写后：每个 top-level `ChunkInner` 迭代一次 submit

因此 runtime / AICore 可以基于这些子任务重新调度：

- `Parallel` 场景下，不同窗口的任务可以独立调度
- `Sequential` 场景下，任务链会按 orchestration 里的显式顺序串起来

## 正确性约束

改写必须保持：

- 数值结果不变
- orchestration 层父 tensor 的 SSA 不丢失
- 默认行为不变，也就是不开开关时仍走原来的大 kernel 路径
- 内层 reduction 仍保留在单次迭代 kernel 内部

不能做的事情：

- 不能静默默认开启
- 不能删除 orchestration 侧的 `tensor.assemble(...)`
- 不能把未证明安全的复杂写回场景强行拆开

## 验证建议

至少需要以下三类验证：

1. Pass UT：证明 orch-visible loop 和 `__iter_windowed` clone 出现了
2. Codegen UT：证明最终 orchestration C++ 里出现了 hoist 后的 loop、`view(...)` 和按迭代 submit
3. Runtime/ST：证明数值正确，并能从保存的 artifact 里直接看到多个子任务
