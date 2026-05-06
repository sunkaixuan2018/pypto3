# Control-Flow-Aware Stable Regions

本文档把 manual-scope 模板设计从“直线型 orchestration region”扩展到“控制流感知的 loop body region”。目标场景是 `paged_attention` 这类稳定 task 模式位于最内层 `for` 循环中、同时夹带少量 flag 型 `if` 语句以及 loop-carried state 的 orchestration。

## 目标

目标是在不把模板分析下沉到 codegen、也不要求 `simpler` 理解模板元数据的前提下，让 PyPTO 能把最内层 orchestration loop body lower 成 `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`。

第一版仍然保持保守：

- 只匹配最内层 `ForStmt` 的 body。
- 只允许模板显式声明过的 `IfStmt` 形态出现在 task stage 之间。
- loop-carried state 通过模板声明和 IR 校验表达，不做自动推断。
- 任意不匹配都原样 fallback 到现有 AUTO 路径。

## 问题形态

当前直线型 matcher 期望的 region 是：

```text
qk -> softmax -> pv -> update
```

而真实的 `paged_attention` loop body 更接近：

```text
qk
softmax
pv
if bn == 0: yield 1 else yield 0
if bn == bn_this_batch - 1: yield 1 else yield 0
update
yield next_iter(mi, li, oi)
```

这意味着成功命中需要同时理解：

- loop body 内的 task call
- stage 之间有限的控制流节点
- loop-carried state（`mi`、`li`、`oi`）

因此，单独“放过 `IfStmt`”并不够，因为当前 matcher 还没有把 loop body 当成一等候选 region。

## 架构

整体仍保持三段式：

1. `IdentifyStableRegions`
   - 下钻到候选最内层 `ForStmt.body`
   - 检查 control-flow-aware 模板契约
   - 给命中的 task call 打标
2. `LowerStableRegionsToManualScope`
   - 把命中的 loop-body region 包成 `ManualScopeStmt`
   - 保留 `ForStmt` 结构作为 scope body 的一部分
3. orchestration codegen
   - 消费 `ManualScopeStmt` 和 task attrs / loop 元数据
   - 发射 manual-scope 提交代码和显式依赖

这样可以继续保持“pass 做分析，codegen 只做结构化 lowering”的分层原则。

## 模板契约

每个 control-flow-aware 模板都采用声明式契约。第一版建议支持以下字段：

```yaml
template_key: paged_attention_loop_body_v1

match:
  region_kind: loop_body
  loop_nest: innermost_for
  tokens: [qk, softmax, pv, update]

  stage_shape:
    kind: linear
    counts:
      qk: 1
      softmax: 1
      pv: 1
      update: 1

  arg_directions:
    qk: [Input, Input, OutputExisting]
    softmax: [Input, Scalar, OutputExisting, OutputExisting, OutputExisting]
    pv: [Input, Input, OutputExisting]
    update: [Input, Input, Input, InOut, InOut, InOut, OutputExisting, Scalar, Scalar]

  allowed_between_stages:
    - between: [pv, update]
      kind: if_flag
      count: 2
      rules:
        branch_yield_kind: scalar_only
        task_calls_allowed: false
        return_vars_kind: scalar_only

  loop_shape:
    iter_arg_count: 3
    iter_arg_roles: [mi_state, li_state, oi_state]
    return_var_count: 3

recipe:
  intra_iteration_deps:
    - { task: softmax, deps: [qk] }
    - { task: pv, deps: [softmax] }
    - { task: update, deps: [pv] }

  loop_carried_reads:
    - { task: update, read_from_prev_iter: mi_state }
    - { task: update, read_from_prev_iter: li_state }
    - { task: update, read_from_prev_iter: oi_state }

  loop_carried_writes:
    - { produced_by: update, write_to_next_iter: mi_state }
    - { produced_by: update, write_to_next_iter: li_state }
    - { produced_by: update, write_to_next_iter: oi_state }
```

字段含义：

- `region_kind`
  - 候选 region 的结构类型。第一版固定为 `loop_body`。
- `loop_nest`
  - 限定只在最内层 `ForStmt` 命中。
- `tokens`
  - 逻辑 task 序列。
- `stage_shape`
  - 约束每类 token 的数量。
- `arg_directions`
  - 复用 `DeriveCallDirections` 的结果做结构校验。
- `allowed_between_stages`
  - 声明允许出现在 stage 之间的非 task 控制流节点。
- `loop_shape`
  - 约束 loop-carried state 的形态。
- `recipe.intra_iteration_deps`
  - 单次迭代内部的静态依赖边。
- `recipe.loop_carried_reads`
  - 哪个 task 会读取前一轮的 state。
- `recipe.loop_carried_writes`
  - 哪个 task 会把 state 更新给下一轮。

第一版应坚持声明式契约，不做 loop-carried state 的自动 def-use 推断。

## IR 表达

本方案复用现有 `ManualScopeStmt`，而不是新增一类全新的 control-flow template IR 节点。

manual region 由以下部分共同表达：

- `ManualScopeStmt`
  - 结构化 region 边界
- per-task attrs
  - `stable_region_template_key`
  - `manual_task_index`
  - `manual_dep_indices`

loop-carried 和 flag 结构继续使用现有控制流 IR：

- `ForStmt.iter_args`
- `ForStmt.return_vars`
- `IfStmt.return_vars`
- `YieldStmt`

如果后续需要，也可以在 `ManualScopeStmt` 或 enclosing `ForStmt` 上增加 region-level attrs，但第一版可以从 task attrs + 模板校验开始。

## Pass / Codegen 职责边界

### `IdentifyStableRegions`

负责：

- 候选 region 发现
- 最内层 loop 检测
- token 匹配
- stage shape 校验
- `arg_directions` 校验
- stage 之间允许控制流节点的校验
- loop-carried shape 校验
- 按模板 recipe 展开单次迭代内的 manual dependency indices

不负责：

- C++ 发射
- task handle 生命周期管理
- runtime 依赖推断

### `LowerStableRegionsToManualScope`

负责：

- 把命中的 loop-body candidate 改写为结构化 `ManualScopeStmt`
- 保留 `ForStmt` 作为 scope body 的一部分
- 未命中的 region 保持原样

不负责：

- 模板匹配
- 依赖重推导

### orchestration codegen

负责：

- 把 `ManualScopeStmt` lower 成 `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- 发射 task-handle capture
- 发射 `Arg.add_dep(...)`
- 把 loop-carried 模板元数据翻译成跨 iteration 的依赖代码

不负责：

- 模板识别
- 控制流安全性分析
- 通用 def-use 推断

## 匹配策略

matcher 建议按以下方式工作：

1. 访问 orchestration function。
2. 递归下钻到 `ForStmt`。
3. 只有当 `ForStmt` 是最内层循环时，才把它视为候选。
4. 从它的 `SeqStmts` body 中抽取 task call 和模板声明允许的控制流节点。
5. 只要 body 内出现以下任意情况就拒绝：
   - 仍有更内层循环
   - 未声明允许的 `IfStmt`
   - 允许的 `IfStmt` 分支里含 task call
   - loop-carried shape 与模板不符
6. 只有整个模板契约都通过后，才写入 manual attrs。

这样可以避免“表面上 `pv -> update` 能接上，但 loop-carried 语义已经错了”的危险部分命中。

## Codegen 语义

当 `ManualScopeStmt.body` 中包含允许的 `ForStmt` 时，codegen 应：

- 在外层只发一次 manual scope
- 保留循环结构
- 按 `manual_dep_indices` 发射单次迭代内的 `add_dep(...)`
- 按模板中的 loop-carried recipe 发射跨 iteration 依赖
- 对首轮迭代做保护，避免依赖不存在的前一轮 task handle

第一版建议显式保存前一轮 relevant task handle / state，不要现场尝试自动推断最小依赖集合。

## 分阶段落地

### Phase 0：只做识别

- 增加最内层 `ForStmt` 下钻。
- 增加测试，证明可以识别 loop body。
- 暂不 lower，不改 codegen。

### Phase 1：支持 loop-body manual scope

- 增加 `allowed_between_stages` 对 flag-only `IfStmt` 的支持。
- 把命中的 loop-body region lower 成 `ManualScopeStmt`。
- codegen 先支持单次迭代内部的 manual deps。

### Phase 2：接入 loop-carried deps

- 校验 `loop_shape`。
- 按模板声明发射跨 iteration 依赖。
- 覆盖完整 `paged_attention` loop-body 路径。

### Phase 3：补变体与回归

- 扩展到 unaligned softmax 等近邻模板。
- 增加性能和 no-regression 校验。

## 风险

1. 在把跨 iteration codegen 形态视为稳定前，必须先确认 `simpler` 对带循环的 manual scope 语义。
2. 第一版如果做 loop-carry 自动推断，风险很高，应避免。
3. loop 下钻必须严格停在最内层候选，否则外层循环误命中会生成无效 orchestration C++。

## 测试

最小覆盖建议包括：

- loop-body `qk -> softmax -> pv -> update` 模式的 pass 级命中
- 未声明控制流的拒绝命中
- loop-carried arity 不匹配的拒绝命中
- 最内层 `ForStmt` 被 lower 成 `ManualScopeStmt`
- 含 flag `IfStmt` 的 loop body manual-scope codegen
- 模板契约不满足时保持 AUTO fallback
