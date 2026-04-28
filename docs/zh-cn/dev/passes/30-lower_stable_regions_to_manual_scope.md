# LowerStableRegionsToManualScope Pass

把 `IdentifyStableRegions` 标记过的 call 包成结构化 manual-scope IR。

## 概览

`LowerStableRegionsToManualScope` 消费 `IdentifyStableRegions` 产生的 attrs，并在命中的直线型 region 外创建 `ManualScopeStmt`。每个 call 上的依赖 attrs 会保留，供 orchestration codegen 发射显式依赖。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::LowerStableRegionsToManualScope()` | `passes.lower_stable_regions_to_manual_scope()` | Program-level |

## 输入

call 需要携带：

- `stable_region_template_key`
- `manual_task_index`
- `manual_dep_indices`

## 输出

命中的语句会被包成：

```text
ManualScopeStmt(template_key=..., body=SeqStmts([...]))
```

pass 会保留 region 前后的未标记语句，包括尾部 `ReturnStmt`。如果候选范围包含控制流或已有 scope，则拒绝包装。

## Codegen 契约

orchestration codegen 会把 `ManualScopeStmt` lower 成：

```cpp
PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
  ...
}
```

scope 内会把 submit 返回值保存为 `task_result_N`，并根据 `manual_dep_indices` 生成：

```cpp
params_t1.add_dep(task_result_0.task_id());
```

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`
**实现**：`src/ir/transforms/lower_stable_regions_to_manual_scope_pass.cpp`

## 测试

**测试**：`tests/ut/ir/transforms/test_stable_region_manual_scope.py`、`tests/ut/codegen/test_orchestration_codegen.py`
