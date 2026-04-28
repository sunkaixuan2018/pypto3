# IdentifyStableRegions Pass

标记可以下沉到 PTO2 manual scope 的稳定 orchestration call region。

## 概览

`IdentifyStableRegions` 运行在 call direction 推导和最后一轮 `Simplify` 之后。第一阶段内置的模板识别 PagedAttention 风格的序列：

```text
qk -> softmax -> pv -> update
```

匹配依据是 kernel name 语义 token 和调用顺序，不要求也不校验 `func_id` 序列。

## API

| C++ | Python | 层级 |
| --- | ------ | ---- |
| `pass::IdentifyStableRegions()` | `passes.identify_stable_regions()` | Program-level |

## 输出

命中的 call 会带上这些 attrs：

| Attr | 含义 |
| ---- | ---- |
| `stable_region_template_key` | 命中的模板键 |
| `manual_task_index` | manual region 内的 task 索引 |
| `manual_dep_indices` | 当前 task 依赖的前序 task 索引 |

初始模板的依赖是线性的：task `1` 依赖 `0`，task `2` 依赖 `1`，task `3` 依赖 `2`。

## 边界策略

第一阶段只接受闭合的直线型 region：

- 允许输入来自函数参数、`tensor.create` 和 region 内部前序 task。
- 如果输入依赖 region 外 task，则拒绝命中。
- 如果 region 输出被 region 外 task 消费，则拒绝命中。
- 如果窗口包含 `for`、`while`、`if` 或已有 scope，则拒绝命中。

被拒绝的 region 保持原样，继续走现有 AUTO codegen 路径。

## 实现

**头文件**：`include/pypto/ir/transforms/passes.h`
**实现**：`src/ir/transforms/identify_stable_regions_pass.cpp`
**模板注册表**：`include/pypto/codegen/orchestration/template_registry.h`、`src/codegen/orchestration/template_registry.cpp`

## 测试

**测试**：`tests/ut/ir/transforms/test_stable_region_manual_scope.py`
