# 模板命中后 Manual Scope 下沉的代码修改方案

## 背景

这版方案基于一次 `codex + claude-code` 的交叉审查整理而成，目标不是在 device 侧做学习型缓存，而是在 PyPTO3 的编译期识别稳定的 orchestration call region。

命中后，PyPTO3 直接生成带显式依赖的 orchestration C++，让 `simpler` 按已有的 manual-scope 执行路径跑，不再依赖运行时的动态图命中。

这里的“命中”不是单看输入输出 args，而是看一段稳定的调用序列/模式。以 PagedAttention 为例，像 `QK -> Softmax -> PV -> Update` 这种 region，只要当前编译里的 kernel name 语义 token 和调用顺序稳定，即使具体 tensor args 变了，也仍然认为是同一个模板命中。第一阶段不强制 `func_id` 序列；`func_id` 只能作为调试信息或后续更严格模板的可选条件。

## 目标

第一阶段只解决一件事：让上层在 codegen 前识别出稳定 region，并把它 lower 成 manual scope + explicit deps。

范围约束如下：

- 只覆盖同一词法作用域内的直线型 call 序列
- 不在第一阶段支持跨 `for/while/if` 的 region
- 不做 device 侧学习缓存
- 不要求 `simpler` 理解模板系统本身
- 命中失败必须安全回退到现有 AUTO 路径

## 方案对比

### 方案 A：只在 orchestration codegen 里做匹配

做法是把模板查找、region 分析、依赖推导都塞进 `orchestration_codegen.cpp`。

优点：

- 少加 pass
- 改动入口集中

缺点：

- 违背 codegen 只做 1-to-1 lowering 的现有设计原则
- 匹配逻辑、边界分析、代码生成会缠在一起
- 很难单独做 pass 级测试

不推荐。

### 方案 B：一个 pass 同时做识别和 lower

做法是增加一个 pass，既负责找模板，也直接把 IR 改成 manual scope 形式。

优点：

- 比方案 A 清晰
- 落地速度快

缺点：

- pass 责任偏重
- 后续再加模板、加边界规则时容易膨胀

可以做备选，但不是最稳的组织方式。

### 方案 C：双 pass + codegen 配合（推荐）

拆成两步：

1. `IdentifyStableRegions` 负责识别候选 region、做模板命中、收集边界信息
2. `LowerStableRegionsToManualScope` 负责把命中的 region 改写成结构化 IR
3. orchestration codegen 只消费 pass 产出的结构和依赖信息

优点：

- 分层清楚，符合现有 codegen 设计原则
- 更容易做命中/拒绝测试
- 后续可以逐步扩模板，不会把 codegen 变成分析器

最终推荐采用方案 C。

## 推荐方案

### 1. 命中规则

模板命中键建议先以逻辑模板键 `template_key` 为主，内置模板记录稳定的 kernel name token 有序序列。

其中：

- `kernel_name` 用于表达逻辑意义，如 `qk`, `softmax`, `pv`, `update`
- `core_type` 可作为后续更严格模板的可选条件，用来区分 AIC/AIV 路径
- `func_id` 不作为第一阶段命中约束

命中时允许输入输出 args 的具体变量名变化，但以下条件必须一致：

- 调用顺序一致
- 参数方向一致
- region 的边界契约一致

这意味着模板不是跨所有构建永久稳定的全局 id 规则，而是“当前编译结果里的稳定调用模式规则”。

### 2. 模板来源

第一阶段不建议一上来引入 YAML/JSON 外部模板库，而是先做成编译期内置模板注册表。

原因很简单：

- 当前需求还在快速收敛
- C++ pass 直接消费内置模板更容易测试和调试
- 可以先把 PagedAttention 这类高价值模板做通，再考虑是否外部化

建议新增一个模板注册模块，维护少量高价值模板：

- `include/pypto/codegen/orchestration/template_registry.h`
- `src/codegen/orchestration/template_registry.cpp`

模板定义至少包含：

- `template_key`
- `kernel_name` 序列
- 可选的 `core_type` 序列
- region 内部依赖 recipe
- region 边界契约

后续如果模板数量变多，再把注册表外移成可维护的文本格式。

### 3. 命中发生的位置

命中不应放在 runtime，也不应直接塞进 codegen。

推荐做法是：

- 在 orchestration IR 已经稳定之后做命中
- 依赖 `DeriveCallDirections` 之后的显式方向信息
- 尽量贴近 orchestration codegen 之前

推荐在 `python/pypto/ir/pass_manager.py` 中，把新 pass 放在最后一轮 `Simplify` 之后、codegen 之前：

1. `DeriveCallDirections`
2. `Simplify`
3. `IdentifyStableRegions`
4. `LowerStableRegionsToManualScope`

这样拿到的是已经规整过的 call 序列，命中逻辑会更稳定。

### 4. IR 表达方式

不建议用“函数 attrs 里记 begin/end index”的方式表示 region，太脆弱。

推荐新增一个结构化 scope 结点，例如：

- 在 `ScopeKind` 中增加 `Manual`
- 新增 `ManualScopeStmt : ScopeStmt`

这个 scope 只表达“这段 region 要按 manual scope 生成”，它本身至少带：

- `template_key`
- `template_version`（可选）
- `body`

region 内每个 call 再带轻量 attrs，例如：

- `manual_task_index`
- `manual_dep_indices`

这样分工更清楚：

- `ManualScopeStmt` 表达 region 边界
- call attrs 表达 region 内部的任务依赖

### 5. 模板首尾边界怎么处理

这是整个方案里最容易出错的地方。

需要区分三类边：

1. region 内部边：模板内 producer -> consumer
2. region 入口边：region 外 producer -> region 内首批 task
3. region 出口边：region 内最后 task -> region 外 consumer

推荐规则如下：

#### 5.1 入口边

`IdentifyStableRegions` 在构图时，需要先做一次局部 def-use 分析。

如果 region 内某个 task 的输入来自：

- 函数参数
- `tensor.create`
- region 内部前驱 task

则允许直接进入候选集。

如果输入来自 region 外其他 kernel call，则要继续判断：

- 这个外部 producer 能不能一起并入当前 region
- 如果不能并入，当前模板先拒绝命中

第一阶段建议保守一些：不支持“外部 producer 在模板外、但模板内还要显式追它的 task_id”的开放边界场景。

#### 5.2 出口边

如果 region 输出只流向：

- `return`
- `Out/InOut` 参数
- 模板内部后续 task

则可以接受。

如果 region 输出还会在模板外继续被其他 kernel call 消费，第一阶段也建议保守处理：

- 要么把这些紧邻 consumer 一起并入模板
- 要么拒绝命中

这样做的目的，是先只支持“闭合 region”，避免 manual scope 和外部 AUTO 路径混合时出现依赖断边。

换句话说，模板前后两头的连接关系，第一阶段靠 def-use 分析找；找到了以后，如果边界不闭合，就不命中。

### 6. orchestration analysis 需要补的能力

当前 `orchestration_analysis.{h,cpp}` 已经有一部分 codegen 共享分析能力，例如 `VarLineageCollector`、`BufferRootCollector`。

这一层建议继续扩展，新增两类分析能力：

1. 稳定 call 序列提取
   从 orchestration function 中按顺序提取非 builtin call，并抽取用于模板匹配的 kernel name token 序列。

2. 局部边界分析
   对候选 call window 收集：
   - region 输入变量
   - region 输出变量
   - 内部 producer/consumer 关系
   - 不能闭合的入口/出口边

这样后面的 pass 和 codegen 都可以复用同一套分析结果。

### 7. orchestration codegen 需要补的能力

当前分支的 `EmitTaskSubmitAndBind()` 只是直接 submit，没有保留 task result 句柄；`orchestration_codegen.h` 里的注释也还写着“不做手动依赖管理”。

如果要支持 manual scope，需要至少补三件事：

1. 支持生成 `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
2. 支持在 `Arg` 上发射 `add_dep(...)`
3. 支持保留 submit 返回的 task 句柄，用于获取 `task_id()`

推荐修改方式：

- 让 `EmitTaskSubmitAndBind()` 不再只发射一行 submit
- 在 manual scope 内，对每个 task 生成可引用的结果句柄，例如 `task_result_0`
- 根据 `manual_dep_indices` 发射：
  - `params_t1.add_dep(task_result_0.task_id())`
  - `params_t3.add_dep(task_result_1.task_id())`

第一阶段只要求支持模板内部静态边，不要求把 manual 区和外部 AUTO 区混合编排得很激进。

### 8. backend 输出与 simpler 对接

`simpler` 不需要理解模板系统。

它只需要继续执行 PyPTO3 产出的：

- `orchestration/<orch>.cpp`
- `kernel_config.py`

如果需要调试和复现，可以在 backend 侧额外输出一个可选文件，例如：

- `report/template_manifest.json`

内容记录：

- 命中的 `template_key`
- 命中的 kernel name token 序列，以及可选的 core type / func_id 调试信息
- region 范围
- 是否因为开放边界被拒绝

这个 manifest 只用于调试，不应成为 `simpler` 的输入前提。

## 具体改动模块

### 1. IR 结点

必改：

- `include/pypto/ir/stmt.h`

建议新增：

- `ScopeKind::Manual`
- `ManualScopeStmt`

如果要让 Python 侧能观察到新结点，还需要同步：

- `python/bindings/modules/ir.cpp`
- `python/pypto/pypto_core/ir.pyi`

### 2. orchestration 分析层

建议扩展：

- `include/pypto/codegen/orchestration/orchestration_analysis.h`
- `src/codegen/orchestration/orchestration_analysis.cpp`

新增职责：

- 提取候选 call 序列
- 计算 region 边界和内部依赖

### 3. 模板注册层

建议新增：

- `include/pypto/codegen/orchestration/template_registry.h`
- `src/codegen/orchestration/template_registry.cpp`

职责：

- 维护内置模板定义
- 提供按序列匹配模板的查询接口

### 4. IR pass 层

必改：

- `include/pypto/ir/transforms/passes.h`
- `python/bindings/modules/passes.cpp`
- `python/pypto/pypto_core/passes.pyi`
- `python/pypto/ir/pass_manager.py`

建议新增两个 pass 源文件：

- `src/ir/transforms/identify_stable_regions.cpp`
- `src/ir/transforms/lower_stable_regions_to_manual_scope.cpp`

职责拆分：

- `IdentifyStableRegions`：找候选 region、做模板命中、记录边界分析结果
- `LowerStableRegionsToManualScope`：把命中的 region 改写成 `ManualScopeStmt` + call attrs

### 5. orchestration codegen 层

必改：

- `include/pypto/codegen/orchestration/orchestration_codegen.h`
- `src/codegen/orchestration/orchestration_codegen.cpp`

职责：

- 识别 `ManualScopeStmt`
- 生成 `PTO2_SCOPE(PTO2ScopeMode::MANUAL)`
- 发射 `add_dep(...)`
- 保留 task result 句柄

### 6. backend 输出层

按需改：

- `python/pypto/backend/pto_backend.py`
- `python/bindings/modules/codegen.cpp`
- `python/pypto/pypto_core/codegen.pyi`

职责：

- 如果需要，把模板命中结果带进 `OrchestrationResult`
- 输出 `template_manifest.json`

### 7. Python attrs 转换层

如果 `manual_dep_indices` 继续用通用 attrs 表达，则建议同步扩展：

- `python/bindings/modules/ir.cpp`

原因是当前 `ConvertKwargsDict()` 对 list/tuple 的支持基本只覆盖 `arg_directions`，后续若要让 `list[int]` 的依赖索引在 Python 里可见，需要补齐转换逻辑。

## 测试计划

### 1. pass 级测试

新增：

- `tests/ut/ir/transforms/test_identify_stable_regions.py`
- `tests/ut/ir/transforms/test_lower_stable_regions_to_manual_scope.py`

覆盖点：

- 命中直线型模板
- 开放入口边拒绝命中
- 开放出口边拒绝命中
- 不跨 `for/if` 命中

### 2. codegen 级测试

扩展：

- `tests/ut/codegen/test_orchestration_codegen.py`

重点校验：

- `PTO2_SCOPE(PTO2ScopeMode::MANUAL)` 是否生成
- `params_tN.add_dep(...)` 是否按模板 recipe 发射
- 未命中模板时仍保持现有 `PTO2_SCOPE()` 输出

### 3. backend 输出测试

如果引入 manifest，再补：

- `kernel_config.py` 原有内容不回归
- `template_manifest.json` 内容与命中结果一致

## 风险与落地顺序

最大风险不是“命不中”，而是“命中了但边连错了”。

所以落地顺序建议非常保守：

1. 先只支持 1 个高价值模板
2. 先只支持闭合、直线型 region
3. 先只生成模板内部依赖
4. 入口/出口边界不满足闭合条件就直接拒绝命中
5. 所有未命中场景回退到现有 AUTO 路径

## 结论

最终推荐方案是：在 PyPTO3 中新增“模板识别 pass + manual scope lowering pass”，让 orchestration codegen 只负责消费结构化 IR 并发射 manual-scope 代码。

这样做的好处是：

- 匹配逻辑留在 pass 层
- codegen 仍然保持 1-to-1 lowering
- `simpler` 只负责执行，不必理解模板系统
- 第一阶段可以用闭合 region 策略把风险压到最低
