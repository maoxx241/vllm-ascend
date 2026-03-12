# 09. Supersede and Migration

## 1. 本文件的地位

本文件是规范性文件。它定义本包如何替换旧包与旧稿。

---

## 2. supersede 规则

`vllm-ascend-agent-design-package-v3_3-final` 正式 supersede：

1. `vllm-ascend-agent-design-package-v3_2-final`
2. `vllm-ascend-agent-design-package-v3_1-final`
3. `vllm-ascend-agent-design-package-v3_0-final`
4. 任何 repo 内仍残留的 `v2` / `v2.3` handoff、draft、notes

替换规则：

- 旧稿只能作为历史背景保留在 `legacy_reference/` 或归档目录
- 旧稿不得继续作为实现入口
- 任何与本包冲突的旧字段、旧流程一律失效

---

## 3. runtime object 迁移说明

### 3.1 `selector_seed/v2 -> selector_seed/v3`

新增：

- `confirmation_reason_codes`

强化：

- `confirmation_required` 与 `confirmation_status` 的条件约束进入 schema

迁移动作：

1. 若旧对象 `confirmation_required = false`，强制写 `confirmation_status = not_needed`
2. 若旧对象 `confirmation_required = true`，补 `confirmation_reason_codes`

### 3.2 `selector_plan/v3 -> selector_plan/v4`

新增硬约束：

- `origin_stage = intake && query_stage = atomic => execution_mode = direct_atomic_workflow`
- `origin_stage = spec_plan && query_stage = atomic => execution_mode = spec_plan_workflow`

保留：

- `query_stage` 仍是预算矩阵的唯一正式驱动字段
- `budget_class` 与 `capsule_type` 仍不允许不合法组合

迁移动作：

1. 所有旧 `selector_plan/v3` 必须改写 `schema_version = selector-plan/v4`
2. 若旧 plan 满足 `origin_stage = intake && query_stage = atomic`，强制写 `execution_mode = direct_atomic_workflow`
3. 若旧 plan 满足 `origin_stage = spec_plan && query_stage = atomic`，强制写 `execution_mode = spec_plan_workflow`
4. 若无法满足以上关系，则旧对象作废，不得静默兼容

### 3.3 `atomic_result_card/v2 -> atomic_result_card/v3`

变化：

- reroute 约束正式进入 schema
- 非 reroute 状态强制 `reroute = null`
- `flush_required = true` 时强制 `update_bundle_files` 包含 `progress.md`

迁移动作：

1. 旧 `needs_reroute` card 必须补齐：
   - `resolution_code`
   - `reroute`
   - `next_action.kind = reroute_task`
   - `flush_required = true`
2. 旧非 reroute card 中若带 `reroute`，一律改成 `null`

### 3.4 `continuation_state/v3 -> continuation_state/v4`

变化：

- 去掉 `light_bundle`
- continuation 一旦存在，`persistence_mode` 固定为 `full_bundle`
- 四件套 bundle 约束不再允许半开状态

迁移动作：

1. 所有旧 `continuation_state/v3` 必须改写 `schema_version = continuation-state/v4`
2. `persistence_mode` 统一改为 `full_bundle`
3. 若对象没有四件套 bundle 路径，则该 continuation 不能迁移，必须删除并回到 bundle 生成逻辑

### 3.5 `kb-pack-request/v1 -> kb-pack-request/v2`

新增：

- `intent = model_expectation`

变化：

- `performance_analysis` 不再只映射到 `perf_breakdown`
- `consumer_id = model-expected-performance-estimator` 时必须映射到 `model_expectation`

迁移动作：

1. 所有旧 pack request 样例统一改写为 `schema_version = kb-pack-request/v2`
2. loader intent compiler 必须升级到新的 mapping 表

---

### 3.6 `context-governor` 调用合同 `v3.2 -> v3.3`

变化：

- 删除 `evaluate_governor(...)` 的显式 `stage` 输入
- query-bearing governor 调用时，`selector_plan/v4` 变为必填
- `governor-decision.stage` 改为派生回显字段，必须等于 `selector_plan.query_stage`
- `Public Entry` 与 `Intake` 的 `direct_answer/no-query` 路径不再调用 governor

迁移动作：

1. 删除所有 wrapper / adapter / tests 中的显式 `stage` 参数
2. 调 governor 前必须先有 query-bearing `selector_plan/v4`
3. 若没有 `selector_plan`，只能停在 route/intake，不得调用 governor
4. 若内部 legacy helper 仍保留 `stage`，必须在 contract boundary 前消除，不得暴露为正式接口

## 4. MVP 范围迁移说明

`v3.1` 的首波目标仍按“4 个 MVP family”描述；`v3.2` 改成“4 个 MVP capability”：

1. `deployment_execution / feature-policy-resolver`
2. `performance_analysis / single-profile-breakdown`
3. `performance_analysis / model-expected-performance-estimator`
4. `validation_strategy / change-impact-test-selector`

说明：

- `debugging` 仍是 stable family，不被删除
- `debugging` 被延后到 `P3B`
- 排期与验收都必须按 capability 而不是 family 解释首波 MVP

---

## 5. repo 侧替换步骤

1. 先把旧设计文件整体移到归档目录。
2. 安装本包的 `schema/`、`docs/`、`examples/`、`sql/`、`tasks/`、`tools/`、`requirements.txt`。
3. 先运行：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python tools/validate_design_package.py
```

4. validator 全绿后，再开始 shared runtime 与 KB 实现。

---

## 6. Codex 的最低迁移要求

在开始任何实现前，Codex 必须完成：

1. `selector_plan` 升级到 `v4`
2. `continuation_state` 升级到 `v4`
3. `kb-pack-request` 升级到 `v2`
4. Intake 执行 confirmation gate
5. governor 执行 `origin_stage + query_stage + execution_mode` 一致性检查
6. continuation 与 bundle 使用 `full_bundle` 单一路径
7. governor 公开接口删除显式 `stage` 参数，且输出 `stage` 只来自 `selector_plan.query_stage`
