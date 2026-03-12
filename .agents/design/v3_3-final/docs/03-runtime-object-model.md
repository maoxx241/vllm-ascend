# 03. Runtime Object Model

## 1. 总览

本包只冻结 4 个核心 runtime object：

1. `selector_seed/v3`
2. `selector_plan/v4`
3. `atomic_result_card/v3`
4. `continuation_state/v4`

辅助接口对象保持：

- `governor-decision/v1`
- `kb-resolve-result/v1`
- `kb-pack-request/v2`
- `kb-pack-response/v1`

---

## 2. 对象流向

```text
Public Entry
  -> selector_seed/v3

Intake
  -> selector_plan/v4 (optional)
  -> direct_answer OR Atomic OR Spec/Plan

Spec/Plan
  -> selector_plan/v4[]
  -> continuation_state/v4

Atomic
  -> atomic_result_card/v3
```

---

## 3. `selector_seed/v3`

### 3.1 生产者 / 消费者

- 生产者：`Public Entry`
- 消费者：`Intake`

### 3.2 设计目的

1. 固定首跳最小槽位。
2. 显式表达缺口与不确定性。
3. 显式表达 confirmation gate。

### 3.3 关键字段

| 字段 | 含义 | 规则 |
| --- | --- | --- |
| `objective` | 用户真正要完成什么 | 必须是动词化目标 |
| `requested_artifact` | 用户期望产物 | 必须落在固定 deliverable enum |
| `task_family_candidates` | 首跳 family 候选 | 最多 3 个，按优先级排序 |
| `normalized_entities` | 已归一化的 files/symbols/models/features/hw | 只放短标识 |
| `evidence_inventory` | 当前证据篮子 | 只描述类型和有无 |
| `code_change_expectation` | 是否可能改代码 | adaptation / deployment 分界关键输入 |
| `confirmation_required` | 是否必须确认 | 正式阻断器 |
| `confirmation_status` | 当前确认状态 | `pending`/`user_declined` 时不得越过 Intake |
| `confirmation_reason_codes` | 为什么需要确认 | 仅在 `confirmation_required = true` 时出现 |
| `smallest_next_step` | 下一跳最小动作 | 必须可执行 |

### 3.4 schema 级强约束

1. `confirmation_required = false` 时，`confirmation_status` 必须是 `not_needed`。
2. `confirmation_required = true` 时，`confirmation_status` 只能是 `pending / confirmed / user_declined`。
3. `confirmation_required = true` 时，`confirmation_reason_codes` 至少 1 项。

---

## 4. `selector_plan/v4`

### 4.1 生产者 / 消费者

- 生产者：`Intake` 或 `Spec/Plan`
- 消费者：`_shared/context-governor`、`_shared/repo-kb-loader`、目标 atomic 或下游 stage

### 4.2 本版关键修复

`selector_plan/v3` 已经把“谁创建了 plan”和“谁消耗预算”拆开，但还留了一个实现歧义口：

- `origin_stage = intake`
- `query_stage = atomic`
- `execution_mode = spec_plan_workflow`

这种组合会让实现者在“Intake 直接进 Atomic”与“应该先走 Spec/Plan”之间各自补隐式规则。

`v4` 直接把这个口封死：

- `origin_stage = intake && query_stage = atomic => execution_mode = direct_atomic_workflow`
- `origin_stage = spec_plan && query_stage = atomic => execution_mode = spec_plan_workflow`

### 4.3 关键字段

| 字段 | 含义 | 规则 |
| --- | --- | --- |
| `origin_stage` | plan 创建者 | 只能是 `intake` / `spec_plan` |
| `query_stage` | 当前适用哪套 governor 规则 | 只能是 `intake` / `spec_plan` / `atomic`；也是 governor 唯一正式 stage 输入 |
| `task_family` | 当前 family | 固定 8 个 enum |
| `execution_mode` | 整体执行模式 | 与 `origin_stage + query_stage` 联动 |
| `logical_domains` | 当前允许查的知识域 | 至少 1 个 |
| `query_trigger_codes` | 为什么现在允许查 | 没有 trigger code 一律拒绝 |
| `selectors` | 命中 selector 集 | 只允许轻量 ID/名称 |
| `must_have` | pack 排序正式输入 | 预算不足时优先保留 |
| `nice_to_have` | pack 排序次级输入 | 预算不足时优先牺牲 |
| `requested_token_cap` | 本次预算请求上限 | governor 可压低，不可被下游无视 |
| `max_capsules` | 当前 query_stage 允许的 capsule 数 | 是 stage 约束，不是 pack 细节 |
| `max_deep_refs` | 当前 query_stage 允许的 deep refs 数 | 是 stage 约束，不是 pack 细节 |

### 4.4 schema 级强约束矩阵

| `query_stage` | 允许的 `budget_class` | 允许的 `capsule_type` | `requested_token_cap` | `max_capsules` | `max_deep_refs` |
| --- | --- | --- | --- | --- | --- |
| `intake` | `intake` | `intake_capsule` | `<= 1200` | `= 1` | `= 0` |
| `spec_plan` | `spec` / `burst_spec` | `spec_capsule` / `delta_capsule` | `<= 2400` 或 `<= 16000` | `1..2` | `<= 1` |
| `atomic` | `atomic` / `burst_atomic` | `atomic_capsule` / `evidence_capsule` | `<= 1500` 或 `<= 4800` | `1..2` | `<= 2` |

额外规则：

1. `query_stage = intake` 时：
   - `origin_stage` 必须是 `intake`
   - `stop_after_first_sufficient = true`
   - `burst_reason_code = null`
2. `origin_stage = intake && query_stage = atomic` 时：
   - `execution_mode` 必须是 `direct_atomic_workflow`
3. `origin_stage = spec_plan && query_stage = atomic` 时：
   - `execution_mode` 必须是 `spec_plan_workflow`
4. `budget_class = burst_spec/burst_atomic` 时：
   - `burst_reason_code` 必填
   - `why_default_focus_not_enough` 必填
5. `query_stage = atomic` 时，`execution_mode` 不允许是 `direct_answer`。

---

## 5. `atomic_result_card/v3`

### 5.1 生产者 / 消费者

- 生产者：每个 atomic
- 消费者：`Intake`、`Spec/Plan`、bundle 更新逻辑、compaction 逻辑

### 5.2 设计目的

1. 把长分析压成可跨轮保留的小对象。
2. 显式记录 work package 是否闭环、blocked、缺证据或 reroute。
3. 指导 bundle 更新，而不是替代 bundle。

### 5.3 关键字段

| 字段 | 含义 | 规则 |
| --- | --- | --- |
| `result_status` | 工作包状态 | 5 个固定状态 |
| `resolution_code` | 更细粒度的结构化原因 | 必须与 `result_status` 联动 |
| `next_action` | 下一步动作 | 必须结构化 |
| `flush_required` | 是否必须先更新 bundle | `complete/blocked/needs_reroute` 为硬约束 |
| `update_bundle_files` | 需要更新哪些文件 | `flush_required = true` 时必须包含 `progress.md` |
| `reroute` | family/stage 重判载体 | 非 reroute 状态必须为 `null` |

### 5.4 schema 级状态机

| `result_status` | 合法 `resolution_code` | `reroute` | `next_action.kind` |
| --- | --- | --- | --- |
| `complete` | `work_package_closed` | `null` | 非 `reroute_task` |
| `partial` | `partial_findings_only` | `null` | 非 `reroute_task` |
| `blocked` | `blocker_confirmed` / `unknown_failure_mode` | `null` | 非 `reroute_task` |
| `needs_more_evidence` | `evidence_gap` / `kb_miss` / `resolver_unknown` / `validation_gap` | `null` | 非 `reroute_task` |
| `needs_reroute` | `reroute_family_boundary` / `reroute_stage_boundary` | 必填对象 | `reroute_task` |

### 5.5 reroute 的正式硬约束

当 `result_status = needs_reroute` 时，schema 强制：

1. `resolution_code` 必须是 reroute 专用 code。
2. `reroute` 必须是非空对象。
3. `next_action.kind` 必须是 `reroute_task`。
4. `next_action.owner_stage` 必须是 `intake` 或 `spec_plan`。
5. `flush_required` 必须为 `true`。
6. `update_bundle_files` 必须包含 `progress.md`。

---

## 6. `continuation_state/v4`

### 6.1 生产者 / 消费者

- 生产者：`Spec/Plan`；复杂 direct atomic 跨轮时也可生成
- 消费者：下轮 `Spec/Plan`、governor compaction/flush、bundle 恢复逻辑

### 6.2 设计目的

1. 让复杂任务在上下文折叠后仍能恢复。
2. 明确当前阶段、待办 work packages、已完成 work packages。
3. 明确 bundle 才是真正的 source of truth。

### 6.3 v4 的关键收敛

`v3` 仍保留了 `light_bundle` 这样的死枚举，但其他文档和 schema 条件已经事实上要求固定四件套。`v4` 不再保留这种半开状态：

- continuation 一旦存在，`persistence_mode` 固定为 `full_bundle`
- 若任务没有 bundle，就不应该序列化 `continuation_state`

### 6.4 schema 级强约束

1. `persistence_mode` 固定为 `full_bundle`。
2. `canonical_source_files` 必须包含：
   - `spec.md`
   - `plan.md`
   - `checklist.md`
   - `progress.md`
3. `continuation_state` 只保存 ref、顺序、门槛和压缩视图，不保存长正文。

---

## 7. 派生关系

### 7.1 `selector_seed -> selector_plan`
由 Intake 或 Spec/Plan 派生。新增的决定包括：

- final family
- final execution mode
- query_stage
- query domains
- query budget

### 7.2 `selector_plan -> kb-pack-request`
由 loader 编译，是 query 的唯一合法编译入口。

### 7.3 `atomic_result_card -> bundle`
card 只指导 bundle 更新；不能替代 bundle。

### 7.4 `bundle -> continuation_state`
`continuation_state` 必须从 bundle 反向派生；不能反过来主导 bundle。
