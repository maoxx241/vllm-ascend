# 04. Context Governor and Persistence

## 1. 目标

governor 只负责五件事：

1. 判断当前 query 是否允许。
2. 解析预算。
3. 判断是否必须先 flush。
4. 判断是否必须 compaction。
5. 生成 dedupe key。

---

## 2. governor 输入

governor 的判断依赖：

- `selector_seed/v3`
- `selector_plan/v4`（必填；没有 plan 就不允许调 governor）
- `continuation_state/v4`（复杂任务）
- 进度状态：
  - `bundle_exists`
  - `has_unflushed_findings`
  - `query_count_in_stage`
  - `opened_deep_refs_in_stage`
  - `last_flush_at`
  - `session_budget_used`

### 2.1 调用前提

1. governor 只在“已经形成 query-bearing `selector_plan/v4`，并准备实际发 query”时调用。
2. `Public Entry` 不调用 governor。
3. `Intake` 若走 `direct_answer` 或其他 no-query 分支，也不调用 governor。
4. governor 不接受任何外部 `stage` override；effective stage 完全由 `selector_plan.query_stage` 派生。
5. `governor-decision.stage` 只是上述派生值的回显字段，不构成第二真相源。

---

## 3. 预算规则

### 3.1 stage hard cap

| 载体 | 硬上限 |
| --- | --- |
| `routing_capsule` | `<= 400` |
| `intake_capsule` | `<= 1200` |
| `spec_capsule` | `<= 2400` |
| `atomic_capsule` | `<= 1500` |
| `deep_reference slice` | `<= 800` |

### 3.2 workflow soft cap

| workflow | 默认上限 | stretch 上限 |
| --- | --- | --- |
| `direct_atomic_workflow` | `<= 3200` | `<= 4800` |
| `spec_plan_workflow` | `<= 9600` | `<= 16000` |

### 3.3 session uncompacted cap

- `session_uncompacted_cap <= 51200`

---

## 4. `query_stage` 规则

本版 governor 的 effective stage 正式且唯一地来自 `selector_plan.query_stage`；不会接受任何额外 stage 输入。

### 4.1 `query_stage = intake`

固定规则：

1. 只允许一次 `intake_capsule`。
2. `requested_token_cap <= 1200`。
3. `max_capsules = 1`。
4. `max_deep_refs = 0`。
5. 不能为未来 atomic 预抓材料。

### 4.2 `query_stage = spec_plan`

固定规则：

1. 默认 1 个 `spec_capsule` 或 `delta_capsule`。
2. 若 `max_capsules = 2`，必须满足：
   - 已有 work package 拆分
   - `why_default_focus_not_enough` 非空
   - 当前阶段结论已 flush
3. `burst_spec` 只在默认预算不足后成立。

### 4.3 `query_stage = atomic`

固定规则：

1. 默认 1 个 `atomic_capsule` 或 `evidence_capsule`。
2. 允许第二个 capsule 或 deep ref 的前提：
   - 当前 work package 的核心未知项仍未覆盖
   - 当前结果已 flush
   - 不扩大到不相关 domain
3. `burst_atomic` 只在默认预算路径失败后成立。
4. `origin_stage = intake && query_stage = atomic` 时，当前 plan 必须属于 `direct_atomic_workflow`。
5. `origin_stage = spec_plan && query_stage = atomic` 时，当前 plan 必须属于 `spec_plan_workflow`。

---

## 5. burst 规则

允许的 `burst_reason_code`：

- `cross_surface_conflict`
- `multi_version_delta`
- `operator_constraint_dense`
- `validation_matrix_dense`
- `upstream_release_dense`
- `topology_policy_dense`

burst 前置条件：

1. 已尝试默认预算路径且仍不足。
2. `why_default_focus_not_enough` 已写明。
3. 当前已尝试路径、关键发现、open questions 已 flush 到 bundle。

---

## 6. flush 规则

满足任意一条都必须先 flush bundle：

1. 一个 atomic 完成。
2. 一个候选路径被排除。
3. 一个 blocker 被确认。
4. `Spec/Plan` 更新了 work package 顺序。
5. `Spec/Plan` 更新了 success / stop / reroute conditions。
6. family 或 stage 发生 reroute。
7. 需要申请第二个 capsule。
8. 需要进入 burst。
9. 当前轮结束但任务未完成。

`flush_required = true` 的实现含义：

- 先更新 bundle，再继续 query。
- 未完成 flush 前，governor 必须拒绝后续 query。

---

## 7. confirmation gate 与 governor 的关系

1. `confirmation_status = pending` 时：
   - governor 必须拒绝 `query_stage = atomic/spec_plan` 的 query。
   - 拒绝原因归到 `stage_disallows_query` 或更上层流程阻断。
2. `confirmation_status = user_declined` 时：
   - governor 必须拒绝任何会继续推进该高成本路径的 query。
3. `confirmation_status = confirmed/not_needed` 时：
   - governor 按正常预算规则工作。

---

## 8. persistence 规则

### 8.1 bundle 文件固定为四件套

- `spec.md`
- `plan.md`
- `checklist.md`
- `progress.md`

### 8.2 source of truth 规则

1. bundle 与 runtime object 冲突时，以 bundle 为准。
2. 不允许只更新 `continuation_state` 而不更新 bundle。
3. `atomic_result_card` 不能替代 `progress.md`。

### 8.3 continuation refresh

只有 bundle 已 flush，才允许刷新 `continuation_state`。

### 8.4 v4 额外规则

- continuation 一旦存在，就必须绑定 `full_bundle`。
- 不存在合法的 `light_bundle` continuation。
- 若任务不需要 bundle，则根本不应创建 `continuation_state`。

---

## 9. dedupe 与 compaction

1. 相同 `dedupe_key` 的 plan 不重复执行。
2. 相同 `selector_plan` 命中缓存时直接复用 `kb-pack-response`。
3. 达到 `session_uncompacted_cap` 前，允许继续但必须安排 compaction。
4. 达到 `session_uncompacted_cap` 后，不允许继续累积长历史而不压缩。
