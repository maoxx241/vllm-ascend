# 01. Master Spec

## 1. 设计目标

本系统只追求五个结果：

1. 把 vLLM-Ascend 请求稳定路由到正确 family。
2. 把知识访问限制成小、准、可追踪的 budgeted slice。
3. 让复杂任务依赖 bundle 恢复，而不是依赖上下文记忆。
4. 让 reroute、flush、confirmation 都变成结构化协议。
5. 让 Codex 可按单一设计包直接实现，不再从多份草稿中猜。

---

## 2. 顶层不变量

1. 公开入口只有一个：`vllm-ascend-assistant`。
2. stable family 固定为 8 个。
3. `direct_answer` 由 `Intake` 直接完成，不派生独立 atomic。
4. 只有 `Spec/Plan` 可以编排多个 atomic。
5. stable skill 不允许直接读取 raw catalog、raw sqlite、raw shard。
6. `Public Entry` 默认零查询。
7. 查询是结果，不是前提。
8. 复杂任务必须依赖 persistence bundle。
9. reroute 必须显式结构化输出。
10. bundle markdown 是唯一 source of truth。
11. confirmation gate 是正式阻断器：`pending` 或 `user_declined` 时，不得进入 `Atomic` 或 `Spec/Plan`。

---

## 3. 共享运行时不变量

shared runtime 固定为三件套：

- `_shared/evidence-normalizer`
- `_shared/context-governor`
- `_shared/repo-kb-loader`

职责不可互换：

- normalizer 负责收口、归一化、confirmation seed
- governor 负责许可、预算、flush、compaction
- loader 负责 `resolve -> build-local -> pack` 桥接

---

## 4. 预算不变量

在 256k context 前提下，预算固定为：

### 4.1 阶段硬上限

- `routing_capsule <= 400`
- `intake_capsule <= 1200`
- `spec_capsule <= 2400`
- `atomic_capsule <= 1500`
- `deep_reference slice <= 800`

### 4.2 工作流软上限

- `direct_atomic_workflow default <= 3200`
- `direct_atomic_workflow stretch <= 4800`
- `spec_plan_workflow default <= 9600`
- `spec_plan_workflow stretch <= 16000`
- `session_uncompacted_cap <= 51200`

---

## 5. 四层运行模型

### 5.1 Public Entry

职责：

1. 收敛 `objective / target_object / requested_artifact / constraints`。
2. 识别 family candidates。
3. 估算 evidence level。
4. 判断 confirmation gate 是否成立。
5. 产出 `selector_seed/v3`。

禁止：

- 长分析
- 多 atomic 编排
- 大 query
- 假装已经完成 intake

### 5.2 Intake

职责：

1. 把 `selector_seed` 收敛成正式 family。
2. 决定：`direct_answer` / `direct_atomic_workflow` / `spec_plan_workflow`。
3. 决定 `analysis_depth` 与 `deliverable_contract`。
4. 必要时生成 `selector_plan/v4`。
5. 强制执行 confirmation gate。

固定规则：

- `confirmation_status = pending` 时，Intake 只能停在确认门，不能产生 `query_stage = atomic/spec_plan` 的 plan。
- `confirmation_status = user_declined` 时，Intake 必须终止 code-change / remote path，不得查询。

### 5.3 Spec/Plan

职责：

1. 拆 work packages。
2. 决定执行顺序。
3. 维护 success / stop / reroute conditions。
4. 维护 `continuation_state/v4`。
5. 为 spec 或 atomic 产出 `selector_plan/v4`。

### 5.4 Atomic

职责：

1. 完成单 work package。
2. 消费一个主 capsule。
3. 只在 governor 允许下打开少量 deep refs。
4. 输出 `atomic_result_card/v3`。

---

## 6. Confirmation gate 总规则

### 6.1 何时成立

满足任意一条，`confirmation_required = true`：

1. 目标 deliverable 是 `code_change_pack`。
2. 目标 family 落在 `adaptation` 或 `operator_development`。
3. `upstream_sync` 的目标是代码同步而不是纯分析。
4. 需要进入 remote execution / repo mutation / destructive action。

### 6.2 状态机

- `not_needed`：无需确认，可正常推进。
- `pending`：等待确认，禁止越过 Intake。
- `confirmed`：允许进入 `Atomic` 或 `Spec/Plan`。
- `user_declined`：终止该条高成本路径，只能返回分析说明或等待用户重开。

---

## 7. 共享枚举

### 7.1 execution_mode
- `direct_answer`
- `direct_atomic_workflow`
- `spec_plan_workflow`

### 7.2 analysis_depth
- `none`
- `lightweight_design_note`
- `full_spec_plan`

### 7.3 deliverable_contract
- `reference_answer`
- `deployment_artifact_pack`
- `analysis_report`
- `design_note`
- `spec_plan`
- `code_change_pack`
