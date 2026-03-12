# 07. Examples and Acceptance

## 1. 为什么这份文件存在

本文件同时承担两个角色：

1. 讲清关键 flow，避免实现者只看 schema 却误解运行语义。
2. 给出最终验收标准，避免“实现出来但不可落地”。

---

## 2. 端到端示例 A：deployment_execution

场景：

- 用户要在 A2 上确认 `qwen3-next` 的默认 prefill policy
- 目标是部署交付物
- 不允许改代码

正确 flow：

```text
Public Entry
  -> selector-seed.deployment.json

Intake
  -> selector-plan.deployment.intake.json

Governor
  -> 允许一次 intake_capsule

Loader
  -> kb-pack-request
  -> kb-pack-response

Atomic: feature-policy-resolver
  -> atomic-result-card.complete.json
```

正确结论：

- family 保持在 `deployment_execution`
- 不需要 `Spec/Plan`
- 可以继续生成配置、脚本与最小验证步骤

---

## 3. 端到端示例 B：performance_analysis / profile breakdown

场景：

- 用户给出一次 prefill regression profile
- 目标是解释 TTFT / 吞吐差异
- 问题只需要一个 atomic work package

正确 flow：

```text
Public Entry
  -> selector_seed

Intake
  -> selector-plan.performance.atomic.json

Governor
  -> 允许 1 个 atomic_capsule

Atomic: single-profile-breakdown
  -> atomic-result-card.performance.partial.json
```

正确降级：

- baseline 缺失时可以输出 `partial`
- 不允许把缺口包装成“已确认根因”

---

## 4. 端到端示例 C：performance_analysis / model expected performance

场景：

- 用户尚未给出真实 profile
- 目标是估算给定模型、硬件、并行度与 context 下的预期 TTFT / 吞吐 / 显存头寸
- 问题只需要一个 atomic work package

正确 flow：

```text
Public Entry
  -> selector-seed.performance.expectation.json

Intake
  -> selector-plan.performance.expectation.atomic.json

Governor
  -> 允许 1 个 atomic_capsule

Loader
  -> kb-pack-request.model-expectation.json
  -> kb-pack-response.model-expectation.json

Atomic: model-expected-performance-estimator
  -> atomic-result-card.performance.expectation.complete.json
```

正确边界：

- 这是 expectation path，不是实测 path
- 可以输出范围和关键假设，不得伪造单点精确值
- 当用户真正要解释异常 profile 时，应回到 profile breakdown capability

---

## 5. 端到端示例 D：validation_strategy

场景：

- 用户给出一个 diff
- 目标是收口最小必跑集
- 不要求 bug root cause

正确 flow：

```text
Public Entry
  -> selector_seed

Intake
  -> selector-plan.validation.atomic.json

Governor
  -> 允许 1 个 atomic_capsule

Atomic: change-impact-test-selector
  -> atomic-result-card.validation.complete.json
```

正确边界：

- 可以输出最小测试集与补采建议
- 不得越界去做 bug triage 或性能归因

---

## 6. 端到端示例 E：design_analysis

场景：

- 多 feature 组合冲突
- 主问题是“路线怎么选”

正确 flow：

```text
Public Entry
  -> selector_seed

Intake
  -> family = design_analysis
  -> execution_mode = spec_plan_workflow

Spec/Plan
  -> selector-plan.design.spec.json
  -> continuation_state
```

正确行为：

- 路线未定时不能跳过 `Spec/Plan`
- atomic 若发现 family 已变，必须 `needs_reroute`

---

## 7. 端到端示例 F：upstream_sync

场景：

- 已知必须同步一段上游 delta
- 需要整理影响面与验证窗口

正确 flow：

```text
Public Entry
  -> selector_seed

Intake
  -> family = upstream_sync

Spec/Plan
  -> continuation-state.upstream-sync.json
```

正确 bundle 更新：

- `spec.md`：同步目标与边界
- `plan.md`：work packages 与顺序
- `checklist.md`：当前待做与 blocker
- `progress.md`：每次 delta / mapping 尝试记录

---

## 8. confirmation gate 示例

### 8.1 pending

`selector-seed.adaptation.pending-confirmation.json` 表示：

- family 倾向 `adaptation`
- deliverable 是 `code_change_pack`
- `confirmation_status = pending`

正确行为：

- 允许：停在 Intake，请用户确认
- 禁止：直接发 `query_stage = atomic/spec_plan` 的 plan

### 8.2 user_declined

`selector-seed.upstream.user-declined.json` 表示：

- 用户拒绝进入 code-change path

正确行为：

- 不得发任何继续推进该路径的 query
- 只能返回影响面说明或等待用户重开

---

## 9. acceptance 总则

Codex 交付必须同时满足：

1. schema 全部可解析
2. examples 全部通过 schema 校验
3. 关键负例被 schema 或 validator 拒绝
4. `resolve -> build-local -> pack` 闭环成立
5. governor 能正确拒绝错误时机的 query
6. bundle 写入与 reroute 闭环成立
7. 首波 4 个 MVP capability 都有 canonical flow
8. deferred `debugging` family 也有规范 flow，但不阻塞首波 MVP 开工
9. governor 的 stage 只能从 `selector_plan.query_stage` 派生，不存在第二真相源

---

## 10. phase-level acceptance

### 10.1 P0：contract freeze

- 所有 schema 可加载
- 所有 examples 可校验
- validator 的关键负例全部被拒绝
- `merged_pack.sql` 可初始化空库

### 10.2 P1：shared runtime minimum

- 能从原始请求得到 `selector_seed/v3`
- governor 能给出 `governor-decision/v1`
- governor 只从 `selector_plan.query_stage` 派生 stage
- `Public Entry` 与 Intake 的 no-query path 不调用 governor
- loader 能把 `selector_plan/v4` 编译成 `kb-pack-request/v2`
- Intake 会执行 confirmation gate

### 10.3 P2：repo-only KB

- 无 substrate shard 时，`build-local` 仍可生成最小 merged pack
- `pack` 能从 repo overlay 输出 budgeted capsule
- `unknown` / pack miss 能被显式返回

### 10.4 P3：first-wave MVP capability integration

必须通过：

- `deployment_execution / feature-policy-resolver`
- `performance_analysis / single-profile-breakdown`
- `performance_analysis / model-expected-performance-estimator`
- `validation_strategy / change-impact-test-selector`

四类 canonical capability 全部跑通。

### 10.5 P3B：deferred debugging integration

- `debugging / log-triage`
- `debugging / cross-log-correlation`

这部分在首波 MVP 之后交付，但仍必须遵守同一 contract。

### 10.6 P4+：advanced family integration

- reroute 到 `adaptation` / `operator_development` / `design_analysis`
- `upstream_sync` 与 `design_analysis` 的 `Spec/Plan` 流程成立
- `continuation_state` 与 bundle 同步成立

---

## 11. 必测负例

1. `needs_reroute` 结果卡没有 `reroute`
2. `needs_reroute` 结果卡没有 `next_action.kind = reroute_task`
3. `query_stage = intake` 却给了 `atomic` 预算
4. `query_stage = intake` 却打开 deep refs
5. `origin_stage = intake && query_stage = atomic` 却给 `spec_plan_workflow`
6. `confirmation_required = true` 却仍写 `confirmation_status = not_needed`
7. `confirmation_status = pending` 还继续进入 Atomic
8. `confirmation_status = user_declined` 还继续 query
9. 只更新 `continuation_state` 不更新 bundle
10. continuation_state 仍试图使用 `light_bundle` 或非 `full_bundle`
11. code-change family 仍设置 `analysis_depth = none`
12. `evaluate_governor` 仍暴露显式 `stage` 参数，或允许 caller stage 与 `selector_plan.query_stage` 分叉

---

## 12. 完成定义（Definition of Done）

只有同时满足以下条件，才算正确落地：

1. shared runtime 三件套已实现。
2. `.agents/kb` 的 repo-only 闭环已实现。
3. 首波 MVP capability 的 canonical flow 已实现并通过验收。
4. deferred `debugging` family 已排进后续 phase，不再隐式阻塞首波 MVP。
5. bundle 持久化与 reroute 协议已实现。
6. `selector_seed/v3`、`selector_plan/v4`、`atomic_result_card/v3`、`continuation_state/v4` 都被真实使用。
7. governor 的 stage 单一真相源已经落到正式接口与实现中。
