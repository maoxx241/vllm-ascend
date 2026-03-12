# 02. Routing and Family Boundaries

## 1. Public Entry 必须收敛的槽位

`Public Entry` 只收敛：

1. `objective`
2. `requested_artifact`
3. `target_object`
4. `constraints`
5. `evidence_inventory`
6. `code_change_expectation`
7. `execution_context`
8. `confirmation_required / confirmation_status`

---

## 2. 8 个 family 的最终边界

| family | 主问题 | 什么时候成立 | 什么时候必须离开 |
| --- | --- | --- | --- |
| `deployment_execution` | 在不改代码前提下交付部署结果 | 有稳定 baseline，需求落在脚本 / 配置 / 参数 / 验证层 | 需要改代码，转 `adaptation`；需要算子，转 `operator_development`；路线未定，转 `design_analysis` |
| `performance_analysis` | 解释性能现象或估算预期性能 | 重点是 profile / metrics / baseline 对照，或根据 comparable baseline 做 expected envelope | 主问题变成 crash/root-cause 时转 `debugging`；变成路线设计时转 `design_analysis` |
| `debugging` | 定位、分诊、止血 | 重点是错误、日志、失败模式、最近变更 | 主问题变成纯性能解释时转 `performance_analysis`；变成路线选择时转 `design_analysis` |
| `design_analysis` | 路线怎么选 | 当前矛盾是 trade-off、冲突或架构方案 | 路线收敛后，按落点转 `adaptation` / `operator_development` / `upstream_sync` |
| `adaptation` | 已知路线下接模型/特性并跑通 | 需要代码改动，但关键缺口不是算子能力 | 若发现关键缺口是算子能力，转 `operator_development`；若路线重新失稳，转 `design_analysis` |
| `operator_development` | 实现或补齐算子能力 | 缺口已经确定落在 op contract / backend / binding | 若连是否需要新算子都未定，先回 `design_analysis` |
| `upstream_sync` | 必须同步时，分析怎么同步 | 已有 commit / PR / tag / release 范围 | 若真正问题变成 feature 路线或整体设计矛盾，转 `design_analysis` |
| `validation_strategy` | 设计验证矩阵与裁剪门禁 | 主问题是“测什么、怎么裁剪、怎么治理覆盖” | 若结论收敛成具体 bug，转 `debugging`；若收敛成路线冲突，转 `design_analysis` |

说明：

- `performance_analysis` 在本版内正式包含两类 capability：
  1. `profile breakdown`
  2. `model expected performance analysis`
- `debugging` 仍是 stable family，但不再是首波 MVP 阻塞项。

---

## 3. execution mode 判定规则

### 3.1 `direct_answer`

同时满足才允许：

1. 已有稳定答案或稳定 baseline。
2. 不需要补前置工件。
3. 不需要 code change。
4. 不存在显著兼容性未知项。

### 3.2 `direct_atomic_workflow`

同时满足时优先：

1. 主问题只需要一个 work package。
2. 证据达到 `analysis_ready`。
3. 路线基本收敛。
4. 不需要 continuation state 才能完成主结论。

### 3.3 `spec_plan_workflow`

满足任意两条时必须升级：

1. 路线未定。
2. 需要多个 atomic。
3. 需要多轮补证据或复现实验。
4. 需要显式 stop conditions。
5. 需要显式 reroute conditions。
6. 需要 continuation state。

---

## 4. confirmation gate 的 family 级规则

| family / 情况 | 是否默认需要确认 | 说明 |
| --- | --- | --- |
| `deployment_execution` 分析型 | 否 | 纯部署分析或 artifact synthesis 不需要 |
| `performance_analysis` | 否 | 纯分析型 |
| `debugging` | 否 | 纯分析/triage 型 |
| `validation_strategy` | 否 | 纯测试选择与覆盖分析 |
| `design_analysis` | 否 | 纯路线分析 |
| `adaptation` | 是 | 本质是 code-change path |
| `operator_development` | 是 | 本质是 code-change path |
| `upstream_sync` 纯 delta 分析 | 否 | 仅分析影响面时不需要 |
| `upstream_sync` 代码同步 / 回灌 | 是 | 进入 repo mutation path 时必须确认 |
| 任意 family 但需要 remote execution / destructive action | 是 | family 之外的额外确认门 |

---

## 5. confirmation gate 的流程语义

### 5.1 `pending`

- 允许：停在 Intake、向用户请求确认、返回确认原因。
- 禁止：
  - 进入 `Spec/Plan`
  - 进入 `Atomic`
  - 产生 `query_stage = atomic/spec_plan` 的 `selector_plan`
  - 静默把确认门当成已经通过

### 5.2 `confirmed`

- 允许正常进入 `Atomic` 或 `Spec/Plan`。

### 5.3 `user_declined`

- 必须终止需要确认的那条路径。
- 不得继续查询、不得生成 code-change work package。
- 可返回两类结果：
  1. 分析性解释：告诉用户如果要继续需要确认什么。
  2. 替代性分析：仅输出影响面/风险说明，但不进入变更执行路径。

---

## 6. canonical routes

### 6.1 deployment baseline 确认
`vllm-ascend-assistant -> deployment-intake -> feature-policy-resolver`

结论：`deployment_execution + direct_atomic_workflow`

### 6.2 single profile breakdown
`vllm-ascend-assistant -> perf-intake -> single-profile-breakdown`

结论：`performance_analysis + direct_atomic_workflow`

### 6.3 model expected performance envelope
`vllm-ascend-assistant -> perf-intake -> model-expected-performance-estimator`

结论：`performance_analysis + direct_atomic_workflow`

### 6.4 minimal test selection
`vllm-ascend-assistant -> validation-strategy-intake -> change-impact-test-selector`

结论：`validation_strategy + direct_atomic_workflow`

### 6.5 runtime error triage（deferred after first-wave MVP）
`vllm-ascend-assistant -> debug-intake -> log-triage`

结论：`debugging + direct_atomic_workflow`

### 6.6 multi-feature route conflict
`vllm-ascend-assistant -> design-analysis-intake -> design-analysis-spec`

结论：`design_analysis + spec_plan_workflow`

### 6.7 whole release sync
`vllm-ascend-assistant -> upstream-sync-intake -> upstream-sync-plan`

结论：`upstream_sync + spec_plan_workflow`
