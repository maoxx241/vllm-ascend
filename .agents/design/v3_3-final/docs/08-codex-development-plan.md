# 08. Codex Development Plan

## 1. 开发原则

1. **严格顺序推进**：不允许大规模跨阶段并行修改。
2. **先合同后实现**：先让 schema、validator、SQL、接口、fixtures 成立，再写复杂逻辑。
3. **MVP-first**：优先完成用户高频 capability，不先追求全部 stable family 的完全性。
4. **repo-only first**：先让 repo overlay + minimal validation 闭环成立，再补 vLLM pair 与 substrate。
5. **unknown over guess**：遇到缺口，优先输出 `unknown` 和显式缺口。

---

## 2. 阶段总览

| Phase | 目标 | 为什么现在做 |
| --- | --- | --- |
| P0 | 合同冻结与 validator | 先让规范真正可执行 |
| P1 | shared runtime minimum | 所有 family 都依赖它 |
| P2 | repo-only KB 闭环 | MVP 必须有可用 KB |
| P3 | first-wave MVP capability integration | 先满足高频用户能力 |
| P3B | deferred debugging integration | `debugging` 稳定但不阻塞首波 MVP |
| P4 | vLLM pair enrichment | 为 upstream/design/adaptation 提供可靠上游语义 |
| P5 | advanced family integration | 扩展到代码改动与复杂规划 |
| P6 | substrate + hardening | 完成高复杂度事实层与系统加固 |

---

## 3. P0：contract freeze

### 3.1 目标

固定 package 骨架、schema、examples、SQL、validator、测试入口。

### 3.2 必交付

- `schema/*.json`
- `examples/*.json`
- `sql/merged_pack.sql`
- `tools/validate_design_package.py`
- `requirements.txt`
- `tasks/acceptance_matrix.md`
- schema validation tests
- SQLite init smoke test

### 3.3 完成标准

1. 所有 examples 通过 schema 校验。
2. validator 的关键负例全部失败。
3. SQL DDL 可初始化空库。
4. 文档与 schema 无冲突。
5. bootstrap 命令可直接把 validator 跑起来。

---

## 4. P1：shared runtime minimum

### 4.1 目标

先实现任何 family 都要复用的最小运行时。

### 4.2 子任务

#### P1-T001 evidence-normalizer
要求：

- 能构建 `selector_seed/v3`
- 归一化 fields deterministic
- 能正确设置 confirmation gate

#### P1-T002 context-governor
要求：

- 实现 `query_stage` 矩阵
- 实现 stage caps / workflow caps
- 实现 flush gating / burst gating
- 实现 pending confirmation 阻断
- 实现 `origin_stage + query_stage + execution_mode` 一致性检查
- governor 只能从 `selector_plan.query_stage` 派生 effective stage；不接受显式 `stage` 输入

#### P1-T003 repo-kb-loader shell
要求：

- 实现 `selector_plan/v4 -> kb-pack-request/v2` 编译器
- 支持调用 `resolve/build-local/pack`
- 支持 `unknown` 降级

#### P1-T004 generic fallbacks
要求：

- `generic-task-intake` 使用正式 runtime objects
- `generic-task-intake` 执行 confirmation gate
- `generic-spec` 产出 `continuation_state/v4`
- `generic-analysis-checklist` 产出 `atomic_result_card/v3`

---

## 5. P2：repo-only KB 闭环

目标：在没有 substrate、没有完整上游 pair 的前提下，先让 repo 侧知识可用。

必做：

- `resolve`
- `build-local`
- `pack`
- `doctor`

完成标准：

1. `resolve -> build-local -> pack` repo-only 闭环成立。
2. 无 substrate 时也能生成最小 pack。
3. deployment / perf-breakdown / model-expectation / validation 的最小查询可跑通。

---

## 6. P3：first-wave MVP capability integration

### 6.1 首波 MVP capability

1. `deployment_execution / feature-policy-resolver`
2. `performance_analysis / single-profile-breakdown`
3. `performance_analysis / model-expected-performance-estimator`
4. `validation_strategy / change-impact-test-selector`

### 6.2 子任务与验收

#### deployment_execution
- `deployment-intake`
- `feature-policy-resolver`
- `deployment-config-synthesizer`
- `deployment-artifact-packager`

验收：
- 能确认 baseline/policy
- 能生成最小配置/脚本草案
- 遇到 code change requirement 会 reroute

#### performance_analysis / profile breakdown
- `perf-intake`
- `single-profile-breakdown`
- `comparative-profile-breakdown`

验收：
- 能做单 profile breakdown
- 能做 baseline/current 对照
- 证据不足时退化为 `partial`

#### performance_analysis / model expectation
- `perf-intake`
- `model-expected-performance-estimator`

验收：
- 能根据 model/hw/parallelism/config 输出 expected envelope
- 能显式写出关键假设与敏感因子
- 不把 expectation 冒充为实测结果

#### validation_strategy
- `validation-strategy-intake`
- `change-impact-test-selector`
- `coverage-gap-analyzer`

验收：
- 能根据 diff / feature / asset 选择最小必跑集
- 资产不足时能输出低置信补采建议
- 不越界去做 bug root cause

---

## 7. P3B：deferred debugging integration

### 7.1 目标

把 `debugging` family 放到首波 MVP 之后实现，避免它占用当前排期，但保留同一 contract。

### 7.2 子任务与验收

#### debugging
- `debug-intake`
- `log-triage`
- `cross-log-correlation`

验收：
- 能处理错误签名、日志、最近变更
- 缺 exact 证据时能显式输出 compatible triage
- 第二个 source 只能在 flush 之后请求

---

## 8. P4：vLLM pair enrichment

子任务：

- `vllm_semantics_ingest`
- `vllm_symbols_ingest`
- `vllm_release_delta_ingest`
- resolve matrix update
- pack ranking update

---

## 9. P5：advanced family integration

family：

- `adaptation`
- `upstream_sync`
- `design_analysis`
- `operator_development`

核心要求：

- code-change family 必须受 confirmation gate 管控
- route 未定时必须能回退到 `design_analysis`
- `Spec/Plan` 与 bundle 同步必须成立

---

## 10. P6：substrate + hardening

目标：

- 接入 substrate / CANN / runtime constraint 层
- 做 doctor / validator / cache / pack ranking 的系统加固
