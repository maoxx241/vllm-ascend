# CODEX_START_HERE

## 1. 先读什么

必须按这个顺序读：

1. `README.md`
2. `docs/00-governance-and-glossary.md`
3. `docs/01-master-spec.md`
4. `docs/02-routing-and-family-boundaries.md`
5. `docs/03-runtime-object-model.md`
6. `docs/04-context-governor-and-persistence.md`
7. `docs/05-knowledge-base-architecture.md`
8. `docs/06-interface-contracts.md`
9. `docs/07-examples-and-acceptance.md`
10. `docs/08-codex-development-plan.md`
11. `docs/09-supersede-and-migration.md`
12. `tasks/codex_backlog.yaml`

不要跳过 `docs/02`、`docs/07`、`docs/09`。

---

## 2. 第一组命令

先运行：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python tools/validate_design_package.py
```

validator 没全绿前，不要开始写 family-local code。

---

## 3. 先做什么

第一批动作必须是：

1. 安装 schema 到 repo
2. 安装 examples 到测试夹具
3. 安装 `merged_pack.sql`
4. 安装 package validator
5. 安装 `requirements.txt`
6. 写 schema validation test
7. 写 SQL init smoke test

---

## 4. 绝对不要做什么

1. 不要直接让 skill 读 `.agents/kb/local/merged/*.sqlite`
2. 不要在 `Public Entry` 做大查询
3. 不要把 `atomic_result_card` 当成 source of truth
4. 不要只更新 `continuation_state` 而不更新 bundle
5. 不要让 `deployment_execution` 越过 code-change 边界
6. 不要在 resolver `unknown` 时猜 `compatible`
7. 不要绕开 `selector_plan -> governor -> loader -> pack` 链路
8. 不要绕过 confirmation gate 直接进入 `Atomic` 或 `Spec/Plan`
9. 不要让 `origin_stage = intake && query_stage = atomic` 的 plan 带 `spec_plan_workflow`
10. 不要生成任何非 `full_bundle` 的 continuation
11. 不要在 `evaluate_governor` 暴露显式 `stage` 输入；stage 只能来自 `selector_plan.query_stage`

---

## 5. MVP 优先级

Phase P3 的 4 个 capability 是首波 MVP：

- `deployment_execution / feature-policy-resolver`
- `performance_analysis / single-profile-breakdown`
- `performance_analysis / model-expected-performance-estimator`
- `validation_strategy / change-impact-test-selector`

`debugging` 仍是 stable family，但延后到 `P3B`。
