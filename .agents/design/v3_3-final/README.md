# vLLM-Ascend Agent Design Package v3.3 Final

本包是**可直接交给 Codex 开发**的最终设计产物。它是一个独立包，不依赖旧 draft；旧稿只可作为历史参考，不能反向修改本包。

这版修复了 v3.2 剩余的最后一个 shared-runtime 分叉口：

1. `context-governor` 的 stage 真相源彻底收敛为一个：`evaluate_governor` 不再接受显式 `stage` 输入；正式只从 `selector_plan.query_stage` 派生 effective stage。
2. `governor-decision.stage` 被明确为派生回显字段，不是第二真相源。
3. `Public Entry` 与 `Intake` 的 `direct_answer/no-query` 路径不再调用 governor，避免为了凑 stage 而制造伪调用。
4. contract validator 新增单一 stage 真相源 lint，防止文档和接口回退。
5. 首波 MVP capability 维持不变，`debugging` 继续留在 `P3B`。

---

## 1. 单一主真相源规则

优先级从高到低固定如下：

1. `schema/*.json`
2. `docs/06-interface-contracts.md`
3. `docs/01-master-spec.md`
4. `docs/02-routing-and-family-boundaries.md`
5. `docs/03-runtime-object-model.md`
6. `docs/04-context-governor-and-persistence.md`
7. `docs/05-knowledge-base-architecture.md`
8. `docs/07-examples-and-acceptance.md`
9. `docs/08-codex-development-plan.md`
10. `docs/09-supersede-and-migration.md`
11. `examples/*.json`
12. `sql/merged_pack.sql`
13. `tasks/codex_backlog.yaml`
14. `requirements.txt`

规则：

- schema 与文档冲突时，以 schema 为准。
- docs/06 与其他文档冲突时，以 docs/06 为准。
- examples 只用于说明与验收，不得反向创造 schema 外语义。
- Codex 不允许从任何旧文档回填字段或行为。

---

## 2. 本版冻结结论

1. 公开入口只有一个：`vllm-ascend-assistant`。
2. stable family 固定为 8 个：
   - `deployment_execution`
   - `performance_analysis`
   - `debugging`
   - `design_analysis`
   - `adaptation`
   - `operator_development`
   - `upstream_sync`
   - `validation_strategy`
3. 四层运行模型固定为：`Public Entry -> Intake -> Spec/Plan -> Atomic`。
4. 默认主路径仍是：
   - `Public Entry -> Intake`
   - `Public Entry -> Intake -> Atomic`
5. 只有 `Spec/Plan` 可以编排多个 atomic。
6. bundle markdown 是唯一 source of truth；runtime objects 只是压缩视图。
7. shared runtime 固定为：
   - `_shared/evidence-normalizer`
   - `_shared/context-governor`
   - `_shared/repo-kb-loader`
8. 所有 stable skill 只能通过 `selector_plan -> governor -> loader -> pack` 访问知识。
9. governor 的唯一 stage 输入源是 `selector_plan.query_stage`；`governor-decision.stage` 只是派生回显。
10. `Public Entry` 与 `Intake` 的 `direct_answer/no-query` 路径不调用 governor。
11. 实施顺序固定为 **MVP-first**。
12. confirmation gate 是正式协议：`pending` 或 `user_declined` 时，不得越过 Intake 进入 `Atomic` 或 `Spec/Plan`。
13. continuation 一旦存在，就必须绑定固定四件套 bundle；不存在“light bundle continuation”。
14. 首波 MVP 不再把 `debugging` 作为阻塞项；`debugging` 是 stable family，但排在 P3B。

---

## 3. 本版 runtime object 版本

本包冻结的 4 个核心 runtime object 为：

- `selector_seed/v3`
- `selector_plan/v4`
- `atomic_result_card/v3`
- `continuation_state/v4`

辅助接口对象保持：

- `governor-decision/v1`
- `kb-resolve-result/v1`
- `kb-pack-request/v2`
- `kb-pack-response/v1`

---

## 4. 本包目录

```text
vllm-ascend-agent-design-package-v3_3-final/
  README.md
  requirements.txt
  docs/
  schema/
  examples/
  sql/
  tasks/
  tools/
  references/
  vllm_ascend_design_package_v3_3_summary.md
```

`docs/09-supersede-and-migration.md` 明确规定了如何替换旧稿。

`tools/validate_design_package.py` 是正式的包级校验脚本：

- 校验全部 schema 是否可加载
- 校验全部 examples 是否通过
- 跑关键负例，确保条件约束真正生效
- 初始化 `merged_pack.sql` 做 SQLite smoke test
- 解析 `codex_backlog.yaml`
- lint governor 的单一 stage 真相源合同，防止接口回退

---

## 5. Codex 必读顺序

必须按下面顺序阅读并实现：

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
12. `tasks/CODEX_START_HERE.md`
13. `tasks/codex_backlog.yaml`

---

## 6. 第一组命令

在把任何文件落到目标 repo 之前，先运行：

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python tools/validate_design_package.py
```

只有 validator 全绿，才允许开始实现。

---

## 7. 一句话实施策略

**先冻结 contract 与 validator，再实现 shared runtime，再跑通 repo-only KB，再跑通首波 MVP capability，随后补 deferred debugging family 与高级 family。**
