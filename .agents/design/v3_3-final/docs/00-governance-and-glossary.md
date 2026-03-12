# 00. Governance and Glossary

## 1. 范围

本文件定义三件事：

1. 包内文档之间的优先级。
2. 术语与命名的统一含义。
3. 版本升级与变更控制规则。

---

## 2. 变更控制

### 2.1 何时必须升 schema 版本

满足任意一条都必须升对象版本：

1. 新增 required 字段。
2. 删除现有字段。
3. 改变字段类型或枚举。
4. 改变条件约束，使此前合法对象变为非法。
5. 改变 stage 责任边界。

### 2.2 何时只改文档

以下情况不要求升 schema：

1. 新增解释性文字。
2. 增加 valid examples。
3. 增加 acceptance case。
4. 增加 CLI 示例但不改参数语义。

### 2.3 Codex 必须遵守的规则

1. 不能为实现方便新增未定义的隐式状态。
2. 不能绕过 `_shared/context-governor` 直接查 KB。
3. 不能绕过 `_shared/repo-kb-loader` 直接读 sqlite。
4. 不能把 `continuation_state` 当成 source of truth。
5. 不能把 `atomic_result_card` 当成 source of truth。
6. 不能以“上下文够大”为由放宽 query discipline。
7. `unknown` 永远优于猜测。

---

## 3. 术语表

### 3.1 family
顶层任务边界。family 决定“这个问题属于哪类决策边界”。

### 3.2 stage
运行阶段，固定为：`public_entry`、`intake`、`spec_plan`、`atomic`。

### 3.3 work package
最小可闭环工作包。

### 3.4 capsule
预算化、压缩过的知识切片。

### 3.5 deep reference
capsule 中附带的 source stub，可按需打开的小切片。

### 3.6 bundle
跨轮任务的持久化文件目录。真正的 source of truth 在 bundle markdown 中。

### 3.7 selector seed
`Public Entry` 的正式输出。保存首跳最小槽位和确认门信息。

### 3.8 selector plan
`Intake` 或 `Spec/Plan` 的正式查询许可对象。现在显式区分：

- `origin_stage`：谁创建了 plan
- `query_stage`：谁消费预算、适用哪套 governor 规则

### 3.9 atomic result card
单个 atomic 的轻量结果对象。用于 flush、reroute、compaction，不取代文件。

### 3.10 continuation state
复杂任务的最小续跑对象。必须从 bundle 派生。

### 3.11 confirmation gate
进入高成本、多步、代码改动或远端动作前的显式确认协议。`pending`/`user_declined` 时不得越过 Intake。

### 3.12 reroute
family 或 stage 重判。必须结构化表达，不能靠自然语言暗示。

### 3.13 logical domain
面向 skill 的知识域，例如：`deployment_config`、`troubleshooting`、`vllm_upstream`。

### 3.14 physical shard
KB 物理存储层的 shard family，例如：`repo_semantics`、`vllm_symbols`、`cann_op_constraints`。

### 3.15 resolve / build-local / pack
`resolve` 负责选 shard；`build-local` 负责合并成本地 SQLite；`pack` 负责把 facts 编译成预算化 capsule。

---

## 4. 命名规范

1. schema version 使用 `object-name/vN`。
2. package version 使用 `vX.Y-final`。
3. stage、family、logical domain 一律用小写 snake_case。
4. skill id 一律用小写 kebab-case。

---

## 5. source of truth 层次

1. 对象层：`schema/*.json`
2. 合同层：`docs/06-interface-contracts.md`
3. 规则层：master spec / routing / governor / persistence
4. 事实层：KB compiled facts、capsules、bundle markdown

层间规则：

- 对象层只管结构与条件约束。
- 规则层只管允许做什么。
- 事实层不允许反向修改对象语义。
