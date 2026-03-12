# 05. Knowledge Base Architecture

## 1. 目标

`.agents/kb` 的目标不是“把所有文档塞给模型”，而是：

1. 以 repo 为中心构建一个 deterministic 的事实编译系统。
2. 同时纳入：
   - `vllm-ascend` 自身语义
   - 配套 `vLLM` 语义与 symbol 面
   - Ascend substrate / CANN / torch_npu 约束
   - validation 事实
3. 让 skills 只读取预算化 capsule，不读取事实原件。

---

## 2. 分层模型

```text
Layer 1  shared_substrate
  hw_soc_detail
  hw_runtime_caps
  cann_op_proto
  cann_op_constraints
  torch_npu_bindings

Layer 2  vllm_pair
  vllm_semantics
  vllm_symbols
  vllm_release_delta

Layer 3  repo_overlay
  repo_custom_ops
  repo_semantics

Layer 4  validation
  local / ci / nightly facts

Layer 5  runtime_pack
  merged.sqlite

Layer 6  context_projection
  capsules / atoms / deep reference stubs
```

固定结论：

- merged SQLite 是本地唯一查询入口。
- capsule 是 skills 的唯一默认读物。
- pack 是 stable skills 允许依赖的唯一正式知识读路径。

---

## 3. 逻辑 domain 与物理 shard

## 3.1 logical domain

| logical domain | 含义 |
| --- | --- |
| `knowledge_governance` | catalog、schema、resolver、provenance |
| `vllm_upstream` | vLLM 语义、symbols、release delta |
| `vllm_ascend_core` | repo 自身语义、patch、custom ops overlay |
| `ascend_foundation` | SoC、runtime、CANN、torch_npu bindings |
| `integration_core` | 上游语义与 repo overlay 的组合事实 |
| `deployment_config` | 配置、策略、已知 baseline、部署经验 |
| `troubleshooting` | 失败模式、错误签名、workaround、triage 证据 |
| `validation_evidence` | smoke / e2e / CI / nightly 事实 |

## 3.2 physical shard family

| shard family | 主职责 |
| --- | --- |
| `hw_soc_detail` | SoC 静态画像 |
| `hw_runtime_caps` | runtime 能力与环境摘要 |
| `cann_op_proto` | op 原型 |
| `cann_op_constraints` | op 约束、limit、variant |
| `torch_npu_bindings` | aten / torch API / backend symbol 绑定图 |
| `vllm_semantics` | 上游 feature / config / engine 语义 |
| `vllm_symbols` | 上游 symbol / file path / signature 索引 |
| `vllm_release_delta` | 上游版本差异 |
| `repo_custom_ops` | repo 自定义 op overlay |
| `repo_semantics` | repo 语义三元组 |
| `validation` | 本地 / CI / nightly 事实 |

## 3.3 映射规则

- `logical domain` 面向 skill 与设计层。
- `physical shard` 面向 resolver、loader 与 build-local。
- `integration_core` 默认不单独持久化为大 shard，而是在 pack 期动态投影。

---

## 4. resolver

## 4.1 输入

- 当前 repo branch / sha
- 配套 `vLLM` ref 或版本
- 当前 runtime tuple：
  - soc
  - CANN
  - PyTorch
  - torch-npu
  - Python
- catalog index
- `matrix.lock.json`

## 4.2 输出

- `kb-resolve-result/v1`

核心字段：

- `match_level`
- `selected_shards`
- `warnings`
- `missing`

## 4.3 匹配优先级

1. `exact`
2. `compatible`
3. `unknown`

### exact
版本 / sha / runtime tuple 全匹配。

### compatible
落在 `matrix.lock.json` 的 fallback 规则内。

### unknown
没有可信匹配。必须显式暴露，不能静默猜测。

## 4.4 重要结论

1. `unknown` 优于猜测。
2. `compatible` 不是“差不多就行”，而是明确被规则允许。
3. resolver 只负责选 shard，不负责 pack。

---

## 5. build-local

## 5.1 输入

- `kb-resolve-result/v1`
- selected shared shards
- repo overlay extractors
- validation extractors

## 5.2 输出

- `.agents/kb/local/merged/<pack_id>.sqlite`

## 5.3 规则

1. deterministic
2. 不把本地缓存反写进 repo catalog
3. repo-only 路径必须成立：即使 shared substrate 缺失，也能产出最小 pack
4. `pack_meta` 表必须记录 provenance

---

## 6. pack

## 6.1 输入

- merged SQLite
- `kb-pack-request/v2`

## 6.2 输出

- `kb-pack-response/v1`

## 6.3 pack 必须完成的事情

1. 根据 selectors 精确命中实体、事实、symbol、validation。
2. 按 `must_have` 优先排序。
3. 按 `nice_to_have` 做预算内补充。
4. 受 `budget_token_cap` 限制输出 capsule。
5. 返回：
   - `capsule_text`
   - `atoms`
   - `deep_reference_stubs`
   - `unknowns`

## 6.4 pack 明确不做的事情

1. 不做无界全文检索。
2. 不做长链推理。
3. 不把 sqlite 结果全量拼进 prompt。
4. 不伪造精确知识。

---

## 7. 为什么必须有 repo-only 闭环

MVP-first 的落地需要 repo-only 闭环先成立，原因有三：

1. 高价值任务很多只需要 repo + minimal validation 就能走通。
2. shared substrate harvest 不应阻塞 MVP capability set。
3. 先让 loader / governor / pack 行为稳定，再叠加 substrate 复杂度更安全。

因此：

- Phase 1 先做 `repo_semantics` + `repo_custom_ops` + minimal `validation`
- Phase 2 再接 MVP capability set
- 之后再补 `vllm_pair` 与 shared substrate

---

## 8. 本地目录规范

```text
.agents/
  kb/
    schema/
    catalog/
    extractors/
    rules/
    tools/
    sql/
    local/
      resolve.json
      merged/
        <pack_id>.sqlite
      capsules/
        <request_id>.json
```

固定规则：

1. `local/` 必须 gitignore。
2. repo 内提交 catalog；运行时产物不提交。
3. stable skill 不得直接读取 `catalog/` 或 `merged/`。

---

## 9. SQL 模型

正式 DDL 见 `sql/merged_pack.sql`。  
核心表为：

- `pack_meta`
- `sources`
- `entities`
- `facts`
- `edges`
- `symbol_index`
- `validations`
- `capsules`

设计目的：

- `entities` / `facts` / `edges` 负责通用事实图
- `symbol_index` 负责 code surface 命中
- `validations` 负责测试与失败模式
- `capsules` 负责 pack 缓存与 provenance

---

## 10. 失败降级

| 失败点 | 允许行为 | 禁止行为 |
| --- | --- | --- |
| resolve `unknown` | 返回 `unknowns`，必要时退到 repo-only | 猜 compatible |
| build-local 失败 | 保留 repo-only fallback 或 generic path | 假装 local pack 可用 |
| pack 超预算 | 缩小 domains / selectors / atoms | 强行返回超长 capsule |
| 某域 facts 缺失 | 显式 knowledge miss | 伪造 facts |
| deep ref 太大 | 返回 stub，不直接展开 | 直接把长 source 塞进上下文 |

---

## 11. 缓存规则

1. `resolve` 可缓存，key 至少包含：
   - repo sha
   - paired vLLM ref
   - runtime tuple
2. `build-local` 可缓存，key 至少包含：
   - resolve result hash
   - repo overlay extractor version
3. `pack` 可缓存，key 至少包含：
   - `selector_plan`
   - `governor_decision.resolved_token_cap`
   - merged pack id

---

## 12. 对高级 family 的支持

### 12.1 `design_analysis`
最依赖 `vllm_pair` 与 `integration_core` 投影。  
不能只靠 repo overlay。

### 12.2 `operator_development`
最依赖 `ascend_foundation`。  
shared substrate 不完善时，只能给 contract gap，而不能 pretending 可直接实现。

### 12.3 `upstream_sync`
最依赖 `vllm_release_delta` 与 `vllm_symbols`。  
否则影响面分析会漂。

---

## 13. 最终结论

1. `.agents/kb` 是编译型事实系统，不是检索即席拼装系统。
2. `resolve -> build-local -> pack` 是唯一正式运行时链路。
3. logical domain 面向任务语义，physical shard 面向实现。
4. repo-only 闭环是 MVP 的前提，不是退而求其次的临时方案。
