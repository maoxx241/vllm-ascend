# 06. Interface Contracts

## 1. 总原则

shared runtime 与 KB 的接口全部采用 **contract-first**：

1. 先稳定输入输出对象。
2. 再实现内部逻辑。
3. 内部实现可替换，但不能破坏 I/O 语义。

---

## 2. `_shared/evidence-normalizer`

### 2.1 职责

1. 从用户请求、附件元信息、路径、日志片段中提取最小槽位。
2. 归一化 models / features / hw / files / symbols / errors。
3. 估算 `evidence_level`。
4. 计算 confirmation gate。
5. 产出 `selector_seed/v3`。

### 2.2 输入

```python
class RawRequest(NamedTuple):
    request_id: str
    user_text: str
    attachment_refs: list[str]
    inline_paths: list[str]
    inline_symbols: list[str]
    inline_errors: list[str]
    execution_context_hint: str | None
```

### 2.3 输出

- `selector-seed/v3`

### 2.4 强约束

1. 不能做 deep query。
2. 不能产出长设计说明。
3. 不能替 Intake 冻结 final family。
4. 若 `confirmation_required = true`，必须同时填 `confirmation_reason_codes`。

---

## 3. `_shared/context-governor`

### 3.1 职责

1. 判断 query 是否允许。
2. 解析预算。
3. 判断是否必须先 flush。
4. 判断是否必须 compaction。
5. 生成 dedupe key。

### 3.2 输入

```python
class ProgressState(TypedDict):
    bundle_exists: bool
    has_unflushed_findings: bool
    query_count_in_stage: int
    opened_deep_refs_in_stage: int
    seen_dedupe_keys: list[str]
    last_flush_at: str | None
    session_budget_used: int

def evaluate_governor(
    *,
    selector_seed: dict,        # selector-seed/v3
    selector_plan: dict,        # selector-plan/v4; required for any real query
    continuation_state: dict | None,
    progress_state: ProgressState,
) -> dict:  # governor-decision/v1
    ...
```

### 3.3 核心输入语义

1. `evaluate_governor(...)` 只接受 query-bearing `selector_plan/v4`；没有 plan 时不允许调用。
2. 正式接口中不存在显式 `stage` 输入参数。
3. `governor-decision.stage` 必须等于 `selector_plan.query_stage`，且只是回显，不是第二真相源。
4. `Public Entry` 与 `Intake` 的 `direct_answer/no-query` 路径不调用 governor。

### 3.4 拒绝规则

`denial_reason_code` 只允许：

- `missing_trigger_code`
- `stage_disallows_query`
- `pending_flush`
- `budget_exceeded`
- `duplicate_plan`
- `unrelated_domain_expansion`
- `missing_persistence_bundle`

额外流程规则：

- `confirmation_status = pending/user_declined` 时，禁止通过 governor 放行 `query_stage = atomic/spec_plan`。
- `origin_stage = intake && query_stage = atomic` 的 plan 若带 `spec_plan_workflow`，必须拒绝。

---

## 4. `_shared/repo-kb-loader`

### 4.1 职责

1. 接受 `selector_plan/v4` + `governor-decision/v1`
2. 确保本地 resolve / merged pack 可用
3. 把 `selector_plan` 编译成 `kb-pack-request/v2`
4. 调用 `kb.py pack`
5. 返回 `kb-pack-response/v1`

### 4.2 输入输出

```python
def load_capsule(
    *,
    selector_plan: dict,        # selector-plan/v4
    governor_decision: dict,    # governor-decision/v1
    repo_root: str = ".",
) -> dict:                      # kb-pack-response/v1
    ...
```

### 4.3 规则

1. 若 local resolve 缺失，loader 可触发 `resolve`。
2. 若 merged pack 缺失，loader 可触发 `build-local`。
3. loader 不得绕开 governor 私自扩大预算。
4. loader 不得直接把 sqlite 结果全展开返回给 skill。
5. loader 必须显式返回 `match_level`、`warnings`、`unknowns`。

---

## 5. `.agents/kb/tools/kb.py`

### 5.1 `doctor`

```bash
python .agents/kb/tools/kb.py doctor --repo-root .
```

语义：

- 校验 schema 是否可加载
- 校验 examples 是否可通过 schema
- 校验 resolve/merged pack 是否健康

### 5.2 `resolve`

```bash
python .agents/kb/tools/kb.py resolve --repo-root . --emit .agents/kb/local/resolve.json
```

输出：`kb-resolve-result/v1`

### 5.3 `build-local`

```bash
python .agents/kb/tools/kb.py build-local \
  --repo-root . \
  --resolve .agents/kb/local/resolve.json \
  --emit-sqlite .agents/kb/local/merged/current.sqlite
```

### 5.4 `pack`

模型预期性能分析示例：

```bash
python .agents/kb/tools/kb.py pack \
  --repo-root . \
  --resolve .agents/kb/local/resolve.json \
  --merged-pack .agents/kb/local/merged/current.sqlite \
  --intent model_expectation \
  --domains validation_evidence deployment_config ascend_foundation \
  --models qwen3-next-32b \
  --features prefill decode tp4 bf16 ctx8k \
  --hw A2 \
  --versions vllm-ascend@0.13.0 \
  --configs tp4_bf16_ctx8k \
  --must-have "expected TTFT range" "expected throughput range" "memory headroom assumptions" \
  --nice-to-have "closest comparable baseline" "top sensitivity factors" \
  --budget-token-cap 1500 \
  --max-atoms 10 \
  --max-hops 1 \
  --include-evidence-stubs \
  --stop-after-first-sufficient \
  --emit .agents/kb/local/capsules/perf-exp-003.json
```

---

## 6. `selector_plan/v4 -> kb-pack-request/v2` 正式映射

| `selector_plan/v4` 字段 | `kb-pack-request/v2` 字段 | 映射规则 |
| --- | --- | --- |
| `request_id` | `request_id` | 直接复制 |
| `created_at` | `created_at` | loader 填当前时间或沿用 plan 时间 |
| `logical_domains` | `logical_domains` | 直接复制 |
| `physical_shard_hints` | `physical_shard_hints` | 直接复制 |
| `selectors` | `selectors` | 直接复制 |
| `must_have` | `must_have` | 直接复制 |
| `nice_to_have` | `nice_to_have` | 直接复制 |
| `requested_token_cap` | `budget_token_cap` | 先经 governor 解析，再写入 resolved cap |
| `hop_limit` | `max_hops` | 直接复制 |
| `stop_after_first_sufficient` | `stop_after_first_sufficient` | 直接复制 |
| `query_stage + task_family + consumer_id` | `intent` | 由 loader 按 intent mapping 编译 |
| `capsule_type` | `include_evidence_stubs` | 由 capsule 类型决定 stub 策略 |

### 6.1 intent mapping

| 条件 | `kb-pack-request.intent` |
| --- | --- |
| `query_stage = intake` | `intake_lookup` |
| `task_family = deployment_execution` 且 `query_stage = atomic` | `deployment_lookup` |
| `task_family = performance_analysis` 且 `consumer_id in {single-profile-breakdown, comparative-profile-breakdown}` | `perf_breakdown` |
| `task_family = performance_analysis` 且 `consumer_id = model-expected-performance-estimator` | `model_expectation` |
| `task_family = debugging` | `debug_triage` |
| `task_family = design_analysis` 且 `query_stage = spec_plan` | `design_lookup` |
| `task_family = adaptation` | `adaptation_codegen` |
| `task_family = operator_development` | `operator_codegen` |
| `task_family = upstream_sync` | `upstream_delta` |
| `task_family = validation_strategy` | `validation_selection` |

### 6.2 `include_evidence_stubs` 规则

- `intake_capsule`: `false`
- `spec_capsule`: `true`
- `delta_capsule`: `true`
- `atomic_capsule`: `true`
- `evidence_capsule`: `true`

---

## 7. bundle 写入接口（推荐实现）

```python
def ensure_task_bundle(task_id: str, persistence_mode: str) -> str:
    ...

def append_progress_entry(task_id: str, entry_markdown: str) -> None:
    ...

def update_plan_section(task_id: str, patch: dict) -> None:
    ...

def save_atomic_card(task_id: str, card: dict) -> str:
    ...

def save_continuation_state(task_id: str, state: dict) -> str:
    ...
```

约束：

1. `save_atomic_card` 不能替代 `append_progress_entry`。
2. `save_continuation_state` 不能替代 `update_plan_section`。
3. 所有写接口都必须幂等或可重入。
