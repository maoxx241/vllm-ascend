# Demo Talk Track (10-15 min)

## 0. Opening (1 min)

- 今天演示的是“弱模型也能稳定执行”的部署 skill。
- 输入是中文自然语言，输出是可执行部署包。

## 1. Natural Language Input (2 min)

Demo input:

```text
帮我部署Qwen3-32B-W8A8，开图、开量化、开权重预取，tp4，最后给我验证命令。
```

Expected behavior:

1. Normalize terms to canonical features.
2. If ambiguous, ask one clarification.
3. Render deterministic deployment package.

## 2. Term Normalization (2 min)

Run:

```bash
python .agents/skills/vllm-ascend-deployment-assistant/scripts/normalize_terms.py \
  --text "帮我部署Qwen3-32B-W8A8，开图、开量化、开权重预取，tp4"
```

Explain:

- `开图 -> graph_mode`
- `开量化 -> quantization`
- `开权重预取 -> weight_prefetch`
- `tp4 -> tensor_parallel`

## 3. Deployment Package Rendering (3 min)

Run:

```bash
python .agents/skills/vllm-ascend-deployment-assistant/scripts/render_deploy_package.py \
  --text "帮我部署Qwen3-32B-W8A8，开图、开量化、开权重预取，tp4" \
  --model-profile qwen3-32b-w8a8 \
  --output-dir /tmp/vllm_demo_pkg
```

Show generated files:

- `/tmp/vllm_demo_pkg/start.sh`
- `/tmp/vllm_demo_pkg/validate.sh`
- `/tmp/vllm_demo_pkg/rollback.sh`
- `/tmp/vllm_demo_pkg/deployment_plan.json`

## 4. Start + Validate (3 min)

```bash
bash /tmp/vllm_demo_pkg/start.sh
bash /tmp/vllm_demo_pkg/validate.sh
```

Success criteria:

- `/v1/models` returns 200
- `/v1/chat/completions` returns 200 and non-empty text

## 5. Failure Rollback Demo (2 min)

- Simulate an error (port occupied or invalid model path).
- Show one-step rollback:

```bash
bash /tmp/vllm_demo_pkg/rollback.sh
```

Then rerender with corrected args and restart.

## 6. Closing (1 min)

- 弱模型可用的关键不是“更聪明”，而是“更强约束 + 更强知识索引 + 更确定脚本”。
