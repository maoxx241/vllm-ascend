# Code Knowledge Map (L0)

This file maps runtime/deployment code paths to shared knowledge docs.

| Code or doc path | Knowledge doc | Why it matters |
| --- | --- | --- |
| `docs/source/user_guide/feature_guide/quantization.md` | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | Quantization terms and CLI flags |
| `docs/source/user_guide/feature_guide/graph_mode.md` | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | Graph-mode related aliases |
| `docs/source/user_guide/feature_guide/context_parallel.md` | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | Context-parallel semantics |
| `docs/source/tutorials/models/Qwen3-Dense.md` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | Primary profile defaults |
| `docs/source/tutorials/models/Qwen3-Next.md` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | Backup profile defaults |
| `tests/e2e/nightly/single_node/models/configs/Qwen3-32B-Int8.yaml` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | CLI baseline for 32B W8A8 |
| `tests/e2e/nightly/single_node/models/configs/Qwen3-Next-80B-A3B-Instruct-W8A8.yaml` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | CLI baseline for 80B-Next |
| `examples/run_dp_server.sh` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | DP/TP launch reference |
| `.agents/skills/vllm-ascend-deployment-assistant/scripts/normalize_terms.py` | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | Canonical feature mapping |
| `.agents/skills/vllm-ascend-deployment-assistant/scripts/render_deploy_package.py` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | Deterministic package generation |

Back to [INDEX.md](INDEX.md).
