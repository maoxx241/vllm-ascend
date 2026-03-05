# Error Index (L0)

Use this file to map frequent deployment failures to a deterministic first action.

| Error keyword | Likely cause | First action | Deep link |
| --- | --- | --- | --- |
| `ModuleNotFoundError: torch` | Python environment mismatch | Run env bootstrap first | [env-bootstrap-baseline](ascend-foundation/procedures/env-bootstrap-baseline.md) |
| `No module named vllm` | Editable install missing | Re-run install stage in bootstrap | [env-bootstrap-baseline](ascend-foundation/procedures/env-bootstrap-baseline.md) |
| `Address already in use` | Port occupied | Switch port or stop old service | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) |
| `HCCL` / `NPU init` failures | NPU env variables missing | Re-load shell env and verify cards | [env-bootstrap-baseline](ascend-foundation/procedures/env-bootstrap-baseline.md) |
| `OOM` / memory errors | Over-sized model or config | Lower `max-model-len`/batch, disable heavy features | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) |
| `/v1/models` non-200 | Service not ready or crashed | Check server log + health probe flow | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) |
| Spec decode crash | Unsupported model/backend path | Disable speculative decode and retry | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) |
| `qwen3-32b-w8a8 + int4` | Unsupported feature-profile combo | Block request and suggest W4A4 artifact/profile switch | [unsupported cases](troubleshooting/procedures/unsupported-feature-cases.md) |
| `qwen3-32b-w8a8 + ep` | Dense model cannot use EP | Block EP and keep TP/DP tuning path | [compatibility matrix](vllm-ascend-core/concepts/model-feature-compatibility-matrix.md) |

Back to [INDEX.md](INDEX.md).
