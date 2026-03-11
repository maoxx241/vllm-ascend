# Code Knowledge Map (L0)

This file maps runtime/deployment code paths to shared knowledge docs.

| Code or doc path | Knowledge doc | Why it matters |
| --- | --- | --- |
| `docs/source/user_guide/feature_guide/quantization.md` | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | Quantization terms and CLI flags |
| `docs/source/user_guide/feature_guide/graph_mode.md` | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | Graph-mode related aliases |
| `docs/source/user_guide/feature_guide/context_parallel.md` | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | Context-parallel semantics |
| `docs/source/user_guide/support_matrix/supported_features.md` | [repo-full-knowledge-map](vllm-ascend-core/references/repo-full-knowledge-map.md) | Global feature support truth |
| `docs/source/user_guide/support_matrix/supported_models.md` | [compatibility matrix](vllm-ascend-core/concepts/model-feature-compatibility-matrix.md) | Model-feature constraints |
| `docs/source/tutorials/models/Qwen3-Dense.md` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | Primary profile defaults |
| `docs/source/tutorials/models/Qwen3-Next.md` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | Backup profile defaults |
| `tests/e2e/nightly/single_node/models/configs/Qwen3-32B-Int8.yaml` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | CLI baseline for 32B W8A8 |
| `tests/e2e/nightly/single_node/models/configs/Qwen3-Next-80B-A3B-Instruct-W8A8.yaml` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | CLI baseline for 80B-Next |
| `examples/run_dp_server.sh` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | DP/TP launch reference |
| `vllm/envs.py` | [vllm global inputs](vllm-upstream/references/vllm-inputs-and-envs-global.md) | Full vLLM env var inventory |
| `vllm/vllm/entrypoints/openai/cli_args.py` | [vllm global inputs](vllm-upstream/references/vllm-inputs-and-envs-global.md) | Full OpenAI serve arg inventory |
| `vllm/vllm/engine/arg_utils.py` | [global parameter feature map](deployment-config/references/global-parameter-feature-map.md) | Feature tags, usage and combo mapping for engine args |
| `vllm_ascend/envs.py` | [vllm-ascend global inputs](vllm-ascend-core/references/vllm-ascend-inputs-and-envs-global.md) | Full vLLM-Ascend env var inventory |
| `docs/source/tutorials/**/*.md` | [global combination guide](deployment-config/references/global-parameter-combination-guide.md) | Co-occurrence evidence for real-world flag stacks |
| `.agents/skills/vllm-ascend-deployment-assistant/scripts/build_global_param_kb.py` | [global parameter feature map](deployment-config/references/global-parameter-feature-map.md) | Deterministic global KB generation pipeline |
| `.agents/skills/vllm-ascend-deployment-assistant/scripts/build_global_param_kb.py` | [verification report](deployment-config/references/global-parameter-verification-report.md) | Dual-baseline verification and confidence scoring |
| `.agents/skills/vllm-ascend-deployment-assistant/scripts/render_deploy_package.py` | [global parameter feature map](deployment-config/references/global-parameter-feature-map.md) | Deployment output includes evidence and conflict alerts |
| `.agents/skills/vllm-ascend-deployment-assistant/scripts/normalize_terms.py` | [feature dictionary](deployment-config/concepts/feature-semantic-dictionary.md) | Canonical feature mapping |
| `.agents/skills/vllm-ascend-deployment-assistant/scripts/render_deploy_package.py` | [deployment-playbook](deployment-config/procedures/deployment-playbook.md) | Deterministic package generation |
| `.agents/skills/vllm-ascend-deployment-assistant/scripts/render_deploy_package.py` | [compatibility matrix](vllm-ascend-core/concepts/model-feature-compatibility-matrix.md) | Hard blocking for unsupported features |
| `.agents/skills/vllm-ascend-knowledge-index-maintainer/scripts/validate_shared_knowledge.py` | [imported knowledge report](knowledge-governance/generated/imported_knowledge_report.json) | Rootless validation, domain adjudication, provenance refresh, and generated export |
| `.agents/skills/_shared/knowledge-governance/provenance/verification_manifest.json` | [imported knowledge manifest](knowledge-governance/generated/imported_knowledge_manifest.json) | Per-entry verdicts, domain ownership, evidence, and migration decisions |
| `.agents/skills/_shared/knowledge-governance/provenance/execution_state.json` | [task-skill index](knowledge-governance/generated/task_skill_index.json) | Machine-readable resume state for folded-context continuation |
| `.agents/skills/_shared/knowledge-governance/provenance/import_manifest.json` | [design-analysis index](knowledge-governance/generated/design_analysis_index.json) | Validated entries routed into developer/design-analysis retrieval surfaces |
| `.agents/skills/_shared/{vllm-upstream,vllm-ascend-core,integration-core}/references/source/knowledge/*/*.json` | [domain index](knowledge-governance/generated/domain_index.json) | Canonical source knowledge split across domain-owned roots and indexed through governance |
| `.agents/skills/vllm-ascend-debug-assistant/SKILL.md` | [task-index.md](task-index.md) | Composer workflow for crash, graph, env, and runtime debugging on vLLM Ascend |
| `.agents/skills/vllm-ascend-perf-assistant/SKILL.md` | [task-index.md](task-index.md) | Composer workflow for profiling interpretation, bottleneck ranking, and next-step experiment planning |
| `.agents/skills/vllm-ascend-perf-hunter/SKILL.md` | [design-analysis index](knowledge-governance/generated/design_analysis_index.json) | Atomic bottleneck attribution over profiling artifacts, benchmark logs, and service summaries |
| `.agents/skills/vllm-ascend-test-matrix-planner/SKILL.md` | [global parameter feature map](deployment-config/references/global-parameter-feature-map.md) | Minimal experiment-matrix planning for profiling and performance isolation |
| `.agents/skills/vllm-ascend-graph-analyzer/SKILL.md` | [design-analysis index](knowledge-governance/generated/design_analysis_index.json) | Graph-mode, capture/replay, and shape-stability analysis for profiling and tuning |
| `.agents/skills/vllm-ascend-developer-assistant/SKILL.md` | [task-index.md](task-index.md) | Development/design/debug/release/perf routing entry point |

Back to [INDEX.md](INDEX.md).
