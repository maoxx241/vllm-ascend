---
knowledge_id: vllm-ascend-core.repo-full-knowledge-map
domain: vllm-ascend-core
knowledge_type: reference
summary: Repository-wide knowledge map for vllm-ascend code, docs, tests, and deployment assets.
applicable_vllm_versions: [">=0.15.0", "<0.17.0"]
applicable_cann_versions: [">=8.0.0"]
last_verified: "2026-03-06"
watch_files:
  - "docs/source/index.md"
  - "docs/source/user_guide/feature_guide/index.md"
  - "docs/source/user_guide/support_matrix/supported_features.md"
  - "docs/source/user_guide/support_matrix/supported_models.md"
  - "docs/source/tutorials/models/index.md"
  - "examples/run_dp_server.sh"
  - "tests/e2e/nightly/single_node/models/scripts/single_node_config.py"
  - "vllm_ascend/platform.py"
  - "vllm_ascend/envs.py"
depends_on:
  - "../../INDEX.md"
  - "../concepts/model-feature-compatibility-matrix.md"
source_commit: "workspace-head"
freshness: "fresh"
---

# vLLM-Ascend Full Knowledge Map

This document serves as the high-coverage index for `vllm-ascend` repository knowledge.

## 1. Product and User Docs

- Installation: `docs/source/installation.md`
- Feature guides: `docs/source/user_guide/feature_guide/*.md`
- Support matrix: `docs/source/user_guide/support_matrix/*.md`
- Model tutorials: `docs/source/tutorials/models/*.md`
- Deployment guides: `docs/source/user_guide/deployment_guide/*.md`

## 2. Deployment-Related Runtime Knowledge

- Official example launch script: `examples/run_dp_server.sh`
- External DP template: `examples/external_online_dp/run_dp_template.sh`
- Disaggregated prefill examples: `examples/disaggregated_prefill_v1/*`

## 3. Code Areas for Platform and Runtime Behavior

- Environment variables: `vllm_ascend/envs.py`
- Platform integration: `vllm_ascend/platform.py`
- Worker/runtime logic: `vllm_ascend/worker/`
- Distributed components: `vllm_ascend/distributed/`
- Quantization methods: `vllm_ascend/quantization/`
- Spec decode: `vllm_ascend/spec_decode/`

## 4. Test Knowledge

- Single-node model nightly configs: `tests/e2e/nightly/single_node/models/configs/`
- Multi-node deployment configs: `tests/e2e/nightly/multi_node/config/`
- Base model config templates: `tests/e2e/models/configs/`
- Deployment test scripts: `tests/e2e/nightly/single_node/models/scripts/`

## 5. Weak-Model-Friendly Retrieval Order

1. Start from `_shared/INDEX.md`.
2. Resolve terms via `deployment-config/concepts/feature-semantic-dictionary.md`.
3. Check hard compatibility in `vllm-ascend-core/concepts/model-feature-compatibility-matrix.md`.
4. Select detailed doc path from this map.

Back to [INDEX](../../INDEX.md).
