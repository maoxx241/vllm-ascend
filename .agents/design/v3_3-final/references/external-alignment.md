# External Alignment Notes

> 目的：说明本设计为什么采用 “hardware plugin + paired vLLM + release-aware KB” 的形态。  
> 这些不是实现合同，而是外部现实约束的对齐说明。

## 1. 对齐结论

### 1.1 vLLM Ascend 的官方定位是 hardware plugin
官方仓库与文档都明确把 `vllm-ascend` 定位为 Ascend 上运行 vLLM 的 community maintained hardware plugin。  
因此本设计采用：

- `vllm_pair`
- `repo_overlay`
- `integration_core`

而不是把 `vllm-ascend` 当成完全独立于 vLLM 的孤立项目。

### 1.2 版本配对是正式约束
官方 README 把 `vLLM (the same version as vllm-ascend)` 作为前置条件之一。  
因此 resolver 设计里必须把“paired vLLM ref”作为一级输入，而不是可有可无的注释。

### 1.3 分支与 release 节奏是现实存在的
官方仓库说明 `main` 对应 vLLM main branch，同时存在 `releases/vX.Y.Z` 分支。  
这意味着：

- `vllm_release_delta` 需要正式进入 shard family
- `upstream_sync` 不能只做单次 diff，而要对 branch/release 友好

### 1.4 最近版本强调 custom ops / graph mode / release delta
官方 release note 明确提到：
- custom ops built 已成为正式约束
- 新 graph mode `xlite` 已引入
- 某些旧配置与 scheduler 开关已移除

因此本设计把以下内容设为正式知识层，而不是零散注释：

- `repo_custom_ops`
- `vllm_release_delta`
- `deployment_config`
- `troubleshooting`

### 1.5 文档与模型教程扩展很快
官方开发预览文档中的 model tutorial 与 support matrix 扩展很快。  
这进一步说明：

- 不应把“当前文档里有哪些模型教程”手写进 skill prompt
- 应该用 shard + pack 的方式来吸收持续变化的外部事实

---

## 2. 参考来源（2026-03-12 检视）

1. Official GitHub repository  
   https://github.com/vllm-project/vllm-ascend

2. Official docs landing page  
   https://docs.vllm.ai/projects/ascend/en/latest/

3. Official installation page  
   https://docs.vllm.ai/projects/ascend/en/latest/installation.html

4. Official release notes in repository  
   https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/release_notes.md

5. Official supported features page  
   https://docs.vllm.ai/projects/ascend/en/main/user_guide/support_matrix/supported_features.html

---

## 3. 对实现者的含义

1. 任何涉及 branch / release / paired vLLM 的逻辑都不要 hardcode 成单版本方案。
2. 任何涉及 custom ops、graph mode、removed configs 的逻辑都应进入 KB facts，而不是只写在提示词里。
3. `upstream_sync`、`design_analysis`、`adaptation` 这些 family 在没有 `vllm_pair` 前只能部分可用，不应 pretending fully grounded。
