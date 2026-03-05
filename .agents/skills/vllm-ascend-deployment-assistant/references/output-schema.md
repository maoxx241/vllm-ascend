# Output Schema

Use this fixed response skeleton:

## 1) 参数表

- 模型: `<model_name_or_path>`
- 硬件: `<hardware_type>`
- NPU 数: `<npu_count>`
- Canonical Features: `<feature_1, feature_2, ...>`
- 兼容性结果:
  - Allowed: `<allowed_features>`
  - Blocked: `<blocked_features_with_reasons>`

## 2) 命令块

- `start.sh`: `<absolute_path>`
- `validate.sh`: `<absolute_path>`
- `rollback.sh`: `<absolute_path>`

## 3) 验证块

- `/v1/models` should return HTTP 200
- `/v1/chat/completions` should return HTTP 200 and non-empty text

## 4) 风险块

- incompatibility warnings
- unsupported/not-applicable feature reasons
- next single action if validation fails

## 5) 证据块

- 每个请求特性至少给出 1-3 个参数/环境变量证据
- 每条证据包含:
  - `confidence`
  - `status(aligned/upstream_delta/needs_manual_review)`
  - `definition_ref`（代码文件+行）
  - `web_refs`（官方优先，可包含外部补充来源）

## 6) 冲突告警块

- 列出低置信或上游差异条目
- 规则:
  - 不阻断建议输出
  - 必须显式告警并给出保守回退动作
