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
