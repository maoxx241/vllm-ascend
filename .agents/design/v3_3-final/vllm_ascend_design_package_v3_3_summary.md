# vLLM-Ascend Agent Design Package v3.3 Summary

## 1. 这版解决了什么

v3.3 收掉了 v3.2 最后一个还可能让实现分叉的 shared-runtime 口子：

1. `context-governor` 不再接受显式 `stage` 输入。
2. governor 的 effective stage 只能从 `selector_plan.query_stage` 派生。
3. `governor-decision.stage` 被固定为派生回显字段，不再允许成为第二真相源。
4. `Public Entry` 与 `Intake` 的 `direct_answer/no-query` 路径不再调用 governor。
5. validator 新增 contract lint，用来防止 stage 双真相源回退。

## 2. 核心版本

- `selector_seed/v3`
- `selector_plan/v4`
- `atomic_result_card/v3`
- `continuation_state/v4`
- `kb-pack-request/v2`
- package version: `v3.3-final`

## 3. 首波 MVP capability

1. `deployment_execution / feature-policy-resolver`
2. `performance_analysis / single-profile-breakdown`
3. `performance_analysis / model-expected-performance-estimator`
4. `validation_strategy / change-impact-test-selector`

`debugging` 继续延后到 `P3B`。

## 4. 直接开工前的命令

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python tools/validate_design_package.py
```
