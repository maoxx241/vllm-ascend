# Kimi K3 Attention Residual 算子说明

## 功能

`apply_attn_res` 将 Kimi K3 每个 token 的有效 block residual 与
`prefix_sum` residual 做可学习的 softmax 加权融合。实现位于
`vllm_ascend/ops/triton/kimi_k3/attention_residual.py`。

对每条 residual stream `v_s`，算子先计算 RMSNorm，再通过
`norm.weight * proj.weight` 得到标量分数：

```text
score_s = sum(RMSNorm(v_s) * norm.weight * proj.weight)
weight_s = softmax(score)_s
output = sum(weight_s * v_s)
```

## 输入与输出

| 参数 | 形状 | 说明 |
| --- | --- | --- |
| `prefix_sum` | `[num_tokens, hidden_size]` | 每个 token 的 prefix-sum residual，也是最后一条参与融合的 stream。 |
| `block_residual` | `[num_tokens, block_capacity, hidden_size]` | vLLM 预分配的 residual buffer。只有前 `num_valid_blocks` 个 block 已初始化。 |
| `proj` | `[1, hidden_size]` | 将归一化 residual 投影为标量分数的线性层。 |
| `norm` | `[hidden_size]` | RMSNorm 权重及 epsilon。 |
| `num_valid_blocks` | `int` | `block_residual` 中有效 block 的数量。 |
| `addend` | `[num_tokens, hidden_size]`，可选 | 与 prefix 相加的 attention 输出。 |
| `prefix_sum_out` | `[num_tokens, hidden_size]`，可选 | 提供 `addend` 时必填；保存更新后的 prefix，供后续 MLP 残差相加使用。 |
| 返回值 | `[num_tokens, hidden_size]` | 所有有效 residual stream 的加权和。 |

## 实现约束

- kernel 的 `B` 等于 `num_valid_blocks`，`BLOCK_CAPACITY` 来自预分配
  buffer 的第二维；kernel 只读取 `[0, B)`，不会读取未初始化容量。
- `prefix_sum` 使用逻辑索引 `s == B`。启动端将 stream 数设置为
  `next_power_of_2(B + 1)`，因此 `NB` 始终覆盖 `B` 个 block residual
  加一条 prefix stream。
- softmax 计算使用 FP32，输出再转换为 `prefix_sum` 的 dtype。
- 融合加法先舍入到 prefix 的 dtype，再参与 RMSNorm，与独立 BF16 add
  一致；写入独立的 `prefix_sum_out`，保留 DSpark 已保存的输入 prefix。
- 93 层、residual block size 12 时，85 个非写块层的 attention 残差加法
  融入当前 kernel；层尾的 93 个 MLP 残差加法保持不变。
- 启动 grid 使用设备 vector core 数，每个 program 处理一段连续 token。
