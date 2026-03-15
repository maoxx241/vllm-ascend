# Validation checklist

- 确认权重 dtype/quantization 与命令一致。
- 先做启动 smoke 和吞吐试跑。
- 观察 TP4 切分下的 KV / 吞吐表现，再决定是否需要更激进调参。
