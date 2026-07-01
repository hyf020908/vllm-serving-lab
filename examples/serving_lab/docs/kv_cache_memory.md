# KV Cache Memory

KV Cache 保存每层 attention 已经计算过的 key/value。生成下一个 token
时，模型不需要重复计算完整上下文，只需要复用历史 KV。

推理必须关注 KV Cache，因为它随上下文长度、并发数、层数、hidden size
和 dtype 增长。长上下文请求会占用更多 KV 块；高并发会让多个请求同时
保留 KV；长输出也会持续追加 token。

不要盲目把 `max_model_len` 拉到 128K。即使单个请求不一定用满，服务也
需要按配置规划可承载的 KV 空间，最终可能降低并发或直接 OOM。

核心关系：

- `max_model_len` 越大，单请求最坏 KV 占用越高。
- `max_num_seqs` 越大，同时驻留请求越多。
- `max_num_batched_tokens` 越大，prefill 峰值吞吐更高但资源压力更大。
- 输出越长，请求在 decode 阶段保留 KV 的时间越久。

降低 KV Cache 压力：

- 按真实业务设置 `max_model_len`。
- 降低 `max_num_seqs` 或 `max_num_batched_tokens`。
- 限制客户端 `max_tokens`。
- 开启 prefix caching 并复用稳定 system prompt。
- 降低 `gpu_memory_utilization` 留出碎片和运行时余量。
- 大模型使用合适的 tensor parallel。

观察风险时重点看 `/metrics` 中的 GPU cache usage、waiting requests、
request latency、TTFT 和 time per output token。cache usage 长期接近
上限且 waiting requests 增长，通常说明上下文、并发或输出长度配置过激。
