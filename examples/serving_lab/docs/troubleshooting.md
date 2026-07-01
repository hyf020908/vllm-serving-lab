# Troubleshooting

### OOM

**现象：** 启动或压测时 CUDA out of memory。

**常见原因：** 模型太大、上下文太长、并发太高或显存利用率设置过高。

**排查步骤：** 降低 `max_model_len` 和 `max_num_seqs`，观察 GPU cache。

**解决方向：** 使用更小模型、更多 GPU、tensor parallel 或更保守参数。

### `max_model_len` 过大

**现象：** 启动慢、可用并发低、KV Cache 不足。

**常见原因：** 配置远高于真实请求长度。

**排查步骤：** 对比 config 中上下文长度和业务请求长度。

**解决方向：** 从 4K、8K、16K 逐步压测。

### KV Cache 不足

**现象：** waiting requests 增长，长请求容易失败。

**常见原因：** 上下文、并发和输出长度共同推高 KV 占用。

**排查步骤：** 查看 `/metrics` 中 cache usage、TTFT 和 P99。

**解决方向：** 限制输出长度，降低并发，启用 prefix caching。

### `trust_remote_code` 缺失

**现象：** 模型架构加载失败或 tokenizer 行为异常。

**常见原因：** 模型仓库依赖自定义代码。

**排查步骤：** 查看 `config.json` 的 `auto_map` 和模型文档。

**解决方向：** 对可信模型增加 `--trust-remote-code`。

### 模型架构无法识别

**现象：** vLLM 报 unsupported architecture。

**常见原因：** vLLM 版本太旧或模型 config 不兼容。

**排查步骤：** 检查 `architectures`、`model_type` 和 vLLM 支持列表。

**解决方向：** 升级 vLLM 或换用已支持模型版本。

### MoE config 不兼容

**现象：** MoE 模型启动失败或显存超预期。

**常见原因：** experts 字段、并行策略或 kernel 支持不匹配。

**排查步骤：** 检查 `num_experts`、`num_experts_per_tok` 等字段。

**解决方向：** 使用推荐 vLLM 版本，降低并发并调整 TP。

### Transformers 版本冲突

**现象：** tokenizer、config 或 remote code 导入失败。

**常见原因：** 模型要求的 transformers 版本和环境不一致。

**排查步骤：** 查看模型 README 和异常栈。

**解决方向：** 使用项目推荐依赖，避免混用系统 Python 环境。

### FlashAttention / FlashInfer 后端冲突

**现象：** attention backend 初始化失败。

**常见原因：** CUDA、GPU 架构、wheel 或后端版本不匹配。

**排查步骤：** 查看启动日志中的 selected backend 和 import error。

**解决方向：** 换兼容镜像、升级 wheel，或显式切换后端。

### 端口占用

**现象：** 启动时报 address already in use。

**常见原因：** 旧服务未停止或多实例端口重复。

**排查步骤：** 用 preflight 检查端口。

**解决方向：** 停止旧进程或更换 `PORT`。

### 模型路径挂载错误

**现象：** 找不到 `config.json` 或权重文件。

**常见原因：** 容器挂载路径和启动参数不一致。

**排查步骤：** 在容器内列出 `MODEL_PATH`。

**解决方向：** 修正挂载路径和环境变量。

### tokenizer / chat template 不匹配

**现象：** 输出格式异常、角色混乱或中文效果差。

**常见原因：** tokenizer 文件缺失或模板不适配模型。

**排查步骤：** 检查 `tokenizer_config.json` 和 chat template。

**解决方向：** 使用官方模型目录并固定 tokenizer 文件。

### OpenAI API 返回格式异常

**现象：** 客户端找不到 `choices`、`usage` 或 content。

**常见原因：** 请求失败、模型名错误或网关改写响应。

**排查步骤：** 运行 `api_compat_check.py`。

**解决方向：** 修正 model 名称、请求字段和代理配置。

### 首 token 慢

**现象：** TTFT 高。

**常见原因：** prompt 太长、prefill 拥塞或冷启动。

**排查步骤：** 对比短 prompt 和长 prompt 的 TTFT。

**解决方向：** 降低 prefill 压力，使用 prefix caching 或预热。

### 流式输出卡顿

**现象：** chunk 间隔不稳定。

**常见原因：** decode 负载高、网络代理缓冲或客户端读取慢。

**排查步骤：** 直接运行 `stream_chat_test.py` 绕过业务网关。

**解决方向：** 调整代理缓冲、降低并发并观察 TPOT。

### P99 延迟高

**现象：** 平均延迟正常但尾延迟很差。

**常见原因：** 请求长度差异大、batch 拥塞或资源接近上限。

**排查步骤：** 按输入长度和输出长度分组 benchmark。

**解决方向：** 做请求分流，限制超长请求，调低并发上限。
