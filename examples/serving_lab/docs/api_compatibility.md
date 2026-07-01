# OpenAI-Compatible API Compatibility

vLLM 的 OpenAI-Compatible API 让现有 OpenAI SDK、LangChain、LlamaIndex
和业务网关可以用较小改动接入本地或私有化模型服务。

## `/v1/models`

该接口返回服务可用模型列表。客户端通常用它确认服务是否启动，以及
`served-model-name` 是否匹配。

## `/v1/chat/completions`

该接口接受 `messages`、`model`、`temperature`、`top_p`、`max_tokens`
等字段，返回 chat completion。

非流式响应应关注：

- `choices` 是否存在且非空。
- `choices[0].message.content` 是否有文本。
- `usage` 是否存在；某些失败或流式场景可能没有完整 usage。
- `finish_reason` 是否符合预期。

流式响应使用 Server-Sent Events，每个 chunk 通常包含
`choices[0].delta.content`。健康检查应记录首 chunk 等待时间、chunk 数量
和总耗时。

常见兼容性坑：

- 请求中的 `model` 名称和 `--served-model-name` 不一致。
- tokenizer 或 chat template 不匹配。
- 某些模型需要 `--trust-remote-code`。
- 流式响应最后一个 chunk 可能只包含 finish 信息。
- tool calling 和 structured output 支持取决于模型、模板和 vLLM 版本。

OpenAI SDK 接入时，把 `base_url` 指向 `http://host:port/v1`，并使用服务
端配置的 model 名称。API key 可填任意占位值，除非外层网关强制鉴权。
