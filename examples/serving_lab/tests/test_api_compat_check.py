from __future__ import annotations

from examples.serving_lab.scripts import api_compat_check


def test_parse_chat_response_with_usage() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "vLLM is an inference server.",
                }
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }

    parsed = api_compat_check.parse_chat_response(data)

    assert parsed["has_choices"] is True
    assert parsed["has_usage"] is True
    assert parsed["has_text"] is True


def test_parse_sse_chunks_ignores_done() -> None:
    chunks = api_compat_check.parse_sse_chunks(
        [
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            "data: [DONE]",
        ]
    )

    assert len(chunks) == 1


def test_error_summary_from_openai_error() -> None:
    summary = api_compat_check.extract_error_summary(
        {"error": {"message": "model not found"}},
        "",
    )

    assert summary == "model not found"
