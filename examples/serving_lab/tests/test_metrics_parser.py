from __future__ import annotations

from examples.serving_lab.scripts import benchmark_openai_api, metrics_snapshot


def test_parse_prometheus_text() -> None:
    text = """
# HELP vllm:num_requests_running Running requests.
vllm:num_requests_running{model_name="qwen"} 2
vllm:num_requests_waiting{model_name="qwen"} 3
vllm:gpu_cache_usage_perc 0.75
"""

    metrics = metrics_snapshot.parse_prometheus_text(text)
    summary = dict(metrics_snapshot.summarize_metrics(metrics))

    assert metrics["vllm:num_requests_running"][0].value == 2
    assert summary["running requests"] == "2"
    assert summary["KV cache usage"] == "0.75"
    assert summary["request latency count"] == "N/A"


def test_parse_labels() -> None:
    labels = metrics_snapshot.parse_labels('model_name="qwen",engine="0"')

    assert labels == {"model_name": "qwen", "engine": "0"}


def test_benchmark_percentiles() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert benchmark_openai_api.percentile(values, 50) == 3.0
    assert benchmark_openai_api.percentile(values, 95) == 5.0
    assert benchmark_openai_api.percentile(values, 99) == 5.0


def test_benchmark_summary() -> None:
    results = [
        benchmark_openai_api.RequestResult(True, 1.0, 0.2, 10),
        benchmark_openai_api.RequestResult(True, 2.0, 0.4, 10),
        benchmark_openai_api.RequestResult(False, 0.5, None, 0, "boom"),
    ]

    summary = benchmark_openai_api.summarize(results, elapsed_s=2.0)

    assert summary["total"] == 3
    assert summary["success"] == 2
    assert summary["failure"] == 1
    assert summary["requests_s"] == 1.0
