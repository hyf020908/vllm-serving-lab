from __future__ import annotations

import argparse

from examples.serving_lab.scripts import performance_advisor


def test_high_ttft_gives_advice() -> None:
    report = performance_advisor.analyze(
        {"ttft_ms": 1200.0, "prompt_tokens": 2048, "output_tokens": 128}
    )

    assert report.status == "WARN"
    assert any("TTFT is high" in reason for reason in report.reasons)
    assert any("prefix caching" in item for item in report.suggestions)


def test_high_tpot_gives_advice() -> None:
    report = performance_advisor.analyze({"tpot_ms": 120.0, "concurrency": 8})

    assert report.status == "WARN"
    assert any("TPOT / ITL is high" in reason for reason in report.reasons)
    assert any("--max-num-seqs" in item for item in report.suggestions)


def test_high_kv_cache_usage_gives_advice() -> None:
    report = performance_advisor.analyze({"kv_cache_usage": 0.9})

    assert report.status == "WARN"
    assert any("KV Cache usage is high" in reason for reason in report.reasons)
    assert any("--max-model-len" in item for item in report.suggestions)


def test_waiting_requests_gives_advice() -> None:
    report = performance_advisor.analyze({"waiting_requests": 3.0})

    assert report.status == "WARN"
    assert any("waiting requests" in reason for reason in report.reasons)
    assert any("--max-num-batched-tokens" in item for item in report.suggestions)


def test_normal_metrics_return_ok() -> None:
    report = performance_advisor.analyze(
        {
            "latency_ms": 500.0,
            "ttft_ms": 120.0,
            "tpot_ms": 20.0,
            "kv_cache_usage": 0.4,
            "waiting_requests": 0.0,
        }
    )

    assert report.status == "OK"
    assert any("within" in reason for reason in report.reasons)


def test_prometheus_metrics_text_parsing() -> None:
    text = """
# HELP vllm:num_requests_waiting waiting
vllm:num_requests_waiting 5
vllm:gpu_cache_usage_perc 0.92
vllm:time_to_first_token_seconds_sum 2.4
vllm:time_to_first_token_seconds_count 2
vllm:request_time_per_output_token_seconds_sum 0.24
vllm:request_time_per_output_token_seconds_count 2
vllm:e2e_request_latency_seconds_bucket{le="1"} 1
vllm:e2e_request_latency_seconds_bucket{le="5"} 8
vllm:e2e_request_latency_seconds_bucket{le="10"} 10
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 10
"""
    metrics = performance_advisor.parse_prometheus_text(text)
    args = argparse.Namespace(
        latency_ms=None,
        ttft_ms=None,
        output_tokens=None,
        prompt_tokens=None,
        concurrency=None,
        max_tokens=None,
    )

    summary = performance_advisor.derive_metric_summary(metrics, args)

    assert summary["waiting_requests"] == 5
    assert summary["kv_cache_usage"] == 0.92
    assert summary["ttft_ms"] == 1200
    assert summary["tpot_ms"] == 120
    assert "p95_latency_ms" in summary


def test_advisor_does_not_emit_restart_instruction() -> None:
    report = performance_advisor.analyze({"ttft_ms": 1500.0})
    rendered = performance_advisor.render_report(report).lower()

    assert "kill" not in rendered
    assert "restart" not in "\n".join(report.suggestions).lower()
    assert "only prints advice" in rendered
