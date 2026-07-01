#!/usr/bin/env python
"""Lightweight benchmark for vLLM OpenAI-compatible chat completions."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestResult:
    ok: bool
    latency_s: float
    ttft_s: float | None
    output_tokens: int
    error: str | None = None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) if " " in text else len(text))


def extract_text_and_usage(data: dict[str, Any]) -> tuple[str, int | None]:
    choices = data.get("choices")
    text = ""
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                text = str(message.get("content") or "")
            else:
                text = str(first.get("text") or "")
    usage = data.get("usage")
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int):
            return text, completion_tokens
    return text, None


def post_json(url: str, payload: dict[str, Any], timeout: float) -> RequestResult:
    start = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return RequestResult(False, time.perf_counter() - start, None, 0, str(exc))

    latency = time.perf_counter() - start
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return RequestResult(False, latency, None, 0, "non-json response")
    text, usage_tokens = extract_text_and_usage(data if isinstance(data, dict) else {})
    output_tokens = usage_tokens if usage_tokens is not None else estimate_tokens(text)
    return RequestResult(bool(text), latency, latency, output_tokens, None)


def post_stream(url: str, payload: dict[str, Any], timeout: float) -> RequestResult:
    start = time.perf_counter()
    first_token_s: float | None = None
    chunks = 0
    chars = 0
    req = urllib.request.Request(
        url,
        data=json.dumps(payload | {"stream": True}).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                if first_token_s is None:
                    first_token_s = time.perf_counter() - start
                chunks += 1
                chars += len(data)
    except urllib.error.URLError as exc:
        return RequestResult(
            False, time.perf_counter() - start, first_token_s, 0, str(exc)
        )
    latency = time.perf_counter() - start
    return RequestResult(chunks > 0, latency, first_token_s, max(1, chars // 4), None)


def build_payload(args: argparse.Namespace, index: int) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": f"{args.prompt} Request number {index}.",
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }


def run_one(args: argparse.Namespace, index: int) -> RequestResult:
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    payload = build_payload(args, index)
    if args.stream:
        return post_stream(url, payload, args.timeout)
    return post_json(url, payload, args.timeout)


def summarize(results: list[RequestResult], elapsed_s: float) -> dict[str, float | int]:
    successes = [result for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    latencies = [result.latency_s for result in successes]
    ttfts = [result.ttft_s for result in successes if result.ttft_s is not None]
    output_tokens = sum(result.output_tokens for result in successes)
    return {
        "total": len(results),
        "success": len(successes),
        "failure": len(failures),
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "ttft": statistics.mean(ttfts) if ttfts else 0.0,
        "tpot": sum(latencies) / output_tokens if output_tokens else 0.0,
        "output_tokens_s": output_tokens / elapsed_s if elapsed_s else 0.0,
        "requests_s": len(successes) / elapsed_s if elapsed_s else 0.0,
    }


def print_summary(summary: dict[str, float | int], errors: list[str]) -> None:
    print("| metric | value |")
    print("| --- | ---: |")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"| {key} | {value:.4f} |")
        else:
            print(f"| {key} | {value} |")
    if errors:
        print()
        print("Error summary:")
        for error in sorted(set(errors))[:10]:
            print(f"- {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--prompt", default="Briefly explain vLLM serving.")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(run_one, args, index) for index in range(args.num_requests)
        ]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    elapsed_s = time.perf_counter() - start
    errors = [result.error or "unknown" for result in results if not result.ok]
    print_summary(summarize(results, elapsed_s), errors)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
