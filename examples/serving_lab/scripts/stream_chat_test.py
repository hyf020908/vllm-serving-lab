#!/usr/bin/env python
"""Minimal streaming chat test for a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


def extract_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        return str(delta.get("content") or "")
    message = first.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(first.get("text") or "")


def run_stream(args: argparse.Namespace) -> int:
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_chunk_s: float | None = None
    chunk_count = 0
    text_count = 0

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            print(f"HTTP status: {resp.status}")
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                chunk_count += 1
                if first_chunk_s is None:
                    first_chunk_s = time.perf_counter() - start
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    print(f"[chunk {chunk_count}] <non-json>")
                    continue
                text = extract_delta_text(chunk)
                if text:
                    text_count += 1
                print(f"[chunk {chunk_count}] {text}", flush=True)
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}")
        return 1

    total_s = time.perf_counter() - start
    print()
    first_chunk_value = first_chunk_s if first_chunk_s is not None else "N/A"
    print(f"first_chunk_wait_s: {first_chunk_value}")
    print(f"total_elapsed_s: {total_s:.2f}")
    print(f"chunk_count: {chunk_count}")
    healthy = first_chunk_s is not None and chunk_count > 0 and text_count > 0
    print(f"stream_health: {'PASS' if healthy else 'WARN'}")
    return 0 if healthy else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main() -> int:
    return run_stream(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
