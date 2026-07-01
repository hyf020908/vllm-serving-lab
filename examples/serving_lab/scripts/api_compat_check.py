#!/usr/bin/env python
"""Check a vLLM OpenAI-compatible API endpoint."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpResult:
    status: int
    elapsed_s: float
    body: str
    error: str | None = None


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def parse_chat_response(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    usage = data.get("usage")
    text = ""
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                text = str(message.get("content") or "")
            else:
                text = str(first.get("text") or "")
    return {
        "has_choices": isinstance(choices, list) and bool(choices),
        "has_usage": isinstance(usage, dict),
        "text": text,
        "has_text": bool(text.strip()),
    }


def extract_error_summary(data: dict[str, Any] | None, raw: str) -> str:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or error)
        if isinstance(error, str):
            return error
    return raw.strip().replace("\n", " ")[:240]


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> HttpResult:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return HttpResult(resp.status, time.perf_counter() - start, body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return HttpResult(exc.code, time.perf_counter() - start, body)
    except urllib.error.URLError as exc:
        return HttpResult(0, time.perf_counter() - start, "", str(exc.reason))


def parse_sse_chunks(lines: list[str]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            chunks.append(data)
    return chunks


def stream_chat(base_url: str, payload: dict[str, Any]) -> HttpResult:
    url = f"{normalize_base_url(base_url)}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload | {"stream": True}).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            lines = [line.decode("utf-8", errors="replace") for line in resp]
            chunks = parse_sse_chunks(lines)
            body = json.dumps({"chunks": len(chunks)}, ensure_ascii=False)
            return HttpResult(resp.status, time.perf_counter() - start, body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return HttpResult(exc.code, time.perf_counter() - start, body)
    except urllib.error.URLError as exc:
        return HttpResult(0, time.perf_counter() - start, "", str(exc.reason))


def chat_payload(
    model: str,
    max_tokens: int,
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }


def print_json_result(name: str, result: HttpResult) -> bool:
    ok = 200 <= result.status < 300 and not result.error
    data: dict[str, Any] | None = None
    if result.body:
        try:
            parsed = json.loads(result.body)
            data = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            data = None
    parsed_chat = parse_chat_response(data or {})
    detail = (
        f"status={result.status} elapsed={result.elapsed_s:.2f}s "
        f"choices={parsed_chat['has_choices']} usage={parsed_chat['has_usage']} "
        f"text={parsed_chat['has_text']}"
    )
    if result.error:
        detail += f" error={result.error}"
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def run_checks(args: argparse.Namespace) -> int:
    base_url = normalize_base_url(args.base_url)
    failures = 0

    models = request_json("GET", f"{base_url}/v1/models")
    if not print_json_result("GET /v1/models", models):
        failures += 1

    cases = [
        ("chat basic", chat_payload(args.model, args.max_tokens, "Hello.")),
        (
            "chat zh",
            chat_payload(args.model, args.max_tokens, "请用一句话介绍 vLLM。"),
        ),
        ("chat short", chat_payload(args.model, 8, "Hi")),
        (
            "chat sampling",
            chat_payload(args.model, args.max_tokens, "List 2 facts.", 0.7),
        ),
    ]
    for name, payload in cases:
        result = request_json("POST", f"{base_url}/v1/chat/completions", payload)
        if not print_json_result(name, result):
            failures += 1

    stream_result = stream_chat(
        base_url, chat_payload(args.model, args.max_tokens, "Hi")
    )
    stream_ok = (
        200 <= stream_result.status < 300
        and '"chunks": 0' not in stream_result.body
    )
    detail = f"status={stream_result.status} elapsed={stream_result.elapsed_s:.2f}s"
    print(f"[{'PASS' if stream_ok else 'FAIL'}] chat stream: {detail}")
    failures += 0 if stream_ok else 1

    bad_payload = chat_payload("__missing_model__", 8, "Hi")
    bad = request_json("POST", f"{base_url}/v1/chat/completions", bad_payload)
    try:
        bad_data = json.loads(bad.body) if bad.body else None
    except json.JSONDecodeError:
        bad_data = None
    error_summary = extract_error_summary(
        bad_data if isinstance(bad_data, dict) else None, bad.body
    )
    bad_ok = bad.status >= 400 and bool(error_summary)
    print(
        f"[{'PASS' if bad_ok else 'FAIL'}] bad model error: "
        f"status={bad.status} elapsed={bad.elapsed_s:.2f}s error={error_summary}"
    )
    failures += 0 if bad_ok else 1
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser


def main() -> int:
    failures = run_checks(build_parser().parse_args())
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
