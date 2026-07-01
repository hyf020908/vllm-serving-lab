#!/usr/bin/env python
"""Static preflight checks for launching a vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

COMMON_DTYPES = {
    "auto",
    "half",
    "float16",
    "bfloat16",
    "float",
    "float32",
}
MOE_FIELDS = (
    "num_experts",
    "num_local_experts",
    "moe_intermediate_size",
    "n_routed_experts",
    "num_experts_per_tok",
)
CONTEXT_KEYS = (
    "max_position_embeddings",
    "seq_length",
    "max_sequence_length",
    "model_max_length",
    "n_positions",
)


@dataclass(frozen=True)
class CheckResult:
    status: str
    name: str
    detail: str


def load_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return None, f"{path.name} is missing"
    except json.JSONDecodeError as exc:
        return None, f"{path.name} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{path.name} must contain a JSON object"
    return data, None


def detect_moe_fields(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config[field] for field in MOE_FIELDS if field in config}


def get_context_length(config: dict[str, Any]) -> int | None:
    for key in CONTEXT_KEYS:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    rope = config.get("rope_scaling")
    if isinstance(rope, dict):
        value = rope.get("original_max_position_embeddings")
        if isinstance(value, int) and value > 0:
            return value
    return None


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def needs_trust_remote_code(config: dict[str, Any]) -> bool:
    auto_map = config.get("auto_map")
    if isinstance(auto_map, dict) and auto_map:
        return True
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(item).lower() for item in config.get("architectures", [])]
    names = " ".join([model_type, *architectures])
    return any(marker in names for marker in ("chatglm", "glm", "deepseek_vl"))


def check_model_path(model_path: Path) -> tuple[list[CheckResult], dict[str, Any]]:
    results: list[CheckResult] = []
    if not model_path.exists():
        results.append(CheckResult(FAIL, "model path", f"{model_path} does not exist"))
        return results, {}
    if not model_path.is_dir():
        results.append(CheckResult(FAIL, "model path", f"{model_path} is not a dir"))
        return results, {}

    results.append(CheckResult(PASS, "model path", str(model_path)))
    config, error = load_json_file(model_path / "config.json")
    if error:
        results.append(CheckResult(FAIL, "config.json", error))
        return results, {}

    assert config is not None
    results.append(CheckResult(PASS, "config.json", "parsed successfully"))
    for file_name in ("tokenizer_config.json", "generation_config.json"):
        file_path = model_path / file_name
        status = PASS if file_path.exists() else WARN
        detail = (
            "found" if file_path.exists() else "missing; vLLM may still infer defaults"
        )
        results.append(CheckResult(status, file_name, detail))

    for key in (
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
    ):
        value = config.get(key)
        if value is None:
            results.append(CheckResult(WARN, key, "not found in config.json"))
        else:
            results.append(CheckResult(PASS, key, str(value)))

    moe_fields = detect_moe_fields(config)
    if moe_fields:
        detail = ", ".join(f"{key}={value}" for key, value in moe_fields.items())
        results.append(CheckResult(PASS, "MoE fields", detail))
    else:
        results.append(CheckResult(WARN, "MoE fields", "no MoE fields detected"))
    return results, config


def run_checks(args: argparse.Namespace) -> list[CheckResult]:
    model_path = Path(args.model_path).expanduser()
    results, config = check_model_path(model_path)

    if not 1 <= args.port <= 65535:
        results.append(CheckResult(FAIL, "port", f"{args.port} is outside 1-65535"))
    elif is_port_available(args.port):
        results.append(CheckResult(PASS, "port", f"{args.port} is available"))
    else:
        results.append(CheckResult(FAIL, "port", f"{args.port} is already in use"))

    if config:
        context_len = get_context_length(config)
        if context_len is None:
            results.append(CheckResult(WARN, "context length", "not found in config"))
        elif args.max_model_len > context_len:
            detail = f"requested {args.max_model_len}, config suggests {context_len}"
            results.append(CheckResult(WARN, "max_model_len", detail))
        else:
            detail = f"requested {args.max_model_len}, config suggests {context_len}"
            results.append(CheckResult(PASS, "max_model_len", detail))

        if needs_trust_remote_code(config) and not args.trust_remote_code:
            detail = "model config suggests custom code; add --trust-remote-code"
            results.append(CheckResult(WARN, "trust_remote_code", detail))
        elif args.trust_remote_code:
            results.append(CheckResult(PASS, "trust_remote_code", "enabled"))
        else:
            results.append(CheckResult(PASS, "trust_remote_code", "not required"))

    if args.gpu_memory_utilization <= 0 or args.gpu_memory_utilization > 1:
        detail = "must be within (0, 1]"
        results.append(CheckResult(FAIL, "gpu_memory_utilization", detail))
    elif args.gpu_memory_utilization > 0.95:
        detail = "above 0.95 leaves little headroom for fragmentation"
        results.append(CheckResult(WARN, "gpu_memory_utilization", detail))
    else:
        results.append(
            CheckResult(
                PASS, "gpu_memory_utilization", str(args.gpu_memory_utilization)
            )
        )

    dtype = args.dtype.lower()
    if dtype in COMMON_DTYPES:
        results.append(CheckResult(PASS, "dtype", dtype))
    else:
        detail = f"{args.dtype} is uncommon; expected one of {sorted(COMMON_DTYPES)}"
        results.append(CheckResult(WARN, "dtype", detail))
    return results


def summarize(results: list[CheckResult]) -> str:
    has_fail = any(result.status == FAIL for result in results)
    has_warn = any(result.status == WARN for result in results)
    if has_fail:
        return "Recommendation: fix FAIL items before starting vLLM."
    if has_warn:
        return "Recommendation: service can start, but review WARN items first."
    return "Recommendation: static checks passed; ready to start vLLM."


def print_report(results: list[CheckResult]) -> None:
    print("vLLM Serving Lab preflight report")
    print("=" * 38)
    for result in results:
        print(f"[{result.status:<4}] {result.name:<24} {result.detail}")
    print()
    print(summarize(results))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local model directory")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results = run_checks(args)
    print_report(results)
    return 2 if any(result.status == FAIL for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
