#!/usr/bin/env python
"""Adaptive outer launcher for vLLM OpenAI-compatible serving.

This tool does not change vLLM internals. It starts ``vllm serve``, waits for
``/v1/models``, classifies startup failures from logs, adjusts launch
parameters, and retries only while the service is not ready yet.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MOE_FIELDS = (
    "num_experts",
    "num_local_experts",
    "moe_intermediate_size",
    "n_routed_experts",
    "num_experts_per_tok",
)
SUMMARY_FIELDS = (
    "model_type",
    "architectures",
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
)
CONTEXT_KEYS = (
    "max_position_embeddings",
    "seq_length",
    "max_sequence_length",
    "model_max_length",
    "n_positions",
)


@dataclass
class LaunchConfig:
    model_path: str
    served_model_name: str
    host: str = "0.0.0.0"
    port: int = 8000
    dtype: str = "auto"
    max_model_len: int = 8192
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False
    extra_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelSummary:
    fields: dict[str, Any]
    moe_fields: dict[str, Any]
    context_length: int | None
    config_error: str | None = None


@dataclass
class AttemptRecord:
    round_index: int
    command: list[str]
    parameters: dict[str, Any]
    ready: bool
    failure_type: str | None = None
    adjustment: str | None = None
    log_path: str | None = None
    log_summary: str = ""
    models_summary: str | None = None
    pid: int | None = None


TERMINAL_FAILURES = {
    "model_architecture_unrecognized",
    "moe_config_or_architecture_incompatible",
    "transformers_version_error",
    "attention_backend_conflict",
}


def load_model_summary(model_path: Path) -> ModelSummary:
    config_path = model_path / "config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ModelSummary({}, {}, None, f"{config_path} is missing")
    except json.JSONDecodeError as exc:
        return ModelSummary({}, {}, None, f"{config_path} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        return ModelSummary({}, {}, None, f"{config_path} must contain a JSON object")

    fields = {key: data.get(key) for key in SUMMARY_FIELDS if key in data}
    moe_fields = {key: data[key] for key in MOE_FIELDS if key in data}
    return ModelSummary(fields, moe_fields, get_context_length(data))


def get_context_length(config: dict[str, Any]) -> int | None:
    for key in CONTEXT_KEYS:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    rope_scaling = config.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        value = rope_scaling.get("original_max_position_embeddings")
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


def find_next_available_port(start_port: int, host: str = "127.0.0.1") -> int:
    for port in range(max(1, start_port), 65536):
        if is_port_available(port, host):
            return port
    raise RuntimeError("no available port found in range 1-65535")


def command_for_config(config: LaunchConfig) -> list[str]:
    command = [
        "vllm",
        "serve",
        config.model_path,
        "--served-model-name",
        config.served_model_name,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--dtype",
        config.dtype,
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--max-num-batched-tokens",
        str(config.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
    ]
    if config.trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(config.extra_args)
    return command


def shell_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def ready_url(config: LaunchConfig) -> str:
    host = "127.0.0.1" if config.host in {"0.0.0.0", "::"} else config.host
    return f"http://{host}:{config.port}/v1/models"


def wait_until_ready(
    config: LaunchConfig,
    timeout: float,
    process: subprocess.Popen[bytes] | None = None,
) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout
    url = ready_url(config)
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False, f"process exited with code {process.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                body = response.read().decode("utf-8", errors="replace")
            if 200 <= response.status < 300:
                return True, summarize_models_response(body)
            last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)
    return False, f"readiness timeout: {last_error or url}"


def summarize_models_response(body: str, limit: int = 500) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body[:limit]
    data = parsed.get("data") if isinstance(parsed, dict) else None
    if isinstance(data, list):
        ids = [str(item.get("id")) for item in data if isinstance(item, dict)]
        return f"models={ids[:5]} count={len(ids)}"
    return json.dumps(parsed, ensure_ascii=False)[:limit]


def summarize_log(log_text: str, max_lines: int = 16) -> str:
    interesting = []
    markers = (
        "error",
        "exception",
        "traceback",
        "out of memory",
        "kv cache",
        "max_model_len",
        "trust_remote_code",
        "architecture",
        "transformers",
        "flashattention",
        "flashinfer",
        "address already in use",
    )
    for line in log_text.splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in markers):
            interesting.append(line.strip())
    selected = interesting[-max_lines:] if interesting else log_text.splitlines()[-8:]
    return "\n".join(selected)


def classify_failure(log_text: str, readiness_message: str | None = None) -> str:
    text = f"{log_text}\n{readiness_message or ''}".lower()
    if "address already in use" in text or "errno 98" in text or "errno 48" in text:
        return "port_occupied"
    if "cuda out of memory" in text or "torch.cuda.outofmemoryerror" in text:
        return "cuda_oom"
    if "kv cache" in text and any(
        phrase in text for phrase in ("not enough", "no available", "larger than")
    ):
        return "kv_cache_insufficient"
    if "max_model_len" in text or "max model len" in text:
        too_large = ("larger", "exceed", "greater", "too long")
        if any(phrase in text for phrase in too_large):
            return "max_model_len_too_large"
    if "max seq len" in text and any(phrase in text for phrase in ("larger", "exceed")):
        return "max_model_len_too_large"
    if "bfloat16" in text and any(
        phrase in text for phrase in ("not supported", "unsupported", "not available")
    ):
        return "dtype_incompatible"
    if "unsupported dtype" in text or "dtype is not supported" in text:
        return "dtype_incompatible"
    if "trust_remote_code" in text or "requires you to execute" in text:
        return "trust_remote_code_missing"
    if "unsupported architecture" in text or "not supported for architecture" in text:
        return "model_architecture_unrecognized"
    if "architectures" in text and "not recognized" in text:
        return "model_architecture_unrecognized"
    if "moe" in text or "expert" in text:
        unsupported = ("unsupported", "not supported", "invalid")
        if any(phrase in text for phrase in unsupported):
            return "moe_config_or_architecture_incompatible"
    if "transformers" in text and any(
        phrase in text
        for phrase in ("version", "requires", "cannot import", "attributeerror")
    ):
        return "transformers_version_error"
    if ("flashattention" in text or "flashinfer" in text) and any(
        phrase in text
        for phrase in ("conflict", "failed", "not installed", "importerror")
    ):
        return "attention_backend_conflict"
    if "readiness timeout" in text:
        return "readiness_timeout"
    return "unknown_startup_failure"


def reduce_capacity(config: LaunchConfig) -> list[str]:
    actions: list[str] = []
    new_len = max(1024, config.max_model_len // 2)
    if new_len < config.max_model_len:
        actions.append(f"max_model_len {config.max_model_len} -> {new_len}")
        config.max_model_len = new_len
    new_seqs = max(1, config.max_num_seqs // 2)
    if new_seqs < config.max_num_seqs:
        actions.append(f"max_num_seqs {config.max_num_seqs} -> {new_seqs}")
        config.max_num_seqs = new_seqs
    new_tokens = max(512, config.max_num_batched_tokens // 2)
    if new_tokens < config.max_num_batched_tokens:
        actions.append(
            "max_num_batched_tokens "
            f"{config.max_num_batched_tokens} -> {new_tokens}"
        )
        config.max_num_batched_tokens = new_tokens
    config.gpu_memory_utilization = min(max(config.gpu_memory_utilization, 0.5), 0.95)
    return actions


def dtype_fallback(dtype: str) -> str | None:
    chain = ["bfloat16", "float16", "auto"]
    normalized = dtype.lower()
    if normalized == "half":
        normalized = "float16"
    if normalized not in chain:
        return "auto" if normalized != "auto" else None
    index = chain.index(normalized)
    return chain[index + 1] if index + 1 < len(chain) else None


def adjust_config(
    config: LaunchConfig,
    failure_type: str,
    model_summary: ModelSummary,
) -> tuple[bool, str]:
    if failure_type == "port_occupied":
        old_port = config.port
        config.port = find_next_available_port(old_port + 1)
        return True, f"port {old_port} -> {config.port}"

    if failure_type == "max_model_len_too_large":
        context = model_summary.context_length
        if context and config.max_model_len > context:
            old_len = config.max_model_len
            config.max_model_len = context
            return True, f"max_model_len {old_len} -> config context {context}"
        actions = reduce_capacity(config)
        return bool(actions), "; ".join(actions) or "no safe reduction available"

    if failure_type in {"cuda_oom", "kv_cache_insufficient"}:
        actions = reduce_capacity(config)
        return bool(actions), "; ".join(actions) or "no safe reduction available"

    if failure_type == "dtype_incompatible":
        next_dtype = dtype_fallback(config.dtype)
        if next_dtype is None:
            return False, f"dtype is already {config.dtype}; manual check required"
        old_dtype = config.dtype
        config.dtype = next_dtype
        return True, f"dtype {old_dtype} -> {next_dtype}"

    if failure_type == "trust_remote_code_missing":
        if config.trust_remote_code:
            return False, "trust_remote_code already enabled; manual check required"
        config.trust_remote_code = True
        return True, "enabled --trust-remote-code for a trusted local model path"

    if failure_type == "readiness_timeout":
        old_seqs = config.max_num_seqs
        old_tokens = config.max_num_batched_tokens
        config.max_num_seqs = max(1, config.max_num_seqs // 2)
        config.max_num_batched_tokens = max(512, config.max_num_batched_tokens // 2)
        return (
            config.max_num_seqs != old_seqs
            or config.max_num_batched_tokens != old_tokens,
            "readiness timeout backoff: "
            f"max_num_seqs {old_seqs} -> {config.max_num_seqs}; "
            "max_num_batched_tokens "
            f"{old_tokens} -> {config.max_num_batched_tokens}",
        )

    if failure_type in TERMINAL_FAILURES:
        return False, (
            "terminal startup diagnosis; upgrade/check vLLM, Transformers, model "
            "architecture support, MoE fields, or selected attention backend"
        )

    return False, "unknown failure; inspect log before retrying manually"


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def write_json(path: Path, data: Any) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def write_final_files(
    output_dir: Path,
    config: LaunchConfig,
    model_summary: ModelSummary,
    attempts: list[AttemptRecord],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = command_for_config(config)
    write_json(
        output_dir / "final_config.json",
        {
            "launch_config": asdict(config),
            "model_summary": asdict(model_summary),
            "attempts": [asdict(attempt) for attempt in attempts],
        },
    )
    final_command = (
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + shell_command(command)
        + "\n"
    )
    command_path = output_dir / "final_command.sh"
    command_path.write_text(final_command, encoding="utf-8")
    command_path.chmod(0o755)
    (output_dir / "launch_report.md").write_text(
        render_report(config, model_summary, attempts), encoding="utf-8"
    )


def render_report(
    config: LaunchConfig,
    model_summary: ModelSummary,
    attempts: list[AttemptRecord],
) -> str:
    lines = [
        "# Adaptive Launch Report",
        "",
        "## Final Command",
        "",
        "```bash",
        shell_command(command_for_config(config)),
        "```",
        "",
        "## Model Summary",
        "",
        f"- config error: {model_summary.config_error or 'none'}",
        f"- context length: {model_summary.context_length or 'unknown'}",
        f"- fields: `{json.dumps(model_summary.fields, ensure_ascii=False)}`",
        f"- MoE fields: `{json.dumps(model_summary.moe_fields, ensure_ascii=False)}`",
        "",
        "## Attempts",
        "",
    ]
    for attempt in attempts:
        lines.extend(
            [
                f"### Round {attempt.round_index}",
                "",
                f"- ready: {attempt.ready}",
                f"- pid: {attempt.pid or 'n/a'}",
                f"- failure type: {attempt.failure_type or 'none'}",
                f"- adjustment: {attempt.adjustment or 'none'}",
                f"- log path: {attempt.log_path or 'n/a'}",
                f"- models summary: {attempt.models_summary or 'n/a'}",
                "",
                "```bash",
                shell_command(attempt.command),
                "```",
                "",
                "Log summary:",
                "",
                "```text",
                attempt.log_summary or "n/a",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def run_launcher(
    args: argparse.Namespace,
) -> tuple[LaunchConfig, list[AttemptRecord], Path]:
    config = LaunchConfig(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        extra_args=args.extra_arg or [],
    )
    model_summary = load_model_summary(Path(config.model_path).expanduser())
    output_dir = Path(args.output_dir).expanduser()
    if not args.dry_run:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / f"run_{timestamp}"
    attempts: list[AttemptRecord] = []

    if args.dry_run:
        attempt = AttemptRecord(
            round_index=0,
            command=command_for_config(config),
            parameters=asdict(config),
            ready=False,
            failure_type="dry_run",
            adjustment="no process started",
            log_summary="dry-run generated command and model summary only",
        )
        attempts.append(attempt)
        write_final_files(output_dir, config, model_summary, attempts)
        return config, attempts, output_dir

    for round_index in range(args.max_retries + 1):
        output_dir.mkdir(parents=True, exist_ok=True)
        command = command_for_config(config)
        log_path = output_dir / f"round_{round_index}.log"
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
            ready, models_summary = wait_until_ready(
                config, args.ready_timeout, process
            )

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        attempt = AttemptRecord(
            round_index=round_index,
            command=command,
            parameters=asdict(config),
            ready=ready,
            log_path=str(log_path),
            log_summary=summarize_log(log_text),
            models_summary=models_summary,
            pid=process.pid,
        )
        attempts.append(attempt)

        if ready:
            write_final_files(output_dir, config, model_summary, attempts)
            return config, attempts, output_dir

        terminate_process(process)
        failure_type = classify_failure(log_text, models_summary)
        can_retry, adjustment = adjust_config(config, failure_type, model_summary)
        attempt.failure_type = failure_type
        attempt.adjustment = adjustment
        if not can_retry or round_index >= args.max_retries:
            write_final_files(output_dir, config, model_summary, attempts)
            return config, attempts, output_dir

    write_final_files(output_dir, config, model_summary, attempts)
    return config, attempts, output_dir


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local model directory")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=positive_int, default=8192)
    parser.add_argument("--max-num-seqs", type=positive_int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=positive_int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=positive_int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--ready-timeout", type=float, default=90)
    parser.add_argument(
        "--output-dir",
        default="examples/serving_lab/runs/adaptive_launcher",
        help="Directory for final_config.json, final_command.sh, and reports",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        help="Additional raw argument passed to vllm serve; repeat as needed",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be within (0, 1]")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    if args.ready_timeout <= 0:
        raise SystemExit("--ready-timeout must be > 0")

    config, attempts, output_dir = run_launcher(args)
    final_attempt = attempts[-1]
    print("Adaptive launcher result")
    print("========================")
    print(f"Ready: {final_attempt.ready}")
    if final_attempt.ready:
        print(f"Service PID: {final_attempt.pid}")
        print(f"Port: {config.port}")
        print(f"Models: {final_attempt.models_summary}")
    else:
        print(f"Failure type: {final_attempt.failure_type}")
        print(f"Adjustment: {final_attempt.adjustment}")
    print(f"Final command: {shell_command(command_for_config(config))}")
    print(f"Report: {output_dir / 'launch_report.md'}")
    print(f"Final config: {output_dir / 'final_config.json'}")
    print(f"Final command file: {output_dir / 'final_command.sh'}")
    return 0 if final_attempt.ready or args.dry_run else 2


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
