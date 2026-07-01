from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from examples.serving_lab.scripts import adaptive_launcher


def write_config(path: Path, data: dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(data), encoding="utf-8")


def test_config_reading_and_moe_detection(tmp_path: Path) -> None:
    write_config(
        tmp_path,
        {
            "model_type": "qwen2_moe",
            "architectures": ["Qwen2MoeForCausalLM"],
            "num_hidden_layers": 24,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32768,
            "num_experts": 60,
            "num_experts_per_tok": 4,
        },
    )

    summary = adaptive_launcher.load_model_summary(tmp_path)

    assert summary.fields["model_type"] == "qwen2_moe"
    assert summary.context_length == 32768
    assert summary.moe_fields == {"num_experts": 60, "num_experts_per_tok": 4}


def test_port_occupied_detection() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        assert not adaptive_launcher.is_port_available(port)
        assert adaptive_launcher.find_next_available_port(port + 1) >= port + 1


def test_dtype_fallback() -> None:
    config = adaptive_launcher.LaunchConfig(
        model_path="/models/qwen",
        served_model_name="qwen",
        dtype="bfloat16",
    )

    changed, action = adaptive_launcher.adjust_config(
        config,
        "dtype_incompatible",
        adaptive_launcher.ModelSummary({}, {}, None),
    )

    assert changed
    assert config.dtype == "float16"
    assert "bfloat16 -> float16" in action


def test_max_model_len_fallback() -> None:
    config = adaptive_launcher.LaunchConfig(
        model_path="/models/qwen",
        served_model_name="qwen",
        max_model_len=32768,
    )
    summary = adaptive_launcher.ModelSummary({}, {}, 8192)

    changed, action = adaptive_launcher.adjust_config(
        config, "max_model_len_too_large", summary
    )

    assert changed
    assert config.max_model_len == 8192
    assert "config context 8192" in action


def test_oom_and_kv_cache_classification() -> None:
    assert (
        adaptive_launcher.classify_failure(
            "torch.cuda.OutOfMemoryError: CUDA out of memory"
        )
        == "cuda_oom"
    )
    assert (
        adaptive_launcher.classify_failure("ValueError: KV Cache is not enough")
        == "kv_cache_insufficient"
    )


def test_readiness_timeout_classification() -> None:
    assert (
        adaptive_launcher.classify_failure("", "readiness timeout: connection refused")
        == "readiness_timeout"
    )


def test_command_generation() -> None:
    config = adaptive_launcher.LaunchConfig(
        model_path="/models/qwen",
        served_model_name="qwen",
        port=8010,
        dtype="float16",
        trust_remote_code=True,
    )

    command = adaptive_launcher.command_for_config(config)

    assert command[:3] == ["vllm", "serve", "/models/qwen"]
    assert "--served-model-name" in command
    assert "qwen" in command
    assert "--trust-remote-code" in command


def test_dry_run_does_not_start_vllm(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    output_dir = tmp_path / "out"
    write_config(
        model_path,
        {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "max_position_embeddings": 4096,
        },
    )
    args = argparse.Namespace(
        model_path=str(model_path),
        served_model_name="qwen",
        host="127.0.0.1",
        port=8000,
        dtype="auto",
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        trust_remote_code=False,
        max_retries=4,
        ready_timeout=1,
        output_dir=str(output_dir),
        dry_run=True,
        extra_arg=None,
    )

    _, attempts, _ = adaptive_launcher.run_launcher(args)

    assert attempts[0].failure_type == "dry_run"
    assert (output_dir / "final_config.json").exists()
    assert (output_dir / "final_command.sh").exists()
    assert (output_dir / "launch_report.md").exists()
