from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from examples.serving_lab.scripts import preflight_check


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_config_parsing_and_context_length(tmp_path: Path) -> None:
    write_json(
        tmp_path / "config.json",
        {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "max_position_embeddings": 32768,
        },
    )
    write_json(tmp_path / "tokenizer_config.json", {})

    results, config = preflight_check.check_model_path(tmp_path)

    assert config["model_type"] == "qwen2"
    assert preflight_check.get_context_length(config) == 32768
    assert any(result.name == "config.json" and result.status == "PASS"
               for result in results)


def test_moe_field_detection() -> None:
    config = {
        "model_type": "deepseek",
        "num_experts": 64,
        "num_experts_per_tok": 6,
        "hidden_size": 4096,
    }

    assert preflight_check.detect_moe_fields(config) == {
        "num_experts": 64,
        "num_experts_per_tok": 6,
    }


def test_run_checks_warns_for_large_max_model_len(tmp_path: Path) -> None:
    write_json(
        tmp_path / "config.json",
        {
            "model_type": "glm",
            "architectures": ["ChatGLMForConditionalGeneration"],
            "max_position_embeddings": 4096,
            "auto_map": {"AutoModel": "modeling.ChatGLM"},
        },
    )
    args = argparse.Namespace(
        model_path=str(tmp_path),
        port=0,
        max_model_len=8192,
        gpu_memory_utilization=0.99,
        dtype="bfloat16",
        trust_remote_code=False,
    )

    results = preflight_check.run_checks(args)

    assert any(result.name == "max_model_len" and result.status == "WARN"
               for result in results)
    assert any(result.name == "trust_remote_code" and result.status == "WARN"
               for result in results)


def test_invalid_json_fails(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{broken", encoding="utf-8")

    results, config = preflight_check.check_model_path(tmp_path)

    assert config == {}
    assert any(result.name == "config.json" and result.status == "FAIL"
               for result in results)


@pytest.mark.parametrize("dtype", ["auto", "float16", "bfloat16"])
def test_common_dtype_values(dtype: str) -> None:
    assert dtype in preflight_check.COMMON_DTYPES
