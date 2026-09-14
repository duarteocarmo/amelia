import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from amelia_evals.config import (
    GenerationConfig,
    ModelConfig,
    TaskConfig,
    load_model_registry,
    load_task_registry,
)
from amelia_evals.evaluation import generation_config_for

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = load_model_registry(path=PROJECT_ROOT / "configs/models.yaml")
TASKS = load_task_registry(path=PROJECT_ROOT / "configs/tasks.yaml")


@pytest.mark.parametrize(
    argnames=(
        "model_name",
        "temperature",
        "top_p",
        "max_tokens",
        "thinking",
        "timeout",
        "concurrency",
    ),
    argvalues=(
        ("smollm3-3b", 0.6, 0.95, 32768, True, 120, 32),
        ("smollm3-3b-no-think", 0.6, 0.95, 16384, False, 120, 32),
        ("amalia-9b-sft", 0.0, 1.0, 16384, None, 300, 16),
        ("amalia-9b-dpo", 0.0, 1.0, 16384, None, 300, 16),
        ("lfm2.5-2.6b", 0.1, 1.0, 32768, None, 120, 32),
        ("nemotron-3-nano-4b", 1.0, 0.95, 32768, True, 120, 32),
    ),
)
def test_generation_config(
    *,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    thinking: bool | None,
    timeout: int,
    concurrency: int,
) -> None:
    model_config = MODELS[model_name]
    config = generation_config_for(model_config=model_config)
    assert config.temperature == temperature
    assert config.top_p == top_p
    assert config.seed == 42
    assert config.timeout == timeout
    assert config.attempt_timeout == timeout
    assert config.max_retries == 0
    assert config.max_tokens == max_tokens
    assert model_config.model_args.max_model_len == max_tokens + 8192
    assert model_config.generation.max_connections == concurrency
    assert model_config.model_args.max_num_seqs == concurrency
    assert tuple(config.stop_seqs or ()) == ("</s>", "<|im_end|>", "<|endoftext|>")
    expected_body = {}
    if thinking is not None:
        expected_body["chat_template_kwargs"] = {"enable_thinking": thinking}
    if model_name == "lfm2.5-2.6b":
        expected_body.update(top_k=50, repetition_penalty=1.1)
    assert config.extra_body == (expected_body or None)


@pytest.mark.parametrize(
    argnames="model_name",
    argvalues=("qwen3.5-2b", "qwen3.5-4b", "qwen3.5-4b-no-think"),
)
def test_qwen_generation_override_is_cli_json(model_name: str) -> None:
    model_args = MODELS[model_name].model_args.model_dump(
        mode="json", exclude_none=True
    )
    override = model_args["override_generation_config"]
    assert isinstance(override, str)
    assert json.loads(s=override) == {"presence_penalty": 1.5}


@pytest.mark.parametrize(
    argnames=("model_name", "concurrency"),
    argvalues=(("qwen3.5-2b", 48), ("qwen3.5-4b", 32), ("qwen3.5-4b-no-think", 32)),
)
def test_qwen_runtime_limits(model_name: str, concurrency: int) -> None:
    model = MODELS[model_name]
    config = generation_config_for(model_config=model)
    assert config.timeout == 300
    assert config.attempt_timeout == 300
    assert model.generation.max_connections == concurrency
    assert model.model_args.max_num_seqs == concurrency
    assert config.max_tokens == 32768
    assert model.gpu == "A100-80GB"


def test_qwen_4b_variants_only_differ_in_thinking() -> None:
    thinking = MODELS["qwen3.5-4b"]
    no_thinking = MODELS["qwen3.5-4b-no-think"]
    assert thinking.model_dump(exclude={"thinking"}) == no_thinking.model_dump(
        exclude={"thinking"}
    )
    assert thinking.thinking.enabled
    assert not no_thinking.thinking.enabled
    assert no_thinking.thinking.template_configurable
    config = generation_config_for(model_config=no_thinking)
    assert config.extra_body == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_smollm_variants_only_differ_in_thinking_and_budgets() -> None:
    thinking = MODELS["smollm3-3b"]
    no_thinking = MODELS["smollm3-3b-no-think"]
    excluded = {
        "thinking": True,
        "generation": {"max_tokens"},
        "model_args": {"max_model_len"},
    }
    assert thinking.model_dump(exclude=excluded) == no_thinking.model_dump(
        exclude=excluded
    )
    assert thinking.thinking.enabled
    assert not no_thinking.thinking.enabled
    assert thinking.thinking.template_configurable
    assert no_thinking.thinking.template_configurable


def test_amalia_variants_share_generation_settings() -> None:
    assert MODELS["amalia-9b-sft"].generation == MODELS["amalia-9b-dpo"].generation


@pytest.mark.parametrize(argnames="task_name", argvalues=tuple(TASKS))
def test_tasks_cannot_override_generation(task_name: str) -> None:
    task = TASKS[task_name].model_dump()
    assert "generation" not in task
    with pytest.raises(expected_exception=ValidationError, match="Extra inputs"):
        TaskConfig.model_validate(obj={**task, "generation": {"max_tokens": 128}})


@pytest.mark.parametrize(
    argnames="settings",
    argvalues=(
        {"timeout": 0},
        {"timeout": -1},
        {"max_retries": -1},
        {"temperature": -0.1},
        {"top_p": 0},
        {"top_p": 1.1},
        {"seed": -1},
        {"max_tokens": 0},
        {"max_connections": 0},
        {"top_k": 0},
        {"repetition_penalty": 0},
        {"stop_sequences": []},
        {"max_model_len": 40960},
    ),
)
def test_invalid_generation_config(settings: dict) -> None:
    generation = MODELS["smollm3-3b"].generation.model_dump()
    with pytest.raises(expected_exception=ValidationError):
        GenerationConfig.model_validate(obj={**generation, **settings})


@pytest.mark.parametrize(argnames="max_tokens", argvalues=(40960, 40961))
def test_model_budget_must_leave_room_for_input(max_tokens: int) -> None:
    model = MODELS["smollm3-3b"].model_dump()
    model["generation"]["max_tokens"] = max_tokens
    with pytest.raises(
        expected_exception=ValidationError, match="leave room for input"
    ):
        ModelConfig.model_validate(obj=model)


def test_thinking_switch_preserves_vllm_sampling_fields() -> None:
    model = MODELS["lfm2.5-2.6b"].model_dump()
    model["thinking"] = {"enabled": False, "template_configurable": True}
    config = generation_config_for(model_config=ModelConfig.model_validate(obj=model))
    assert config.extra_body == {
        "top_k": 50,
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
