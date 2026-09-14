import os
import subprocess
import sys
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model
from inspect_ai.solver import generate

from amelia_evals.config import PROJECT_ROOT, load_model_registry, load_task_registry
from amelia_evals.evaluation import (
    evaluation_metadata,
    generation_config_for,
    multiple_choice_scorer,
    run_evaluation,
)
from amelia_evals.rig_runner import check_endpoint_model


@pytest.mark.parametrize(
    argnames=("arguments", "expected"),
    argvalues=(
        (
            ["eval", "MODEL=x", "TASK=y"],
            "uv run modal run -m amelia_evals.modal_runner --model 'x' --task 'y'",
        ),
        (
            ["eval", "MODEL=x", "TASK=y", "LIMIT=2", "BASE_URL=http://example.test/v1"],
            "uv run modal run -m amelia_evals.modal_runner --model 'x' --task 'y' --limit '2'",
        ),
        (
            ["eval-rig", "MODEL=x", "TASK=y"],
            "uv run python -m amelia_evals.rig_runner --model 'x' --task 'y' --base-url 'http://localhost:8000/v1'",
        ),
        (
            ["eval-rig", "MODEL=x", "TASK=y", "LIMIT=2", "BASE_URL=http://example.test/v1"],
            "uv run python -m amelia_evals.rig_runner --model 'x' --task 'y' --base-url 'http://example.test/v1' --limit '2'",
        ),
    ),
)
def test_make_commands(arguments: list[str], expected: str) -> None:
    result = subprocess.run(
        args=["make", "-n", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == expected


def test_rig_cli_help() -> None:
    result = subprocess.run(
        args=[sys.executable, "-m", "amelia_evals.rig_runner", "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Evaluate an existing vLLM server" in result.stdout


@pytest.mark.parametrize(
    argnames=("option", "message"),
    argvalues=(
        (["--limit", "0"], "limit must be greater"),
        (["--limit", "-1"], "limit must be greater"),
        (["--limit", "abc"], "invalid positive_limit value"),
        (["--base-url", "https://user:pass@example.test/v1"], "without credentials"),
        (["--base-url", "file:///tmp/model"], "HTTP(S)"),
        (["--base-url", "http://localhost/v1?api_key=secret"], "without credentials"),
    ),
)
def test_invalid_arguments(option: list[str], message: str) -> None:
    result = subprocess.run(
        args=[sys.executable, "-m", "amelia_evals.rig_runner", "--model", "x", "--task", "y", *option],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert message in result.stderr


def test_metadata_separates_unverified_server_config() -> None:
    model_config = load_model_registry(path=PROJECT_ROOT / "configs/models.yaml")["qwen3.5-2b"]
    task_config = load_task_registry(path=PROJECT_ROOT / "configs/tasks.yaml")["pt-exams"]
    original = model_config.model_dump(mode="json")
    settings = dict(model_name="qwen3.5-2b", model_config=model_config, task_config=task_config, limit=1)
    modal_metadata = evaluation_metadata(**settings)
    rig_metadata = evaluation_metadata(**settings, base_url="http://localhost:8000/v1")
    assert modal_metadata["model_config"] == original
    assert modal_metadata["execution"] == {"backend": "modal"}
    assert "requested_server_config" not in modal_metadata
    assert set(rig_metadata["model_config"]) == {"model_name", "thinking", "generation"}
    assert rig_metadata["model_config"]["generation"] == original["generation"]
    assert rig_metadata["model_config"]["thinking"] == original["thinking"]
    assert rig_metadata["requested_server_config"] == {
        key: original[key] for key in ("gpu", "model_args", "volume_prefix")
    }
    assert rig_metadata["execution"] == {
        "backend": "endpoint",
        "base_url": "http://localhost:8000/v1",
        "server_config_verified": False,
    }
    assert model_config.model_dump(mode="json") == original


@pytest.mark.skipif(not os.environ.get("AMELIA_TEST_BASE_URL"), reason="real endpoint test is opt-in")
def test_real_endpoint_smoke(tmp_path: Path) -> None:
    base_url = os.environ["AMELIA_TEST_BASE_URL"].rstrip("/")
    model_variant = os.environ.get("AMELIA_TEST_MODEL", "qwen3.5-2b")
    model_config = load_model_registry(path=PROJECT_ROOT / "configs/models.yaml")[model_variant]
    task_config = load_task_registry(path=PROJECT_ROOT / "configs/tasks.yaml")["pt-exams"]
    check_endpoint_model(base_url=base_url, model_name=model_config.model_name)
    task = Task(
        dataset=[Sample(input=r"Qual é a capital de Portugal? (A) Porto (B) Lisboa. Responde com \boxed{A} ou \boxed{B}.", target="B")],
        solver=generate(),
        scorer=multiple_choice_scorer(letters="AB"),
        config=generation_config_for(model_config=model_config),
        name="rig-smoke",
    )
    metadata = evaluation_metadata(
        model_name=model_variant,
        model_config=model_config,
        task_config=task_config,
        limit=1,
        base_url=base_url,
    )
    with get_model(model=f"vllm/{model_config.model_name}", base_url=base_url, lazy_init=False) as model:
        summaries = run_evaluation(
            model=model,
            task=task,
            model_config=model_config,
            metadata=metadata,
            log_dir=str(tmp_path),
        )
    assert summaries[0]["status"] == "success"
    assert summaries[0]["sample_errors"] == 0
    log = read_eval_log(log_file=str(tmp_path / summaries[0]["log_file"]))
    assert log.eval.metadata == metadata
    expected_config = generation_config_for(model_config=model_config).model_copy(
        update={"max_connections": model_config.generation.max_connections}
    )
    assert log.plan.config == expected_config
    assert log.samples and log.samples[0].output.usage
    assert log.samples[0].output.usage.output_tokens > 0
