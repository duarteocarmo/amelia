import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import modal
from inspect_ai.model import get_model

from amelia_evals.config import (
    PROJECT_ROOT,
    ModelConfig,
    TaskConfig,
    config_named,
    load_model_registry,
    load_task_registry,
)
from amelia_evals.evaluation import (
    evaluation_metadata,
    multiple_choice_task,
    run_evaluation,
)

if TYPE_CHECKING:
    from pathlib import PurePosixPath

RUNNER_CONFIG = SimpleNamespace(
    **{  # noqa: PIE804
        "app_name": "amelia-evals-vllm",
        "models_path": PROJECT_ROOT / "configs/models.yaml",
        "tasks_path": PROJECT_ROOT / "configs/tasks.yaml",
        "function_timeout_minutes": 120,
        "scaledown_window_minutes": 15,
        "cuda_image": "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        "python_version": "3.12",
        "vllm_version": "0.21.0",
        "inspect_ai_version": "0.3.261",
        "datasets_version": "5.0.1",
        "transformers_version": "5.16.1",
        "pydantic_version": "2.13.5",
        "pyyaml_version": "6.0.3",
        "hf_secret_name": "huggingface",
        "hf_cache_dir": "/root/.cache/huggingface",
        "vllm_cache_dir": "/root/.cache/vllm",
        "remote_log_dir": "/logs",
        "local_log_dir": PROJECT_ROOT / "logs",
    }
)

app = modal.App(name=RUNNER_CONFIG.app_name)
image = (
    modal.Image.from_registry(
        tag=RUNNER_CONFIG.cuda_image,
        add_python=RUNNER_CONFIG.python_version,
    )
    .entrypoint(entrypoint_commands=[])
    .uv_pip_install(
        f"vllm=={RUNNER_CONFIG.vllm_version}",
        f"inspect-ai=={RUNNER_CONFIG.inspect_ai_version}",
        f"datasets=={RUNNER_CONFIG.datasets_version}",
        f"transformers=={RUNNER_CONFIG.transformers_version}",
        f"pydantic=={RUNNER_CONFIG.pydantic_version}",
        f"pyyaml=={RUNNER_CONFIG.pyyaml_version}",
    )
    .env(
        vars={
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "10",
        }
    )
)
hf_secret = modal.Secret.from_name(name=RUNNER_CONFIG.hf_secret_name)


@app.function(image=image, secrets=[hf_secret])
def run_eval(
    model_name: str,
    task_name: str,
    model_config: ModelConfig,
    task_config: TaskConfig,
    limit: int | None,
) -> list[dict]:
    model_args = model_config.model_args.model_dump(mode="json", exclude_none=True)
    try:
        # Start vLLM outside request deadlines and close it after evaluation.
        with get_model(
            model=f"vllm/{model_config.model_name}",
            lazy_init=False,
            **model_args,
        ) as model:
            return run_evaluation(
                model=model,
                task=multiple_choice_task(
                    task_name=task_name,
                    task_config=task_config,
                    model_config=model_config,
                    limit=limit,
                ),
                model_config=model_config,
                metadata=evaluation_metadata(
                    model_name=model_name,
                    model_config=model_config,
                    task_config=task_config,
                    limit=limit,
                ),
                log_dir=RUNNER_CONFIG.remote_log_dir,
            )
    finally:
        modal.Volume.from_name(name=model_config.volumes.logs).commit()


@app.local_entrypoint()
def main(model: str, task: str, limit: int | None = None) -> None:
    models = load_model_registry(path=RUNNER_CONFIG.models_path)
    tasks = load_task_registry(path=RUNNER_CONFIG.tasks_path)
    model_config = config_named(registry=models, name=model, kind="model")
    task_config = config_named(registry=tasks, name=task, kind="task")
    selected_limit = limit if limit is not None else task_config.limit
    log_volume = modal.Volume.from_name(
        name=model_config.volumes.logs,
        create_if_missing=True,
    )
    volumes: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
        RUNNER_CONFIG.hf_cache_dir: modal.Volume.from_name(
            name=model_config.volumes.huggingface,
            create_if_missing=True,
        ),
        RUNNER_CONFIG.vllm_cache_dir: modal.Volume.from_name(
            name=model_config.volumes.vllm,
            create_if_missing=True,
        ),
        RUNNER_CONFIG.remote_log_dir: log_volume,
    }

    summaries = run_eval.with_options(
        gpu=model_config.gpu,
        timeout=RUNNER_CONFIG.function_timeout_minutes * 60,
        scaledown_window=RUNNER_CONFIG.scaledown_window_minutes * 60,
        volumes=volumes,
    ).remote(
        model_name=model,
        task_name=task,
        model_config=model_config,
        task_config=task_config,
        limit=selected_limit,
    )
    task_log_dir = RUNNER_CONFIG.local_log_dir / task
    task_log_dir.mkdir(parents=True, exist_ok=True)

    for summary in summaries:
        log_file = summary["log_file"]
        local_path = task_log_dir / log_file
        with local_path.open(mode="wb") as destination:
            for chunk in log_volume.read_file(path=log_file):
                destination.write(chunk)
        summary["local_log"] = str(local_path)

    print(json.dumps(summaries, indent=2, ensure_ascii=False))
