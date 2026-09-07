import json
import re
from functools import partial
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import modal
from inspect_ai import Task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample, hf_dataset
from inspect_ai.log import read_eval_log_sample_summaries
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate

from amelia_evals.config import (
    ModelConfig,
    TaskConfig,
    config_named,
    load_model_registry,
    load_task_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_CONFIG = SimpleNamespace(
    **{  # noqa: PIE804
        "app_name": "amelia-evals-vllm",
        "models_path": PROJECT_ROOT / "configs/models.yaml",
        "tasks_path": PROJECT_ROOT / "configs/tasks.yaml",
        "function_timeout_minutes": 120,
        "scaledown_window_minutes": 15,
        "fail_on_error": False,
        "score_on_error": True,
        "retry_on_error": 1,
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


def value_at(record: dict, field: str):
    value = record
    for key in field.split("."):
        value = value[key]
    return value


def format_letters(letters: str) -> str:
    return f"{', '.join(letters[:-1])} ou {letters[-1]}"


def record_to_sample(record: dict, task_config: TaskConfig) -> Sample | list[Sample]:
    if (
        task_config.filter_field is not None
        and value_at(
            record=record,
            field=task_config.filter_field,
        )
        != task_config.filter_value
    ):
        return []

    letters = task_config.letters
    question = value_at(record=record, field=task_config.question_field)
    if task_config.question_suffix is not None:
        question = question.removesuffix(task_config.question_suffix).rstrip()
    target_value = value_at(record=record, field=task_config.target_field)

    if task_config.target_type == "index":
        target = letters[target_value]
    else:
        target = str(target_value).upper()

    choice_lines = ""
    if task_config.choices_field is not None:
        choices = value_at(record=record, field=task_config.choices_field)
        choice_lines = "\n".join(
            task_config.choice_format.format(letter=letter, choice=choice)
            for letter, choice in zip(letters, choices, strict=True)
        )

    return Sample(
        input=task_config.prompt.format(
            question=question.strip(),
            choices=choice_lines,
            valid_letters=format_letters(letters=letters),
        ),
        target=target,
        metadata={
            field: value_at(record=record, field=field)
            for field in task_config.metadata_fields
        },
    )


def extract_answer(response: str, letters: str) -> str | None:
    letter_class = re.escape(pattern=letters)
    matches = re.findall(
        pattern=rf"\\boxed{{([{letter_class}])}}",
        string=response,
        flags=re.IGNORECASE,
    )
    return matches[-1].upper() if matches else None


def task_dataset(
    task_name: str,
    task_config: TaskConfig,
    limit: int | None,
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> Dataset:
    configs = task_config.dataset_config
    configs = (configs,) if isinstance(configs, str) else configs

    if len(configs) == 1:
        dataset = hf_dataset(
            path=task_config.dataset_path,
            name=configs[0],
            revision=task_config.dataset_revision,
            split=task_config.split,
            sample_fields=partial(record_to_sample, task_config=task_config),
            auto_id=True,
            shuffle=shuffle,
            seed=seed,
        )
        if limit is None:
            return dataset
        return MemoryDataset(
            samples=list(dataset)[:limit],
            name=task_name,
            location=task_config.dataset_path,
        )

    samples = []
    for dataset_config in configs:
        remaining = None if limit is None else limit - len(samples)
        if remaining is not None and remaining <= 0:
            break
        dataset = hf_dataset(
            path=task_config.dataset_path,
            name=dataset_config,
            revision=task_config.dataset_revision,
            split=task_config.split,
            sample_fields=partial(record_to_sample, task_config=task_config),
            auto_id=True,
            shuffle=shuffle,
            seed=seed,
        )
        selected_samples = list(dataset)
        if remaining is not None:
            selected_samples = selected_samples[:remaining]
        samples.extend(
            sample.model_copy(update={"id": f"{dataset_config}:{sample.id}"})
            for sample in selected_samples
        )

    return MemoryDataset(
        samples=samples,
        name=task_name,
        location=task_config.dataset_path,
    )


@scorer(metrics=[accuracy(), stderr()])
def multiple_choice_scorer(letters: str) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = extract_answer(response=state.output.completion, letters=letters)
        return Score(
            value=CORRECT if answer == target.text else INCORRECT,
            answer=answer or "[not extracted]",
            explanation=state.output.completion,
        )

    return score


def generation_config_for(model_config: ModelConfig) -> GenerateConfig:
    generation = model_config.generation
    thinking = model_config.thinking
    # These vLLM sampling parameters are not forwarded by Inspect's OpenAI fields.
    extra_body = generation.model_dump(
        include={"top_k", "repetition_penalty"}, exclude_none=True
    )
    if thinking.template_configurable:
        extra_body["chat_template_kwargs"] = {"enable_thinking": thinking.enabled}
    return GenerateConfig(
        timeout=generation.timeout,
        attempt_timeout=generation.timeout,
        max_retries=generation.max_retries,
        temperature=generation.temperature,
        top_p=generation.top_p,
        seed=generation.seed,
        max_tokens=generation.max_tokens,
        stop_seqs=generation.stop_sequences,
        extra_body=extra_body or None,
    )


def multiple_choice_task(
    task_name: str,
    task_config: TaskConfig,
    model_config: ModelConfig,
    limit: int | None,
) -> Task:
    return Task(
        dataset=task_dataset(
            task_name=task_name,
            task_config=task_config,
            limit=limit,
        ),
        solver=generate(),
        scorer=multiple_choice_scorer(letters=task_config.letters),
        config=generation_config_for(model_config=model_config),
        name=task_name,
    )


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
            logs = eval(
                tasks=multiple_choice_task(
                    task_name=task_name,
                    task_config=task_config,
                    model_config=model_config,
                    limit=limit,
                ),
                model=model,
                max_connections=model_config.generation.max_connections,
                fail_on_error=RUNNER_CONFIG.fail_on_error,
                score_on_error=RUNNER_CONFIG.score_on_error,
                retry_on_error=RUNNER_CONFIG.retry_on_error,
                metadata={
                    "model_variant": model_name,
                    "model_config": model_config.model_dump(mode="json"),
                    "task_config": task_config.model_dump(mode="json"),
                    "limit": limit,
                },
                log_dir=RUNNER_CONFIG.remote_log_dir,
                display="plain",
            )
        return [
            {
                "model": model_name,
                "task": task_name,
                "status": log.status,
                "log_file": Path(log.location).name,
                "results": log.results.model_dump(mode="json") if log.results else None,
                "error": log.error.model_dump(mode="json") if log.error else None,
                "sample_errors": sum(
                    sample.error is not None
                    for sample in read_eval_log_sample_summaries(log_file=log.location)
                ),
            }
            for log in logs
        ]
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
