import re
from functools import partial
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample, hf_dataset
from inspect_ai.log import read_eval_log_sample_summaries
from inspect_ai.model import GenerateConfig, Model
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

from amelia_evals.config import ModelConfig, TaskConfig

FAIL_ON_ERROR = False
SCORE_ON_ERROR = True
RETRY_ON_ERROR = 1


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


def evaluation_metadata(
    *,
    model_name: str,
    model_config: ModelConfig,
    task_config: TaskConfig,
    limit: int | None,
    base_url: str | None = None,
) -> dict:
    metadata: dict = {
        "model_variant": model_name,
        "model_config": model_config.model_dump(mode="json"),
        "task_config": task_config.model_dump(mode="json"),
        "limit": limit,
        "execution": {"backend": "modal" if base_url is None else "endpoint"},
    }
    if base_url is not None:
        metadata["requested_server_config"] = {
            key: metadata["model_config"].pop(key)
            for key in ("gpu", "volume_prefix", "model_args")
        }
        metadata["execution"] = {
            "backend": "endpoint",
            "base_url": base_url,
            "server_config_verified": False,
        }
    return metadata


def run_evaluation(
    *,
    model: Model,
    task: Task,
    model_config: ModelConfig,
    metadata: dict,
    log_dir: str,
) -> list[dict]:
    logs = eval(
        tasks=task,
        model=model,
        max_connections=model_config.generation.max_connections,
        fail_on_error=FAIL_ON_ERROR,
        score_on_error=SCORE_ON_ERROR,
        retry_on_error=RETRY_ON_ERROR,
        metadata=metadata,
        log_dir=log_dir,
        display="plain",
    )
    return [
        {
            "model": metadata["model_variant"],
            "task": task.name,
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
