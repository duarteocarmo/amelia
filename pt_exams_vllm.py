import json
import re
from pathlib import Path
from types import SimpleNamespace

import modal
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
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

VLLM_CONFIG = SimpleNamespace(
    **{  # noqa: PIE804
        "app_name": "lfm25-pt-exams-vllm",
        "model_name": "LiquidAI/LFM2.5-2.6B",
        "gpu": "A100",
        "function_timeout_minutes": 120,
        "scaledown_window_minutes": 15,
        "cuda_image": "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        "python_version": "3.12",
        "vllm_version": "0.21.0",
        "hf_secret_name": "huggingface",
        "hf_cache_volume_name": "lfm25-huggingface-cache",
        "vllm_cache_volume_name": "lfm25-vllm-cache",
        "hf_cache_dir": "/root/.cache/huggingface",
        "vllm_cache_dir": "/root/.cache/vllm",
        "model_args": {
            "revision": "654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "max_num_seqs": 32,
            "tensor_parallel_size": 1,
            "generation_config": "vllm",
        },
    }
)

EVAL_CONFIG = SimpleNamespace(
    **{  # noqa: PIE804
        "dataset_path": "amalia-llm/pt_exams",
        "dataset_config": "default",
        "limit": None,
        "max_connections": 32,
        "max_tokens": 2048,
        "temperature": 0.0,
        "stop_sequences": ("</s>", "<|im_end|>", "<|endoftext|>"),
        "inspect_ai_version": "0.3.261",
        "datasets_version": "5.0.1",
        "log_volume_name": "lfm25-inspect-logs",
        "remote_log_dir": "/logs",
        "local_log_dir": Path("logs"),
        "letters": "ABCD",
        "prompt": r"""Pergunta:
{question}
Opções:
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Escolha a opção correta para a pergunta acima. Apresente o seu raciocínio antes de responder. No final, apresente a sua resposta no formato \boxed{{X}}, onde X é a letra da opção correta (A, B, C ou D).""",
    }
)

app = modal.App(name=VLLM_CONFIG.app_name)
image = (
    modal.Image.from_registry(
        tag=VLLM_CONFIG.cuda_image,
        add_python=VLLM_CONFIG.python_version,
    )
    .entrypoint(entrypoint_commands=[])
    .uv_pip_install(
        f"vllm=={VLLM_CONFIG.vllm_version}",
        f"inspect-ai=={EVAL_CONFIG.inspect_ai_version}",
        f"datasets=={EVAL_CONFIG.datasets_version}",
    )
    .env(
        vars={
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "10",
        }
    )
)
hf_cache_volume = modal.Volume.from_name(
    name=VLLM_CONFIG.hf_cache_volume_name,
    create_if_missing=True,
)
vllm_cache_volume = modal.Volume.from_name(
    name=VLLM_CONFIG.vllm_cache_volume_name,
    create_if_missing=True,
)
log_volume = modal.Volume.from_name(
    name=EVAL_CONFIG.log_volume_name,
    create_if_missing=True,
)
hf_secret = modal.Secret.from_name(name=VLLM_CONFIG.hf_secret_name)


def record_to_sample(record: dict) -> Sample:
    choices = record["choices"]
    return Sample(
        input=EVAL_CONFIG.prompt.format(
            question=record["question"].strip(),
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3],
        ),
        target=EVAL_CONFIG.letters[record["answer"]],
        metadata={
            "year": record["year"],
            "phase": record["phase"],
            "subject": record["subject"],
            "question_group": record["question_group"],
            "question_number": record["question_number"],
            "is_completion": record["is_completion"],
        },
    )


def extract_answer(response: str) -> str | None:
    patterns = (
        (r"\\boxed{([A-D])}", re.IGNORECASE),
        (
            r"(?:resposta correta|resposta certa|resposta verdadeira|answer|opção correta|opção certa|opção verdadeira|correct option).{0,10}([A-D])\b",
            re.IGNORECASE,
        ),
        (r"([A-D])\.?\s*$", 0),
        (r"\b([A-D])\b", 0),
    )
    for pattern, flags in patterns:
        match = re.search(pattern=pattern, string=response, flags=flags)
        if match:
            return match.group(1).upper()
    return None


@scorer(metrics=[accuracy(), stderr()])
def pt_exams_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = extract_answer(response=state.output.completion)
        return Score(
            value=CORRECT if answer == target.text else INCORRECT,
            answer=answer or "[not extracted]",
            explanation=state.output.completion,
        )

    return score


@task
def pt_exams(
    config: str = EVAL_CONFIG.dataset_config,
    limit: int | None = EVAL_CONFIG.limit,
) -> Task:
    return Task(
        dataset=hf_dataset(
            path=EVAL_CONFIG.dataset_path,
            name=config,
            split="test",
            sample_fields=record_to_sample,
            auto_id=True,
            limit=limit,
        ),
        solver=generate(),
        scorer=pt_exams_scorer(),
        config=GenerateConfig(
            temperature=EVAL_CONFIG.temperature,
            max_tokens=EVAL_CONFIG.max_tokens,
            stop_seqs=EVAL_CONFIG.stop_sequences,
        ),
    )


@app.function(
    image=image,
    secrets=[hf_secret],
    gpu=VLLM_CONFIG.gpu,
    timeout=VLLM_CONFIG.function_timeout_minutes * 60,
    scaledown_window=VLLM_CONFIG.scaledown_window_minutes * 60,
    volumes={
        VLLM_CONFIG.hf_cache_dir: hf_cache_volume,
        VLLM_CONFIG.vllm_cache_dir: vllm_cache_volume,
        EVAL_CONFIG.remote_log_dir: log_volume,
    },
)
def run_eval(dataset_config: str, limit: int) -> list[dict]:
    try:
        logs = eval(
            tasks=pt_exams(config=dataset_config, limit=limit),
            model=f"vllm/{VLLM_CONFIG.model_name}",
            model_args=VLLM_CONFIG.model_args,
            max_connections=EVAL_CONFIG.max_connections,
            log_dir=EVAL_CONFIG.remote_log_dir,
            display="plain",
        )
        return [
            {
                "status": log.status,
                "log_file": Path(log.location).name,
                "results": log.results.model_dump(mode="json") if log.results else None,
                "error": log.error.model_dump(mode="json") if log.error else None,
            }
            for log in logs
        ]
    finally:
        log_volume.commit()


@app.local_entrypoint()
def main(
    dataset_config: str = EVAL_CONFIG.dataset_config,
    limit: int = EVAL_CONFIG.limit,
) -> None:
    summaries = run_eval.remote(dataset_config=dataset_config, limit=limit)
    EVAL_CONFIG.local_log_dir.mkdir(parents=True, exist_ok=True)

    for summary in summaries:
        log_file = summary["log_file"]
        local_path = EVAL_CONFIG.local_log_dir / log_file
        with local_path.open(mode="wb") as destination:
            for chunk in log_volume.read_file(path=log_file):
                destination.write(chunk)
        summary["local_log"] = str(local_path)

    print(json.dumps(summaries, indent=2, ensure_ascii=False))
