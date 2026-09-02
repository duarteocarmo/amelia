import re

from inspect_ai import Task, task
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

DATASET_PATH = "amalia-llm/pt_exams"
LETTERS = "ABCD"
PROMPT = r"""Pergunta:
{question}
Opções:
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Escolha a opção correta para a pergunta acima. Apresente o seu raciocínio antes de responder. No final, apresente a sua resposta no formato \boxed{{X}}, onde X é a letra da opção correta (A, B, C ou D)."""


def record_to_sample(record: dict) -> Sample:
    choices = record["choices"]
    return Sample(
        input=PROMPT.format(
            question=record["question"].strip(),
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3],
        ),
        target=LETTERS[record["answer"]],
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
def pt_exams(config: str = "default", limit: int | None = None) -> Task:
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            name=config,
            split="test",
            sample_fields=record_to_sample,
            auto_id=True,
            limit=limit,
        ),
        solver=generate(),
        scorer=pt_exams_scorer(),
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=2048,
            stop_seqs=["</s>", "<|im_end|>", "<|endoftext|>"],
        ),
    )
