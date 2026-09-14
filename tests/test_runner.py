from collections import Counter
from pathlib import Path

import anyio
import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT
from inspect_ai.solver import Generate, Solver, TaskState, solver
from tenacity import retry, stop_after_attempt

from amelia_evals.config import PROJECT_ROOT, load_model_registry
from amelia_evals.evaluation import (
    extract_answer,
    multiple_choice_scorer,
    run_evaluation,
)


@pytest.mark.parametrize(
    argnames=("response", "letters", "expected"),
    argvalues=(
        (r"A resposta é \boxed{B}.", "ABCD", "B"),
        (r"A resposta é \boxed{b}.", "ABCD", "B"),
        (r"Primeiro \boxed{A}, finalmente \boxed{C}.", "ABCD", "C"),
        ("The options are A, B, C, or D.", "ABCD", None),
        ("A resposta correta é B.", "ABCD", None),
        ("B", "ABCD", None),
        (r"\boxed{D}", "ABC", None),
    ),
)
def test_extract_answer(response: str, letters: str, expected: str | None) -> None:
    assert extract_answer(response=response, letters=letters) == expected


@pytest.mark.parametrize(argnames="recovers", argvalues=(False, True))
def test_timeout_retries_once_then_scores(tmp_path: Path, *, recovers: bool) -> None:
    attempts = Counter()

    @retry(stop=stop_after_attempt(max_attempt_number=1))
    async def expire_request() -> None:
        with anyio.fail_after(delay=0.01):
            await anyio.sleep(delay=1)

    @solver
    def timed_answer() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            attempts[state.sample_id] += 1
            if state.sample_id == 1 and (not recovers or attempts[1] == 1):
                await expire_request()
            state.output = ModelOutput.from_content(
                model="none/none", content=r"\boxed{B}"
            )
            return state

        return solve

    model_config = load_model_registry(path=PROJECT_ROOT / "configs/models.yaml")["qwen3.5-2b"]
    summaries = run_evaluation(
        task=Task(
            dataset=[
                Sample(id=1, input="Timeout", target="B"),
                Sample(id=2, input="Answer", target="B"),
            ],
            solver=timed_answer(),
            scorer=multiple_choice_scorer(letters="ABCD"),
        ),
        model=get_model(model="none/none"),
        model_config=model_config,
        metadata={"model_variant": "test-timeouts"},
        log_dir=str(tmp_path),
    )
    assert summaries[0]["sample_errors"] == (0 if recovers else 1)
    log = read_eval_log(log_file=str(tmp_path / summaries[0]["log_file"]))
    assert log.status == "success"
    assert log.samples is not None
    assert len(log.samples) == 2
    retried, completed = sorted(log.samples, key=lambda sample: sample.id)
    assert (retried.error is None) == recovers
    assert len(retried.error_retries or []) == 1
    assert "TimeoutError" in retried.error_retries[0].message
    assert completed.error is None
    assert retried.scores is not None
    assert completed.scores is not None
    assert retried.scores["multiple_choice_scorer"].value == (
        CORRECT if recovers else INCORRECT
    )
    assert completed.scores["multiple_choice_scorer"].value == CORRECT
    assert attempts == {1: 2, 2: 1}
    assert log.results is not None
    assert log.results.total_samples == 2
    assert log.results.scores[0].metrics["accuracy"].value == (1.0 if recovers else 0.5)
    summaries = read_eval_log_sample_summaries(log_file=log.location)
    assert sum(sample.error is not None for sample in summaries) == (
        0 if recovers else 1
    )
