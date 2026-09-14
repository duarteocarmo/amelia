import argparse
import json
import os
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from inspect_ai.model import get_model

from amelia_evals.config import (
    PROJECT_ROOT,
    config_named,
    load_model_registry,
    load_task_registry,
)
from amelia_evals.evaluation import (
    evaluation_metadata,
    multiple_choice_task,
    run_evaluation,
)


def endpoint_url(value: str) -> str:
    url = urlsplit(url=value)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise argparse.ArgumentTypeError(
            "Use an HTTP(S) API base URL without credentials, query or fragment. "
            "Use VLLM_API_KEY for authentication."
        )
    return value.rstrip("/")


def positive_limit(value: str) -> int:
    limit = int(value)
    if limit <= 0:
        raise argparse.ArgumentTypeError("limit must be greater than zero")
    return limit


def check_endpoint_model(*, base_url: str, model_name: str) -> None:
    request = Request(
        url=f"{base_url}/models",
        headers={
            "Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'inspectai')}"
        },
    )
    with urlopen(url=request, timeout=10) as response:
        models = json.load(fp=response)
    available = [model["id"] for model in models["data"]]
    if model_name not in available:
        raise ValueError(
            f"Endpoint does not serve {model_name}. Available: {available}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing vLLM server.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--limit", type=positive_limit)
    parser.add_argument(
        "--base-url", type=endpoint_url, default="http://localhost:8000/v1"
    )
    args = parser.parse_args(args=argv)
    try:
        model_config = config_named(
            registry=load_model_registry(path=PROJECT_ROOT / "configs/models.yaml"),
            name=args.model,
            kind="model",
        )
        task_config = config_named(
            registry=load_task_registry(path=PROJECT_ROOT / "configs/tasks.yaml"),
            name=args.task,
            kind="task",
        )
        check_endpoint_model(base_url=args.base_url, model_name=model_config.model_name)
    except (OSError, ValueError) as error:
        parser.error(message=str(error))

    limit = args.limit if args.limit is not None else task_config.limit
    log_dir = PROJECT_ROOT / "logs" / args.task
    with get_model(
        model=f"vllm/{model_config.model_name}",
        base_url=args.base_url,
        lazy_init=False,
    ) as model:
        summaries = run_evaluation(
            model=model,
            task=multiple_choice_task(
                task_name=args.task,
                task_config=task_config,
                model_config=model_config,
                limit=limit,
            ),
            model_config=model_config,
            metadata=evaluation_metadata(
                model_name=args.model,
                model_config=model_config,
                task_config=task_config,
                limit=limit,
                base_url=args.base_url,
            ),
            log_dir=str(log_dir),
        )
    for summary in summaries:
        summary["local_log"] = str(log_dir / summary["log_file"])
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    if any(
        summary["status"] != "success" or summary["sample_errors"]
        for summary in summaries
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
