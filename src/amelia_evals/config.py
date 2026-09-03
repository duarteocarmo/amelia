from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VolumeConfig(Config):
    huggingface: str = Field(min_length=1)
    vllm: str = Field(min_length=1)
    logs: str = Field(min_length=1)

    @classmethod
    def from_prefix(cls, prefix: str) -> Self:
        return cls(
            huggingface=f"{prefix}-huggingface-cache",
            vllm=f"{prefix}-vllm-cache",
            logs=f"{prefix}-inspect-logs",
        )


class ThinkingConfig(Config):
    enabled: bool
    template_configurable: bool


class ModelArgsConfig(Config):
    model_config = ConfigDict(extra="allow", frozen=True)

    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_num_seqs: int = Field(gt=0)
    tensor_parallel_size: int = Field(gt=0)
    generation_config: str = Field(min_length=1)
    trust_remote_code: bool | None = None
    mamba_ssm_cache_dtype: Literal["auto", "float16", "float32"] | None = None


class ModelConfig(Config):
    model_name: str = Field(min_length=1)
    gpu: str = Field(min_length=1)
    volume_prefix: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    thinking: ThinkingConfig
    model_args: ModelArgsConfig

    @property
    def volumes(self) -> VolumeConfig:
        return VolumeConfig.from_prefix(prefix=self.volume_prefix)


class GenerationConfig(Config):
    max_model_len: int = Field(gt=0)
    max_connections: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)
    stop_sequences: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_token_budget(self) -> Self:
        if self.max_tokens > self.max_model_len:
            raise ValueError("max_tokens cannot exceed max_model_len")
        return self


class TaskConfig(Config):
    dataset_path: str = Field(min_length=1)
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_config: str = Field(min_length=1)
    split: str = Field(min_length=1)
    limit: int | None = Field(default=None, gt=0)
    question_field: str = Field(min_length=1)
    choices_field: str = Field(min_length=1)
    target_field: str = Field(min_length=1)
    target_type: Literal["index", "letter"]
    letters: str
    metadata_fields: tuple[str, ...]
    generation: GenerationConfig
    prompt: str = Field(min_length=1)

    @field_validator("letters")
    @classmethod
    def validate_letters(cls, letters: str) -> str:
        if len(letters) < 2 or not letters.isalpha() or not letters.isupper():
            raise ValueError("letters must contain at least two uppercase letters")
        if len(set(letters)) != len(letters):
            raise ValueError("letters must be unique")
        return letters

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, prompt: str) -> str:
        placeholders = ("{question}", "{choices}", "{valid_letters}")
        missing = [
            placeholder for placeholder in placeholders if placeholder not in prompt
        ]
        if missing:
            raise ValueError(f"prompt is missing placeholders: {', '.join(missing)}")
        return prompt


MODEL_REGISTRY = TypeAdapter(dict[str, ModelConfig])
TASK_REGISTRY = TypeAdapter(dict[str, TaskConfig])


def load_yaml(path: Path):
    with path.open(mode="r", encoding="utf-8") as source:
        return yaml.safe_load(stream=source)


def load_model_registry(path: Path) -> dict[str, ModelConfig]:
    return MODEL_REGISTRY.validate_python(load_yaml(path=path))


def load_task_registry(path: Path) -> dict[str, TaskConfig]:
    return TASK_REGISTRY.validate_python(load_yaml(path=path))


def config_named[ConfigType: Config](
    registry: dict[str, ConfigType], name: str, kind: str
) -> ConfigType:
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown {kind} '{name}'. Available: {available}")
    return registry[name]
