"""Request and response models for the API.

Request bodies are typed because that is where a bad value causes damage;
list responses are plain dicts assembled by the handlers, because pinning a
response schema to every projection the UI needs buys duplication rather than
safety.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    role: str
    team_id: int
    team_name: str
    team_slug: str


class LoginResponse(BaseModel):
    token: str
    user: UserOut


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["sft", "preference", "prompts", "benchmark"]
    description: str = ""


class LoraConfig(BaseModel):
    r: int = Field(default=16, ge=1, le=256)
    alpha: float = Field(default=32, gt=0, le=512)
    dropout: float = Field(default=0.05, ge=0, le=0.9)
    targets: list[str] | None = None


class RunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    method: str
    base_model: str
    params: dict[str, Any] = Field(default_factory=dict)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    train_dataset_id: int | None = None
    eval_dataset_id: int | None = None
    review_batch_id: int | None = None
    parent_model_version_id: int | None = None
    reward_model_version_id: int | None = None
    backend: Literal["auto", "torch", "tiny"] = "auto"
    priority: int = Field(default=0, ge=-10, le=10)


class ReviewBatchCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    protocol: Literal["pairwise", "rank", "rubric"] = "pairwise"
    prompt_dataset_id: int
    policy_model_version_id: int
    candidates_per_prompt: int = Field(default=2, ge=2, le=8)
    annotations_per_item: int = Field(default=1, ge=1, le=7)
    ai_assist_fraction: float = Field(default=0.0, ge=0, le=1)
    judge_model: str = ""
    limit: int = Field(default=100, ge=1, le=5000)


class AnnotationRequest(BaseModel):
    item_id: int
    choice: str | None = None                       # candidate id, "tie", "both_bad"
    ranking: list[str] = Field(default_factory=list)
    rubric_scores: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    confidence: float = Field(default=1.0, ge=0, le=1)
    latency_ms: int = 0


class AdjudicationRequest(BaseModel):
    item_id: int
    choice: str
    rationale: str = ""


class PromoteRequest(BaseModel):
    to: Literal["staging", "production", "archived"]
    notes: str = ""


class EvaluateRequest(BaseModel):
    dataset_id: int


class PlaygroundRequest(BaseModel):
    model_version_id: int | None = None
    base_model: str | None = None
    prompt: str = Field(min_length=1)
    system: str = ""
    max_new_tokens: int = Field(default=160, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0, le=2)
    compare_to_base: bool = False


class TeamCreate(BaseModel):
    slug: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9-]+$")
    name: str
    concurrency: int = Field(default=2, ge=1, le=64)


class UserCreate(BaseModel):
    email: str
    name: str
    password: str = Field(min_length=8)
    role: Literal["viewer", "member", "reviewer", "lead", "admin"] = "member"
    team_id: int


# ------------------------------------------------------------ serving & export

class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    # 'serve' can only call /v1. A credential baked into a production service
    # should not be able to delete a dataset.
    scope: Literal["serve", "full"] = "serve"


class ExportRequest(BaseModel):
    fmt: Literal["adapter", "merged"] = "adapter"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-shaped, so existing clients work unchanged.

    ``model`` accepts ``name`` (whatever is promoted), ``name@4``, ``name@staging``
    or ``#17``. Unsupported OpenAI fields are accepted and ignored rather than
    rejected — a client that always sends ``frequency_penalty`` should not fail.
    """
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0.01, le=1)
    n: int = Field(default=1, ge=1, le=4)
    stream: bool = False

    model_config = {"extra": "ignore", "protected_namespaces": ()}
