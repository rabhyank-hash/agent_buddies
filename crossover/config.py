"""Experiment configuration: YAML file -> nested dataclasses, with dotted overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "openai/gpt-oss-20b"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "VLLM_API_KEY"
    reasoning_effort: str | None = "high"
    temperature: float | None = None  # None = the model's default sampling
    max_tokens_per_call: int = 16384
    context_window: int = 131072
    # Carry each turn's reasoning back into the next request. Verify in the
    # pilot that prompt_tokens actually grows when this is on: servers differ
    # in which field they read (reasoning_field).
    carry_reasoning: bool = True
    reasoning_field: str = "reasoning_content"
    request_timeout: float = 900.0


@dataclass
class DataConfig:
    queries_path: str = "data/browsecomp_plus_decrypted.jsonl"
    corpus_path: str = "data/corpus.jsonl"
    subset_size: int | None = 500
    subset_seed: int = 768
    # Queries used for prompt tuning; excluded from every evaluation subset.
    pilot_size: int = 100


@dataclass
class RetrievalConfig:
    backend: str = "bm25"  # bm25 | dense | http
    http_url: str = "http://localhost:8001"
    dense_index_glob: str = "data/index/qwen3-embedding-8b/corpus.*.pkl"
    dense_model: str = "Qwen/Qwen3-Embedding-8B"
    dense_query_prefix: str = ""  # must match how the index was built
    k: int = 5
    snippet_tokens: int = 512


@dataclass
class PressureConfig:
    levels: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    protect: str = "gold"  # gold | gold+evidence
    seed: int = 768


@dataclass
class BudgetConfig:
    B: int = 60000  # set after the pilot to ~p75 of L0's natural generated tokens
    wrap_up_frac: float = 0.9
    definition: str = "generated"  # generated | total (generated + prefill)


@dataclass
class AgentConfig:
    max_turns: int = 80
    max_subtask_turns: int = 30
    relay_cap_tokens: int | None = None  # 64 / 256 / 1024 for the bandwidth dial
    parent_can_search: bool = True
    max_parallel_workers: int = 4
    force_branch_after_searches: int | None = None
    summary_trigger_tokens: int = 90000  # L0 + summarization compacts above this
    sc_samples: int = 4


@dataclass
class JudgeConfig:
    model: str = "gpt-4.1"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class OutputConfig:
    run_dir: str = "runs/default"
    save_final_context: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    pressure: PressureConfig = field(default_factory=PressureConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    tokenizer: str = "o200k_harmony"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, data: dict[str, Any]):
    kwargs = {}
    names = {f.name for f in dataclasses.fields(cls)}
    for key, value in (data or {}).items():
        if key not in names:
            raise KeyError(f"Unknown config key {cls.__name__}.{key}")
        sub = _SECTIONS.get(key) if cls is Config else None
        kwargs[key] = _build(sub, value) if sub else value
    return cls(**kwargs)


_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "retrieval": RetrievalConfig,
    "pressure": PressureConfig,
    "budget": BudgetConfig,
    "agents": AgentConfig,
    "judge": JudgeConfig,
    "output": OutputConfig,
}


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply `section.key=value` overrides; values are parsed as YAML."""
    for item in overrides:
        dotted, _, value = item.partition("=")
        node = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return raw


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    raw = apply_overrides(raw, overrides or [])
    return _build(Config, raw)
