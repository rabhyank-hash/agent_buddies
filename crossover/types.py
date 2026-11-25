"""Plain data types shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Query:
    query_id: str
    question: str
    answer: str
    gold_docids: list[str] = field(default_factory=list)
    evidence_docids: list[str] = field(default_factory=list)
    negative_docids: list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class LLMResponse:
    content: str | None
    reasoning: str | None
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None
    cached_tokens: int | None = None  # None = the server did not report it
    reasoning_tokens: int | None = None


@dataclass
class Stop:
    """Why a thread's loop ended.

    kind: "finish" (the thread called its finish tool or replied without a
    tool call), "budget", "overflow" or "max_turns".
    """

    kind: str
    args: dict[str, Any] = field(default_factory=dict)
