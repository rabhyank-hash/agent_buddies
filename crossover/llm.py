"""Chat-completions clients.

OpenAIChatLLM talks to any OpenAI-compatible server (e.g. vLLM serving
gpt-oss, Sail, or OpenAI itself). ScriptedLLM replays a policy function and is
used by the tests and --dry-run.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from typing import Any, Callable, Protocol

from crossover.types import LLMResponse, ToolCall


class ContextOverflow(Exception):
    """The request exceeded the model's context window."""


class LLM(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        tool_choice: Any = None,
    ) -> LLMResponse: ...


def _parse_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"_value": value}
    except json.JSONDecodeError:
        return {"_raw": raw}


_OVERFLOW_HINTS = ("context length", "maximum context", "context_length", "too long", "max_model_len")


class OpenAIChatLLM:
    def __init__(self, cfg):
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=os.environ.get(cfg.api_key_env, "EMPTY"),
            timeout=cfg.request_timeout,
            max_retries=3,
        )

    def chat(self, messages, tools, max_tokens, tool_choice=None) -> LLMResponse:
        import openai

        # max_completion_tokens is the current OpenAI name; vLLM accepts it and
        # Sail rejects the older max_tokens.
        kwargs: dict[str, Any] = {"model": self.cfg.name, "messages": messages, "max_completion_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if self.cfg.reasoning_effort:
            kwargs["reasoning_effort"] = self.cfg.reasoning_effort
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        if self.cfg.service_tier:
            kwargs["extra_body"] = {"service_tier": self.cfg.service_tier}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            if any(h in str(e).lower() for h in _OVERFLOW_HINTS):
                raise ContextOverflow(str(e)) from e
            raise
        choice = resp.choices[0]
        msg = choice.message
        extra = msg.model_extra or {}
        reasoning = getattr(msg, "reasoning_content", None) or extra.get("reasoning_content") or extra.get("reasoning")
        calls = [
            ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:8]}", name=tc.function.name,
                     arguments=_parse_args(tc.function.arguments), raw_arguments=tc.function.arguments or "")
            for tc in (msg.tool_calls or [])
        ]
        usage = resp.usage
        return LLMResponse(
            content=msg.content,
            reasoning=reasoning,
            tool_calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
            cached_tokens=_detail(usage, "prompt_tokens_details", "cached_tokens"),
            reasoning_tokens=_detail(usage, "completion_tokens_details", "reasoning_tokens"),
        )


def _detail(usage, group: str, key: str) -> int | None:
    """Read e.g. usage.prompt_tokens_details.cached_tokens, whether the SDK
    parsed it as an object or a dict; None if the server didn't report it."""
    obj = getattr(usage, group, None) if usage else None
    if obj is None:
        return None
    value = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return int(value) if value is not None else None


Policy = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None, Any], LLMResponse]


class ScriptedLLM:
    """Calls `policy(messages, tools, tool_choice)`; records every request."""

    def __init__(self, policy: Policy, context_window: int | None = None):
        self.policy = policy
        self.context_window = context_window
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def chat(self, messages, tools, max_tokens, tool_choice=None) -> LLMResponse:
        snapshot = json.loads(json.dumps(messages))
        with self._lock:
            self.requests.append({"messages": snapshot, "tools": tools, "tool_choice": tool_choice, "max_tokens": max_tokens})
        resp = self.policy(snapshot, tools, tool_choice)
        resp.completion_tokens = min(resp.completion_tokens, max_tokens)  # a real server enforces this
        if self.context_window and resp.prompt_tokens > self.context_window:
            raise ContextOverflow(f"prompt of {resp.prompt_tokens} tokens exceeds {self.context_window}")
        return resp


def build_llm(model_cfg) -> LLM:
    return OpenAIChatLLM(model_cfg)
