"""Per-episode token ledger over all agents.

Primary budget: total generated tokens across every agent (reasoning,
actions, summaries, plan writes), capped at B. At wrap_up_frac * B the
harness forces each thread to its finish tool; the last (1 - wrap_up_frac) * B
is reserved for that. Prefill (prompt) tokens are logged per call so the same
trajectories can be re-scored under the "total tokens" definition
(crossover.analysis.budget).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass

COORDINATION_TOOLS = {"start_subtask", "finish_subtask", "update_plan", "read_plan", "delegate", "report"}


@dataclass
class Step:
    idx: int
    agent_id: str
    thread_id: str
    prompt_tokens: int
    completion_tokens: int
    kind: str  # work | coordination | compaction
    t: float
    sample: int = 0  # self-consistency sample index
    cached_tokens: int | None = None  # prompt tokens served from cache, if reported
    reasoning_tokens: int | None = None  # share of completion_tokens, if reported


class TokenLedger:
    def __init__(self, budget: int, wrap_up_frac: float = 0.9, definition: str = "generated", min_call_tokens: int = 256):
        if definition not in ("generated", "total"):
            raise ValueError(f"Unknown budget definition {definition!r}")
        self.budget = budget
        self.wrap_up_frac = wrap_up_frac
        self.definition = definition
        self.min_call_tokens = min_call_tokens
        self.steps: list[Step] = []
        self.generated = 0
        self.prefill = 0
        self._t0 = time.time()
        self._lock = threading.Lock()

    def record(self, agent_id: str, thread_id: str, prompt_tokens: int, completion_tokens: int, kind: str,
               sample: int = 0, cached_tokens: int | None = None, reasoning_tokens: int | None = None) -> Step:
        with self._lock:
            step = Step(len(self.steps), agent_id, thread_id, prompt_tokens, completion_tokens, kind,
                        time.time() - self._t0, sample, cached_tokens, reasoning_tokens)
            self.steps.append(step)
            self.generated += completion_tokens
            self.prefill += prompt_tokens
            return step

    def absorb(self, other: "TokenLedger", sample: int) -> None:
        """Merge a sub-episode's ledger (self-consistency samples)."""
        for s in other.steps:
            self.record(s.agent_id, s.thread_id, s.prompt_tokens, s.completion_tokens, s.kind,
                        sample=sample, cached_tokens=s.cached_tokens, reasoning_tokens=s.reasoning_tokens)

    @property
    def spent(self) -> int:
        return self.generated if self.definition == "generated" else self.generated + self.prefill

    @property
    def wrap_up_at(self) -> int:
        return int(self.wrap_up_frac * self.budget)

    def wrap_up(self) -> bool:
        return self.spent >= self.wrap_up_at

    def exhausted(self) -> bool:
        return self.spent >= self.budget

    def max_tokens_for_next_call(self, per_call_cap: int) -> int:
        """Before wrap-up a call may not eat into the reserved answer budget."""
        limit = self.budget if self.wrap_up() else self.wrap_up_at
        return max(self.min_call_tokens, min(per_call_cap, limit - self.spent))

    def summary(self) -> dict:
        per_agent: dict[str, dict[str, int]] = defaultdict(lambda: {"generated": 0, "prefill": 0, "calls": 0})
        by_kind: dict[str, int] = defaultdict(int)
        for s in self.steps:
            a = per_agent[s.agent_id]
            a["generated"] += s.completion_tokens
            a["prefill"] += s.prompt_tokens
            a["calls"] += 1
            by_kind[s.kind] += s.completion_tokens
        cached = [s.cached_tokens for s in self.steps if s.cached_tokens is not None]
        reasoning = [s.reasoning_tokens for s in self.steps if s.reasoning_tokens is not None]
        return {
            "budget": self.budget,
            # None = the server never reported it (check this in the pilot).
            "cached": sum(cached) if cached else None,
            "reasoning": sum(reasoning) if reasoning else None,
            "definition": self.definition,
            "generated": self.generated,
            "prefill": self.prefill,
            "calls": len(self.steps),
            "coordination_share": by_kind.get("coordination", 0) / self.generated if self.generated else 0.0,
            "generated_by_kind": dict(by_kind),
            "per_agent": dict(per_agent),
        }

    def steps_as_dicts(self) -> list[dict]:
        return [asdict(s) for s in self.steps]


def estimate_cost(summary: dict, prices: dict[str, float] | None) -> float | None:
    """USD from a ledger summary and prices per 1M tokens {input, cached, output}.
    Cached prompt tokens are billed at `cached`, the rest of the prompt at `input`."""
    if not prices:
        return None
    cached = summary.get("cached") or 0
    uncached = summary["prefill"] - cached
    return (uncached * prices.get("input", 0) + cached * prices.get("cached", prices.get("input", 0))
            + summary["generated"] * prices.get("output", 0)) / 1e6
