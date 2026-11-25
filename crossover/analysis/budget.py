"""Budget-capped re-scoring under two budget definitions.

The runs are capped on generated tokens. The same trajectories are re-scored
at a budget B under either definition:
  generated  sum of completion tokens across all agents
  total      generated + prefill (multi-agent systems re-read prompts)

Rule (fixed before the runs): a trajectory counts as correct at budget B only
if it was correct AND the cumulative spend up to and including the call that
produced the final answer is <= B. Otherwise it scores 0 (truncated).
"""

from __future__ import annotations

from typing import Sequence


def spend_at_answer(record: dict, definition: str) -> float:
    idx = record.get("answer_step")
    if idx is None:
        return float("inf")
    total = 0
    for s in record["steps"][: idx + 1]:
        total += s["completion_tokens"] + (s["prompt_tokens"] if definition == "total" else 0)
    return float(total)


def correct_at_budget(record: dict, budget: float, definition: str) -> bool:
    return bool(record.get("correct")) and spend_at_answer(record, definition) <= budget


def accuracy_curve(rows: Sequence[dict], budgets: Sequence[float], definition: str) -> list[dict]:
    out = []
    for b in budgets:
        n = len(rows)
        acc = sum(correct_at_budget(r, b, definition) for r in rows) / n if n else 0.0
        out.append({"budget": b, "accuracy": acc, "n": n})
    return out


def rescore(rows: Sequence[dict], budget: float, definition: str) -> list[dict]:
    """Copies of rows with `correct` replaced by the budget-capped outcome."""
    return [{**r, "correct": correct_at_budget(r, budget, definition)} for r in rows]
