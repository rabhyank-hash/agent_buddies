"""The length-neutral pressure dial.

search(query, level) -> k results, each cut to `snippet_tokens`. Every non-gold
result slot is replaced by one of the query's labelled hard negatives with
probability `level`. Gold documents are never removed and keep their natural
ranks, so for any fixed sequence of search queries the gold documents shown
are identical at every level (tests/test_pressure.py checks this).

Design choices (from the gap-check doc, "The pressure dial needs exact rules"):
  * Replacement is slot-wise. The coin for (query, call index, slot) is a fixed
    uniform draw, so P25's replacements are a subset of P50's, and so on.
  * Negatives are drawn without replacement within a trajectory, in a fixed
    per-query shuffled order, and repeat only once the pool runs out.
    Repeats are counted and logged.
  * A substituted negative never duplicates a doc already in the same list.
  * One PressureSearch instance = one trajectory (all agents in an episode
    share it, so the negative pool is not reused across workers).
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from typing import Mapping

from crossover.retrieval import Retriever
from crossover.types import Query


@dataclass
class SearchRecord:
    call_idx: int
    agent_id: str
    thread_id: str
    query: str
    base: list[str]
    shown: list[str]
    replaced: list[bool]
    gold_shown: list[str] = field(default_factory=list)


class PressureSearch:
    def __init__(
        self,
        retriever: Retriever,
        corpus: Mapping[str, str],
        query: Query,
        level: float,
        tokenizer,
        k: int = 5,
        snippet_tokens: int = 512,
        seed: int = 768,
        protect: str = "gold",
    ):
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"pressure level must be in [0, 1], got {level}")
        self.retriever, self.corpus, self.query = retriever, corpus, query
        self.level, self.k, self.snippet_tokens = level, k, snippet_tokens
        self.tokenizer, self.seed = tokenizer, seed
        self.protect = protect
        self.gold = set(query.gold_docids)
        self.protected = set(self.gold)
        if protect == "gold+evidence":
            self.protected |= set(query.evidence_docids)
        elif protect != "gold":
            raise ValueError(f"Unknown protect mode {protect!r}")

        pool = [d for d in dict.fromkeys(query.negative_docids) if d not in self.protected]
        random.Random(f"{seed}:{query.query_id}:negpool").shuffle(pool)
        self._pool = pool
        self._cursor = 0
        self._drawn: set[str] = set()
        self.draws = 0
        self.repeats = 0
        self.unfilled = 0  # slots that should have been replaced but no negative was available
        self.log: list[SearchRecord] = []
        self._calls = 0
        self._lock = threading.Lock()

    # -- dial -----------------------------------------------------------------

    def _coin(self, call_idx: int, slot: int) -> float:
        return random.Random(f"{self.seed}:{self.query.query_id}:{call_idx}:{slot}").random()

    def _next_negative(self, exclude: set[str]) -> str | None:
        # Fresh negatives first, in the fixed shuffled order...
        for docid in self._pool:
            if docid not in self._drawn and docid not in exclude:
                self._drawn.add(docid)
                self.draws += 1
                return docid
        # ...then, once the pool has run out, cycle through it again.
        n = len(self._pool)
        for _ in range(n):
            docid = self._pool[self._cursor % n]
            self._cursor += 1
            if docid not in exclude:
                self.draws += 1
                self.repeats += 1
                return docid
        return None

    def apply(self, base: list[str], call_idx: int) -> tuple[list[str], list[bool]]:
        shown: list[str] = []
        replaced: list[bool] = []
        exclude = set(base)
        for slot, docid in enumerate(base):
            if docid in self.protected or self.level <= 0.0 or self._coin(call_idx, slot) >= self.level:
                shown.append(docid)
                replaced.append(False)
                continue
            neg = self._next_negative(exclude | set(shown))
            if neg is None:
                self.unfilled += 1
                shown.append(docid)
                replaced.append(False)
            else:
                shown.append(neg)
                replaced.append(True)
        return shown, replaced

    # -- tool -----------------------------------------------------------------

    def search(self, query_text: str, agent_id: str = "", thread_id: str = "") -> tuple[list[str], str]:
        with self._lock:
            call_idx = self._calls
            self._calls += 1
            base = self.retriever.search(query_text, self.k)
            shown, replaced = self.apply(base, call_idx)
            self.log.append(
                SearchRecord(
                    call_idx=call_idx,
                    agent_id=agent_id,
                    thread_id=thread_id,
                    query=query_text,
                    base=base,
                    shown=shown,
                    replaced=replaced,
                    gold_shown=[d for d in shown if d in self.gold],
                )
            )
        return shown, self.format(query_text, shown)

    def format(self, query_text: str, docids: list[str]) -> str:
        if not docids:
            return f"No results for {query_text!r}."
        blocks = [f"Results for {query_text!r}:"]
        for i, docid in enumerate(docids, 1):
            snippet = self.tokenizer.truncate(self.corpus.get(docid, ""), self.snippet_tokens)
            blocks.append(f"[{i}] docid: {docid}\n{snippet}")
        return "\n\n".join(blocks)

    def stats(self) -> dict:
        slots = sum(len(r.shown) for r in self.log)
        return {
            "n_searches": len(self.log),
            "slots": slots,
            "replaced": sum(sum(r.replaced) for r in self.log),
            "negative_draws": self.draws,
            "negative_repeats": self.repeats,
            "negative_repeat_rate": self.repeats / self.draws if self.draws else 0.0,
            "unfilled_slots": self.unfilled,
            "gold_seen": sorted({d for r in self.log for d in r.gold_shown}),
        }
