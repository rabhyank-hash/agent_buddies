"""Self-consistency: n independent L0 samples at B/n each, majority vote
(Jwalapuram et al., 2026). The strongest simple baseline without teammates."""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from crossover.agents.base import Episode
from crossover.agents.react import ReactAgent
from crossover.ledger import TokenLedger
from crossover.pressure import PressureSearch


def normalize(answer: str) -> str:
    return re.sub(r"[^\w\s]", "", answer.lower()).strip()


class SelfConsistency:
    name = "SC"

    def __init__(self, n: int | None = None):
        self.n = n

    def run(self, ep: Episode) -> None:
        cfg = ep.cfg
        n = self.n or cfg.agents.sc_samples
        subs = [self._fork(ep, i, n) for i in range(n)]
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda s: ReactAgent().run(s), subs))

        for i, s in enumerate(subs):
            ep.ledger.absorb(s.ledger, sample=i)
            ep.search.log.extend(s.search.log)
            ep.search.draws += s.search.draws
            ep.search.repeats += s.search.repeats
            ep.search.unfilled += s.search.unfilled
            ep.seen_docids |= s.seen_docids
            ep.cited_docids |= s.cited_docids
            ep.threads.extend(s.threads)
            ep.cap_hit |= s.cap_hit
            ep.overflow |= s.overflow
            ep.peak_context = max(ep.peak_context, s.peak_context)

        answered = [s for s in subs if s.final and s.final["answered"] and s.final["exact_answer"]]
        votes = Counter(normalize(s.final["exact_answer"]) for s in answered)
        if votes:
            top = max(votes.values())
            tied = [s for s in answered if votes[normalize(s.final["exact_answer"])] == top]
            winner = max(tied, key=lambda s: float(s.final.get("confidence") or 0))
        else:
            winner = subs[0]
        ep.final = dict(winner.final)
        ep.stop_reason = winner.stop_reason
        ep.final_context = winner.final_context
        ep.final_context_docids = winner.final_context_docids
        # The answer is only "in budget" once every sample has finished.
        ep.answer_step = len(ep.ledger.steps) - 1 if winner.final["answered"] else None
        ep.extra["sc_votes"] = dict(votes)
        ep.extra["sc_samples"] = [
            {"answer": s.final["exact_answer"], "answered": s.final["answered"], "generated": s.ledger.generated}
            for s in subs
        ]

    def _fork(self, ep: Episode, i: int, n: int) -> Episode:
        cfg = ep.cfg
        ledger = TokenLedger(ep.ledger.budget // n, ep.ledger.wrap_up_frac, ep.ledger.definition)
        src = ep.search
        search = PressureSearch(src.retriever, src.corpus, src.query, src.level, src.tokenizer, src.k,
                                src.snippet_tokens, seed=src.seed * 1000 + i + 1, protect=src.protect)
        return Episode(cfg, ep.llm, search, ledger, ep.tokenizer, ep.query, sample=i)
