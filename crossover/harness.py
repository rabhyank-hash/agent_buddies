"""Episode runner and resumable sweeps.

run_episode() is the single entry point every arm plugs into. It builds the
per-trajectory pressure search and token ledger, runs the arm, and returns one
JSON-serializable record: tokens per agent, context length per call, every
search (what the base retriever returned vs. what the agent saw), doc IDs
cited, the final answering context, relay summaries, and the final answer.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping

from crossover.agents import make_system
from crossover.agents.base import Episode, response_text
from crossover.ledger import TokenLedger
from crossover.pressure import PressureSearch
from crossover.retrieval import Retriever
from crossover.types import Query


def traj_id(system: str, level: float, query_id: str, relay_cap: int | None) -> str:
    return f"{system}|P{round(level * 100)}|r{relay_cap or 'none'}|{query_id}"


def run_episode(cfg, llm, retriever: Retriever, corpus: Mapping[str, str], query: Query, system: str,
                level: float, tokenizer, context_dir: Path | None = None) -> dict:
    search = PressureSearch(retriever, corpus, query, level, tokenizer, cfg.retrieval.k,
                            cfg.retrieval.snippet_tokens, cfg.pressure.seed, cfg.pressure.protect)
    ledger = TokenLedger(cfg.budget.B, cfg.budget.wrap_up_frac, cfg.budget.definition)
    ep = Episode(cfg, llm, search, ledger, tokenizer, query)
    t0 = time.time()
    make_system(system).run(ep)
    wall = time.time() - t0

    tid = traj_id(system, level, query.query_id, cfg.agents.relay_cap_tokens)
    context_path = None
    if context_dir is not None and cfg.output.save_final_context:
        context_dir.mkdir(parents=True, exist_ok=True)
        context_path = context_dir / (tid.replace("|", "__") + ".txt.gz")
        with gzip.open(context_path, "wt") as f:
            f.write(ep.final_context)

    root = next((t for t in ep.threads if t.thread_id == "root"), None)
    gold = set(query.gold_docids)
    sstats = search.stats()
    return {
        "traj_id": tid,
        "query_id": query.query_id,
        "system": system,
        "level": level,
        "relay_cap": cfg.agents.relay_cap_tokens,
        "model": cfg.model.name,
        "budget": dataclasses.asdict(cfg.budget),
        "final": ep.final,
        "response": response_text(ep.final or {}),
        "answered": bool(ep.final and ep.final["answered"]),
        "stop_reason": ep.stop_reason,
        "overflow": ep.overflow,
        "cap_hit": ep.cap_hit,
        "ledger": ledger.summary(),
        "steps": ledger.steps_as_dicts(),
        "answer_step": ep.answer_step,
        "peak_context": ep.peak_context,
        "root_context_trace": root.prompt_trace if root else [],
        "answer_context_tokens": (root.prompt_trace[-1] if root and root.prompt_trace else 0),
        "searches": [dataclasses.asdict(r) for r in search.log],
        "search_stats": sstats,
        "seen_docids": sorted(ep.seen_docids),
        "cited_docids": sorted(ep.cited_docids),
        "gold_seen": sorted(gold & ep.seen_docids),
        "gold_cited": sorted(gold & ep.cited_docids),
        "gold_in_final_context": sorted(gold & set(ep.final_context_docids)),
        "final_context_docids": ep.final_context_docids,
        "final_context_path": str(context_path) if context_path else None,
        "relays": ep.relays,
        "branches": ep.branches,
        "rejected_branches": ep.rejected_branches,
        "compactions": ep.compactions,
        "threads": [{"agent_id": t.agent_id, "thread_id": t.thread_id, "depth": t.depth,
                     "calls": len(t.prompt_trace), "peak_context": max(t.prompt_trace, default=0)} for t in ep.threads],
        "extra": ep.extra,
        "wall_time": wall,
    }


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["traj_id"])
    return ids


def run_sweep(cfg, llm, retriever: Retriever, corpus: Mapping[str, str], queries: Iterable[Query],
              systems: list[str], levels: list[float], tokenizer, workers: int = 8, log=print) -> Path:
    """Run every (system, level, query) cell once, appending to
    <run_dir>/trajectories.jsonl. Re-running skips finished cells."""
    run_dir = Path(cfg.output.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    out = run_dir / "trajectories.jsonl"
    errors = run_dir / "errors.jsonl"
    finished = done_ids(out)
    jobs = [(s, lv, q) for s in systems for lv in levels for q in queries
            if traj_id(s, lv, q.query_id, cfg.agents.relay_cap_tokens) not in finished]
    log(f"{len(jobs)} episodes to run ({len(finished)} already done) -> {out}")
    lock = threading.Lock()

    def one(job):
        system, level, q = job
        return run_episode(cfg, llm, retriever, corpus, q, system, level, tokenizer, run_dir / "contexts")

    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, j): j for j in jobs}
        for fut in as_completed(futures):
            system, level, q = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # keep the sweep alive; failures are logged and retried on resume
                with lock, open(errors, "a") as f:
                    f.write(json.dumps({"system": system, "level": level, "query_id": q.query_id,
                                        "error": repr(e), "traceback": traceback.format_exc()}) + "\n")
                log(f"ERROR {system} P{level} {q.query_id}: {e!r}")
                continue
            with lock, open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            n_done += 1
            if n_done % 25 == 0 or n_done == len(jobs):
                log(f"{n_done}/{len(jobs)} done")
    return out
