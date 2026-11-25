"""Grade trajectories with the gpt-4.1 judge, then attribute every failure to (a)-(d).

  uv run python scripts/grade.py runs/sweep            # writes runs/sweep/graded.jsonl (resumable)
  uv run python scripts/grade.py runs/sweep --no-attribution
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossover.config import load_config
from crossover.data import load_corpus, load_queries
from crossover.eval.attribution import attribute
from crossover.eval.judge import Judge
from crossover.harness import done_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-attribution", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    run_dir = Path(args.run_dir)
    queries = {q.query_id: q for q in load_queries(cfg.data.queries_path)}
    corpus = None if args.no_attribution else load_corpus(cfg.data.corpus_path)
    judge = Judge(cfg.judge)
    out = run_dir / "graded.jsonl"
    finished = done_ids(out)
    with open(run_dir / "trajectories.jsonl") as f:
        todo = [r for r in map(json.loads, filter(str.strip, f)) if r["traj_id"] not in finished]
    print(f"grading {len(todo)} trajectories -> {out}")

    def one(rec: dict) -> dict:
        q = queries[rec["query_id"]]
        rec = {**rec, **judge.grade(q.question, rec["response"], q.answer)}
        if corpus is not None:
            rec["attribution"] = attribute(rec, q.question, q.answer, corpus, judge.complete)
        return rec

    with ThreadPoolExecutor(args.workers) as pool, open(out, "a") as f:
        for rec in pool.map(one, todo):
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
