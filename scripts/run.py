"""Run a sweep: every (system, pressure level, query) cell once, resumable.

  uv run python scripts/run.py --split pilot --systems L0 M-iso --levels 0 1        # RQ1 kill test
  uv run python scripts/run.py --split core --systems L0 S-iso M-iso                 # crossover sweep
  uv run python scripts/run.py --split ablation --set agents.relay_cap_tokens=256 ... # an ablation
  uv run python scripts/run.py --dry-run                                            # offline smoke test

Splits (fixed seed, non-overlapping; see crossover.data.split_queries):
pilot (50), core (500), ablation (first 150 of core), rest (~280 after core),
nonpilot (core + rest, for the P100 S-iso vs. M-iso contrast).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossover.agents import SYSTEMS
from crossover.config import load_config
from crossover.data import SPLITS, load_corpus, load_queries, split_queries
from crossover.harness import run_sweep
from crossover.llm import ScriptedLLM, build_llm
from crossover.retrieval import BM25Retriever, build_retriever
from crossover.tokenizer import get_tokenizer
from crossover.toy import TOY_CORPUS, TOY_QUERIES, toy_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. budget.B=80000")
    ap.add_argument("--systems", nargs="+", default=["L0", "S-iso", "M-iso"], choices=sorted(SYSTEMS))
    ap.add_argument("--levels", nargs="+", type=float, help="pressure levels in [0,1]; default from config")
    ap.add_argument("--split", default="core", choices=SPLITS, help="which query set to run")
    ap.add_argument("--n", type=int, help="use only the first n queries of the split (quick tests)")
    ap.add_argument("--workers", type=int, default=8, help="episodes in flight at once")
    ap.add_argument("--dry-run", action="store_true", help="toy corpus + scripted policy, no model or GPU")
    args = ap.parse_args()

    if args.dry_run:
        args.set += ["tokenizer=whitespace", "output.run_dir=runs/dry_run", "budget.B=5000"]
    cfg = load_config(args.config, args.set)
    levels = args.levels or cfg.pressure.levels

    if args.dry_run:
        corpus, queries = TOY_CORPUS, TOY_QUERIES
        llm, retriever = ScriptedLLM(toy_policy), BM25Retriever(TOY_CORPUS)
        systems = sorted(SYSTEMS)
    else:
        corpus = load_corpus(cfg.data.corpus_path)
        queries = split_queries(load_queries(cfg.data.queries_path), cfg.data)[args.split]
        llm, retriever = build_llm(cfg.model), build_retriever(cfg, corpus)
        systems = args.systems
    if args.n:
        queries = queries[: args.n]

    run_sweep(cfg, llm, retriever, corpus, queries, systems, levels, get_tokenizer(cfg.tokenizer), args.workers)


if __name__ == "__main__":
    main()
