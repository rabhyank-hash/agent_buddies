"""Check that query encoding matches the prebuilt index before running agents.

Encodes every BrowseComp-Plus question with the configured retriever and scores
the ranking against the labelled evidence and gold documents. If the numbers
are far below the references, the query prefix, pooling or model is wrong.

  uv run python scripts/check_retriever.py --config configs/sail_modal.yaml          # via the HTTP server
  uv run python scripts/check_retriever.py --set retrieval.backend=dense --n 100     # local GPU, quick

References (BrowseComp-Plus): the project page reports 20.3 nDCG@10 for
Qwen3-Embedding-8B, i.e. about 0.203 as printed here. The repo's Qwen3-Embedding-0.6B example gives evidence
nDCG@10 0.0802 / recall@100 0.2641, gold nDCG@10 0.0740 / recall@100 0.3023,
so 8B should land clearly above those.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossover.config import load_config
from crossover.data import load_queries
from crossover.retrieval import build_retriever


def ndcg_at(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--n", type=int, help="only the first n queries")
    ap.add_argument("--depth", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    queries = load_queries(cfg.data.queries_path)[: args.n]
    retriever = build_retriever(cfg)
    if hasattr(retriever, "search_many"):
        rankings = []
        for i in range(0, len(queries), 32):
            rankings += retriever.search_many([q.question for q in queries[i:i + 32]], args.depth)
    else:
        rankings = [retriever.search(q.question, args.depth) for q in queries]

    for name, rel in (("evidence", lambda q: set(q.evidence_docids)), ("gold", lambda q: set(q.gold_docids))):
        rows = [(r, rel(q)) for r, q in zip(rankings, queries) if rel(q)]
        n = len(rows)
        print(f"{name:9s} n={n}  nDCG@10={sum(ndcg_at(r, s) for r, s in rows) / n:.4f}  "
              + "  ".join(f"recall@{k}={sum(recall_at(r, s, k) for r, s in rows) / n:.4f}" for k in (5, 100, 1000)))


if __name__ == "__main__":
    main()
