"""BrowseComp-Plus loading.

The HF release (Tevatron/browsecomp-plus) is encrypted; run the decryption
script from the BrowseComp-Plus repo first, which writes one JSON object per
query. Field names below follow that release (query_id, query, answer,
gold_docs, evidence_docs, negative_docs); check them against your decrypted
file in week 1 and adjust FIELD_ALIASES if they differ.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from crossover.types import Query

FIELD_ALIASES = {
    "query_id": ("query_id", "qid", "id"),
    "question": ("query", "question", "problem"),
    "answer": ("answer", "gold_answer"),
    "gold": ("gold_docs", "gold_docids"),
    "evidence": ("evidence_docs", "evidence_docids"),
    "negative": ("negative_docs", "negative_docids", "hard_negatives"),
}


def _first(row: dict[str, Any], key: str, default=None):
    for name in FIELD_ALIASES[key]:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _docids(items: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    for item in items or []:
        if isinstance(item, dict):
            docid = item.get("docid", item.get("doc_id", item.get("id")))
        else:
            docid = item
        if docid is not None:
            out.append(str(docid))
    return out


def query_from_row(row: dict[str, Any]) -> Query:
    return Query(
        query_id=str(_first(row, "query_id")),
        question=str(_first(row, "question")),
        answer=str(_first(row, "answer", "")),
        gold_docids=_docids(_first(row, "gold", [])),
        evidence_docids=_docids(_first(row, "evidence", [])),
        negative_docids=_docids(_first(row, "negative", [])),
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_queries(path: str | Path) -> list[Query]:
    return [query_from_row(r) for r in read_jsonl(path)]


def load_corpus(path: str | Path) -> dict[str, str]:
    """JSONL with docid + text (optionally title/url). Returns docid -> text."""
    corpus: dict[str, str] = {}
    for row in read_jsonl(path):
        text = row.get("text") or row.get("contents") or ""
        title = row.get("title")
        corpus[str(row["docid"])] = f"{title}\n{text}" if title else text
    return corpus


def export_corpus_from_hf(out_path: str | Path, name: str = "Tevatron/browsecomp-plus-corpus") -> int:
    """Download the corpus once and write it as JSONL (needs the `hf` extra)."""
    from datasets import load_dataset

    ds = load_dataset(name, split="train")
    n = 0
    with open(out_path, "w") as f:
        for row in ds:
            f.write(json.dumps({"docid": str(row["docid"]), "text": row["text"], "url": row.get("url")}) + "\n")
            n += 1
    return n


SPLITS = ("pilot", "core", "ablation", "rest", "nonpilot")


def split_queries(queries: list[Query], data_cfg) -> dict[str, list[Query]]:
    """Fixed-seed, deterministic splits of the 830 queries:

      pilot     first `pilot_size` of the shuffle; prompt tuning only, never evaluated
      core      next `subset_size` (500): every core experiment
      ablation  the first `ablation_size` (150) of core: every ablation
      rest      everything after core (~280): only the P100 S-iso vs. M-iso contrast
      nonpilot  core + rest: the full P100 contrast
    """
    order = sorted(queries, key=lambda q: q.query_id)
    random.Random(data_cfg.subset_seed).shuffle(order)
    pilot, nonpilot = order[: data_cfg.pilot_size], order[data_cfg.pilot_size:]
    n = len(nonpilot) if data_cfg.subset_size is None else data_cfg.subset_size
    core = nonpilot[:n]
    return {
        "pilot": pilot,
        "core": core,
        "ablation": core[: data_cfg.ablation_size],
        "rest": nonpilot[n:],
        "nonpilot": nonpilot,
    }
