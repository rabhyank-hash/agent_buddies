"""Analysis of graded trajectories: pilot diagnostics, accuracy table, p* (H2),
TOST (H3), 2x2 contrasts with Holm correction, failure modes, and the
two budget definitions.

  uv run python scripts/analyze.py runs/sweep        # prints a report, writes runs/sweep/analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossover.analysis.budget import rescore
from crossover.analysis.stats import CONTRASTS, bootstrap_crossover, holm, paired, paired_bootstrap, tost_paired


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def diagnostics(rows: list[dict]) -> dict:
    """Pilot gates: peak context, tokens per trajectory, searches, branching, cap/overflow rates."""
    out = {}
    by_sys = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    for s, rs in sorted(by_sys.items()):
        gen = [r["ledger"]["generated"] for r in rs]
        out[s] = {
            "n": len(rs),
            "generated_p50": pct(gen, 50),
            "generated_p75": pct(gen, 75),  # set B to L0's p75
            "peak_context_p95": pct([r["peak_context"] for r in rs], 95),  # >110K -> drop carried reasoning
            "searches_mean": float(np.mean([r["search_stats"]["n_searches"] for r in rs])),
            "branches_mean": float(np.mean([r["branches"] for r in rs])),
            "relay_tokens_mean": float(np.mean([x["raw_tokens"] for r in rs for x in r["relays"]] or [0])),
            "cap_hit_rate": float(np.mean([r["cap_hit"] for r in rs])),
            "overflow_rate": float(np.mean([r["overflow"] for r in rs])),
            "answered_rate": float(np.mean([r["answered"] for r in rs])),
            "neg_repeat_rate": float(np.mean([r["search_stats"]["negative_repeat_rate"] for r in rs])),
            "coordination_share": float(np.mean([r["ledger"]["coordination_share"] for r in rs])),
        }
    return out


def accuracy_table(rows: list[dict]) -> dict:
    """Accuracy per system x level, plus gold recall (failure category a) per level."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["system"], r["level"])].append(r)
    table = defaultdict(dict)
    for (s, lv), rs in sorted(cells.items()):
        table[s][f"P{round(lv * 100)}"] = {
            "acc": float(np.mean([bool(r["correct"]) for r in rs])),
            "gold_recall": float(np.mean([bool(r["gold_seen"]) for r in rs])),
            "answer_context_tokens": float(np.mean([r["answer_context_tokens"] for r in rs])),
            "n": len(rs),
        }
    return dict(table)


def analyze(rows: list[dict]) -> dict:
    levels = sorted({r["level"] for r in rows})
    systems = {r["system"] for r in rows}
    report: dict = {"diagnostics": diagnostics(rows), "accuracy": accuracy_table(rows)}

    if {"L0", "M-iso"} <= systems and len(levels) > 1:
        report["H2_p_star"] = bootstrap_crossover(rows, "L0", "M-iso")
    if {"S-iso", "M-iso"} <= systems and 1.0 in levels:
        report["H3_tost_P100"] = tost_paired(*paired(rows, "S-iso", "M-iso", 1.0))

    contrasts, pvals = {}, {}
    for name, (a, b) in CONTRASTS.items():
        if not {a, b} <= systems:
            continue
        for lv in levels:
            res = paired_bootstrap(*paired(rows, a, b, lv))
            if res.get("n"):
                key = f"{name}@P{round(lv * 100)}"
                contrasts[key] = res
                pvals[key] = res["p"]
    report["contrasts"] = contrasts
    report["holm"] = holm(pvals) if pvals else {}

    modes = defaultdict(Counter)
    for r in rows:
        cat = (r.get("attribution") or {}).get("category")
        if cat:
            modes[f"{r['system']}@P{round(r['level'] * 100)}"][cat] += 1
    report["failure_modes"] = {k: dict(v) for k, v in sorted(modes.items())}

    # Budget accounting check: does p* move when the budget counts input tokens too?
    # The total-token budget is B scaled by L0's median (generated + prefill) / generated.
    if "H2_p_star" in report:
        B = rows[0]["budget"]["B"]
        l0 = [r["ledger"] for r in rows if r["system"] == "L0"]
        scale = float(np.median([(x["generated"] + x["prefill"]) / max(1, x["generated"]) for x in l0]))
        report["p_star_by_budget_definition"] = {
            "generated": bootstrap_crossover(rescore(rows, B, "generated"), n_boot=500),
            "total": bootstrap_crossover(rescore(rows, B * scale, "total"), n_boot=500),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    path = Path(args.run_dir) / "graded.jsonl"
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    report = analyze(rows)
    (Path(args.run_dir) / "analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k != "contrasts"}, indent=2))


if __name__ == "__main__":
    main()
