"""Grade trajectories, then attribute every failure to (a)-(d).

  uv run python scripts/grade.py runs/sweep --config configs/sail_modal.yaml
  uv run python scripts/grade.py runs/sweep --no-attribution

Two passes, each sent to the judge as one batch: (1) accuracy grades for every
ungraded trajectory, (2) key-fact checks for failures that cited a gold doc.
Writes runs/<name>/graded.jsonl; re-running grades only new trajectories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossover.config import load_config
from crossover.data import load_corpus, load_queries
from crossover.eval.attribution import categorize, keyfact_prompts
from crossover.eval.judge import grader_prompt, make_judge, parse_grade
from crossover.harness import done_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--no-attribution", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    run_dir = Path(args.run_dir)
    queries = {q.query_id: q for q in load_queries(cfg.data.queries_path)}
    judge = make_judge(cfg.judge, state_dir=run_dir / "judge")
    out = run_dir / "graded.jsonl"
    finished = done_ids(out)
    with open(run_dir / "trajectories.jsonl") as f:
        todo = [r for r in map(json.loads, filter(str.strip, f)) if r["traj_id"] not in finished]
    print(f"grading {len(todo)} trajectories with {cfg.judge.model} ({cfg.judge.mode})")

    # Pass 1: accuracy. Empty responses are wrong without asking the judge.
    prompts = {f"grade:{r['traj_id']}": grader_prompt(queries[r["query_id"]].question, r["response"],
                                                       queries[r["query_id"]].answer)
               for r in todo if r["response"].strip()}
    texts = judge.complete_many(prompts)
    for r in todo:
        raw = texts.get(f"grade:{r['traj_id']}", "")
        r.update(parse_grade(raw) if raw else {"correct": False, "parse_error": False, "extracted_final_answer": None})
        r["judge_raw"] = raw

    # Pass 2: key-fact checks for failures that cited a gold document.
    if not args.no_attribution:
        corpus = load_corpus(cfg.data.corpus_path)
        verbatim, kf_prompts = {}, {}
        for r in todo:
            if r["correct"] or not r["gold_seen"] or not r["gold_cited"]:
                continue
            q = queries[r["query_id"]]
            verbatim[r["traj_id"]], per_doc = keyfact_prompts(r, q.question, q.answer, corpus)
            kf_prompts.update({f"keyfact:{r['traj_id']}:{d}": p for d, p in per_doc.items()})
        kf = judge.complete_many(kf_prompts) if kf_prompts else {}
        by_traj: dict[str, dict[str, str]] = {}
        for key, text in kf.items():
            tid, docid = key.removeprefix("keyfact:").rsplit(":", 1)
            by_traj.setdefault(tid, {})[docid] = text
        for r in todo:
            r["attribution"] = categorize(r, verbatim.get(r["traj_id"]), by_traj.get(r["traj_id"]))

    with open(out, "a") as f:
        for r in todo:
            f.write(json.dumps(r) + "\n")
    parse_errors = sum(r.get("parse_error", False) for r in todo)
    print(f"wrote {len(todo)} graded trajectories to {out} ({parse_errors} judge parse errors)")


if __name__ == "__main__":
    main()
