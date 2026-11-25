"""Judge through Sail's Batch API (docs.sailresearch.com/requests_at_scale).

Needed for Flex-only judge models such as Qwen/Qwen3.6-35B-A3B, which Sail
does not serve on Chat Completions. Batches always run on Flex (first result
within about an hour).

Protocol, as documented:
  POST /v1/batches {"endpoint": "/v1/responses", "requests": [{"custom_id", "params"}], "label"}
  GET  /v1/batches/{id}               -> status "in_progress" | "completed", inline `results`
  GET  /v1/batches/{id}/{custom_id}   -> one result, when inline results are too large
Results can expire 10 minutes after first delivery, so they are written to
disk as soon as they arrive, and submitted batch ids are saved so an
interrupted grading run resumes instead of paying twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

MAX_BATCH_BYTES = 200 * 1024 * 1024  # docs: 256 MB per request body
MAX_BATCH_REQUESTS = 50_000  # docs: 100,000


def response_text(body: dict) -> str:
    """Concatenate the output_text parts of a Responses API object."""
    out = body.get("output")
    if isinstance(out, str):
        return out
    parts = []
    for item in out or []:
        for c in item.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    return "".join(parts)


def custom_id(key: str) -> str:
    """Sail requires ^[a-z0-9][a-z0-9_-]*$, at most 64 characters."""
    return "g" + hashlib.sha1(key.encode()).hexdigest()[:40]


class SailBatchJudge:
    def __init__(self, judge_cfg, state_dir: str | Path | None, poll_seconds: float = 60.0):
        self.cfg = judge_cfg
        self.base = (judge_cfg.base_url or "https://api.sailresearch.com/v1").rstrip("/")
        self.key = os.environ.get(judge_cfg.api_key_env, "")
        self.poll_seconds = poll_seconds
        self.state_dir = Path(state_dir or ".")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.state_dir / "judge_results.jsonl"
        self.batches_path = self.state_dir / "judge_batches.json"

    # -- HTTP ---------------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None, headers: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())

    # -- state --------------------------------------------------------------------

    def _done(self) -> dict[str, str]:
        if not self.results_path.exists():
            return {}
        with open(self.results_path) as f:
            return {r["custom_id"]: r["text"] for r in map(json.loads, f)}

    def _save(self, cid: str, text: str) -> None:
        with open(self.results_path, "a") as f:
            f.write(json.dumps({"custom_id": cid, "text": text}) + "\n")

    def _batches(self) -> list[str]:
        return json.loads(self.batches_path.read_text()) if self.batches_path.exists() else []

    # -- API ----------------------------------------------------------------------

    def complete_many(self, prompts: dict[str, str], workers: int = 0) -> dict[str, str]:
        ids = {k: custom_id(k) for k in prompts}
        if self._batches():  # finish batches left by an interrupted run first
            self._collect(set(self._batches()))
        done = self._done()
        todo = [k for k in prompts if ids[k] not in done]
        if todo:
            self._collect(set(self._submit({ids[k]: prompts[k] for k in todo})))
        done = self._done()
        missing = [k for k in prompts if ids[k] not in done]
        if missing:
            raise RuntimeError(f"{len(missing)} judge requests have no result; re-run grading to retry them")
        return {k: done[ids[k]] for k in prompts}

    def _submit(self, requests: dict[str, str]) -> list[str]:
        chunks, cur, size = [], [], 0
        for cid, prompt in requests.items():
            item = {"custom_id": cid, "params": {"model": self.cfg.model, "input": prompt,
                                                 "max_output_tokens": self.cfg.max_output_tokens}}
            n = len(json.dumps(item))
            if cur and (size + n > MAX_BATCH_BYTES or len(cur) >= MAX_BATCH_REQUESTS):
                chunks.append(cur)
                cur, size = [], 0
            cur.append(item)
            size += n
        if cur:
            chunks.append(cur)
        batch_ids = []
        for chunk in chunks:
            key = hashlib.sha1("".join(i["custom_id"] for i in chunk).encode()).hexdigest()
            resp = self._request("POST", "/batches", {"endpoint": "/v1/responses", "requests": chunk,
                                                      "label": "crossover-judge"}, {"Idempotency-Key": key})
            batch_ids.append(resp["id"])
            self.batches_path.write_text(json.dumps(batch_ids))  # saved before polling: resumable
        print(f"submitted {len(requests)} judge requests in {len(batch_ids)} batch(es)")
        return batch_ids

    def _collect(self, batch_ids: set[str]) -> None:
        remaining = set(batch_ids)
        while remaining:
            for bid in sorted(remaining):
                status = self._request("GET", f"/batches/{bid}")
                counts = status.get("request_counts")
                if status.get("status") != "completed":
                    print(f"batch {bid}: {status.get('status')} {counts}")
                    continue
                results = status.get("results")
                if results is None:  # too large to inline: fetch one by one
                    results = [self._request("GET", f"/batches/{bid}/{cid}") for cid in (status.get("request_status") or {})]
                for r in results:
                    body = (r.get("response") or {}).get("body") or {}
                    if not r.get("error") and body:  # failed requests are left out and retried next run
                        self._save(r["custom_id"], response_text(body))
                remaining.discard(bid)
            if remaining:
                time.sleep(self.poll_seconds)
        self.batches_path.unlink(missing_ok=True)
