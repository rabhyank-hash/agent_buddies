"""Minimal retrieval server for `retrieval.backend: http`.

Loads the dense index once and answers POST /search {"query", "k"} -> {"docids": [...]}.
Run it wherever the index fits; agents point retrieval.http_url at it.

  uv run python scripts/serve_retrieval.py --port 8001
  uv run python scripts/serve_retrieval.py --backend bm25 --set data.corpus_path=data/dev_corpus.jsonl   # small dev corpus
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossover.config import load_config
from crossover.data import load_corpus
from crossover.retrieval import build_retriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--backend", default="dense", choices=["dense", "bm25"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set + [f"retrieval.backend={args.backend}"])
    corpus = load_corpus(cfg.data.corpus_path) if args.backend == "bm25" else None
    retriever = build_retriever(cfg, corpus)
    lock = threading.Lock()  # one encoder on one GPU: serialize queries

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/search":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                docids = retriever.search(body["query"], int(body.get("k", cfg.retrieval.k)))
            out = json.dumps({"docids": docids}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *_):
            pass

    print(f"retrieval server ({args.backend}) on {args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
