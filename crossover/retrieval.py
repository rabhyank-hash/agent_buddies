"""Base retrievers. All return a ranked list of docids; the pressure wrapper
(crossover.pressure) decides what the agent actually sees."""

from __future__ import annotations

import glob
import json
import math
import pickle
import re
import urllib.request
from collections import Counter, defaultdict
from typing import Mapping, Protocol


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[str]: ...


_TOKEN = re.compile(r"\w+")


def _terms(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Pure-Python BM25 for toy data and small dev corpora only.

    Too slow for the full 100K-document corpus; use the dense index or a
    retrieval server (HTTPRetriever) there.
    """

    def __init__(self, corpus: Mapping[str, str], k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b
        self.doc_len: dict[str, int] = {}
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for docid, text in corpus.items():
            tf = Counter(_terms(text))
            self.doc_len[docid] = sum(tf.values())
            for term, count in tf.items():
                self.postings[term].append((docid, count))
        self.n_docs = len(self.doc_len)
        self.avg_len = sum(self.doc_len.values()) / max(1, self.n_docs)

    def search(self, query: str, k: int) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        for term in set(_terms(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for docid, tf in plist:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[docid] / self.avg_len)
                scores[docid] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [docid for docid, _ in ranked[:k]]


class HTTPRetriever:
    """Client for a retrieval server: POST {url}/search {"query", "k"} -> {"docids": [...]}."""

    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url.rstrip("/") + "/search"
        self.timeout = timeout

    def search(self, query: str, k: int) -> list[str]:
        body = json.dumps({"query": query, "k": k}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return [str(d) for d in json.loads(resp.read())["docids"]]


class DenseRetriever:
    """Exact inner-product search over precomputed document embeddings.

    Loads Tevatron-style shards (pickled (embeddings, docid_list) tuples), the
    format BrowseComp-Plus uses for its released indexes. Queries are encoded
    with sentence-transformers; `query_prefix` must match the instruction the
    index was built with, or accuracy will be off. Check this against the
    paper's numbers on the dev split (kickoff step 1).
    """

    def __init__(self, index_glob: str, model_name: str, query_prefix: str = "", device: str | None = None):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        reps, ids = [], []
        for path in sorted(glob.glob(index_glob)):
            with open(path, "rb") as f:
                emb, lookup = pickle.load(f)
            reps.append(np.asarray(emb, dtype=np.float32))
            ids.extend(str(x) for x in lookup)
        if not reps:
            raise FileNotFoundError(f"No index shards match {index_glob}")
        self._np = np
        self.emb = np.concatenate(reps)
        self.docids = ids
        self.model = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix

    def search(self, query: str, k: int) -> list[str]:
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]
        scores = self.emb @ q.astype(self._np.float32)
        top = self._np.argpartition(-scores, k)[:k]
        top = top[self._np.argsort(-scores[top])]
        return [self.docids[i] for i in top]


def build_retriever(cfg, corpus: Mapping[str, str] | None = None) -> Retriever:
    backend = cfg.retrieval.backend
    if backend == "bm25":
        if corpus is None:
            raise ValueError("bm25 backend needs the corpus")
        return BM25Retriever(corpus)
    if backend == "http":
        return HTTPRetriever(cfg.retrieval.http_url)
    if backend == "dense":
        return DenseRetriever(cfg.retrieval.dense_index_glob, cfg.retrieval.dense_model, cfg.retrieval.dense_query_prefix)
    raise ValueError(f"Unknown retrieval backend {backend!r}")
