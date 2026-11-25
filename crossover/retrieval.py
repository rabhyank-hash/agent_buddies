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
    """Exact inner-product search over the prebuilt BrowseComp-Plus index.

    The index (Tevatron/browsecomp-plus-indexes on the HF hub) holds the
    document embeddings: 4 pickled (embeddings, docid_list) shards for
    Qwen3-Embedding-8B, already L2-normalized. Only the agent's search queries
    are encoded at run time, which is the one step that needs the 8B model.

    Query encoding mirrors BrowseComp-Plus (searcher/searchers/faiss_searcher.py
    and scripts_build_index/qwen3-embed.md): prefix + query, max length 512,
    fp16, last-token ("eos") pooling, L2 normalization. Check that it matches
    with scripts/check_retriever.py before any run.
    """

    def __init__(self, index_glob: str, model_name: str, query_prefix: str, max_len: int = 512, device: str | None = None):
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        reps, ids = [], []
        for path in sorted(glob.glob(index_glob)):
            with open(path, "rb") as f:
                emb, lookup = pickle.load(f)
            reps.append(np.asarray(emb, dtype=np.float32))
            ids.extend(str(x) for x in lookup)
        if not reps:
            raise FileNotFoundError(f"No index shards match {index_glob}")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.emb = torch.from_numpy(np.concatenate(reps)).to(self.device, dtype)
        self.docids = ids
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.query_prefix, self.max_len = query_prefix, max_len

    def encode(self, queries: list[str]):
        torch = self.torch
        batch = self.tokenizer([self.query_prefix + q for q in queries], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
        reps = hidden[:, -1]  # left padding: the last position is each sequence's final token
        return torch.nn.functional.normalize(reps.float(), dim=-1).to(self.emb.dtype)

    def search_many(self, queries: list[str], k: int) -> list[list[str]]:
        scores = self.encode(queries) @ self.emb.T
        top = scores.topk(k, dim=-1).indices.tolist()
        return [[self.docids[i] for i in row] for row in top]

    def search(self, query: str, k: int) -> list[str]:
        return self.search_many([query], k)[0]


def build_retriever(cfg, corpus: Mapping[str, str] | None = None) -> Retriever:
    backend = cfg.retrieval.backend
    if backend == "bm25":
        if corpus is None:
            raise ValueError("bm25 backend needs the corpus")
        return BM25Retriever(corpus)
    if backend == "http":
        return HTTPRetriever(cfg.retrieval.http_url)
    if backend == "dense":
        r = cfg.retrieval
        return DenseRetriever(r.dense_index_glob, r.dense_model, r.dense_query_prefix, r.dense_query_max_len)
    raise ValueError(f"Unknown retrieval backend {backend!r}")
