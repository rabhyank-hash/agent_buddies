"""Qwen3-Embedding-8B retrieval server on Modal.

The document index is prebuilt (Tevatron/browsecomp-plus-indexes); this only
encodes the agents' search queries and scores them against it. It serves the
same contract as scripts/serve_retrieval.py, so the harness just points
`retrieval.http_url` at the URL Modal prints.

  uv run --with modal modal deploy deploy/modal_retriever.py

The first start downloads the index (~1.6 GB) and the model (~16 GB) into a
Modal Volume, so later cold starts are fast. The container scales to zero
when idle, so you pay only while sweeps are sending searches.

The endpoint is a public URL (the corpus is public, but don't share it). Check
Modal's docs if a decorator name has changed in your Modal version.
"""

import modal

GPU = "L4"  # 24 GB: fits the 8B encoder in fp16 plus the 100K x 4096 index
INDEX_REPO = "Tevatron/browsecomp-plus-indexes"
MODEL = "Qwen/Qwen3-Embedding-8B"
PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

app = modal.App("crossover-retriever")
volume = modal.Volume.from_name("crossover-retriever-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "numpy", "huggingface_hub", "fastapi[standard]")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("crossover")
)


@app.cls(gpu=GPU, image=image, volumes={"/cache": volume}, scaledown_window=600, timeout=3600)
@modal.concurrent(max_inputs=32)
class Retriever:
    @modal.enter()
    def load(self):
        import threading

        from huggingface_hub import snapshot_download

        from crossover.retrieval import DenseRetriever

        snapshot_download(INDEX_REPO, repo_type="dataset", allow_patterns=["qwen3-embedding-8b/*"], local_dir="/cache/index")
        volume.commit()
        self.retriever = DenseRetriever("/cache/index/qwen3-embedding-8b/corpus.*.pkl", MODEL, PREFIX, max_len=512)
        self.lock = threading.Lock()  # one GPU encoder: serialize forward passes

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI

        api = FastAPI()

        @api.post("/search")
        def search(body: dict) -> dict:
            with self.lock:
                return {"docids": self.retriever.search(body["query"], int(body.get("k", 5)))}

        @api.post("/search_many")
        def search_many(body: dict) -> dict:
            with self.lock:
                return {"docids": self.retriever.search_many(body["queries"], int(body.get("k", 5)))}

        return api
