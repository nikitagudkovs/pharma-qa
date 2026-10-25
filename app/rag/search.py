"""Hybrid label search: BM25 + dense embeddings, merged with Reciprocal Rank
Fusion, filtered by the resolved drug, optionally reranked by a cross-encoder.

Build embeddings once:  uv run python -m app.rag.search --build
"""
import json
import re
import sys
from functools import lru_cache
from threading import Lock

import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from app.config import EMBED_MODEL, EMBEDDINGS_PATH, LABELS_PATH, RERANK_MODEL, USE_RERANKER

RRF_K = 60
# BGE models expect this instruction on queries (not on passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CANDIDATES = 30
# Torch models segfault under concurrent inference from FastAPI's threadpool; inference is
# ~10 ms per query, so serializing it costs nothing measurable.
_model_lock = Lock()
_index_lock = Lock()


class Chunk(BaseModel):
    id: str
    drug: str
    section: str
    section_name: str
    label_id: str
    set_id: str | None
    effective_time: str | None
    url: str
    text: str


class Hit(BaseModel):
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


class LabelIndex:
    def __init__(self):
        self.chunks = [Chunk(**json.loads(l)) for l in LABELS_PATH.read_text().splitlines() if l]
        self.bm25 = BM25Okapi([_tokens(c.text) for c in self.chunks])
        self.emb = np.load(EMBEDDINGS_PATH)
        assert len(self.emb) == len(self.chunks), "embeddings out of date: run python -m app.rag.search --build"
        self.by_drug: dict[str, np.ndarray] = {}
        for i, c in enumerate(self.chunks):
            self.by_drug.setdefault(c.drug, []).append(i)
        self.by_drug = {d: np.array(ix) for d, ix in self.by_drug.items()}

    def search(self, query: str, drugs: list[str] | None = None, k: int = 5,
               rerank: bool | None = None) -> list[Hit]:
        rerank = USE_RERANKER if rerank is None else rerank
        allowed = None
        if drugs:
            ix = [self.by_drug[d] for d in drugs if d in self.by_drug]
            if not ix:
                return []  # drug is resolved but not in the corpus: nothing to cite
            allowed = np.concatenate(ix)
        pool = allowed if allowed is not None else np.arange(len(self.chunks))

        bm25_scores = self.bm25.get_scores(_tokens(query))[pool]
        bm25_order = pool[np.argsort(-bm25_scores)][:CANDIDATES]

        with _model_lock:
            q = _encoder().encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        dense_scores = self.emb[pool] @ q
        dense_order = pool[np.argsort(-dense_scores)][:CANDIDATES]

        fused: dict[int, Hit] = {}
        for rank, i in enumerate(bm25_order):
            fused[i] = Hit(chunk=self.chunks[i], score=1 / (RRF_K + rank + 1), bm25_rank=rank)
        for rank, i in enumerate(dense_order):
            h = fused.setdefault(i, Hit(chunk=self.chunks[i], score=0.0))
            h.score += 1 / (RRF_K + rank + 1)
            h.dense_rank = rank
        hits = sorted(fused.values(), key=lambda h: -h.score)[:CANDIDATES]

        if rerank and hits:
            with _model_lock:
                scores = _reranker().predict([(query, h.chunk.text) for h in hits])
            for h, s in zip(hits, scores):
                h.score = float(s)
            hits.sort(key=lambda h: -h.score)
        return hits[:k]


_index: LabelIndex | None = None


def get_index() -> LabelIndex:
    global _index
    with _index_lock:
        if _index is None:
            _index = LabelIndex()
            _encoder()
        return _index


def build_embeddings() -> None:
    chunks = [json.loads(l) for l in LABELS_PATH.read_text().splitlines() if l]
    vecs = _encoder().encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True,
                             show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, vecs.astype(np.float32))
    print(f"{len(chunks)} embeddings -> {EMBEDDINGS_PATH.name}")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_embeddings()
