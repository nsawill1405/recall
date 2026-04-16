from __future__ import annotations

import hashlib
import math

from .base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    name = "local"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or "sentence-transformers/all-MiniLM-L6-v2"
        self._backend = "hash"
        self._model_impl = None
        self.dimension = 384

        if self.model.startswith("hash-"):
            self.dimension = _dimension_from_model_name(self.model, default=384)
            return

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            self.model = "hash-embedding-v1"
            self.dimension = 384
            return

        self._backend = "sentence-transformers"
        self._model_impl = SentenceTransformer(self.model)
        reported = self._model_impl.get_sentence_embedding_dimension()
        self.dimension = int(reported) if reported else 384

    def embed(self, text: str) -> list[float]:
        if self._backend == "sentence-transformers" and self._model_impl is not None:
            vec = self._model_impl.encode([text], normalize_embeddings=True)[0]
            return [float(v) for v in vec]
        return _hash_embed(text, self.dimension)


def _dimension_from_model_name(model: str, default: int) -> int:
    tail = model.split("-")[-1]
    if tail.isdigit():
        return int(tail)
    return default


def _hash_embed(text: str, dimension: int) -> list[float]:
    vector: list[float] = []
    counter = 0
    while len(vector) < dimension:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        for idx in range(0, len(digest), 4):
            chunk = digest[idx : idx + 4]
            if len(chunk) < 4:
                break
            value = int.from_bytes(chunk, byteorder="little", signed=False)
            vector.append((value / 2**32) * 2.0 - 1.0)
            if len(vector) == dimension:
                break
        counter += 1

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]
