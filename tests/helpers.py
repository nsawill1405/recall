from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable


class DeterministicEmbedder:
    def __init__(self, dimension: int = 8, name: str = "mock", model: str = "mock-v1") -> None:
        self.dimension = dimension
        self.name = name
        self.model = model

    def embed(self, text: str) -> list[float]:
        values: list[float] = []
        salt = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(f"{salt}:{text}".encode()).digest()
            for idx in range(0, len(digest), 4):
                chunk = digest[idx : idx + 4]
                if len(chunk) < 4:
                    break
                value = int.from_bytes(chunk, byteorder="little", signed=False)
                values.append((value / 2**32) * 2.0 - 1.0)
                if len(values) == self.dimension:
                    break
            salt += 1

        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            return values
        return [v / norm for v in values]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
