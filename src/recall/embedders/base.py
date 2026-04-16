from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Protocol


class Embedder(Protocol):
    name: str
    model: str
    dimension: int

    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        ...


class BaseEmbedder(ABC):
    name: str
    model: str
    dimension: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]
