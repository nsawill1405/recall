from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .anthropic import AnthropicEmbedder
from .base import Embedder
from .cohere import CohereEmbedder
from .local import LocalEmbedder
from .openai import OpenAIEmbedder


@dataclass(slots=True)
class _CustomEmbedderAdapter:
    _embedder: Any
    name: str
    model: str
    dimension: int

    def embed(self, text: str) -> list[float]:
        values = self._embedder.embed(text)
        return [float(v) for v in values]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        embed_many_fn = getattr(self._embedder, "embed_many", None)
        if callable(embed_many_fn):
            vectors = embed_many_fn(texts)
            return [[float(v) for v in row] for row in vectors]
        return [self.embed(text) for text in texts]


def create_embedder(embedder: str | Embedder | None = None, model: str | None = None) -> Embedder:
    if embedder is not None and not isinstance(embedder, str):
        return _adapt_custom_embedder(embedder)

    provider = _resolve_provider(embedder)
    if provider == "openai":
        return OpenAIEmbedder(model=model)
    if provider == "anthropic":
        return AnthropicEmbedder(model=model)
    if provider == "cohere":
        return CohereEmbedder(model=model)
    if provider == "local":
        return LocalEmbedder(model=model)
    raise ValueError(f"Unsupported embedder '{provider}'.")


def _resolve_provider(embedder: str | None) -> str:
    if embedder:
        return embedder.lower()
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("COHERE_API_KEY"):
        return "cohere"
    return "local"


def _adapt_custom_embedder(embedder: Embedder) -> Embedder:
    dimension = int(embedder.dimension)
    name = str(embedder.name)
    model = str(embedder.model)
    return _CustomEmbedderAdapter(
        _embedder=embedder,
        name=name,
        model=model,
        dimension=dimension,
    )


__all__ = ["create_embedder"]
