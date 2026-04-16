from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from recall import Memory, StoreItem


def test_store_many_roundtrip(db_path: Path) -> None:
    mem = Memory(
        path=str(db_path),
        namespace="default",
        embedder="local",
        model="hash-embedding-v1",
    )
    try:
        ids = mem.store_many(
            [
                StoreItem(text="alpha", tags=["one"]),
                StoreItem(text="beta", tags=["two"], ttl_days=1),
                StoreItem(text="gamma", tags=["two", "three"]),
            ]
        )
        assert len(ids) == 3
        assert len(set(ids)) == 3

        listed = mem.list(limit=10)
        assert len(listed) == 3
        assert {row["text"] for row in listed} == {"alpha", "beta", "gamma"}
    finally:
        mem.close()


class BatchAwareEmbedder:
    name = "batch-aware"
    model = "batch-aware-v1"
    dimension = 8

    def __init__(self) -> None:
        self.embed_calls = 0
        self.embed_many_calls = 0

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return _tiny_hash_embedding(text, self.dimension)

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        self.embed_many_calls += 1
        return [self.embed(text) for text in texts]


class EmbedOnly:
    name = "embed-only"
    model = "embed-only-v1"
    dimension = 8

    def __init__(self) -> None:
        self.embed_calls = 0

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return _tiny_hash_embedding(text, self.dimension)


class BrokenBatchEmbedder:
    name = "broken-batch"
    model = "broken-batch-v1"
    dimension = 8

    def embed(self, text: str) -> list[float]:
        return _tiny_hash_embedding(text, self.dimension)

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        vectors: list[list[float]] = []
        for idx, text in enumerate(text_list):
            if idx == 1:
                vectors.append([])
            else:
                vectors.append(self.embed(text))
        return vectors


def test_store_many_uses_embed_many_path(db_path: Path) -> None:
    embedder = BatchAwareEmbedder()
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        mem.store_many([StoreItem("one"), StoreItem("two"), StoreItem("three")])
        assert embedder.embed_many_calls == 1
    finally:
        mem.close()


def test_store_many_falls_back_to_per_item_embed(db_path: Path) -> None:
    embedder: Any = EmbedOnly()
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        mem.store_many([StoreItem("one"), StoreItem("two"), StoreItem("three")])
        assert embedder.embed_calls == 3
    finally:
        mem.close()


def test_store_many_is_all_or_nothing_on_failure(db_path: Path) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=BrokenBatchEmbedder())
    try:
        with pytest.raises(ValueError, match="Embedding vectors cannot be empty"):
            mem.store_many([StoreItem("good"), StoreItem("bad")])
        assert mem.list(limit=10) == []
    finally:
        mem.close()


def _tiny_hash_embedding(text: str, dimension: int) -> list[float]:
    values = [float((hash((text, idx)) % 2000) - 1000) for idx in range(dimension)]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]
