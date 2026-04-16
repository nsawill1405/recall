from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from .embedders import create_embedder
from .embedders.base import Embedder
from .storage import SQLiteStorage, resolve_db_path
from .types import MemoryResult


class Memory:
    def __init__(
        self,
        path: str | None = None,
        namespace: str = "default",
        embedder: str | Embedder | None = None,
        model: str | None = None,
        *,
        _allow_dimension_mismatch: bool = False,
    ) -> None:
        if not namespace:
            raise ValueError("namespace must be a non-empty string")

        self.path = resolve_db_path(path)
        self.namespace = namespace
        self._embedder = create_embedder(embedder=embedder, model=model)
        self._storage = SQLiteStorage(
            db_path=self.path,
            namespace=namespace,
            embedder_name=self._embedder.name,
            embedder_model=self._embedder.model,
            embedder_dimension=self._embedder.dimension,
            allow_dimension_mismatch=_allow_dimension_mismatch,
        )

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    @property
    def embedder_model(self) -> str:
        return self._embedder.model

    @property
    def embedder_dimension(self) -> int:
        return self._embedder.dimension

    @property
    def vector_backend(self) -> str:
        return self._storage.vector_backend

    def close(self) -> None:
        self._storage.close()

    def store(
        self,
        text: str,
        tags: list[str] | None = None,
        ttl_days: int | None = None,
    ) -> str:
        self._storage.require_compatible_dimensions()

        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")

        unique_tags = sorted({str(tag) for tag in (tags or []) if str(tag).strip()})

        created_at = _now_ts()
        expires_at = None
        if ttl_days is not None:
            if ttl_days <= 0:
                raise ValueError("ttl_days must be greater than zero")
            expires_at = created_at + ttl_days * 86400

        embedding = self._embedder.embed(text)
        return self._storage.insert_memory(
            text=text,
            embedding=embedding,
            tags=unique_tags,
            created_at=created_at,
            expires_at=expires_at,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> list[MemoryResult]:
        self._storage.require_compatible_dimensions()

        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_embedding = self._embedder.embed(query)
        rows = self._storage.search_memories(
            query_embedding=query_embedding,
            top_k=top_k,
            tags=tags,
        )
        return [
            MemoryResult(
                id=row["id"],
                text=row["text"],
                score=row["score"],
                tags=row["tags"],
                created_at=_to_datetime_required(row["created_at"]),
                expires_at=_to_datetime(row["expires_at"]),
            )
            for row in rows
        ]

    def forget(self, id: str | None = None, tag: str | None = None) -> int:
        self._storage.require_compatible_dimensions()

        if (id is None and tag is None) or (id is not None and tag is not None):
            raise ValueError("provide exactly one of id or tag")

        if id is not None:
            return self._storage.delete_memory_by_id(memory_id=id)
        return self._storage.delete_memory_by_tag(tag=tag or "")

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        self._storage.require_compatible_dimensions()
        rows = self._storage.list_memories(limit=limit)
        return [
            {
                **row,
                "created_at": _to_datetime_required(row["created_at"]),
                "expires_at": _to_datetime(row["expires_at"]),
            }
            for row in rows
        ]

    def stats(self) -> dict[str, Any]:
        self._storage.require_compatible_dimensions()
        return self._storage.stats()

    def rebuild_index(self) -> int:
        self._storage.reconfigure_embedding_space(
            provider=self._embedder.name,
            model=self._embedder.model,
            dimension=self._embedder.dimension,
        )

        rows = list(self._storage.iter_memory_texts())
        if not rows:
            return 0

        ids = [row_id for row_id, _ in rows]
        texts = [text for _, text in rows]
        embeddings = self._embedder.embed_many(texts)

        for memory_id, embedding in zip(ids, embeddings, strict=False):
            self._storage.replace_embedding(memory_id=memory_id, embedding=embedding)

        return len(rows)


class AsyncMemory:
    def __init__(
        self,
        path: str | None = None,
        namespace: str = "default",
        embedder: str | Embedder | None = None,
        model: str | None = None,
    ) -> None:
        self._memory = Memory(path=path, namespace=namespace, embedder=embedder, model=model)

    async def store(
        self,
        text: str,
        tags: list[str] | None = None,
        ttl_days: int | None = None,
    ) -> str:
        return await asyncio.to_thread(self._memory.store, text, tags, ttl_days)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> list[MemoryResult]:
        return await asyncio.to_thread(self._memory.search, query, top_k, tags)

    async def forget(self, id: str | None = None, tag: str | None = None) -> int:
        return await asyncio.to_thread(self._memory.forget, id, tag)

    async def list(self, limit: int = 20) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._memory.list, limit)

    async def stats(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._memory.stats)

    async def rebuild_index(self) -> int:
        return await asyncio.to_thread(self._memory.rebuild_index)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._memory.close)


def _now_ts() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def _to_datetime(value: int | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _to_datetime_required(value: int | None) -> datetime:
    converted = _to_datetime(value)
    if converted is None:
        raise ValueError("Expected datetime value but received None.")
    return converted
