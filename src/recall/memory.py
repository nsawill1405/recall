from __future__ import annotations

import asyncio
import re
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
        reranker: str | None = None,
    ) -> list[MemoryResult]:
        self._storage.require_compatible_dimensions()

        query = query.strip()
        if not query:
            raise ValueError("query must be non-empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if reranker not in (None, "local"):
            raise ValueError("reranker must be one of: None, 'local'")

        query_embedding = self._embedder.embed(query)
        rows = self._storage.search_memories(
            query_embedding=query_embedding,
            top_k=top_k,
            tags=tags,
        )
        results = [
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
        if reranker == "local":
            results = _local_lexical_rerank(query=query, results=results)
        return results

    def find(
        self,
        query: str,
        tags: list[str] | None = None,
        min_score: float = 0.0,
        reranker: str | None = None,
    ) -> MemoryResult | None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0.0 and 1.0")
        candidate_k = 20 if reranker == "local" else 1
        results = self.search(
            query=query,
            top_k=candidate_k,
            tags=tags,
            reranker=reranker,
        )
        if not results:
            return None
        best = results[0]
        if best.score < min_score:
            return None
        return best

    def forget(self, id: str | None = None, tag: str | None = None) -> int:
        self._storage.require_compatible_dimensions()

        if (id is None and tag is None) or (id is not None and tag is not None):
            raise ValueError("provide exactly one of id or tag")

        if id is not None:
            return self._storage.delete_memory_by_id(memory_id=id)
        return self._storage.delete_memory_by_tag(tag=tag or "")

    def redact(self, id: str, remove: str) -> str:
        new_id, _removed_count = self._redact_internal(id=id, remove=remove)
        return new_id

    def _redact_internal(self, id: str, remove: str) -> tuple[str, int]:
        self._storage.require_compatible_dimensions()

        memory_id = id.strip()
        if not memory_id:
            raise ValueError("id must be non-empty")
        if remove == "":
            raise ValueError("remove must be non-empty")

        current = self._storage.get_current_memory(memory_id)
        if current is None:
            raise ValueError("memory id not found in current namespace")

        source_text = str(current["text"])
        removed_count = source_text.count(remove)
        if removed_count == 0:
            raise ValueError("remove text not found in memory")

        redacted_text = source_text.replace(remove, "")
        if not redacted_text.strip():
            raise ValueError("redaction would leave empty memory text")

        embedding = self._embedder.embed(redacted_text)
        new_id = self._storage.insert_redacted_memory(
            previous_id=current["id"],
            root_id=current["root_id"],
            redacted_text=redacted_text,
            embedding=embedding,
            tags=current["tags"],
            created_at=_now_ts(),
            expires_at=current["expires_at"],
        )
        return new_id, removed_count

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
        reranker: str | None = None,
    ) -> list[MemoryResult]:
        return await asyncio.to_thread(self._memory.search, query, top_k, tags, reranker)

    async def find(
        self,
        query: str,
        tags: list[str] | None = None,
        min_score: float = 0.0,
        reranker: str | None = None,
    ) -> MemoryResult | None:
        return await asyncio.to_thread(self._memory.find, query, tags, min_score, reranker)

    async def forget(self, id: str | None = None, tag: str | None = None) -> int:
        return await asyncio.to_thread(self._memory.forget, id, tag)

    async def redact(self, id: str, remove: str) -> str:
        return await asyncio.to_thread(self._memory.redact, id, remove)

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


def _local_lexical_rerank(query: str, results: list[MemoryResult]) -> list[MemoryResult]:
    query_tokens = _tokenize(query)
    reranked = sorted(
        results,
        key=lambda item: _blend_scores(
            item.score,
            _token_overlap(query_tokens, _tokenize(item.text)),
        ),
        reverse=True,
    )
    normalized: list[MemoryResult] = []
    for item in reranked:
        lexical = _token_overlap(query_tokens, _tokenize(item.text))
        blended = _blend_scores(item.score, lexical)
        normalized.append(
            MemoryResult(
                id=item.id,
                text=item.text,
                score=blended,
                tags=item.tags,
                created_at=item.created_at,
                expires_at=item.expires_at,
            )
        )
    return normalized


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token}


def _token_overlap(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(text_tokens))
    return overlap / len(query_tokens)


def _blend_scores(vector_score: float, lexical_score: float) -> float:
    blended = (vector_score * 0.75) + (lexical_score * 0.25)
    return max(0.0, min(1.0, blended))
