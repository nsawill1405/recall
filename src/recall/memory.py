from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .embedders import create_embedder
from .embedders.base import Embedder
from .storage import SQLiteStorage, resolve_db_path
from .types import MemoryResult, StoreItem


class Memory:
    def __init__(
        self,
        path: str | None = None,
        namespace: str = "default",
        embedder: str | Embedder | None = None,
        model: str | None = None,
        max_scan_rows: int | None = None,
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
            max_scan_rows=max_scan_rows,
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

        unique_tags = _normalize_tags(tags)

        created_at = _now_ts()
        expires_at = _resolve_store_ttl(ttl_days=ttl_days, created_at=created_at)

        embedding = self._embedder.embed(text)
        return self._storage.insert_memory(
            text=text,
            embedding=embedding,
            tags=unique_tags,
            created_at=created_at,
            expires_at=expires_at,
        )

    def store_many(self, items: list[StoreItem]) -> list[str]:
        self._storage.require_compatible_dimensions()
        if not items:
            return []

        prepared: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, StoreItem):
                raise TypeError("items must contain StoreItem values")

            text = item.text.strip()
            if not text:
                raise ValueError("item.text must be non-empty")

            created_at = _now_ts()
            prepared.append(
                {
                    "text": text,
                    "tags": _normalize_tags(item.tags),
                    "created_at": created_at,
                    "expires_at": _resolve_store_ttl(ttl_days=item.ttl_days, created_at=created_at),
                }
            )

        embeddings = _embed_many(self._embedder, [entry["text"] for entry in prepared])
        if len(embeddings) != len(prepared):
            raise RuntimeError("embed_many returned an unexpected number of embeddings")

        for entry, embedding in zip(prepared, embeddings, strict=True):
            entry["embedding"] = embedding

        return self._storage.insert_many_memories(prepared)

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

    def edit(
        self,
        id: str,
        text: str | None = None,
        tags: list[str] | None = None,
        ttl_days: int | None = None,
    ) -> str:
        self._storage.require_compatible_dimensions()

        memory_id = id.strip()
        if not memory_id:
            raise ValueError("id must be non-empty")
        if text is None and tags is None and ttl_days is None:
            raise ValueError("edit requires at least one change")

        current = self._storage.get_current_memory(memory_id)
        if current is None:
            raise ValueError("memory id not found in current namespace")

        updated_text = str(current["text"])
        text_changed = False
        if text is not None:
            updated_text = text.strip()
            if not updated_text:
                raise ValueError("text must be non-empty")
            text_changed = updated_text != str(current["text"])

        updated_tags = list(current["tags"])
        tags_changed = False
        if tags is not None:
            updated_tags = _normalize_tags(tags)
            tags_changed = updated_tags != list(current["tags"])

        now_ts = _now_ts()
        updated_expires_at = (
            int(current["expires_at"]) if current["expires_at"] is not None else None
        )
        ttl_changed = False
        if ttl_days is not None:
            if ttl_days < 0:
                raise ValueError("ttl_days must be zero or greater")
            updated_expires_at = None if ttl_days == 0 else now_ts + ttl_days * 86400
            ttl_changed = updated_expires_at != current["expires_at"]

        if not (text_changed or tags_changed or ttl_changed):
            raise ValueError("edit produced no changes")

        new_embedding = self._embedder.embed(updated_text) if text_changed else None
        return self._storage.insert_edited_memory(
            previous_id=str(current["id"]),
            root_id=str(current["root_id"]),
            text=updated_text,
            tags=updated_tags,
            created_at=now_ts,
            expires_at=updated_expires_at,
            new_embedding=new_embedding,
        )

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

    def prune_expired(self) -> int:
        self._storage.require_compatible_dimensions()
        return self._storage.prune_expired_lineages()

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

    def export_jsonl(
        self,
        path: str,
        namespace: str | None = None,
        include_history: bool = False,
    ) -> int:
        self._storage.require_compatible_dimensions()
        export_namespace = _resolve_namespace(namespace, self.namespace)
        export_path = Path(path).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self._storage.export_rows(export_namespace, include_history=include_history)
        with export_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
        return len(rows)

    def import_jsonl(
        self,
        path: str,
        namespace: str | None = None,
        conflict: str = "skip",
    ) -> int:
        self._storage.require_compatible_dimensions()
        if conflict not in {"skip", "overwrite", "new"}:
            raise ValueError("conflict must be one of: skip, overwrite, new")

        import_namespace = _resolve_namespace(namespace, self.namespace)
        import_path = Path(path).expanduser().resolve()
        if not import_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {import_path}")

        rows: list[dict[str, Any]] = []
        with import_path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    raw = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_num}") from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"Line {line_num} must contain a JSON object")
                rows.append(_normalize_import_row(raw, line_num))

        if not rows:
            return 0

        embeddings = _embed_many(self._embedder, [row["text"] for row in rows])
        if len(embeddings) != len(rows):
            raise RuntimeError("embed_many returned an unexpected number of embeddings")

        prepared_rows: list[dict[str, Any]] = []
        for row, embedding in zip(rows, embeddings, strict=True):
            prepared_rows.append(
                {
                    "id": row["id"],
                    "root_id": row["root_id"],
                    "replaces_id": row["replaces_id"],
                    "is_current": row["is_current"],
                    "text": row["text"],
                    "tags": row["tags"],
                    "created_at": row["created_at"],
                    "expires_at": row["expires_at"],
                    "embedding": embedding,
                }
            )

        return self._storage.import_prepared_rows(
            prepared_rows,
            import_namespace,
            conflict=conflict,
        )


class AsyncMemory:
    def __init__(
        self,
        path: str | None = None,
        namespace: str = "default",
        embedder: str | Embedder | None = None,
        model: str | None = None,
        max_scan_rows: int | None = None,
    ) -> None:
        self._memory = Memory(
            path=path,
            namespace=namespace,
            embedder=embedder,
            model=model,
            max_scan_rows=max_scan_rows,
        )

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

    async def edit(
        self,
        id: str,
        text: str | None = None,
        tags: list[str] | None = None,
        ttl_days: int | None = None,
    ) -> str:
        return await asyncio.to_thread(self._memory.edit, id, text, tags, ttl_days)

    async def redact(self, id: str, remove: str) -> str:
        return await asyncio.to_thread(self._memory.redact, id, remove)

    async def prune_expired(self) -> int:
        return await asyncio.to_thread(self._memory.prune_expired)

    async def list(self, limit: int = 20) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._memory.list, limit)

    async def stats(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._memory.stats)

    async def rebuild_index(self) -> int:
        return await asyncio.to_thread(self._memory.rebuild_index)

    async def store_many(self, items: Sequence[StoreItem]) -> Sequence[str]:
        return await asyncio.to_thread(self._memory.store_many, list(items))

    async def export_jsonl(
        self,
        path: str,
        namespace: str | None = None,
        include_history: bool = False,
    ) -> int:
        return await asyncio.to_thread(self._memory.export_jsonl, path, namespace, include_history)

    async def import_jsonl(
        self,
        path: str,
        namespace: str | None = None,
        conflict: str = "skip",
    ) -> int:
        return await asyncio.to_thread(self._memory.import_jsonl, path, namespace, conflict)

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


def _resolve_store_ttl(ttl_days: int | None, created_at: int) -> int | None:
    if ttl_days is None:
        return None
    if ttl_days <= 0:
        raise ValueError("ttl_days must be greater than zero")
    return created_at + ttl_days * 86400


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if tags is None:
        return []
    return sorted({str(tag).strip() for tag in tags if str(tag).strip()})


def _embed_many(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    embed_many_fn = getattr(embedder, "embed_many", None)
    if callable(embed_many_fn):
        vectors = embed_many_fn(texts)
        return [[float(value) for value in row] for row in vectors]
    return [embedder.embed(text) for text in texts]


def _resolve_namespace(candidate: str | None, fallback: str) -> str:
    namespace = fallback if candidate is None else candidate
    namespace = namespace.strip()
    if not namespace:
        raise ValueError("namespace must be non-empty")
    return namespace


def _normalize_import_row(raw: dict[str, Any], line_num: int) -> dict[str, Any]:
    text = str(raw.get("text", "")).strip()
    if not text:
        raise ValueError(f"Line {line_num} has empty text")

    tags_raw = raw.get("tags")
    tags: list[str]
    if tags_raw is None:
        tags = []
    elif isinstance(tags_raw, list):
        tags = _normalize_tags([str(tag) for tag in tags_raw])
    else:
        raise ValueError(f"Line {line_num} has invalid tags payload")

    memory_id = str(raw.get("id", "")).strip()
    if not memory_id:
        raise ValueError(f"Line {line_num} is missing id")

    root_id_raw = raw.get("root_id")
    root_id = memory_id if root_id_raw is None else str(root_id_raw).strip()
    if not root_id:
        root_id = memory_id

    replaces_id_raw = raw.get("replaces_id")
    replaces_id = None if replaces_id_raw in (None, "") else str(replaces_id_raw).strip()

    created_at = _coerce_int(raw.get("created_at"), "created_at", line_num)
    expires_at_raw = raw.get("expires_at")
    expires_at = (
        _coerce_int(expires_at_raw, "expires_at", line_num) if expires_at_raw is not None else None
    )

    return {
        "id": memory_id,
        "root_id": root_id,
        "replaces_id": replaces_id,
        "is_current": bool(raw.get("is_current", True)),
        "text": text,
        "tags": tags,
        "created_at": created_at,
        "expires_at": expires_at,
    }


def _coerce_int(value: Any, field: str, line_num: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Line {line_num} has invalid {field}") from exc


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
