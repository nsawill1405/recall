from __future__ import annotations

import importlib
import json
import math
import re
import sqlite3
import struct
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .errors import DimensionMismatchError


def default_db_path() -> Path:
    return Path("~/.recall/default.db").expanduser()


def resolve_db_path(path: str | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return default_db_path().resolve()


class SQLiteStorage:
    def __init__(
        self,
        db_path: Path,
        namespace: str,
        embedder_name: str,
        embedder_model: str,
        embedder_dimension: int,
        *,
        allow_dimension_mismatch: bool = False,
        vec_module: Any | None = None,
        max_scan_rows: int | None = None,
    ) -> None:
        if max_scan_rows is not None and max_scan_rows <= 0:
            raise ValueError("max_scan_rows must be greater than zero when provided")

        self.db_path = db_path
        self.namespace = namespace
        self.embedder_name = embedder_name
        self.embedder_model = embedder_model
        self.embedder_dimension = embedder_dimension
        self.allow_dimension_mismatch = allow_dimension_mismatch
        self.max_scan_rows = max_scan_rows

        self._lock = threading.RLock()
        self._vector_backend = "sqlite"
        self._vec_module = vec_module if vec_module is not None else _import_vec_module()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._configure_connection()
        self._vec_loaded = self._load_vec_extension()
        self._run_base_migrations()
        self._dimension_mismatch: DimensionMismatchError | None = self._validate_metadata()
        self._ensure_vec_index_table(self.current_dimension)

    @property
    def vector_backend(self) -> str:
        return self._vector_backend

    @property
    def dimension_mismatch(self) -> DimensionMismatchError | None:
        return self._dimension_mismatch

    @property
    def current_dimension(self) -> int:
        value = self._get_meta("embedding_dimension")
        if value is None:
            return self.embedder_dimension
        return int(value)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def insert_memory(
        self,
        text: str,
        embedding: list[float],
        tags: list[str],
        created_at: int,
        expires_at: int | None,
    ) -> str:
        memory_id = uuid.uuid4().hex
        memory_blob, vector_payload = self._embedding_payloads(embedding)
        with self._lock, self._conn:
            self._insert_memory_row(
                memory_id=memory_id,
                root_id=memory_id,
                replaces_id=None,
                is_current=True,
                namespace=self.namespace,
                text=text,
                memory_blob=memory_blob,
                tags=tags,
                created_at=created_at,
                expires_at=expires_at,
                vector_payload=vector_payload,
            )
        return memory_id

    def insert_many_memories(self, items: list[dict[str, Any]]) -> list[str]:
        if not items:
            return []

        memory_ids: list[str] = []
        with self._lock, self._conn:
            for item in items:
                memory_id = uuid.uuid4().hex
                memory_blob, vector_payload = self._embedding_payloads(item["embedding"])
                self._insert_memory_row(
                    memory_id=memory_id,
                    root_id=memory_id,
                    replaces_id=None,
                    is_current=True,
                    namespace=self.namespace,
                    text=str(item["text"]),
                    memory_blob=memory_blob,
                    tags=[str(tag) for tag in item.get("tags", [])],
                    created_at=int(item["created_at"]),
                    expires_at=(
                        int(item["expires_at"])
                        if item.get("expires_at") is not None
                        else None
                    ),
                    vector_payload=vector_payload,
                )
                memory_ids.append(memory_id)

        return memory_ids

    def get_current_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, root_id, namespace, text, embedding, tags, created_at, expires_at
                FROM memories
                WHERE id = ?
                  AND namespace = ?
                  AND is_current = 1
                """,
                (memory_id, self.namespace),
            ).fetchone()

        if row is None:
            return None

        return {
            "id": str(row["id"]),
            "root_id": str(row["root_id"]),
            "namespace": str(row["namespace"]),
            "text": str(row["text"]),
            "embedding": bytes(row["embedding"]),
            "tags": _parse_tags(row["tags"]),
            "created_at": int(row["created_at"]),
            "expires_at": int(row["expires_at"]) if row["expires_at"] is not None else None,
        }

    def insert_redacted_memory(
        self,
        previous_id: str,
        root_id: str,
        redacted_text: str,
        embedding: list[float],
        tags: list[str],
        created_at: int,
        expires_at: int | None,
    ) -> str:
        return self._insert_versioned_memory(
            previous_id=previous_id,
            root_id=root_id,
            text=redacted_text,
            tags=tags,
            created_at=created_at,
            expires_at=expires_at,
            new_embedding=embedding,
        )

    def insert_edited_memory(
        self,
        previous_id: str,
        root_id: str,
        text: str,
        tags: list[str],
        created_at: int,
        expires_at: int | None,
        new_embedding: list[float] | None,
    ) -> str:
        return self._insert_versioned_memory(
            previous_id=previous_id,
            root_id=root_id,
            text=text,
            tags=tags,
            created_at=created_at,
            expires_at=expires_at,
            new_embedding=new_embedding,
        )

    def _insert_versioned_memory(
        self,
        previous_id: str,
        root_id: str,
        text: str,
        tags: list[str],
        created_at: int,
        expires_at: int | None,
        new_embedding: list[float] | None,
    ) -> str:
        new_id = uuid.uuid4().hex

        with self._lock, self._conn:
            previous = self._conn.execute(
                """
                SELECT id, embedding
                FROM memories
                WHERE id = ?
                  AND namespace = ?
                  AND is_current = 1
                """,
                (previous_id, self.namespace),
            ).fetchone()
            if previous is None:
                raise ValueError("memory id not found in current namespace")

            if new_embedding is None:
                memory_blob = bytes(previous["embedding"])
                vector_row = self._conn.execute(
                    "SELECT embedding FROM vec_index WHERE memory_id = ?",
                    (previous_id,),
                ).fetchone()
                if vector_row is not None and vector_row["embedding"] is not None:
                    vector_payload = bytes(vector_row["embedding"])
                else:
                    vector_payload = self._vector_param(
                        _deserialize_embedding(
                            memory_blob,
                            expected_dimension=self.current_dimension,
                        )
                    )
            else:
                memory_blob, vector_payload = self._embedding_payloads(new_embedding)

            cursor = self._conn.execute(
                """
                UPDATE memories
                SET is_current = 0
                WHERE id = ?
                  AND namespace = ?
                  AND is_current = 1
                """,
                (previous_id, self.namespace),
            )
            if int(cursor.rowcount or 0) != 1:
                raise ValueError("memory id not found in current namespace")

            self._insert_memory_row(
                memory_id=new_id,
                root_id=root_id,
                replaces_id=previous_id,
                is_current=True,
                namespace=self.namespace,
                text=text,
                memory_blob=memory_blob,
                tags=tags,
                created_at=created_at,
                expires_at=expires_at,
                vector_payload=vector_payload,
            )
            self._conn.execute("DELETE FROM vec_index WHERE memory_id = ?", (previous_id,))

        return new_id

    def delete_memory_by_id(self, memory_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT root_id FROM memories WHERE id = ? AND namespace = ?",
                (memory_id, self.namespace),
            ).fetchone()
        if row is None or row["root_id"] is None:
            return 0

        return self._delete_lineages({str(row["root_id"])})

    def delete_memory_by_tag(self, tag: str) -> int:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT root_id, tags
                FROM memories
                WHERE namespace = ?
                  AND is_current = 1
                """,
                (self.namespace,),
            ).fetchall()

        root_ids: set[str] = set()
        for row in rows:
            parsed_tags = _parse_tags(row["tags"])
            if tag in parsed_tags and row["root_id"] is not None:
                root_ids.add(str(row["root_id"]))

        return self._delete_lineages(root_ids)

    def prune_expired_lineages(self) -> int:
        now = _unixepoch()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT DISTINCT root_id
                FROM memories
                WHERE namespace = ?
                  AND is_current = 1
                  AND expires_at IS NOT NULL
                  AND expires_at <= ?
                """,
                (self.namespace, now),
            ).fetchall()

        root_ids = {
            str(row["root_id"])
            for row in rows
            if row["root_id"] is not None
        }
        return self._delete_lineages(root_ids)

    def _delete_lineages(self, root_ids: set[str]) -> int:
        if not root_ids:
            return 0

        lineage_list = sorted(root_ids)
        placeholders = ",".join("?" for _ in lineage_list)

        with self._lock, self._conn:
            deleted_rows = self._conn.execute(
                f"""
                SELECT id
                FROM memories
                WHERE namespace = ?
                  AND root_id IN ({placeholders})
                """,  # noqa: S608
                [self.namespace, *lineage_list],
            ).fetchall()

            ids = [str(row["id"]) for row in deleted_rows]
            if not ids:
                return 0

            self._conn.execute(
                f"""
                DELETE FROM memories
                WHERE namespace = ?
                  AND root_id IN ({placeholders})
                """,  # noqa: S608
                [self.namespace, *lineage_list],
            )

            id_placeholders = ",".join("?" for _ in ids)
            self._conn.execute(
                f"DELETE FROM vec_index WHERE memory_id IN ({id_placeholders})",  # noqa: S608
                ids,
            )

        return len(ids)

    def search_memories(
        self,
        query_embedding: list[float],
        top_k: int,
        tags: list[str] | None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []

        candidate_limit = max(top_k * 5, top_k)
        rows = self._search_candidates(query_embedding, candidate_limit)

        wanted_tags = set(tags or [])
        scored: list[dict[str, Any]] = []
        for row in rows:
            row_tags = _parse_tags(row["tags"])
            if wanted_tags and not wanted_tags.intersection(row_tags):
                continue
            similarity = _cosine_similarity(
                query_embedding,
                _deserialize_embedding(row["embedding"], expected_dimension=self.current_dimension),
            )
            score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            scored.append(
                {
                    "id": str(row["id"]),
                    "text": str(row["text"]),
                    "score": score,
                    "tags": row_tags,
                    "created_at": int(row["created_at"]),
                    "expires_at": int(row["expires_at"]) if row["expires_at"] is not None else None,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def list_memories(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, text, tags, created_at, expires_at
                FROM memories
                WHERE namespace = ?
                  AND is_current = 1
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (self.namespace, limit),
            ).fetchall()

        return [
            {
                "id": str(row["id"]),
                "text": str(row["text"]),
                "tags": _parse_tags(row["tags"]),
                "created_at": int(row["created_at"]),
                "expires_at": int(row["expires_at"]) if row["expires_at"] is not None else None,
            }
            for row in rows
        ]

    def stats(self) -> dict[str, Any]:
        now = _unixepoch()
        with self._lock:
            total = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE namespace = ? AND is_current = 1",
                    (self.namespace,),
                ).fetchone()[0]
            )
            expired = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM memories
                    WHERE namespace = ?
                      AND is_current = 1
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    """,
                    (self.namespace, now),
                ).fetchone()[0]
            )

        return {
            "namespace": self.namespace,
            "path": str(self.db_path),
            "total": total,
            "expired": expired,
            "active": total - expired,
            "embedder": {
                "provider": self._get_meta("embedding_provider") or self.embedder_name,
                "model": self._get_meta("embedding_model") or self.embedder_model,
                "dimension": int(self._get_meta("embedding_dimension") or self.embedder_dimension),
            },
            "vector_backend": self._vector_backend,
        }

    def iter_memory_texts(self) -> Iterator[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, text
                FROM memories
                WHERE namespace = ?
                  AND is_current = 1
                ORDER BY created_at ASC
                """,
                (self.namespace,),
            ).fetchall()
        for row in rows:
            yield str(row["id"]), str(row["text"])

    def replace_embedding(self, memory_id: str, embedding: list[float]) -> None:
        memory_blob, vector_payload = self._embedding_payloads(embedding)
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                (memory_blob, memory_id),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO vec_index (memory_id, embedding) VALUES (?, ?)",
                (memory_id, vector_payload),
            )

    def reconfigure_embedding_space(self, provider: str, model: str, dimension: int) -> None:
        previous_dimension = int(self._get_meta("embedding_dimension") or dimension)
        with self._lock, self._conn:
            if previous_dimension != dimension:
                self._conn.execute("DROP TABLE IF EXISTS vec_index")
                self._vector_backend = "sqlite"
                self._ensure_vec_index_table(dimension)
            self._set_meta("embedding_provider", provider)
            self._set_meta("embedding_model", model)
            self._set_meta("embedding_dimension", str(dimension))

        self.embedder_name = provider
        self.embedder_model = model
        self.embedder_dimension = dimension
        self._dimension_mismatch = None

    def export_rows(self, namespace: str, include_history: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            if include_history:
                rows = self._conn.execute(
                    """
                    SELECT
                        id,
                        root_id,
                        replaces_id,
                        is_current,
                        namespace,
                        text,
                        tags,
                        created_at,
                        expires_at
                    FROM memories
                    WHERE namespace = ?
                    ORDER BY created_at ASC, id ASC
                    """,
                    (namespace,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT
                        id,
                        namespace,
                        text,
                        tags,
                        created_at,
                        expires_at
                    FROM memories
                    WHERE namespace = ?
                      AND is_current = 1
                    ORDER BY created_at ASC, id ASC
                    """,
                    (namespace,),
                ).fetchall()

        exported: list[dict[str, Any]] = []
        for row in rows:
            memory_id = str(row["id"])
            if include_history:
                root_id = str(row["root_id"]) if row["root_id"] else memory_id
                replaces_id = str(row["replaces_id"]) if row["replaces_id"] else None
                is_current = bool(row["is_current"])
            else:
                root_id = memory_id
                replaces_id = None
                is_current = True

            exported.append(
                {
                    "id": memory_id,
                    "root_id": root_id,
                    "replaces_id": replaces_id,
                    "is_current": is_current,
                    "namespace": str(row["namespace"]),
                    "text": str(row["text"]),
                    "tags": _parse_tags(row["tags"]),
                    "created_at": int(row["created_at"]),
                    "expires_at": (
                        int(row["expires_at"]) if row["expires_at"] is not None else None
                    ),
                }
            )

        return exported

    def import_prepared_rows(
        self,
        rows: list[dict[str, Any]],
        target_namespace: str,
        conflict: str,
    ) -> int:
        if conflict not in {"skip", "overwrite", "new"}:
            raise ValueError("conflict must be one of: skip, overwrite, new")
        if not rows:
            return 0

        id_map = self._build_id_map_for_new(rows, target_namespace) if conflict == "new" else {}

        imported = 0
        with self._lock, self._conn:
            for row in rows:
                source_id = str(row["id"])
                memory_id = id_map.get(source_id, source_id)

                source_root = str(row.get("root_id") or source_id)
                source_replaces = row.get("replaces_id")

                root_id = id_map.get(source_root, source_root)
                replaces_id = None
                if source_replaces is not None:
                    replaces_id = id_map.get(str(source_replaces), str(source_replaces))

                exists = self._conn.execute(
                    "SELECT 1 FROM memories WHERE id = ? AND namespace = ?",
                    (memory_id, target_namespace),
                ).fetchone()

                if exists is not None:
                    if conflict == "skip":
                        continue
                    if conflict == "overwrite":
                        self._conn.execute(
                            "DELETE FROM memories WHERE id = ? AND namespace = ?",
                            (memory_id, target_namespace),
                        )
                        self._conn.execute(
                            "DELETE FROM vec_index WHERE memory_id = ?",
                            (memory_id,),
                        )
                    else:
                        raise RuntimeError(
                            "unexpected id collision while importing with conflict='new'"
                        )

                embedding_values = row.get("embedding")
                if embedding_values is None:
                    raise ValueError("prepared import rows must include an embedding")

                memory_blob, vector_payload = self._embedding_payloads(embedding_values)
                is_current = bool(row.get("is_current", True))
                self._insert_memory_row(
                    memory_id=memory_id,
                    root_id=root_id,
                    replaces_id=replaces_id,
                    is_current=is_current,
                    namespace=target_namespace,
                    text=str(row["text"]),
                    memory_blob=memory_blob,
                    tags=[str(tag) for tag in row.get("tags", [])],
                    created_at=int(row["created_at"]),
                    expires_at=(
                        int(row["expires_at"])
                        if row.get("expires_at") is not None
                        else None
                    ),
                    vector_payload=vector_payload,
                )
                imported += 1

            self._normalize_current_versions_locked(target_namespace)

        return imported

    def require_compatible_dimensions(self) -> None:
        if self._dimension_mismatch is not None:
            raise self._dimension_mismatch

    def _configure_connection(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")

    def _run_base_migrations(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recall_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    root_id TEXT,
                    replaces_id TEXT,
                    is_current INTEGER NOT NULL DEFAULT 1,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER
                )
                """
            )

        self._ensure_versioning_columns()

        with self._lock, self._conn:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)"
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_namespace_expiry
                ON memories(namespace, expires_at)
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_namespace_current
                ON memories(namespace, is_current)
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_root_id ON memories(root_id)"
            )

    def _ensure_versioning_columns(self) -> None:
        with self._lock, self._conn:
            columns = {
                str(row["name"])
                for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
            }

            if "root_id" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN root_id TEXT")
            if "is_current" not in columns:
                self._conn.execute(
                    "ALTER TABLE memories ADD COLUMN is_current INTEGER NOT NULL DEFAULT 1"
                )
            if "replaces_id" not in columns:
                self._conn.execute("ALTER TABLE memories ADD COLUMN replaces_id TEXT")

            self._conn.execute(
                "UPDATE memories SET root_id = id WHERE root_id IS NULL OR root_id = ''"
            )
            self._conn.execute("UPDATE memories SET is_current = 1 WHERE is_current IS NULL")

    def _validate_metadata(self) -> DimensionMismatchError | None:
        db_dimension_raw = self._get_meta("embedding_dimension")
        if db_dimension_raw is None:
            self._set_meta("embedding_provider", self.embedder_name)
            self._set_meta("embedding_model", self.embedder_model)
            self._set_meta("embedding_dimension", str(self.embedder_dimension))
            return None

        db_dimension = int(db_dimension_raw)
        db_provider = self._get_meta("embedding_provider") or "unknown"
        db_model = self._get_meta("embedding_model") or "unknown"

        if db_dimension == self.embedder_dimension:
            return None

        mismatch = DimensionMismatchError(
            db_dimension=db_dimension,
            requested_dimension=self.embedder_dimension,
            db_provider=db_provider,
            requested_provider=self.embedder_name,
            db_model=db_model,
            requested_model=self.embedder_model,
        )
        if self.allow_dimension_mismatch:
            return mismatch
        raise mismatch

    def _load_vec_extension(self) -> bool:
        if self._vec_module is None:
            return False
        try:
            self._vec_module.load(self._conn)
            return True
        except Exception:
            return False

    def _ensure_vec_index_table(self, dimension: int) -> None:
        with self._lock:
            existing_sql = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'vec_index'"
            ).fetchone()

            if existing_sql and existing_sql[0]:
                sql = str(existing_sql[0])
                if "VIRTUAL TABLE" in sql.upper():
                    parsed_dim = _parse_vec_dimension(sql)
                    if parsed_dim is not None and parsed_dim != dimension:
                        mismatch = DimensionMismatchError(
                            db_dimension=parsed_dim,
                            requested_dimension=dimension,
                            db_provider=self._get_meta("embedding_provider") or "unknown",
                            requested_provider=self.embedder_name,
                            db_model=self._get_meta("embedding_model") or "unknown",
                            requested_model=self.embedder_model,
                        )
                        if self.allow_dimension_mismatch:
                            self._dimension_mismatch = mismatch
                            return
                        raise mismatch
                    self._vector_backend = "sqlite-vec"
                else:
                    self._vector_backend = "sqlite"
                return

            if self._vec_loaded:
                try:
                    self._conn.execute(
                        f"""
                        CREATE VIRTUAL TABLE vec_index USING vec0(
                            memory_id TEXT,
                            embedding float[{dimension}]
                        )
                        """
                    )
                    self._vector_backend = "sqlite-vec"
                    return
                except sqlite3.OperationalError:
                    self._vector_backend = "sqlite"

            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vec_index (
                    memory_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
                """
            )
            self._vector_backend = "sqlite"

    def _search_candidates(self, query_embedding: list[float], limit: int) -> list[sqlite3.Row]:
        now = _unixepoch()
        if self._vector_backend == "sqlite-vec":
            try:
                with self._lock:
                    return self._conn.execute(
                        """
                        SELECT m.id, m.text, m.tags, m.created_at, m.expires_at, m.embedding,
                               vec_distance_cosine(vi.embedding, ?) AS ann_distance
                        FROM vec_index vi
                        JOIN memories m ON m.id = vi.memory_id
                        WHERE m.namespace = ?
                          AND m.is_current = 1
                          AND (m.expires_at IS NULL OR m.expires_at > ?)
                        ORDER BY ann_distance ASC
                        LIMIT ?
                        """,
                        (self._vector_param(query_embedding), self.namespace, now, limit),
                    ).fetchall()
            except sqlite3.OperationalError:
                self._vector_backend = "sqlite"

        with self._lock:
            if self.max_scan_rows is None:
                return self._conn.execute(
                    """
                    SELECT id, text, tags, created_at, expires_at, embedding
                    FROM memories
                    WHERE namespace = ?
                      AND is_current = 1
                      AND (expires_at IS NULL OR expires_at > ?)
                    """,
                    (self.namespace, now),
                ).fetchall()

            return self._conn.execute(
                """
                SELECT id, text, tags, created_at, expires_at, embedding
                FROM memories
                WHERE namespace = ?
                  AND is_current = 1
                  AND (expires_at IS NULL OR expires_at > ?)
                LIMIT ?
                """,
                (self.namespace, now, self.max_scan_rows),
            ).fetchall()

    def _vector_param(self, embedding: list[float]) -> bytes:
        if self._vector_backend == "sqlite-vec" and self._vec_module is not None:
            serializer = getattr(self._vec_module, "serialize_float32", None)
            if callable(serializer):
                return serializer(embedding)
        return _serialize_embedding(embedding)

    def _embedding_payloads(self, embedding: list[float]) -> tuple[bytes, bytes]:
        memory_blob = _serialize_embedding(embedding)
        vector_payload = self._vector_param(embedding)
        return memory_blob, vector_payload

    def _insert_memory_row(
        self,
        *,
        memory_id: str,
        root_id: str,
        replaces_id: str | None,
        is_current: bool,
        namespace: str,
        text: str,
        memory_blob: bytes,
        tags: list[str],
        created_at: int,
        expires_at: int | None,
        vector_payload: bytes,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO memories (
                id,
                root_id,
                replaces_id,
                is_current,
                namespace,
                text,
                embedding,
                tags,
                created_at,
                expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                root_id,
                replaces_id,
                1 if is_current else 0,
                namespace,
                text,
                memory_blob,
                json.dumps(tags),
                created_at,
                expires_at,
            ),
        )
        if is_current:
            self._conn.execute(
                "INSERT OR REPLACE INTO vec_index (memory_id, embedding) VALUES (?, ?)",
                (memory_id, vector_payload),
            )

    def _build_id_map_for_new(
        self,
        rows: list[dict[str, Any]],
        target_namespace: str,
    ) -> dict[str, str]:
        source_ids = sorted({str(row["id"]) for row in rows})
        id_map: dict[str, str] = {}
        used = set()

        with self._lock:
            for source_id in source_ids:
                candidate = uuid.uuid4().hex
                while (
                    candidate in used
                    or self._conn.execute(
                        "SELECT 1 FROM memories WHERE id = ? AND namespace = ?",
                        (candidate, target_namespace),
                    ).fetchone()
                    is not None
                ):
                    candidate = uuid.uuid4().hex
                id_map[source_id] = candidate
                used.add(candidate)

        return id_map

    def _normalize_current_versions_locked(self, namespace: str) -> None:
        roots = self._conn.execute(
            "SELECT DISTINCT root_id FROM memories WHERE namespace = ?",
            (namespace,),
        ).fetchall()

        for root in roots:
            root_id = root["root_id"]
            if root_id is None:
                continue

            versions = self._conn.execute(
                """
                SELECT id, embedding
                FROM memories
                WHERE namespace = ?
                  AND root_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (namespace, root_id),
            ).fetchall()
            if not versions:
                continue

            winner_id = str(versions[0]["id"])
            self._conn.execute(
                """
                UPDATE memories
                SET is_current = CASE WHEN id = ? THEN 1 ELSE 0 END
                WHERE namespace = ?
                  AND root_id = ?
                """,
                (winner_id, namespace, root_id),
            )

            losers = [str(row["id"]) for row in versions[1:]]
            for loser_id in losers:
                self._conn.execute("DELETE FROM vec_index WHERE memory_id = ?", (loser_id,))

            winner_vec = self._conn.execute(
                "SELECT 1 FROM vec_index WHERE memory_id = ?",
                (winner_id,),
            ).fetchone()
            if winner_vec is None:
                winner_embedding = _deserialize_embedding(
                    bytes(versions[0]["embedding"]),
                    expected_dimension=self.current_dimension,
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO vec_index (memory_id, embedding) VALUES (?, ?)",
                    (winner_id, self._vector_param(winner_embedding)),
                )

    def _set_meta(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO recall_meta(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def _get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM recall_meta WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])


def _import_vec_module() -> Any | None:
    for module_name in ("sqlite_vec", "sqlite_vec_sl_tmp"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def _parse_vec_dimension(sql: str) -> int | None:
    match = re.search(r"float\[(\d+)\]", sql)
    if not match:
        return None
    return int(match.group(1))


def _serialize_embedding(values: Iterable[float]) -> bytes:
    floats = [float(v) for v in values]
    if not floats:
        raise ValueError("Embedding vectors cannot be empty.")
    return struct.pack(f"<{len(floats)}f", *floats)


def _deserialize_embedding(blob: bytes, expected_dimension: int | None = None) -> list[float]:
    if len(blob) % 4 != 0:
        raise ValueError("Invalid embedding payload size.")
    dimension = len(blob) // 4
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"Embedding payload dimension {dimension} does not match expected {expected_dimension}."
        )
    return list(struct.unpack(f"<{dimension}f", blob))


def _parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Embedding dimension mismatch while scoring: {len(left)} vs {len(right)}"
        )
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def _unixepoch() -> int:
    return int(time.time())
