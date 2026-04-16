from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest

from recall import Memory
from tests.helpers import DeterministicEmbedder


def test_redact_creates_new_current_version(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        original_id = mem.store(
            "SSN 123-45-6789 belongs to user. Repeat SSN 123-45-6789.",
            tags=["pii"],
        )
        redacted_id = mem.redact(original_id, "123-45-6789")

        assert redacted_id != original_id

        listed = mem.list(limit=5)
        assert len(listed) == 1
        assert listed[0]["id"] == redacted_id
        assert "123-45-6789" not in listed[0]["text"]

        with mem._storage._lock:
            rows = mem._storage._conn.execute(
                """
                SELECT id, root_id, replaces_id, is_current
                FROM memories
                WHERE namespace = ?
                ORDER BY created_at ASC
                """,
                ("default",),
            ).fetchall()

        assert len(rows) == 2
        assert rows[0]["id"] == original_id
        assert rows[0]["is_current"] == 0
        assert rows[1]["id"] == redacted_id
        assert rows[1]["is_current"] == 1
        assert rows[1]["replaces_id"] == original_id
        assert rows[0]["root_id"] == rows[1]["root_id"]
    finally:
        mem.close()


def test_redact_errors(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        memory_id = mem.store("token token token")

        with pytest.raises(ValueError, match="remove must be non-empty"):
            mem.redact(memory_id, "")

        with pytest.raises(ValueError, match="memory id not found"):
            mem.redact("missing", "token")

        with pytest.raises(ValueError, match="remove text not found"):
            mem.redact(memory_id, "missing")

        single = mem.store("secret")
        with pytest.raises(ValueError, match="redaction would leave empty"):
            mem.redact(single, "secret")
    finally:
        mem.close()


def test_forget_purges_full_lineage_by_id(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        original_id = mem.store("remove me but keep this", tags=["temp"])
        redacted_id = mem.redact(original_id, "remove me")

        deleted = mem.forget(id=redacted_id)
        assert deleted == 2

        assert mem.list(limit=10) == []
        with mem._storage._lock:
            remaining = mem._storage._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE namespace = ?",
                ("default",),
            ).fetchone()[0]
        assert remaining == 0
    finally:
        mem.close()


def test_forget_purges_full_lineage_by_tag(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        first_id = mem.store("alpha private", tags=["private"])
        mem.redact(first_id, "private")
        mem.store("public note", tags=["public"])

        deleted = mem.forget(tag="private")
        assert deleted == 2

        listed = mem.list(limit=10)
        assert len(listed) == 1
        assert listed[0]["tags"] == ["public"]
    finally:
        mem.close()


def test_migration_backfills_versioning_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    embedder = DeterministicEmbedder(dimension=8)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL DEFAULT 'default',
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at INTEGER NOT NULL,
                expires_at INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO memories (id, namespace, text, embedding, tags, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-id",
                "default",
                "legacy text",
                sqlite3.Binary(struct.pack("<8f", *embedder.embed("legacy text"))),
                '["legacy"]',
                1,
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        with mem._storage._lock:
            row = mem._storage._conn.execute(
                """
                SELECT root_id, is_current, replaces_id
                FROM memories
                WHERE id = 'legacy-id'
                """
            ).fetchone()

        assert row is not None
        assert row["root_id"] == "legacy-id"
        assert row["is_current"] == 1
        assert row["replaces_id"] is None
    finally:
        mem.close()
