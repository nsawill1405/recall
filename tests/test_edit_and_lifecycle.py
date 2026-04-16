from __future__ import annotations

from pathlib import Path

import pytest

from recall import Memory
from tests.helpers import DeterministicEmbedder


def test_edit_creates_new_current_version(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        original_id = mem.store("User prefers tea", tags=["profile"], ttl_days=5)
        edited_id = mem.edit(
            id=original_id,
            text="User prefers coffee",
            tags=["profile", "preferences"],
            ttl_days=3,
        )

        assert edited_id != original_id

        listed = mem.list(limit=10)
        assert len(listed) == 1
        assert listed[0]["id"] == edited_id
        assert listed[0]["text"] == "User prefers coffee"
        assert listed[0]["tags"] == ["preferences", "profile"]
        assert listed[0]["expires_at"] is not None

        best = mem.find("coffee preference", min_score=0.0)
        assert best is not None
        assert best.id == edited_id

        results = mem.search("tea", top_k=5)
        assert all(row.id != original_id for row in results)

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
        assert rows[1]["id"] == edited_id
        assert rows[1]["is_current"] == 1
        assert rows[1]["replaces_id"] == original_id
        assert rows[0]["root_id"] == rows[1]["root_id"]
    finally:
        mem.close()


def test_edit_metadata_only_reuses_embedding(
    db_path: Path,
    embedder: DeterministicEmbedder,
) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        original_id = mem.store("Metadata edit target", tags=["old"])
        original = mem._storage.get_current_memory(original_id)
        assert original is not None

        edited_id = mem.edit(id=original_id, tags=["new"])
        updated = mem._storage.get_current_memory(edited_id)
        assert updated is not None

        assert updated["embedding"] == original["embedding"]
        assert updated["tags"] == ["new"]
    finally:
        mem.close()


def test_edit_validation_errors(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        memory_id = mem.store("keep me", tags=["keep"])

        with pytest.raises(ValueError, match="at least one change"):
            mem.edit(memory_id)

        with pytest.raises(ValueError, match="text must be non-empty"):
            mem.edit(memory_id, text="   ")

        with pytest.raises(ValueError, match="memory id not found"):
            mem.edit("missing-id", text="updated")

        with pytest.raises(ValueError, match="no changes"):
            mem.edit(memory_id, text="keep me")

        with pytest.raises(ValueError, match="ttl_days must be zero or greater"):
            mem.edit(memory_id, ttl_days=-1)
    finally:
        mem.close()


def test_prune_expired_removes_full_lineage(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        keep_id = mem.store("keep this", tags=["keep"])
        expiring_id = mem.store("expire this", tags=["temp"], ttl_days=2)
        edited_id = mem.edit(id=expiring_id, text="expire this soon", tags=["temp"])

        with mem._storage._lock, mem._storage._conn:
            mem._storage._conn.execute(
                "UPDATE memories SET expires_at = 1 WHERE id = ?",
                (edited_id,),
            )

        deleted = mem.prune_expired()
        assert deleted == 2

        listed = mem.list(limit=10)
        assert len(listed) == 1
        assert listed[0]["id"] == keep_id
    finally:
        mem.close()


def test_forget_tag_purges_lineage_after_edit(
    db_path: Path,
    embedder: DeterministicEmbedder,
) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        original_id = mem.store("private note", tags=["private"])
        mem.edit(id=original_id, text="private note v2", tags=["private"])
        mem.store("public note", tags=["public"])

        deleted = mem.forget(tag="private")
        assert deleted == 2

        listed = mem.list(limit=10)
        assert len(listed) == 1
        assert listed[0]["tags"] == ["public"]
    finally:
        mem.close()
