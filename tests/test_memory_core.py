from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from recall import Memory
from tests.helpers import DeterministicEmbedder


def test_store_search_forget_roundtrip(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        first_id = mem.store("User prefers short answers", tags=["pref"])
        second_id = mem.store("Project stack is FastAPI", tags=["project"])

        assert first_id
        assert second_id
        assert first_id != second_id

        results = mem.search("what does the user prefer?", top_k=5)
        assert len(results) >= 1
        assert any("prefers short" in r.text for r in results)

        top = results[0]
        assert isinstance(top.created_at, datetime)
        assert top.created_at.tzinfo == timezone.utc
        assert 0.0 <= top.score <= 1.0

        deleted = mem.forget(id=first_id)
        assert deleted == 1

        after_delete = mem.search("what does the user prefer?", top_k=5)
        assert all(r.id != first_id for r in after_delete)
    finally:
        mem.close()


def test_forget_by_tag(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        mem.store("one", tags=["temp"])
        mem.store("two", tags=["temp"])
        mem.store("three", tags=["keep"])

        deleted = mem.forget(tag="temp")
        assert deleted == 2

        listed = mem.list(limit=10)
        assert len(listed) == 1
        assert listed[0]["text"] == "three"
    finally:
        mem.close()


def test_stats_include_backend_and_embedder_info(
    db_path: Path,
    embedder: DeterministicEmbedder,
) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        mem.store("a")
        stats = mem.stats()
        assert stats["total"] == 1
        assert stats["active"] == 1
        assert stats["expired"] == 0
        assert stats["embedder"]["dimension"] == 8
        assert stats["vector_backend"] in {"sqlite", "sqlite-vec"}
    finally:
        mem.close()
