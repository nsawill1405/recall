from __future__ import annotations

from pathlib import Path

from recall import Memory
from recall.errors import DimensionMismatchError
from tests.helpers import DeterministicEmbedder


def test_dimension_mismatch_fails_fast(db_path: Path) -> None:
    first = Memory(
        path=str(db_path),
        namespace="default",
        embedder=DeterministicEmbedder(dimension=8),
    )
    try:
        first.store("alpha")
    finally:
        first.close()

    try:
        Memory(path=str(db_path), namespace="default", embedder=DeterministicEmbedder(dimension=16))
    except DimensionMismatchError as exc:
        assert "Embedding dimension mismatch" in str(exc)
    else:
        raise AssertionError("Expected DimensionMismatchError")


def test_rebuild_index_migrates_dimension(db_path: Path) -> None:
    first = Memory(
        path=str(db_path),
        namespace="default",
        embedder=DeterministicEmbedder(dimension=8),
    )
    try:
        first.store("first")
        first.store("second")
    finally:
        first.close()

    migrator = Memory(
        path=str(db_path),
        namespace="default",
        embedder=DeterministicEmbedder(dimension=16, model="mock-v2"),
        _allow_dimension_mismatch=True,
    )
    try:
        rebuilt = migrator.rebuild_index()
        assert rebuilt == 2
    finally:
        migrator.close()

    reopened = Memory(
        path=str(db_path),
        namespace="default",
        embedder=DeterministicEmbedder(dimension=16, model="mock-v2"),
    )
    try:
        results = reopened.search("first", top_k=3)
        assert len(results) >= 1
        assert reopened.stats()["embedder"]["dimension"] == 16
    finally:
        reopened.close()
