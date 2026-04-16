from __future__ import annotations

from pathlib import Path

import pytest

from recall import Memory
from tests.helpers import DeterministicEmbedder


def test_max_scan_rows_limits_fallback_scan(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(
        path=str(db_path),
        namespace="default",
        embedder=embedder,
        max_scan_rows=1,
    )
    try:
        mem.store("alpha item")
        mem.store("beta item")
        mem.store("gamma item")

        # Force fallback path so max_scan_rows is applied even when sqlite-vec is available.
        mem._storage._vector_backend = "sqlite"

        results = mem.search("item", top_k=5)
        assert len(results) <= 1
    finally:
        mem.close()


def test_max_scan_rows_validation(db_path: Path, embedder: DeterministicEmbedder) -> None:
    with pytest.raises(ValueError, match="max_scan_rows must be greater than zero"):
        Memory(path=str(db_path), namespace="default", embedder=embedder, max_scan_rows=0)
