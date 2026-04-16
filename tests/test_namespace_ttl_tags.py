from __future__ import annotations

from pathlib import Path

from recall import Memory
from tests.helpers import DeterministicEmbedder


def test_namespace_isolation(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem_a = Memory(path=str(db_path), namespace="a", embedder=embedder)
    mem_b = Memory(path=str(db_path), namespace="b", embedder=embedder)

    try:
        mem_a.store("alpha only")
        mem_b.store("beta only")

        a_results = mem_a.search("alpha", top_k=5)
        b_results = mem_b.search("beta", top_k=5)

        assert any("alpha" in row.text for row in a_results)
        assert all("beta" not in row.text for row in a_results)
        assert any("beta" in row.text for row in b_results)
        assert all("alpha" not in row.text for row in b_results)
    finally:
        mem_a.close()
        mem_b.close()


def test_ttl_expiry_filtering(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        mem.store("expires soon", ttl_days=1)
        mem.store("never expires")

        with mem._storage._lock, mem._storage._conn:
            mem._storage._conn.execute(
                "UPDATE memories SET expires_at = 1 WHERE text = ?",
                ("expires soon",),
            )

        results = mem.search("expires", top_k=10)
        texts = [item.text for item in results]
        assert "expires soon" not in texts
        assert "never expires" in texts
    finally:
        mem.close()


def test_tag_filtering(db_path: Path, embedder: DeterministicEmbedder) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        mem.store("frontend preference", tags=["ui"])
        mem.store("backend preference", tags=["api"])

        ui_results = mem.search("preference", top_k=5, tags=["ui"])
        api_results = mem.search("preference", top_k=5, tags=["api"])

        assert len(ui_results) == 1
        assert ui_results[0].text == "frontend preference"
        assert len(api_results) == 1
        assert api_results[0].text == "backend preference"
    finally:
        mem.close()
