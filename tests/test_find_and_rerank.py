from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from recall import Memory


class RankedTestEmbedder:
    name = "ranked-test"
    model = "ranked-test-v1"
    dimension = 2

    def embed(self, text: str) -> list[float]:
        lower = text.lower().strip()
        if lower == "banana preference":
            return [1.0, 0.0]  # query vector
        if "banana" in lower:
            return [0.6, 0.8]
        return [0.75, 0.66]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def test_find_returns_one_or_none(db_path: Path) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=RankedTestEmbedder())
    try:
        good_id = mem.store("banana preference saved", tags=["pref"])
        mem.store("unrelated fallback text", tags=["misc"])

        best = mem.find("banana preference", tags=["pref"], min_score=0.1)
        assert best is not None
        assert best.id == good_id

        missing = mem.find("banana preference", tags=["unknown"], min_score=0.0)
        assert missing is None

        filtered = mem.find("banana preference", tags=["pref"], min_score=0.95)
        assert filtered is None
    finally:
        mem.close()


def test_local_reranker_reorders_results(db_path: Path) -> None:
    mem = Memory(path=str(db_path), namespace="default", embedder=RankedTestEmbedder())
    try:
        mem.store("totally unrelated sentence")
        mem.store("banana preference saved")

        plain = mem.search("banana preference", top_k=2)
        assert plain[0].text == "totally unrelated sentence"

        reranked = mem.search("banana preference", top_k=2, reranker="local")
        assert reranked[0].text == "banana preference saved"

        best = mem.find("banana preference", reranker="local")
        assert best is not None
        assert best.text == "banana preference saved"
    finally:
        mem.close()
