from __future__ import annotations

import asyncio
from pathlib import Path

from recall import AsyncMemory
from tests.helpers import DeterministicEmbedder


def test_async_memory_parity(db_path: Path, embedder: DeterministicEmbedder) -> None:
    async def scenario() -> None:
        mem = AsyncMemory(path=str(db_path), namespace="async", embedder=embedder)
        try:
            memory_id = await mem.store("async memory", tags=["async"])
            results = await mem.search("async", top_k=3)
            assert len(results) >= 1
            assert results[0].id == memory_id

            deleted = await mem.forget(id=memory_id)
            assert deleted == 1
            assert await mem.search("async", top_k=3) == []
        finally:
            await mem.aclose()

    asyncio.run(scenario())
