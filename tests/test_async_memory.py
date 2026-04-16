from __future__ import annotations

import asyncio
from pathlib import Path

from recall import AsyncMemory, StoreItem
from tests.helpers import DeterministicEmbedder


def test_async_memory_parity(db_path: Path, embedder: DeterministicEmbedder) -> None:
    async def scenario() -> None:
        export_path = db_path.parent / "async-export.jsonl"
        imported_db = db_path.parent / "async-imported.db"

        mem = AsyncMemory(path=str(db_path), namespace="async", embedder=embedder)
        try:
            ids = await mem.store_many(
                [
                    StoreItem(text="async memory", tags=["async"]),
                    StoreItem(text="another async memory", tags=["async"]),
                ]
            )
            edited_id = await mem.edit(
                ids[0],
                text="async memory updated",
                tags=["async", "edited"],
            )
            redacted_id = await mem.redact(edited_id, "updated")

            results = await mem.search("async", top_k=3)
            assert len(results) >= 1
            assert results[0].id == redacted_id

            exported = await mem.export_jsonl(str(export_path), include_history=True)
            assert exported == 4
        finally:
            await mem.aclose()

        imported = AsyncMemory(path=str(imported_db), namespace="async", embedder=embedder)
        try:
            imported_count = await imported.import_jsonl(str(export_path), conflict="new")
            assert imported_count == 4

            with imported._memory._storage._lock, imported._memory._storage._conn:
                imported._memory._storage._conn.execute(
                    "UPDATE memories SET expires_at = 1 WHERE namespace = ? AND is_current = 1",
                    ("async",),
                )

            pruned = await imported.prune_expired()
            assert pruned == 4
            assert await imported.search("async", top_k=3) == []
        finally:
            await imported.aclose()

    asyncio.run(scenario())
