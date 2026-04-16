from __future__ import annotations

from pathlib import Path
from typing import Any

from recall.storage import SQLiteStorage


class FakeVecModule:
    def __init__(self) -> None:
        self.loaded = False

    def load(self, _conn: Any) -> None:
        self.loaded = True

    def serialize_float32(self, values: list[float]) -> bytes:
        import struct

        floats = [float(v) for v in values]
        return struct.pack(f"<{len(floats)}f", *floats)


def test_storage_attempts_sqlite_vec_load(tmp_path: Path) -> None:
    fake = FakeVecModule()
    db_path = tmp_path / "vec.db"

    storage = SQLiteStorage(
        db_path=db_path,
        namespace="default",
        embedder_name="mock",
        embedder_model="mock-v1",
        embedder_dimension=8,
        vec_module=fake,
    )
    try:
        assert fake.loaded is True
        assert storage.vector_backend in {"sqlite", "sqlite-vec"}
    finally:
        storage.close()
