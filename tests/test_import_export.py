from __future__ import annotations

import json
from pathlib import Path

from recall import Memory
from tests.helpers import DeterministicEmbedder


def test_export_import_round_trip_current_only(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    target_db = tmp_path / "target.db"
    export_path = tmp_path / "memories.jsonl"
    embedder = DeterministicEmbedder(dimension=8)

    source = Memory(path=str(source_db), namespace="default", embedder=embedder)
    try:
        first_id = source.store("first note", tags=["a"])
        source.edit(first_id, text="first note updated", tags=["a", "b"])
        source.store("second note", tags=["b"])

        exported = source.export_jsonl(str(export_path))
        assert exported == 2
    finally:
        source.close()

    target = Memory(path=str(target_db), namespace="default", embedder=embedder)
    try:
        imported = target.import_jsonl(str(export_path))
        assert imported == 2

        listed = target.list(limit=10)
        assert {row["text"] for row in listed} == {"first note updated", "second note"}

        with target._storage._lock:
            rows = target._storage._conn.execute(
                """
                SELECT id, root_id, replaces_id, is_current
                FROM memories
                WHERE namespace = ?
                ORDER BY created_at ASC
                """,
                ("default",),
            ).fetchall()

        assert len(rows) == 2
        assert all(row["is_current"] == 1 for row in rows)
        assert all(row["root_id"] == row["id"] for row in rows)
        assert all(row["replaces_id"] is None for row in rows)
    finally:
        target.close()


def test_export_import_with_history_preserves_lineage(tmp_path: Path) -> None:
    source_db = tmp_path / "history-source.db"
    target_db = tmp_path / "history-target.db"
    export_path = tmp_path / "history.jsonl"
    embedder = DeterministicEmbedder(dimension=8)

    source = Memory(path=str(source_db), namespace="default", embedder=embedder)
    try:
        original = source.store("lineage secret value", tags=["private"])
        edited = source.edit(original, text="lineage secret value v2")
        source.redact(edited, "secret ")

        exported = source.export_jsonl(str(export_path), include_history=True)
        assert exported == 3
    finally:
        source.close()

    target = Memory(path=str(target_db), namespace="default", embedder=embedder)
    try:
        imported = target.import_jsonl(str(export_path), conflict="new")
        assert imported == 3

        with target._storage._lock:
            rows = target._storage._conn.execute(
                """
                SELECT id, root_id, replaces_id, is_current
                FROM memories
                WHERE namespace = ?
                ORDER BY created_at ASC, id ASC
                """,
                ("default",),
            ).fetchall()

        assert len(rows) == 3
        root_ids = {str(row["root_id"]) for row in rows}
        assert len(root_ids) == 1
        assert sum(int(row["is_current"]) for row in rows) == 1

        replaces = [row["replaces_id"] for row in rows if row["replaces_id"] is not None]
        assert len(replaces) == 2
    finally:
        target.close()


def test_import_conflict_modes(tmp_path: Path) -> None:
    db_path = tmp_path / "conflict.db"
    export_path = tmp_path / "conflict.jsonl"
    embedder = DeterministicEmbedder(dimension=8)

    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        mem.store("conflict row", tags=["conflict"])
        assert mem.export_jsonl(str(export_path)) == 1

        assert mem.import_jsonl(str(export_path), conflict="skip") == 0
        assert mem.stats()["total"] == 1

        assert mem.import_jsonl(str(export_path), conflict="overwrite") == 1
        assert mem.stats()["total"] == 1

        assert mem.import_jsonl(str(export_path), conflict="new") == 1
        assert mem.stats()["total"] == 2
    finally:
        mem.close()


def test_import_rejects_invalid_json(tmp_path: Path) -> None:
    db_path = tmp_path / "invalid.db"
    jsonl_path = tmp_path / "invalid.jsonl"
    embedder = DeterministicEmbedder(dimension=8)

    jsonl_path.write_text(json.dumps({"id": "ok", "text": "ok", "created_at": 1}) + "\nnot-json\n")

    mem = Memory(path=str(db_path), namespace="default", embedder=embedder)
    try:
        try:
            mem.import_jsonl(str(jsonl_path))
        except ValueError as exc:
            assert "Invalid JSON on line 2" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid JSON line")
    finally:
        mem.close()
