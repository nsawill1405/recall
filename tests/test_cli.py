from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from recall import Memory


def _run_cli(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "recall.cli", *args],
        cwd=os.path.dirname(__file__) + "/..",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_search_list_forget_stats_rebuild(db_path: Path) -> None:
    imported_db = db_path.parent / "imported.db"
    export_path = db_path.parent / "memories.jsonl"

    mem = Memory(
        path=str(db_path),
        namespace="default",
        embedder="local",
        model="hash-embedding-v1",
    )
    try:
        memory_id = mem.store("CLI seeded memory", tags=["cli"])
        assert memory_id
    finally:
        mem.close()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    env.pop("OPENAI_API_KEY", None)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("COHERE_API_KEY", None)

    base_args = [
        "--path",
        str(db_path),
        "--namespace",
        "default",
        "--embedder",
        "local",
        "--model",
        "hash-embedding-v1",
    ]

    list_run = _run_cli(*base_args, "list", "--limit", "5", env=env)
    assert list_run.returncode == 0
    assert "CLI seeded memory" in list_run.stdout

    search_run = _run_cli(*base_args, "search", "seeded", "--top-k", "3", env=env)
    assert search_run.returncode == 0
    assert "CLI seeded memory" in search_run.stdout

    find_run = _run_cli(*base_args, "find", "seeded", "--min-score", "0.0", env=env)
    assert find_run.returncode == 0
    assert "CLI seeded memory" in find_run.stdout

    redact_run = _run_cli(
        *base_args,
        "redact",
        "--id",
        memory_id,
        "--remove",
        "seeded",
        env=env,
    )
    assert redact_run.returncode == 0
    assert "removed 1 occurrence" in redact_run.stdout
    redacted_match = re.search(r"Created redacted memory ([0-9a-f]+)", redact_run.stdout)
    assert redacted_match is not None
    redacted_id = redacted_match.group(1)

    redact_fail_run = _run_cli(
        *base_args,
        "redact",
        "--id",
        memory_id,
        "--remove",
        "seeded",
        env=env,
    )
    assert redact_fail_run.returncode == 1
    assert "memory id not found in current namespace" in redact_fail_run.stderr

    edit_run = _run_cli(
        *base_args,
        "edit",
        "--id",
        redacted_id,
        "--text",
        "CLI edited memory",
        "--tag",
        "cli",
        "--ttl-days",
        "7",
        env=env,
    )
    assert edit_run.returncode == 0
    assert "Created edited memory" in edit_run.stdout
    edited_match = re.search(r"Created edited memory ([0-9a-f]+)", edit_run.stdout)
    assert edited_match is not None
    edited_id = edited_match.group(1)

    bad_edit_run = _run_cli(
        *base_args,
        "edit",
        "--id",
        edited_id,
        "--tag",
        "cli",
        "--clear-tags",
        env=env,
    )
    assert bad_edit_run.returncode == 1
    assert "cannot use --tag with --clear-tags" in bad_edit_run.stderr

    no_change_edit_run = _run_cli(
        *base_args,
        "edit",
        "--id",
        edited_id,
        env=env,
    )
    assert no_change_edit_run.returncode == 1
    assert "edit requires at least one change" in no_change_edit_run.stderr

    list_after_edit = _run_cli(*base_args, "list", "--limit", "5", env=env)
    assert list_after_edit.returncode == 0
    assert "CLI edited memory" in list_after_edit.stdout

    prune_run = _run_cli(*base_args, "prune", env=env)
    assert prune_run.returncode == 0
    assert "Pruned 0 memories." in prune_run.stdout

    export_run = _run_cli(*base_args, "export", "--out", str(export_path), env=env)
    assert export_run.returncode == 0
    assert "Exported 1 memories" in export_run.stdout

    imported_base_args = [
        "--path",
        str(imported_db),
        "--namespace",
        "default",
        "--embedder",
        "local",
        "--model",
        "hash-embedding-v1",
    ]
    import_run = _run_cli(
        *imported_base_args,
        "import",
        "--in",
        str(export_path),
        "--conflict",
        "skip",
        env=env,
    )
    assert import_run.returncode == 0
    assert "Imported 1 memories" in import_run.stdout

    imported_list_run = _run_cli(*imported_base_args, "list", "--limit", "5", env=env)
    assert imported_list_run.returncode == 0
    assert "CLI edited memory" in imported_list_run.stdout

    imported_mem = Memory(
        path=str(imported_db),
        namespace="default",
        embedder="local",
        model="hash-embedding-v1",
    )
    try:
        with imported_mem._storage._lock, imported_mem._storage._conn:
            imported_mem._storage._conn.execute(
                "UPDATE memories SET expires_at = 1 WHERE namespace = ? AND is_current = 1",
                ("default",),
            )
    finally:
        imported_mem.close()

    imported_prune = _run_cli(*imported_base_args, "prune", env=env)
    assert imported_prune.returncode == 0
    assert "Pruned 1 memories." in imported_prune.stdout

    stats_run = _run_cli(*base_args, "stats", env=env)
    assert stats_run.returncode == 0
    assert '"total": 1' in stats_run.stdout

    forget_run = _run_cli(*base_args, "forget", "--tag", "cli", env=env)
    assert forget_run.returncode == 0
    assert "Deleted 3 memories." in forget_run.stdout

    rebuild_run = _run_cli(*base_args, "rebuild-index", env=env)
    assert rebuild_run.returncode == 0
    assert "Rebuilt embeddings" in rebuild_run.stdout
