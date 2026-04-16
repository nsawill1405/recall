from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import DeterministicEmbedder


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "recall.db"


@pytest.fixture
def embedder() -> DeterministicEmbedder:
    return DeterministicEmbedder(dimension=8)
