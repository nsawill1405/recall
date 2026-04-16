from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class MemoryResult:
    id: str
    text: str
    score: float
    tags: list[str]
    created_at: datetime
    expires_at: datetime | None
