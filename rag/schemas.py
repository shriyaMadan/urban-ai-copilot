"""Shared schemas for RAG retrieval outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved guidance snippet from the vector store."""

    text: str
    source: str
    chunk_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
