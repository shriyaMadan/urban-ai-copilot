"""Load local knowledge documents for RAG ingestion (txt only)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from utils.config import resolve_project_path


@dataclass(frozen=True)
class KnowledgeDocument:
    """A raw knowledge document loaded from disk."""

    source: str
    text: str


def load_txt_documents(knowledge_dir: str) -> List[KnowledgeDocument]:
    """Load `.txt` knowledge documents from a directory."""
    base_path = Path(resolve_project_path(knowledge_dir))

    if not base_path.exists() or not base_path.is_dir():
        return []

    documents: List[KnowledgeDocument] = []
    for file_path in sorted(base_path.glob("*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue

        if not text:
            continue

        documents.append(KnowledgeDocument(source=file_path.name, text=text))

    return documents
