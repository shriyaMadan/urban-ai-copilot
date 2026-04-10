"""Simple text chunking utilities for RAG ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from rag.loader import KnowledgeDocument


@dataclass(frozen=True)
class TextChunk:
    """Chunked knowledge text with metadata for indexing."""

    chunk_id: str
    source: str
    text: str
    metadata: Dict[str, str]


def chunk_documents(
    documents: List[KnowledgeDocument],
    chunk_size_words: int = 420,
    overlap_words: int = 60,
) -> List[TextChunk]:
    """Split knowledge documents into overlapping word chunks."""
    chunks: List[TextChunk] = []

    for document in documents:
        words = document.text.split()
        if not words:
            continue

        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document.source}::chunk_{chunk_index}",
                        source=document.source,
                        text=chunk_text,
                        metadata={
                            "source": document.source,
                            "chunk_index": str(chunk_index),
                        },
                    )
                )

            if end >= len(words):
                break

            next_start = end - overlap_words
            if next_start <= start:
                next_start = end

            start = next_start
            chunk_index += 1

    return chunks
