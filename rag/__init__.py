"""Lightweight RAG utilities for Urban AI Copilot."""

from rag.schemas import RetrievedChunk

try:
    from rag.retriever import RAGRetriever
except Exception:  # pragma: no cover - optional dependency guard
    RAGRetriever = None  # type: ignore[assignment]

__all__ = ["RetrievedChunk", "RAGRetriever"]
