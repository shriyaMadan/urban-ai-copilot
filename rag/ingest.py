"""Build a local Qdrant index from txt knowledge files."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag.chunker import chunk_documents
from rag.embeddings import OpenAICompatibleEmbedder
from rag.loader import load_txt_documents
from rag.vector_store import QdrantVectorStore
from utils.config import AppConfig, read_config_value


def _resolve_embedding_config(config: AppConfig) -> tuple[str, str, str]:
    """Resolve embedding settings from env/Streamlit secrets consistently."""
    embedding_model = (config.embedding_model or "").strip()
    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL is required for RAG ingestion.")

    openai_api_key = (config.openai_api_key or "").strip()
    openai_base_url = (read_config_value("OPENAI_BASE_URL", "") or "").strip()
    if openai_api_key and openai_base_url:
        return openai_api_key, openai_base_url, embedding_model

    gemini_api_key = (config.gemini_api_key or "").strip()
    if gemini_api_key:
        gemini_base_url = (
            read_config_value(
                "GEMINI_OPENAI_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta/openai",
            )
            or "https://generativelanguage.googleapis.com/v1beta/openai"
        ).strip()
        return gemini_api_key, gemini_base_url, embedding_model

    raise ValueError(
        "No supported embedding credentials found. Set OPENAI_API_KEY/OPENAI_BASE_URL "
        "or GEMINI_API_KEY/GEMINI_OPENAI_BASE_URL."
    )


def ensure_rag_index(force_recreate: bool = False) -> dict[str, object]:
    """Ensure a local Qdrant index exists, creating it when needed."""
    config = AppConfig.from_env()

    vector_store = QdrantVectorStore(
        qdrant_path=config.rag_qdrant_path,
        collection_name=config.rag_collection,
    )
    existing_points = vector_store.count_points()
    if not force_recreate and existing_points > 0:
        return {
            "status": "ready",
            "skipped": True,
            "chunks_indexed": existing_points,
            "collection": config.rag_collection,
            "path": config.rag_qdrant_path,
        }

    documents = load_txt_documents(config.rag_knowledge_dir)
    if not documents:
        raise ValueError(
            f"No .txt knowledge files found in '{config.rag_knowledge_dir}'."
        )

    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("No chunks were produced from knowledge documents.")

    api_key, base_url, model = _resolve_embedding_config(config)
    embedder = OpenAICompatibleEmbedder(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    if not vectors or not vectors[0]:
        raise ValueError("Embedding provider returned no vectors.")

    vector_store.recreate_collection(vector_size=len(vectors[0]))
    vector_store.upsert_chunks(chunks=chunks, vectors=vectors)

    return {
        "status": "created",
        "skipped": False,
        "chunks_indexed": len(chunks),
        "collection": config.rag_collection,
        "path": config.rag_qdrant_path,
    }


def run_ingestion() -> None:
    """Ingest local knowledge docs into local persistent Qdrant index."""
    result = ensure_rag_index(force_recreate=True)

    print(
        f"RAG ingestion complete. Indexed {result['chunks_indexed']} chunks into "
        f"collection '{result['collection']}' at '{result['path']}'."
    )


if __name__ == "__main__":
    run_ingestion()
