"""
Ingestion graph: LOAD -> CLEAN/PREPROCESS -> CHUNK -> EMBED -> QDRANT UPSERT.

Ingestion is mostly deterministic, so this graph is deliberately simple --
LangGraph is used here for a consistent, inspectable pipeline shape, not
because ingestion needs conditional branching the way the chat graph does.
"""

from __future__ import annotations

import logging
import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from config.settings import Settings, get_settings
from ingestion.chunking.chunker import chunk_documents
from ingestion.chunking.semantic_chunker import chunk_documents_semantic
from ingestion.embeddings.dense_embedder import DenseEmbedder
from ingestion.embeddings.sparse_embedder import SparseEmbedder
from ingestion.loaders.csv_loader import load_csv_from_dir
from ingestion.loaders.docx_loader import load_docx_from_dir
from ingestion.loaders.markdown_loader import load_markdown_from_dir
from ingestion.loaders.pdf_loader import load_pdfs_from_dir
from ingestion.loaders.txt_loader import load_txt_from_dir
from ingestion.loaders.web_loader import crawl_website
from ingestion.processors.preprocessor import preprocess_documents
from ingestion.schema import Chunk, EmbeddedChunk, RawDocument
from ingestion.vector_store.qdrant_store import initialize_collection, upsert_embedded_chunks

logger = logging.getLogger(__name__)


class IngestionState(TypedDict, total=False):
    data_dir: str
    include_website: bool
    max_website_pages: int
    raw_documents: list[RawDocument]
    processed_documents: list[RawDocument]
    chunks: list[Chunk]
    embedded_chunks: list[EmbeddedChunk]
    upserted_count: int


def load_node(state: IngestionState) -> dict:
    data_dir = state["data_dir"]
    settings = get_settings()

    documents: list[RawDocument] = []
    documents.extend(load_pdfs_from_dir(os.path.join(data_dir, "pdfs")))
    documents.extend(load_docx_from_dir(os.path.join(data_dir, "documents")))
    documents.extend(load_txt_from_dir(os.path.join(data_dir, "documents")))
    documents.extend(load_markdown_from_dir(os.path.join(data_dir, "documents")))
    documents.extend(load_csv_from_dir(os.path.join(data_dir, "documents")))

    if state.get("include_website"):
        max_pages = state.get("max_website_pages", 50)
        documents.extend(
            crawl_website(settings.website_url, settings.website_domain, max_pages=max_pages)
        )

    logger.info("Loaded %d raw documents", len(documents))
    return {"raw_documents": documents}


def preprocess_node(state: IngestionState) -> dict:
    processed = preprocess_documents(state.get("raw_documents", []))
    logger.info("Preprocessed down to %d usable documents", len(processed))
    return {"processed_documents": processed}


def chunk_node(state: IngestionState) -> dict:
    settings = get_settings()
    documents = state.get("processed_documents", [])

    if settings.use_semantic_chunking:
        chunks = chunk_documents_semantic(documents, embedding_model_name=settings.dense_embedding_model)
    else:
        chunks = chunk_documents(documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    logger.info("Produced %d chunks", len(chunks))
    return {"chunks": chunks}


def embed_node(state: IngestionState) -> dict:
    settings = get_settings()
    chunks = state.get("chunks", [])
    if not chunks:
        return {"embedded_chunks": []}

    dense_embedder = DenseEmbedder(settings.dense_embedding_model)
    sparse_embedder = SparseEmbedder(settings.sparse_embedding_model)

    texts = [c.text for c in chunks]
    dense_vectors = dense_embedder.embed_documents(texts)
    sparse_vectors = sparse_embedder.embed_documents(texts)

    embedded = [
        EmbeddedChunk(
            chunk=chunk,
            dense_vector=dense_vec,
            sparse_indices=sparse_vec.indices,
            sparse_values=sparse_vec.values,
        )
        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors)
    ]
    logger.info("Embedded %d chunks (dense + sparse)", len(embedded))
    return {"embedded_chunks": embedded}


def upsert_node(state: IngestionState) -> dict:
    settings = get_settings()
    initialize_collection(settings=settings)
    count = upsert_embedded_chunks(state.get("embedded_chunks", []), settings=settings)
    return {"upserted_count": count}


def build_ingestion_graph():
    graph = StateGraph(IngestionState)
    graph.add_node("load", load_node)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed", embed_node)
    graph.add_node("upsert", upsert_node)

    graph.set_entry_point("load")
    graph.add_edge("load", "preprocess")
    graph.add_edge("preprocess", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "upsert")
    graph.add_edge("upsert", END)

    return graph.compile()


def run_ingestion(data_dir: str, include_website: bool = True, max_website_pages: int = 50) -> int:
    app = build_ingestion_graph()
    final_state = app.invoke(
        {"data_dir": data_dir, "include_website": include_website, "max_website_pages": max_website_pages}
    )
    return final_state.get("upserted_count", 0)
