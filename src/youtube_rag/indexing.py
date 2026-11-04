"""Split transcripts into chunks and keep their Chroma vectors up to date."""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING

from youtube_rag.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
)
from youtube_rag.transcripts import get_youtube_transcript

if TYPE_CHECKING:
    from langchain_chroma import Chroma

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Create the persistent vector store and its embedding model on demand."""
    from dotenv import load_dotenv
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is required for embeddings. "
            "Add it to your .env file or environment."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )


def index_transcript(source_url: str) -> None:
    """Fetch a transcript and add only changed chunks to the local index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    transcript = get_youtube_transcript(source_url)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.create_documents([transcript])
    if not chunks:
        raise ValueError("Transcript could not be split into indexable chunks.")

    source_id = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
    chunk_ids = [f"{source_id}-{index}" for index in range(len(chunks))]
    vector_store = get_vector_store()

    existing = vector_store.get(
        where={"source_url": source_url},
        include=["metadatas"],
    )
    existing_metadata = existing["metadatas"] or []
    existing_by_id = dict(zip(existing["ids"], existing_metadata))
    legacy_ids = set()

    # Remove matching pre-metadata chunks left by older versions of the project.
    if not existing_by_id:
        legacy = vector_store.get(include=["documents", "metadatas"])
        chunk_contents = {chunk.page_content for chunk in chunks}
        legacy_ids = {
            chunk_id
            for chunk_id, document, metadata in zip(
                legacy["ids"],
                legacy["documents"] or [],
                legacy["metadatas"] or [],
            )
            if document in chunk_contents and not (metadata or {}).get("source_url")
        }

    changed_chunks = []
    changed_ids = []
    for chunk_id, chunk in zip(chunk_ids, chunks):
        content_hash = hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()
        chunk.metadata.update(
            {
                "source_url": source_url,
                "content_hash": content_hash,
            }
        )
        if existing_by_id.get(chunk_id, {}).get("content_hash") != content_hash:
            changed_chunks.append(chunk)
            changed_ids.append(chunk_id)

    if changed_chunks:
        vector_store.add_documents(documents=changed_chunks, ids=changed_ids)

    stale_ids = set(existing["ids"]) - set(chunk_ids)
    stale_ids.update(legacy_ids)
    if stale_ids:
        vector_store.delete(ids=list(stale_ids))

    LOGGER.info(
        "Indexed %d changed chunk(s); skipped %d unchanged chunk(s).",
        len(changed_chunks),
        len(chunks) - len(changed_chunks),
    )
