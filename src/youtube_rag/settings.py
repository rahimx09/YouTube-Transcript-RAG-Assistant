"""Shared model settings and project-local data paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
VECTOR_DB_DIR = PROJECT_ROOT / "chroma.db"

COLLECTION_NAME = "RAG_Collection"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 10
RETRIEVAL_COUNT = 4
TRANSCRIPT_JOB_TIMEOUT_SECONDS = 300
