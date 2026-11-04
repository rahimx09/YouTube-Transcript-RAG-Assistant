# YouTube Transcript RAG Assistant

Ask questions about a YouTube video and get answers grounded in its transcript. The app fetches and caches a transcript, splits it into searchable chunks, stores embeddings in a local Chroma database, and uses Gemini to answer questions from the retrieved context.

## What it does

1. Accepts a YouTube video URL and a question.
2. Reuses a local transcript when available, or downloads one through Supadata.
3. Splits the transcript and updates its vectors in a persistent Chroma collection.
4. Retrieves relevant chunks and asks Gemini to answer using that context.

## Requirements

- Python 3.10 or newer
- A [Supadata](https://supadata.ai/) API key to fetch uncached transcripts
- A Gemini API key to create embeddings and generate answers

## Setup

From the project directory, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project and its development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Create a local environment file and add your API keys:

```powershell
Copy-Item .env.example .env
```

Set `SUPADATA_API_KEY` and `GEMINI_API_KEY` in `.env`. The file is ignored by Git; do not commit API keys.

## Run

Pass the video URL and question on the command line:

```powershell
youtube-rag --url "https://www.youtube.com/watch?v=VIDEO_ID" --question "What is the video about?"
```

You can also run the package directly:

```powershell
python -m youtube_rag --url "https://www.youtube.com/watch?v=VIDEO_ID" --question "What is the video about?"
```

For repeated use, set `YOUTUBE_VIDEO_URL` and `RAG_QUESTION` in `.env`, then run `youtube-rag` without flags. Command-line values take precedence.

The program prints progress as it fetches or reuses the transcript, indexes changed chunks, retrieves context, and generates the answer. If the context does not contain the answer, the assistant is instructed to say it does not know.

## Project layout

```text
src/youtube_rag/
  cli.py          Command-line entry point
  indexing.py     Transcript chunking and Chroma indexing
  rag.py          Context retrieval and Gemini question answering
  settings.py     Models and local data paths
  transcripts.py  YouTube URL parsing and transcript caching
tests/
  test_transcripts.py
```

## Local data and privacy

The transcript cache (`transcripts/`) and Chroma database (`chroma.db/`) are created locally and excluded from Git. To start fresh, stop the app and remove those specific directories. Transcript availability depends on the video and the Supadata service.

## Tests

```powershell
python -m pytest
```
