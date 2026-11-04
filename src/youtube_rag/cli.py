"""Command-line interface for the YouTube transcript RAG assistant."""

import argparse
import logging
import os
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser, using optional .env defaults."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ask a question grounded in a YouTube video's transcript."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("YOUTUBE_VIDEO_URL"),
        help="YouTube video URL (or set YOUTUBE_VIDEO_URL in .env).",
    )
    parser.add_argument(
        "--question",
        default=os.getenv("RAG_QUESTION"),
        help="Question to ask (or set RAG_QUESTION in .env).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run transcript indexing followed by grounded question answering."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("provide --url or set YOUTUBE_VIDEO_URL in .env")
    if not args.question:
        parser.error("provide --question or set RAG_QUESTION in .env")

    from youtube_rag.indexing import index_transcript
    from youtube_rag.rag import answer_question

    logging.info("Preparing transcript index.")
    index_transcript(args.url)
    logging.info("Question: %s", args.question)
    answer = answer_question(args.question)
    print("\nAnswer:\n")
    print(answer)
    return 0
