"""Fetch, validate, and locally cache YouTube video transcripts."""

import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_rag.settings import TRANSCRIPTS_DIR, TRANSCRIPT_JOB_TIMEOUT_SECONDS

LOGGER = logging.getLogger(__name__)
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}


def extract_video_id(video_url: str) -> str:
    """Return the 11-character video ID from a supported YouTube URL."""
    parsed_url = urlparse(video_url)
    hostname = (parsed_url.hostname or "").lower()

    if hostname not in YOUTUBE_HOSTS:
        raise ValueError(f"Unsupported YouTube URL: {video_url}")

    if hostname.endswith("youtu.be"):
        video_id = parsed_url.path.strip("/").split("/", maxsplit=1)[0]
    elif parsed_url.path == "/watch":
        video_id = parse_qs(parsed_url.query).get("v", [""])[0]
    else:
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live"}:
            video_id = path_parts[1]
        else:
            video_id = ""

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(f"Could not find a valid video ID in URL: {video_url}")

    return video_id


def get_youtube_transcript(
    video_url: str,
    transcript_dir: Path = TRANSCRIPTS_DIR,
) -> str:
    """Return a cached transcript or fetch and cache it with Supadata."""
    video_id = extract_video_id(video_url)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_file = transcript_dir / f"{video_id}.txt"

    if transcript_file.exists():
        transcript = transcript_file.read_text(encoding="utf-8").strip()
        if transcript:
            LOGGER.info("Using cached transcript for video %s.", video_id)
            return transcript

    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("SUPADATA_API_KEY")
    if not api_key:
        raise ValueError(
            "SUPADATA_API_KEY is required to fetch a transcript. "
            "Add it to your .env file or environment."
        )

    from supadata import Supadata, SupadataError

    LOGGER.info("Fetching transcript for video %s from Supadata.", video_id)
    client = Supadata(api_key=api_key)

    try:
        result = client.transcript(
            url=video_url,
            lang="en",
            text=True,
            mode="auto",
        )

        if hasattr(result, "content"):
            transcript = result.content
        else:
            LOGGER.info("Transcript job %s started.", result.job_id)
            deadline = time.monotonic() + TRANSCRIPT_JOB_TIMEOUT_SECONDS
            while True:
                job = client.transcript.get_job_status(result.job_id)
                if job.status == "completed":
                    transcript = job.result.content
                    break
                if job.status == "failed":
                    raise RuntimeError(f"Supadata transcription failed: {job}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Supadata transcript job did not finish within "
                        f"{TRANSCRIPT_JOB_TIMEOUT_SECONDS} seconds."
                    )

                LOGGER.info("Transcript job status: %s.", job.status)
                time.sleep(2)
    except SupadataError as error:
        raise RuntimeError(f"Supadata API request failed: {error}") from error

    if not transcript or not transcript.strip():
        raise ValueError("Supadata returned an empty transcript.")

    transcript_file.write_text(transcript, encoding="utf-8")
    LOGGER.info("Transcript cached at %s.", transcript_file)
    return transcript
