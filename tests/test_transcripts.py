import tempfile
import unittest
from pathlib import Path

from youtube_rag.transcripts import extract_video_id, get_youtube_transcript


class ExtractVideoIdTests(unittest.TestCase):
    def test_extracts_ids_from_supported_url_forms(self):
        video_id = "Ub4ZmBuSA7Q"
        urls = [
            f"https://www.youtube.com/watch?v={video_id}",
            f"https://youtu.be/{video_id}?feature=shared",
            f"https://youtube.com/embed/{video_id}",
            f"https://m.youtube.com/shorts/{video_id}",
            f"https://youtube.com/live/{video_id}",
        ]

        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), video_id)

    def test_rejects_non_youtube_and_malformed_urls(self):
        for url in ("https://example.com/watch?v=Ub4ZmBuSA7Q", "not a URL"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    extract_video_id(url)

    def test_uses_cached_transcript_without_api_credentials(self):
        video_id = "Ub4ZmBuSA7Q"
        with tempfile.TemporaryDirectory() as temporary_directory:
            transcript_file = Path(temporary_directory) / f"{video_id}.txt"
            transcript_file.write_text("Cached transcript", encoding="utf-8")

            transcript = get_youtube_transcript(
                f"https://youtu.be/{video_id}",
                transcript_dir=Path(temporary_directory),
            )

        self.assertEqual(transcript, "Cached transcript")


if __name__ == "__main__":
    unittest.main()
