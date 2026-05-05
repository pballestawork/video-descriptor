from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from video_descriptor.formats import seconds_to_srt_time, write_srt, write_txt, write_vtt


class FormatTests(TestCase):
    def test_seconds_to_srt_time(self) -> None:
        self.assertEqual(seconds_to_srt_time(0), "00:00:00,000")
        self.assertEqual(seconds_to_srt_time(3661.25), "01:01:01,250")

    def test_writers_create_expected_files(self) -> None:
        result = {
            "segments": [
                {"start": 0.0, "end": 1.5, "text": " Hola mundo"},
                {"start": 2.0, "end": 3.0, "text": "Segunda linea"},
            ]
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            txt_path = root / "out.txt"
            srt_path = root / "out.srt"
            vtt_path = root / "out.vtt"

            write_txt(result, txt_path)
            write_srt(result, srt_path)
            write_vtt(result, vtt_path)

            self.assertIn("Hola mundo", txt_path.read_text(encoding="utf-8"))
            self.assertIn("00:00:00,000 --> 00:00:01,500", srt_path.read_text(encoding="utf-8"))
            self.assertTrue(vtt_path.read_text(encoding="utf-8").startswith("WEBVTT"))
