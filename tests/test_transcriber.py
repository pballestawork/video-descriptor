import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf

from video_descriptor.transcriber import (
    TranscriptionConfig,
    allow_trusted_whisperx_checkpoints,
    extract_audio,
    resolve_compute_type,
    run_transcription,
)


class TranscriberTests(TestCase):
    def test_missing_input_video_fails(self) -> None:
        with TemporaryDirectory() as directory:
            config = TranscriptionConfig(
                input_video=Path(directory) / "missing.mp4",
                output_dir=Path(directory) / "outputs",
            )
            with self.assertRaises(FileNotFoundError):
                run_transcription(config)

    def test_cpu_compute_type_is_int8(self) -> None:
        self.assertEqual(resolve_compute_type("float16", "cpu"), "int8")
        self.assertEqual(resolve_compute_type("float16", "cuda"), "float16")

    def test_torch_load_patch_forces_weights_only_false(self) -> None:
        calls = []

        class TorchStub:
            @staticmethod
            def load(*args, **kwargs):
                calls.append(kwargs)
                return "ok"

        allow_trusted_whisperx_checkpoints(TorchStub)

        self.assertEqual(TorchStub.load("checkpoint.ckpt"), "ok")
        self.assertEqual(calls[-1]["weights_only"], False)

        TorchStub.load("checkpoint.ckpt", weights_only=True)
        self.assertEqual(calls[-1]["weights_only"], False)

    @skipIf(shutil.which("ffmpeg") is None, "ffmpeg is required")
    def test_extract_audio_from_synthetic_video(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "sample.mp4"
            audio_path = root / "sample.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=128x128:rate=1:duration=1",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=1",
                    "-shortest",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            extract_audio(video_path, audio_path)

            self.assertTrue(audio_path.exists())
            self.assertGreater(audio_path.stat().st_size, 0)
