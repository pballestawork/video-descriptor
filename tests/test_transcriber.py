import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf

from video_descriptor.transcriber import (
    TranscriptionConfig,
    allow_trusted_whisperx_checkpoints,
    apply_person_labels,
    extract_audio,
    hf_diarization_access_message,
    resolve_compute_type,
    run_transcription,
    speaker_count_kwargs,
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

    def test_speaker_count_kwargs_exact_count(self) -> None:
        config = TranscriptionConfig(
            input_video=Path("video.mp4"),
            output_dir=Path("outputs"),
            num_speakers=2,
        )
        self.assertEqual(
            speaker_count_kwargs(config),
            {"min_speakers": 2, "max_speakers": 2},
        )

    def test_apply_person_labels(self) -> None:
        result = {
            "segments": [
                {"speaker": "SPEAKER_00", "words": [{"speaker": "SPEAKER_00"}]},
                {"speaker": "SPEAKER_01", "words": [{"speaker": "SPEAKER_01"}]},
            ]
        }

        speaker_map = apply_person_labels(result)

        self.assertEqual(speaker_map["SPEAKER_00"], "Persona A")
        self.assertEqual(speaker_map["SPEAKER_01"], "Persona B")
        self.assertEqual(result["segments"][0]["speaker_label"], "Persona A")
        self.assertEqual(result["segments"][1]["words"][0]["speaker_label"], "Persona B")

    def test_hf_diarization_access_message_mentions_required_models(self) -> None:
        message = hf_diarization_access_message()

        self.assertIn("pyannote/segmentation-3.0", message)
        self.assertIn("pyannote/speaker-diarization-3.1", message)
        self.assertIn("HF_TOKEN", message)

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
