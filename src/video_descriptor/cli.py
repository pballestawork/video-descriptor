from __future__ import annotations

import argparse
from pathlib import Path

from .transcriber import TranscriptionConfig, run_transcription


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-descriptor",
        description="Extract audio from a video and transcribe it with WhisperX.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser(
        "transcribe",
        help="Extract video audio and generate transcript files.",
    )
    transcribe.add_argument("input_video", type=Path)
    transcribe.add_argument("--output-dir", type=Path, default=Path("outputs"))
    transcribe.add_argument("--model", default="large-v3-turbo")
    transcribe.add_argument("--language", default="es")
    transcribe.add_argument("--batch-size", type=int, default=16)
    transcribe.add_argument("--compute-type", default="float16")
    transcribe.add_argument(
        "--diarize",
        action="store_true",
        help="Assign speaker labels using WhisperX/Pyannote diarization.",
    )
    transcribe.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token for pyannote diarization. Prefer HF_TOKEN env var.",
    )
    transcribe.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers, e.g. 2 for Persona A and Persona B.",
    )
    transcribe.add_argument("--min-speakers", type=int, default=None)
    transcribe.add_argument("--max-speakers", type=int, default=None)
    transcribe.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "transcribe":
        outputs = run_transcription(
            TranscriptionConfig(
                input_video=args.input_video,
                output_dir=args.output_dir,
                model=args.model,
                language=args.language,
                batch_size=args.batch_size,
                compute_type=args.compute_type,
                device=args.device,
                diarize=args.diarize,
                hf_token=args.hf_token,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
            )
        )
        print("Done. Generated:")
        for path in (
            outputs.json_path,
            outputs.txt_path,
            outputs.srt_path,
            outputs.vtt_path,
        ):
            print(f"- {path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
