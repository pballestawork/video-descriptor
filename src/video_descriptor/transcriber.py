from __future__ import annotations

import gc
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .formats import write_srt, write_txt, write_vtt


@dataclass(frozen=True)
class TranscriptionConfig:
    input_video: Path
    output_dir: Path
    model: str = "large-v3-turbo"
    language: str | None = "es"
    batch_size: int = 16
    compute_type: str = "float16"
    device: str = "auto"


@dataclass(frozen=True)
class TranscriptionOutputs:
    audio_path: Path
    json_path: Path
    txt_path: Path
    srt_path: Path
    vtt_path: Path


def extract_audio(input_video: Path, output_audio: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or is not available on PATH.")

    output_audio.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_audio),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"ffmpeg could not extract audio: {detail}") from exc
    return output_audio


def run_transcription(config: TranscriptionConfig) -> TranscriptionOutputs:
    input_video = config.input_video.expanduser().resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_video.stem
    audio_path = output_dir / f"{base_name}.wav"
    json_path = output_dir / f"{base_name}.json"
    txt_path = output_dir / f"{base_name}.txt"
    srt_path = output_dir / f"{base_name}.srt"
    vtt_path = output_dir / f"{base_name}.vtt"

    print(f"Extracting audio to {audio_path}")
    extract_audio(input_video, audio_path)

    print("Loading WhisperX")
    result = transcribe_audio_with_whisperx(audio_path, config)

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_txt(result, txt_path)
    write_srt(result, srt_path)
    write_vtt(result, vtt_path)

    return TranscriptionOutputs(
        audio_path=audio_path,
        json_path=json_path,
        txt_path=txt_path,
        srt_path=srt_path,
        vtt_path=vtt_path,
    )


def transcribe_audio_with_whisperx(
    audio_path: Path,
    config: TranscriptionConfig,
) -> dict[str, Any]:
    try:
        import torch
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX dependencies are not installed. In Colab, run "
            "`pip install -r requirements-colab.txt` after installing this project."
        ) from exc

    device = resolve_device(config.device, torch)
    compute_type = resolve_compute_type(config.compute_type, device)
    language = None if config.language in {None, "", "auto"} else config.language

    print(
        "Transcribing with "
        f"model={config.model}, language={language or 'auto'}, "
        f"device={device}, compute_type={compute_type}, batch_size={config.batch_size}"
    )

    model = whisperx.load_model(config.model, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_path))
    transcribe_kwargs: dict[str, Any] = {"batch_size": config.batch_size}
    if language:
        transcribe_kwargs["language"] = language
    result = model.transcribe(audio, **transcribe_kwargs)

    del model
    free_memory(torch)

    detected_language = result.get("language") or language
    if not detected_language:
        return result

    print(f"Aligning timestamps for language={detected_language}")
    model_a, metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    result["language"] = detected_language

    del model_a
    free_memory(torch)
    return result


def resolve_device(requested_device: str, torch_module: Any) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def resolve_compute_type(requested_compute_type: str, device: str) -> str:
    if device == "cpu":
        return "int8"
    return requested_compute_type


def free_memory(torch_module: Any) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
