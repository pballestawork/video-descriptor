from __future__ import annotations

import gc
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
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
    log("Running ffmpeg audio extraction")
    start = time.monotonic()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("ffmpeg could not extract audio.") from exc
    log(f"Audio extraction finished in {format_duration(time.monotonic() - start)}")
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

    log(f"Input video: {input_video} ({format_bytes(input_video.stat().st_size)})")
    log(f"Output directory: {output_dir}")
    log(f"Extracting audio to {audio_path}")
    extract_audio(input_video, audio_path)
    log(f"Extracted audio size: {format_bytes(audio_path.stat().st_size)}")

    log("Starting WhisperX pipeline")
    result = transcribe_audio_with_whisperx(audio_path, config)

    log("Writing transcript files")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_txt(result, txt_path)
    write_srt(result, srt_path)
    write_vtt(result, vtt_path)
    log(f"Generated {len(result.get('segments', []))} segments")

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
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX dependencies are not installed. In Colab, run "
            "`pip install -r requirements-colab.txt` after installing this project."
        ) from exc

    allow_trusted_whisperx_checkpoints(torch)

    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX dependencies are not installed. In Colab, run "
            "`pip install -r requirements-colab.txt` after installing this project."
        ) from exc

    device = resolve_device(config.device, torch)
    compute_type = resolve_compute_type(config.compute_type, device)
    language = None if config.language in {None, "", "auto"} else config.language

    log(
        "Transcribing with "
        f"model={config.model}, language={language or 'auto'}, "
        f"device={device}, compute_type={compute_type}, batch_size={config.batch_size}"
    )

    start = time.monotonic()
    log("Loading ASR model")
    model = whisperx.load_model(config.model, device, compute_type=compute_type)
    log(f"ASR model loaded in {format_duration(time.monotonic() - start)}")

    log("Loading audio into WhisperX")
    audio = whisperx.load_audio(str(audio_path))
    transcribe_kwargs: dict[str, Any] = {"batch_size": config.batch_size}
    if language:
        transcribe_kwargs["language"] = language

    start = time.monotonic()
    log("Running transcription")
    result = model.transcribe(audio, **transcribe_kwargs)
    log(
        "Transcription finished in "
        f"{format_duration(time.monotonic() - start)} with "
        f"{len(result.get('segments', []))} raw segments"
    )

    del model
    free_memory(torch)

    detected_language = result.get("language") or language
    if not detected_language:
        return result

    start = time.monotonic()
    log(f"Loading alignment model for language={detected_language}")
    model_a, metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    log("Running timestamp alignment")
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    result["language"] = detected_language
    log(f"Alignment finished in {format_duration(time.monotonic() - start)}")

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


def allow_trusted_whisperx_checkpoints(torch_module: Any) -> None:
    original_load = torch_module.load
    if getattr(original_load, "_video_descriptor_weights_patch", False):
        return

    def patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    patched_load._video_descriptor_weights_patch = True  # type: ignore[attr-defined]
    torch_module.load = patched_load
    log(
        "Configured torch.load(weights_only=False) for trusted "
        "WhisperX/Pyannote checkpoints"
    )


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
