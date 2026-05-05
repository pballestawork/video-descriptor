from __future__ import annotations

from pathlib import Path
from typing import Any


def seconds_to_srt_time(value: float) -> str:
    hours, remainder = divmod(max(value, 0.0), 3600)
    minutes, seconds = divmod(remainder, 60)
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    return f"{int(hours):02}:{int(minutes):02}:{whole_seconds:02},{milliseconds:03}"


def seconds_to_vtt_time(value: float) -> str:
    return seconds_to_srt_time(value).replace(",", ".")


def write_txt(result: dict[str, Any], output_path: Path) -> None:
    lines = []
    for segment in result.get("segments", []):
        start = seconds_to_vtt_time(float(segment.get("start", 0.0)))
        end = seconds_to_vtt_time(float(segment.get("end", 0.0)))
        text = str(segment.get("text", "")).strip()
        speaker = format_speaker(segment)
        speaker_prefix = f" {speaker}:" if speaker else ""
        lines.append(f"[{start} - {end}]{speaker_prefix} {text}")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def write_srt(result: dict[str, Any], output_path: Path) -> None:
    blocks = []
    for index, segment in enumerate(result.get("segments", []), start=1):
        start = seconds_to_srt_time(float(segment.get("start", 0.0)))
        end = seconds_to_srt_time(float(segment.get("end", 0.0)))
        text = str(segment.get("text", "")).strip()
        speaker = format_speaker(segment)
        if speaker:
            text = f"{speaker}: {text}"
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    output_path.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")


def write_vtt(result: dict[str, Any], output_path: Path) -> None:
    blocks = ["WEBVTT"]
    for segment in result.get("segments", []):
        start = seconds_to_vtt_time(float(segment.get("start", 0.0)))
        end = seconds_to_vtt_time(float(segment.get("end", 0.0)))
        text = str(segment.get("text", "")).strip()
        speaker = format_speaker(segment)
        if speaker:
            text = f"{speaker}: {text}"
        blocks.append(f"{start} --> {end}\n{text}")
    output_path.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")


def format_speaker(segment: dict[str, Any]) -> str | None:
    speaker = segment.get("speaker_label") or segment.get("speaker")
    if not speaker:
        return None
    return str(speaker).strip()
