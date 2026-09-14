from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WordTimestamp:
    word: str
    start_sec: float
    end_sec: float
    score: float | None = None


@dataclass
class AlignmentResult:
    words: list[WordTimestamp]
    device: str
    backend: str = "whisperx"


class AlignmentError(RuntimeError):
    pass


def _require_whisperx() -> Any:
    try:
        import whisperx  # type: ignore
    except ImportError as exc:
        raise AlignmentError(
            "Forced alignment dependencies are missing. Install with: "
            "`uv add --optional forced-alignment whisperx torch torchaudio`"
        ) from exc
    return whisperx


def _resolve_device(preferred_device: str) -> str:
    pref = (preferred_device or "auto").strip().lower()
    if pref not in {"auto", "cpu", "cuda"}:
        raise AlignmentError("Alignment device must be one of: auto, cpu, cuda")

    if pref != "auto":
        return pref

    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"


def _audio_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            return 0.0
        return float(frame_count / frame_rate)


def _clean_word(word: str) -> str:
    cleaned = re.sub(r"\s+", " ", word or "").strip()
    cleaned = cleaned.strip(".,!?;:\"()[]{}")
    return cleaned


def align_words(
    audio_path: str | Path,
    transcript_text: str,
    preferred_device: str = "auto",
    language_code: str = "en",
) -> AlignmentResult:
    path = Path(audio_path)
    if not path.exists():
        raise AlignmentError(f"Audio file not found: {path}")

    text = re.sub(r"\s+", " ", transcript_text or "").strip()
    if not text:
        return AlignmentResult(words=[], device=_resolve_device(preferred_device))

    whisperx = _require_whisperx()
    target_device = _resolve_device(preferred_device)

    try:
        audio = whisperx.load_audio(str(path))
        align_model, metadata = whisperx.load_align_model(language_code=language_code, device=target_device)
    except Exception as exc:
        if target_device == "cuda":
            # Fallback keeps the feature usable when CUDA runtime is present but misconfigured.
            target_device = "cpu"
            try:
                audio = whisperx.load_audio(str(path))
                align_model, metadata = whisperx.load_align_model(language_code=language_code, device=target_device)
            except Exception as fallback_exc:
                raise AlignmentError(f"Failed to initialize WhisperX alignment model: {fallback_exc}") from exc
        else:
            raise AlignmentError(f"Failed to initialize WhisperX alignment model: {exc}") from exc

    duration_sec = _audio_duration_seconds(path)

    try:
        aligned = whisperx.align(
            transcript=[{"start": 0.0, "end": duration_sec, "text": text}],
            model=align_model,
            align_model_metadata=metadata,
            audio=audio,
            device=target_device,
            return_char_alignments=False,
        )
    except TypeError:
        # Compatibility with WhisperX versions where argument names differ.
        try:
            aligned = whisperx.align(
                [{"start": 0.0, "end": duration_sec, "text": text}],
                align_model,
                metadata,
                audio,
                target_device,
                return_char_alignments=False,
            )
        except Exception as exc:
            raise AlignmentError(f"WhisperX alignment failed: {exc}") from exc
    except Exception as exc:
        raise AlignmentError(f"WhisperX alignment failed: {exc}") from exc

    raw_words = aligned.get("word_segments", []) if isinstance(aligned, dict) else []
    words: list[WordTimestamp] = []

    for item in raw_words:
        if not isinstance(item, dict):
            continue

        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue

        try:
            start_sec = max(0.0, float(start))
            end_sec = min(duration_sec, float(end))
        except (TypeError, ValueError):
            continue

        if end_sec <= start_sec:
            continue

        word = _clean_word(str(item.get("word", "")))
        if not word:
            continue

        score: float | None = None
        if item.get("score") is not None:
            try:
                score = float(item["score"])
            except (TypeError, ValueError):
                score = None

        words.append(
            WordTimestamp(
                word=word,
                start_sec=start_sec,
                end_sec=end_sec,
                score=score,
            )
        )

    words.sort(key=lambda w: (w.start_sec, w.end_sec))
    return AlignmentResult(words=words, device=target_device)
