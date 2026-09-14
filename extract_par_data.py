from __future__ import annotations

import argparse
import csv
import re
import wave
from dataclasses import dataclass
from pathlib import Path

from alignment import AlignmentError, align_words


TIME_MARK = "\x15"
TIME_RE = re.compile(r"\x15(\d+)_(\d+)\x15")


@dataclass
class ParSegment:
    subject_id: str
    segment_index: int
    start_ms: int
    end_ms: int
    raw_text: str
    clean_text: str

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def unwrap_word(text: str) -> str:
    # Convert CHAT-style partial forms like an(d) -> and, stealin(g) -> stealing.
    return re.sub(r"([A-Za-z]+)\(([A-Za-z]+)\)", r"\1\2", text)

def repeat_words(text):
    count = len(re.findall(r"(\[x\s\d])", text))
    for x in range(count):
        match = re.search(r"(\[x\s\d])", text)
        split = re.split(r"\s", text)
        begin, end = match.span()
        repeat_count = text[end-2]
        preceding_word = split[split.index('[x')-1]
        repeated_words = ' '.join([preceding_word] * (int(repeat_count)-1))
        text = text[:begin] + repeated_words + text[end:]
    return text

def clean_chat_text(text: str) -> str:
    cleaned = text
    cleaned = TIME_RE.sub(" ", cleaned)
    cleaned = cleaned.replace(TIME_MARK, " ")
    cleaned = cleaned.replace("\t", " ")
    cleaned = cleaned.replace("\n", " ")

    # ‘w [x n]’ were replaced by repetitions of w for n times
    # cleaned = repeat_words(cleaned)

    # Remove bracketed CHAT annotations such as [//], [* s:r], [: word], [+ exc].
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)

    # Remove angle brackets used for retracing markers.
    cleaned = cleaned.replace("<", " ").replace(">", " ")

    # Remove event/disfluency markers like &=laughs, &uh, &um.
    # cleaned = re.sub(r"&=?[A-Za-z:]+", " ", cleaned)

    # Remove word begin with `&=` symbols such as "&=laughs"
    cleaned = re.sub(r'&=[^\s]+', '', cleaned)

    # Remove word begin with `&` symbols such as &um
    # transcript = re.sub(r'&[^\s]+', '', transcript)
    # Keep the disfluency markers
    cleaned = re.sub(r'&', '', cleaned)

    # Remove and @ pattern like `@l` , `@s:*`
    cleaned = re.sub(r'@[a-zA-Z0-9:$*]{1,5}(?=\s)', '', cleaned)
    cleaned = re.sub(r'@[a-zA-Z0-9]{1,5}', '', cleaned)

    # Remove quote markers such as +"/. while keeping spoken words.
    cleaned = cleaned.replace('+"', " ")
    cleaned = cleaned.replace("+/", " ")

    # Normalize common CHAT token patterns.
    # cleaned = cleaned.replace("out_of", "out of")
    # Replace `_` with empty space ` `
    cleaned = re.sub(r'_', ' ', cleaned)
    cleaned = cleaned.replace("+//.", " ")

    cleaned = unwrap_word(cleaned)

    # Keep apostrophes and sentence punctuation, drop other annotation symbols.
    cleaned = re.sub(r"[^A-Za-z0-9'.,!?\- ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def parse_semicolon_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Parse metadata files formatted as semi-colon separated text."""
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows

    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    content_lines = [line for line in lines if line.strip()]
    if not content_lines:
        return rows

    header_parts = [part.strip() for part in content_lines[0].split(";")]
    headers = [re.sub(r"\s+", "", part).lower() for part in header_parts]

    for line in content_lines[1:]:
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < len(headers):
            parts.extend([""] * (len(headers) - len(parts)))

        row = {headers[i]: parts[i] for i in range(len(headers))}
        subject_id = row.get("id", "").strip()
        if subject_id:
            rows[subject_id] = row

    return rows


def load_train_metadata(cc_meta_path: Path, cd_meta_path: Path) -> dict[str, dict[str, str]]:
    """Load train metadata from control/case files and assign labels 0/1."""
    train_meta: dict[str, dict[str, str]] = {}

    cc_rows = parse_semicolon_metadata(cc_meta_path)
    for subject_id, row in cc_rows.items():
        train_meta[subject_id] = {
            "age": row.get("age", ""),
            "gender": row.get("gender", ""),
            "label": "0",
            "mmse": row.get("mmse", ""),
            "split": "train",
        }

    cd_rows = parse_semicolon_metadata(cd_meta_path)
    for subject_id, row in cd_rows.items():
        train_meta[subject_id] = {
            "age": row.get("age", ""),
            "gender": row.get("gender", ""),
            "label": "1",
            "mmse": row.get("mmse", ""),
            "split": "train",
        }

    return train_meta


def load_test_metadata(test_meta_path: Path) -> dict[str, dict[str, str]]:
    """Load test metadata from 2020Labels.txt (contains label/mmse)."""
    test_meta: dict[str, dict[str, str]] = {}
    test_rows = parse_semicolon_metadata(test_meta_path)

    for subject_id, row in test_rows.items():
        test_meta[subject_id] = {
            "age": row.get("age", ""),
            "gender": row.get("gender", ""),
            "label": row.get("label", ""),
            "mmse": row.get("mmse", ""),
            "split": "test",
        }

    return test_meta


def extract_par_segments(cha_path: Path) -> list[ParSegment]:
    lines = read_text(cha_path).splitlines()
    subject_id = cha_path.stem
    segments: list[ParSegment] = []
    segment_index = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("*PAR:"):
            utterance_lines = [line[len("*PAR:") :].strip()]
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if next_line.startswith("\t"):
                    utterance_lines.append(next_line.strip())
                    j += 1
                    continue
                break

            full_utterance = " ".join(utterance_lines)
            time_match = TIME_RE.search(full_utterance)
            if time_match:
                start_ms = int(time_match.group(1))
                end_ms = int(time_match.group(2))
                raw_text = TIME_RE.sub(" ", full_utterance)
                raw_text = re.sub(r"\s+", " ", raw_text).strip()
                clean_text = clean_chat_text(raw_text)

                segments.append(
                    ParSegment(
                        subject_id=subject_id,
                        segment_index=segment_index,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        raw_text=raw_text,
                        clean_text=clean_text,
                    )
                )
                segment_index += 1

            i = j
            continue

        i += 1

    return segments


def write_segment_csv(path: Path, all_segments: list[ParSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject_id",
                "segment_index",
                "start_ms",
                "end_ms",
                "duration_ms",
                "raw_text",
                "clean_text",
            ]
        )
        for seg in all_segments:
            writer.writerow(
                [
                    seg.subject_id,
                    seg.segment_index,
                    seg.start_ms,
                    seg.end_ms,
                    seg.duration_ms,
                    seg.raw_text,
                    seg.clean_text,
                ]
            )


def extract_par_audio(source_wav: Path, segments: list[ParSegment], out_wav: Path) -> int:
    if not segments:
        return 0

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(source_wav), "rb") as src:
        n_channels = src.getnchannels()
        sample_width = src.getsampwidth()
        frame_rate = src.getframerate()
        n_frames = src.getnframes()
        comp_type = src.getcomptype()
        comp_name = src.getcompname()

        # Most DementiaBank wav files are PCM; preserve source wave parameters.
        with wave.open(str(out_wav), "wb") as dst:
            dst.setnchannels(n_channels)
            dst.setsampwidth(sample_width)
            dst.setframerate(frame_rate)
            dst.setcomptype(comp_type, comp_name)

            written_frames = 0
            for seg in sorted(segments, key=lambda s: (s.start_ms, s.end_ms)):
                start_frame = max(0, int(seg.start_ms * frame_rate / 1000))
                end_frame = min(n_frames, int(seg.end_ms * frame_rate / 1000))
                if end_frame <= start_frame:
                    continue

                src.setpos(start_frame)
                chunk = src.readframes(end_frame - start_frame)
                if not chunk:
                    continue

                dst.writeframes(chunk)
                written_frames += end_frame - start_frame

    return written_frames


def write_subject_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "split",
                "age",
                "gender",
                "label",
                "mmse",
                "cha_path",
                "source_audio_path",
                "par_audio_path",
                "num_segments",
                "total_par_duration_ms",
                "par_transcript",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_word_timestamp_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "word_index",
                "word",
                "start_sec",
                "end_sec",
                "duration_sec",
                "score",
                "device",
                "backend",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract PAR-only transcripts from CHAT (.cha) files to CSV and "
            "create PAR-only merged wav audio for each subject."
        )
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=Path("data/transcript"),
        help="Directory containing .cha transcript files.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/audio"),
        help="Directory containing source .wav audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/par_only"),
        help="Output base directory for CSV files and PAR-only audio.",
    )
    parser.add_argument(
        "--cc-meta-path",
        type=Path,
        default=Path("data/cc_meta_data.txt"),
        help="Path to control-group train metadata file (label forced to 0).",
    )
    parser.add_argument(
        "--cd-meta-path",
        type=Path,
        default=Path("data/cd_meta_data.txt"),
        help="Path to dementia-group train metadata file (label forced to 1).",
    )
    parser.add_argument(
        "--test-meta-path",
        type=Path,
        default=Path("data/2020Labels.txt"),
        help="Path to test metadata file used to generate test.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of transcript files to process (0 = all).",
    )
    parser.add_argument(
        "--force-align",
        action="store_true",
        help=(
            "Run WhisperX forced alignment on each generated PAR-only audio using "
            "the merged PAR transcript and export word-level timestamps."
        ),
    )
    parser.add_argument(
        "--alignment-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Preferred runtime for forced alignment.",
    )
    parser.add_argument(
        "--alignment-language",
        default="en",
        help="Language code passed to WhisperX align model (default: en).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transcript_dir: Path = args.transcript_dir
    audio_dir: Path = args.audio_dir
    output_dir: Path = args.output_dir

    segment_csv = output_dir / "par_segments.csv"
    subject_csv = output_dir / "par_subjects.csv"
    train_csv = output_dir / "train.csv"
    test_csv = output_dir / "test.csv"
    output_audio_dir = output_dir / "audio"
    word_timestamp_csv = output_dir / "par_word_timestamps.csv"

    train_meta = load_train_metadata(args.cc_meta_path, args.cd_meta_path)
    test_meta = load_test_metadata(args.test_meta_path)
    all_meta = {**train_meta, **test_meta}

    cha_files = sorted(transcript_dir.glob("*.cha"))
    if args.limit and args.limit > 0:
        cha_files = cha_files[: args.limit]

    if not cha_files:
        raise FileNotFoundError(f"No .cha files found in: {transcript_dir}")

    all_segments: list[ParSegment] = []
    subject_rows: list[dict[str, str | int]] = []
    word_timestamp_rows: list[dict[str, str | float | int]] = []

    missing_audio = 0
    no_par = 0
    missing_meta = 0
    alignment_failures = 0

    for cha_path in cha_files:
        subject_id = cha_path.stem
        source_wav = audio_dir / f"{subject_id}.wav"
        out_wav = output_audio_dir / f"{subject_id}_PAR.wav"

        segments = extract_par_segments(cha_path)
        if not segments:
            no_par += 1
            continue

        all_segments.extend(segments)

        transcript = " ".join(seg.clean_text for seg in segments if seg.clean_text).strip()
        total_duration_ms = sum(seg.duration_ms for seg in segments)

        metadata = all_meta.get(subject_id)
        if metadata is None:
            metadata = {"split": "", "age": "", "gender": "", "label": "", "mmse": ""}
            missing_meta += 1

        par_audio_path = ""
        if source_wav.exists():
            written_frames = extract_par_audio(source_wav, segments, out_wav)
            if written_frames > 0:
                par_audio_path = str(out_wav)
        else:
            missing_audio += 1

        subject_rows.append(
            {
                "subject_id": subject_id,
                "split": metadata.get("split", ""),
                "age": metadata.get("age", ""),
                "gender": metadata.get("gender", ""),
                "label": metadata.get("label", ""),
                "mmse": metadata.get("mmse", ""),
                "cha_path": str(cha_path),
                "source_audio_path": str(source_wav),
                "par_audio_path": par_audio_path,
                "num_segments": len(segments),
                "total_par_duration_ms": total_duration_ms,
                "par_transcript": transcript,
            }
        )

        if args.force_align and par_audio_path and transcript:
            try:
                aligned = align_words(
                    audio_path=par_audio_path,
                    transcript_text=transcript,
                    preferred_device=args.alignment_device,
                    language_code=args.alignment_language,
                )
                for word_index, word in enumerate(aligned.words):
                    word_timestamp_rows.append(
                        {
                            "subject_id": subject_id,
                            "word_index": word_index,
                            "word": word.word,
                            "start_sec": round(word.start_sec, 6),
                            "end_sec": round(word.end_sec, 6),
                            "duration_sec": round(word.end_sec - word.start_sec, 6),
                            "score": "" if word.score is None else round(word.score, 6),
                            "device": aligned.device,
                            "backend": aligned.backend,
                        }
                    )
            except AlignmentError as exc:
                alignment_failures += 1
                print(f"[WARN] Alignment failed for {subject_id}: {exc}")

    write_segment_csv(segment_csv, all_segments)
    write_subject_csv(subject_csv, subject_rows)

    train_rows = [row for row in subject_rows if str(row.get("subject_id", "")) in train_meta]
    test_rows = [row for row in subject_rows if str(row.get("subject_id", "")) in test_meta]
    write_subject_csv(train_csv, train_rows)
    write_subject_csv(test_csv, test_rows)

    if args.force_align:
        write_word_timestamp_csv(word_timestamp_csv, word_timestamp_rows)

    print(f"Processed subjects: {len(subject_rows)}")
    print(f"PAR segments: {len(all_segments)}")
    print(f"No-PAR transcripts skipped: {no_par}")
    print(f"Missing source audio files: {missing_audio}")
    print(f"Missing metadata rows: {missing_meta}")
    print(f"Segment CSV: {segment_csv}")
    print(f"All subjects CSV: {subject_csv}")
    print(f"Train CSV: {train_csv}")
    print(f"Test CSV: {test_csv}")
    print(f"PAR-only audio dir: {output_audio_dir}")
    if args.force_align:
        print(f"Word timestamp CSV: {word_timestamp_csv}")
        print(f"Word timestamps written: {len(word_timestamp_rows)}")
        print(f"Alignment failures: {alignment_failures}")

if __name__ == "__main__":
    main()