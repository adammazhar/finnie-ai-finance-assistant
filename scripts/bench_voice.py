"""Transcription speed and accuracy: python scripts/bench_voice.py SAMPLE_DIR [--runs N]

SAMPLE_DIR holds pairs of files: ``name.wav`` (16 kHz mono speech) and ``name.txt`` (what is
said). Each recording is transcribed N times through Finnie's voice module (the same call
the chat makes, with OPENAI_API_KEY from .env), and the script reports the recording's
length, the median and slowest time, and the word error rate of the first transcript.

Recordings with known text can be made on Windows with the built-in speech synthesizer
(see docs/BENCHMARKS.md). The recordings aren't committed.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings
from src.core.voice import transcribe


def words(text: str) -> list[str]:
    """Lower-case words; "401(k)" and "401k" count as the same word."""
    return re.findall(r"[a-z0-9']+", re.sub(r"[()]", "", text.lower()).replace("-", " "))


def word_error_rate(expected: str, heard: str) -> float:
    """Word-level edit distance divided by the number of expected words."""
    ref, hyp = words(expected), words(heard)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1] / max(1, len(ref))


def seconds(path: Path) -> float:
    with wave.open(str(path)) as audio:
        return audio.getnframes() / audio.getframerate()


def main(folder: Path, runs: int) -> None:
    settings = get_settings()
    print(f"Model: {settings.voice.model}, {runs} runs per recording\n")
    print("| Recording | Length | Median | Slowest | Word error rate |")
    print("|---|---|---|---|---|")
    for wav in sorted(folder.glob("*.wav")):
        expected = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
        audio = wav.read_bytes()
        times, first = [], ""
        for _ in range(runs):
            started = time.perf_counter()
            heard = transcribe(audio, settings, filename=wav.name)
            times.append(time.perf_counter() - started)
            first = first or heard
        print(
            f"| {wav.stem} | {seconds(wav):.1f} s | {statistics.median(times):.2f} s | "
            f"{max(times):.2f} s | {word_error_rate(expected, first):.0%} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    main(args.folder, args.runs)
