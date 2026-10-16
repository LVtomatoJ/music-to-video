#!/usr/bin/env python3
"""Synthesize a 30s test track with known beat times.

The track is generated in this repository (no third-party recording):
126 BPM kick on every quarter note, snare on the backbeat, eighth-note hats,
a quiet bass line, and sixteenth-note ghost notes. Seconds covering beats
32-39 drop the kick level so a tracker has to hold tempo through a breakdown.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

SR = 44100
DURATION = 30.0
BPM = 126.0


def synthesize(seed: int = 1) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    n = int(SR * DURATION)
    y = np.zeros(n, dtype=np.float64)
    period = 60.0 / BPM
    beat_times = [i * period for i in range(int(DURATION / period + 1e-9))]
    strong = [t for i, t in enumerate(beat_times) if i % 2 == 1]

    def place(sig: np.ndarray, t: float) -> None:
        i0 = int(round(t * SR))
        if i0 >= n or i0 < 0:
            return
        end = min(n, i0 + len(sig))
        y[i0:end] += sig[: end - i0]

    def kick(amp: float) -> np.ndarray:
        tt = np.arange(int(0.11 * SR)) / SR
        freq = 150.0 * np.exp(-tt * 30.0) + 45.0
        phase = 2.0 * np.pi * np.cumsum(freq) / SR
        env = np.exp(-tt * 22.0)
        click = np.exp(-tt * 250.0)
        return amp * (np.sin(phase) * env + 0.35 * click)

    def snare(amp: float) -> np.ndarray:
        length = int(0.15 * SR)
        tt = np.arange(length) / SR
        noise = rng.normal(0.0, 1.0, length)
        noise = np.diff(noise, prepend=noise[0])
        env = np.exp(-tt * 16.0)
        tone = np.sin(2.0 * np.pi * 196.0 * tt) * np.exp(-tt * 24.0)
        return amp * (0.85 * noise * env + 0.35 * tone)

    def hat(amp: float) -> np.ndarray:
        length = int(0.035 * SR)
        tt = np.arange(length) / SR
        noise = rng.normal(0.0, 1.0, length)
        noise = np.diff(noise, prepend=0.0)
        env = np.exp(-tt * 90.0)
        return amp * noise * env

    def bass(freq: float, dur: float, amp: float) -> np.ndarray:
        tt = np.arange(int(dur * SR)) / SR
        env = np.sin(np.clip(np.pi * tt / dur, 0, np.pi)) ** 1.3
        return amp * np.sin(2.0 * np.pi * freq * tt) * env

    def ghost(amp: float) -> np.ndarray:
        tt = np.arange(int(0.08 * SR)) / SR
        env = np.exp(-tt * 20.0)
        return amp * np.sin(2.0 * np.pi * 98.0 * tt) * env

    bass_notes = [55.0, 65.41, 73.42, 49.0]
    for i, t in enumerate(beat_times):
        breakdown = 32 <= i <= 39
        place(kick(0.22 if breakdown else 0.92), t)
        if i % 2 == 1:
            place(snare(0.38 if breakdown else 0.72), t)
        place(hat(0.11), t)
        place(hat(0.10), t + period * 0.5)
        if i % 4 == 0:
            place(ghost(0.20), t + period * 0.25)
        place(bass(bass_notes[i % len(bass_notes)], period * 0.92, 0.16), t)

    peak = np.max(np.abs(y))
    if peak > 0:
        y *= 0.89 / peak

    truth = {
        "bpm": BPM,
        "sample_rate": SR,
        "duration": DURATION,
        "beats": beat_times,
        "strong_beats": strong,
        "description": "Synthetic 126 BPM drum pattern with a quieter breakdown on beats 32-39.",
    }
    return y, truth


def write_wav(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(y, -1.0, 1.0)
    ints = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(ints.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, default=Path("assets/test-track.wav"))
    parser.add_argument("--truth", type=Path, default=Path("assets/ground-truth.json"))
    args = parser.parse_args()
    y, truth = synthesize()
    write_wav(args.wav, y)
    args.truth.parent.mkdir(parents=True, exist_ok=True)
    args.truth.write_text(json.dumps(truth, indent=2) + "\n")
    print(f"wrote {args.wav} ({truth['duration']}s, {len(truth['beats'])} beats)")


if __name__ == "__main__":
    main()
