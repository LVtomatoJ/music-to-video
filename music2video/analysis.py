"""Audio file in, beat times out. librosa owns onset and tempo tracking."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

HOP = 512
N_FFT = 2048


@dataclass(frozen=True)
class Analysis:
    tempo: float
    beats: list[float]
    strong_beats: list[float]
    duration: float
    sample_rate: int
    wav_path: Path


def decode_wav(source: Path, destination: Path, max_seconds: float | None) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source)]
    if max_seconds is not None:
        cmd.extend(["-t", f"{max_seconds:.3f}"])
    cmd.extend(["-ac", "1", "-ar", "44100", str(destination)])
    subprocess.run(cmd, check=True)


def analyze(source: Path, workdir: Path, max_seconds: float | None = None) -> Analysis:
    """Decode `source` into `workdir` and return beat times in seconds.

    A beat is strong when its high-frequency onset strength is at least 1.35×
    the median strength of all detected beats. That is the same rule measured
    in the Python prototype.
    """
    wav_path = workdir / "analysis.wav"
    decode_wav(source, wav_path, max_seconds)
    y, sr = librosa.load(str(wav_path), sr=44100, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, units="frames")
    tempo_f = float(np.atleast_1d(tempo)[0])
    beat_frames = np.asarray(beat_frames, dtype=int)
    times = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)]
    strong = _strong_beats(y, sr, beat_frames, times)
    result = Analysis(
        tempo=tempo_f,
        beats=times,
        strong_beats=strong,
        duration=float(len(y) / sr),
        sample_rate=int(sr),
        wav_path=wav_path,
    )
    return result


def _strong_beats(y: np.ndarray, sr: int, beat_frames: np.ndarray, times: list[float]) -> list[float]:
    if len(times) == 0:
        return []
    spectrum = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    high = spectrum[freqs >= 2000.0]
    if high.size == 0:
        return []
    onset = librosa.onset.onset_strength(S=high, sr=sr, hop_length=HOP)
    if len(onset) == 0:
        return []
    values = onset[np.clip(beat_frames, 0, len(onset) - 1)]
    threshold = float(np.median(values)) * 1.35
    return [time for time, value in zip(times, values) if value >= threshold and value > 0]
