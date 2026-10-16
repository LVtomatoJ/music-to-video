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
class Section:
    start: float
    end: float
    energy: float
    label: str


@dataclass(frozen=True)
class Analysis:
    tempo: float
    beats: list[float]
    strong_beats: list[float]
    duration: float
    sample_rate: int
    wav_path: Path
    sections: tuple[Section, ...] = ()
    rms_times: tuple[float, ...] = ()
    rms: tuple[float, ...] = ()
    bands: tuple[tuple[float, ...], ...] = ()


def decode_wav(source: Path, destination: Path, max_seconds: float | None) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source)]
    if max_seconds is not None:
        cmd.extend(["-t", f"{max_seconds:.3f}"])
    cmd.extend(["-ac", "1", "-ar", "44100", str(destination)])
    subprocess.run(cmd, check=True)


def analyze(source: Path, workdir: Path, max_seconds: float | None = None, beat_mode: str = "default") -> Analysis:
    """Decode `source` into `workdir` and return beat times in seconds.

    A beat is strong when its high-frequency onset strength is at least 1.35×
    the median strength of all detected beats. That is the same rule measured
    in the Python prototype.

    `beat_mode="soft"` tries a half-time pulse when the tracker locks onto a
    fast strum or hat pattern, and keeps it only when those slower hits land
    on onsets at least as cleanly.
    """
    if beat_mode not in {"default", "soft"}:
        raise SystemExit(f"unknown beat mode {beat_mode}")
    wav_path = workdir / "analysis.wav"
    decode_wav(source, wav_path, max_seconds)
    y, sr = librosa.load(str(wav_path), sr=44100, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, units="frames")
    tempo_f = float(np.atleast_1d(tempo)[0])
    beat_frames = np.asarray(beat_frames, dtype=int)
    times = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)]
    if beat_mode == "soft":
        tempo_f, beat_frames, times = _prefer_slower_pulse(y, int(sr), tempo_f, beat_frames, times)
    strong = _strong_beats(y, sr, beat_frames, times)
    rms_times, rms, bands, sections = _sections_and_bands(y, int(sr))
    return Analysis(
        tempo=tempo_f,
        beats=times,
        strong_beats=strong,
        duration=float(len(y) / sr),
        sample_rate=int(sr),
        wav_path=wav_path,
        sections=tuple(sections),
        rms_times=tuple(rms_times),
        rms=tuple(rms),
        bands=tuple(tuple(row) for row in bands),
    )


def _onset_contrast(y: np.ndarray, sr: int, beats: list[float]) -> float:
    if len(beats) < 4:
        return 0.0
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    grid = librosa.times_like(onset, sr=sr, hop_length=HOP)
    values = [float(onset[int(np.argmin(np.abs(grid - beat)))]) for beat in beats]
    between = []
    for left, right in zip(beats, beats[1:]):
        between.append(float(onset[int(np.argmin(np.abs(grid - (left + right) / 2.0)))]))
    return float(np.mean(values) / (float(np.mean(between)) + 1e-9))


def _prefer_slower_pulse(y: np.ndarray, sr: int, tempo: float, frames: np.ndarray, times: list[float]):
    """Keep a half-time track when it matches onsets at least as well.

    Songs with a soft kick and a busy guitar or hat often get a doubled
    tempo. The slower grid is used only if the tracker actually slows down
    and the hits are not weaker than the fast grid.
    """
    if tempo <= 120.0 or len(times) < 8:
        return tempo, frames, times
    full = _onset_contrast(y, sr, times)
    half_tempo, half_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=HOP,
        start_bpm=float(tempo) / 2.0,
        units="frames",
    )
    half_frames = np.asarray(half_frames, dtype=int)
    half_times = [float(t) for t in librosa.frames_to_time(half_frames, sr=sr, hop_length=HOP)]
    if len(half_times) < 4:
        return tempo, frames, times
    realized = 60.0 / float(np.median(np.diff(half_times)))
    if realized >= tempo * 0.75:
        return tempo, frames, times
    if _onset_contrast(y, sr, half_times) + 1e-6 < full:
        return tempo, frames, times
    return float(np.atleast_1d(half_tempo)[0]), half_frames, half_times


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


def section_at(sections: tuple[Section, ...] | list[Section], time: float) -> Section:
    for section in sections:
        if section.start <= time < section.end:
            return section
    if sections:
        return sections[-1]
    return Section(0.0, 1e9, 0.4, "mid")


def _sections_and_bands(y: np.ndarray, sr: int):
    """RMS, 8 log-frequency bands, and 2s calm/mid/intense sections."""
    frame = 2048
    hop = 1024
    if len(y) < frame:
        return [0.0], [0.0], [[0.0] * 8], [Section(0.0, max(len(y) / sr, 1e-3), 0.4, "mid")]
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    edges = np.geomspace(40.0, 8000.0, 9)
    times: list[float] = []
    rms_vals: list[float] = []
    bands: list[list[float]] = []
    for start in range(0, len(y) - frame, hop):
        chunk = y[start : start + frame] * window
        spec = np.abs(np.fft.rfft(chunk))
        row = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            row.append(float(spec[mask].mean()) if np.any(mask) else 0.0)
        times.append((start + frame / 2) / sr)
        rms_vals.append(float(np.sqrt(np.mean(chunk * chunk))))
        bands.append(row)
    rms_arr = np.asarray(rms_vals, dtype=np.float64)
    band_arr = np.asarray(bands, dtype=np.float64)
    scale = float(np.percentile(rms_arr, 95)) + 1e-9
    rms_norm = np.clip(rms_arr / scale, 0.0, 1.0)
    for col in range(band_arr.shape[1]):
        col_scale = float(np.percentile(band_arr[:, col], 95)) + 1e-9
        band_arr[:, col] = np.clip(band_arr[:, col] / col_scale, 0.0, 1.0)
    duration = len(y) / sr
    sections: list[Section] = []
    cursor = 0.0
    time_arr = np.asarray(times)
    while cursor < duration - 1e-6:
        end = min(duration, cursor + 2.0)
        mask = (time_arr >= cursor) & (time_arr < end)
        energy = float(rms_norm[mask].mean()) if np.any(mask) else 0.0
        sections.append(Section(cursor, end, energy, "mid"))
        cursor = end
    energies = np.asarray([item.energy for item in sections], dtype=np.float64)
    if len(energies) >= 3:
        low, high = np.quantile(energies, [0.34, 0.67])
    else:
        low, high = 0.35, 0.62
    labeled = []
    for item in sections:
        if item.energy <= low:
            label = "calm"
        elif item.energy >= high:
            label = "intense"
        else:
            label = "mid"
        labeled.append(Section(item.start, item.end, item.energy, label))
    return times, rms_norm.tolist(), band_arr.tolist(), labeled
