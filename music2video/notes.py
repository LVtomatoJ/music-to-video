"""Melody notes for the waterfall and the constellations.

librosa supplies onsets, a pitch track, and chroma. A stable voiced pitch
becomes a note. When the pitch track is too thin (a drum loop, a noisy mix),
each beat still gets a pitch class from chroma so the picture is not empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from music2video.analysis import Analysis

HOP = 1024
_CACHE: dict[tuple, tuple] = {}


@dataclass(frozen=True)
class Tone:
    time: float
    duration: float
    midi: int
    source: str


def extract_tones(analysis: Analysis) -> list[Tone]:
    key = ("g3c6-v2", str(analysis.wav_path), round(analysis.duration, 3), len(analysis.beats))
    cached = _CACHE.get(key)
    if cached is not None:
        return list(cached)
    tones = _extract(analysis)
    _CACHE[key] = tuple(tones)
    return list(tones)


def _extract(analysis: Analysis) -> list[Tone]:
    if not analysis.wav_path.is_file():
        return _from_beats(analysis, np.zeros((12, 1)), np.zeros(1))
    y, sr = librosa.load(str(analysis.wav_path), sr=analysis.sample_rate, mono=True)
    if len(y) < 2048:
        return _from_beats(analysis, np.zeros((12, 1)), np.zeros(1))
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, backtrack=True, units="time")
    onset_times = [float(t) for t in np.atleast_1d(onsets)]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    chroma_times = librosa.times_like(chroma, sr=sr, hop_length=HOP)
    f0, voiced, prob = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("G3")),
        fmax=float(librosa.note_to_hz("C6")),
        sr=sr,
        hop_length=HOP,
    )
    frame_times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    pitched = _segments(frame_times, f0, voiced, onset_times)
    if len(pitched) >= 6:
        return pitched
    return _from_beats(analysis, chroma, chroma_times, onset_times)


def _segments(times, f0, voiced, onsets: list[float]) -> list[Tone]:
    # voiced_prob stays near zero on some mixes even when the flag is set,
    # so the flag and a finite pitch are the gate.
    midis: list[int | None] = []
    for freq, flag in zip(f0, voiced):
        if flag and np.isfinite(freq) and float(freq) > 0:
            midis.append(int(round(float(librosa.hz_to_midi(float(freq))))))
        else:
            midis.append(None)
    notes: list[Tone] = []
    index = 0
    hop_s = float(times[1] - times[0]) if len(times) > 1 else HOP / 44100.0
    while index < len(midis):
        midi = midis[index]
        if midi is None:
            index += 1
            continue
        end = index + 1
        while end < len(midis) and midis[end] is not None and abs(midis[end] - midi) <= 1:
            end += 1
        duration = max(0.0, float(times[min(end, len(times) - 1)] - times[index]))
        if end == index + 1:
            duration = hop_s
        if duration >= 0.07:
            start = float(times[index])
            start = _snap_onset(start, onsets)
            notes.append(Tone(start, min(2.4, max(0.12, duration)), midi, "pitch"))
        index = end
    return _dedupe(notes)


def _snap_onset(start: float, onsets: list[float]) -> float:
    """A pitch that begins within 50ms of an onset lands on that onset."""
    if not onsets:
        return start
    nearest = min(onsets, key=lambda onset: abs(onset - start))
    if abs(nearest - start) <= 0.05:
        return nearest
    return start


def _from_beats(analysis: Analysis, chroma, chroma_times, onsets: list[float] | None = None) -> list[Tone]:
    beats = list(analysis.beats) or [0.0]
    gaps = [beats[i] - beats[i - 1] for i in range(1, len(beats))]
    typical = float(np.median(gaps)) if gaps else 0.4
    notes: list[Tone] = []
    for beat in beats:
        pitch_class = _chroma_at(chroma, chroma_times, beat)
        midi = 60 + pitch_class
        notes.append(Tone(float(beat), min(1.4, max(0.16, typical * 0.72)), midi, "chroma"))
    if onsets and len(notes) < 4:
        return notes
    return _dedupe(notes)


def chroma_columns(analysis: Analysis, times: list[float]) -> list[list[float]]:
    """Twelve pitch-class weights at each time. Used to voice chords, not the melody."""
    key = ("chroma-cols-v1", str(analysis.wav_path), round(analysis.duration, 3), len(times), round(times[0], 3) if times else 0)
    cached = _CACHE.get(key)
    if cached is not None:
        return [list(column) for column in cached]
    columns = _chroma_columns(analysis, times)
    _CACHE[key] = tuple(tuple(column) for column in columns)
    return columns


def _chroma_columns(analysis: Analysis, times: list[float]) -> list[list[float]]:
    if not times:
        return []
    if not analysis.wav_path.is_file():
        return [[0.0] * 12 for _ in times]
    y, sr = librosa.load(str(analysis.wav_path), sr=analysis.sample_rate, mono=True)
    if len(y) < 2048:
        return [[0.0] * 12 for _ in times]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    chroma_times = np.asarray(librosa.times_like(chroma, sr=sr, hop_length=HOP))
    if chroma.shape[1] == 0:
        return [[0.0] * 12 for _ in times]
    columns = []
    for when in times:
        index = int(np.argmin(np.abs(chroma_times - when)))
        columns.append([float(value) for value in chroma[:, index]])
    return columns


def _chroma_at(chroma, times, when: float) -> int:
    if chroma is None or len(getattr(chroma, "shape", ())) != 2 or chroma.shape[1] == 0:
        return 0
    index = int(np.argmin(np.abs(np.asarray(times) - when)))
    column = chroma[:, index]
    return int(np.argmax(column))


def _dedupe(notes: list[Tone]) -> list[Tone]:
    kept: list[Tone] = []
    for note in sorted(notes, key=lambda item: item.time):
        if kept and abs(note.time - kept[-1].time) < 0.06 and abs(note.midi - kept[-1].midi) <= 1:
            longer = note if note.duration > kept[-1].duration else kept[-1]
            kept[-1] = longer
            continue
        kept.append(note)
    return kept
