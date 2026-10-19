#!/usr/bin/env python3
"""Reference spectral-flux beat tracker.

TypeScript and Rust prototypes port this file. It is the Ellis dynamic
programming tracker (the same family as librosa) on a positive spectral-flux
onset envelope, with a tempo prior centered at 120 BPM — not at the test
track's tempo.
"""

from __future__ import annotations

import math

import numpy as np

N_FFT = 2048
HOP = 512
HF_HZ = 2000.0


def _flux(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
    window = np.hanning(N_FFT).astype(np.float64)
    if len(y) < N_FFT:
        raise ValueError("audio shorter than one FFT window")
    n_frames = 1 + (len(y) - N_FFT) // HOP
    spec_bins = N_FFT // 2 + 1
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    hf = freqs >= HF_HZ
    flux = np.zeros(n_frames, dtype=np.float64)
    hf_flux = np.zeros(n_frames, dtype=np.float64)
    prev = np.zeros(spec_bins, dtype=np.float64)
    prev_hf = np.zeros(int(np.sum(hf)), dtype=np.float64)
    for i in range(n_frames):
        start = i * HOP
        frame = y[start : start + N_FFT] * window
        mag = np.abs(np.fft.rfft(frame))
        delta = mag - prev
        flux[i] = np.maximum(delta, 0.0).sum()
        mag_hf = mag[hf]
        delta_hf = mag_hf - prev_hf
        hf_flux[i] = np.maximum(delta_hf, 0.0).sum()
        prev = mag
        prev_hf = mag_hf
    return flux, hf_flux, sr / HOP


def estimate_bpm(flux: np.ndarray, frame_rate: float) -> tuple[float, int]:
    centered = flux - float(np.mean(flux))
    ac = np.correlate(centered, centered, mode="full")
    ac = ac[len(ac) // 2 :]
    ac = ac / (ac[0] + 1e-12)
    best_score = -1e18
    best_lag = max(1, int(round(frame_rate * 60.0 / 120.0)))
    best_bpm = 120.0
    for bpm in range(80, 171):
        lag = int(round(frame_rate * 60.0 / bpm))
        if lag <= 0 or lag >= len(ac):
            continue
        lo = max(1, lag - 1)
        hi = min(len(ac), lag + 2)
        peak = float(np.max(ac[lo:hi]))
        prior = math.exp(-((bpm - 120.0) ** 2) / (2.0 * 30.0**2))
        score = peak * prior
        if score > best_score:
            best_score = score
            best_lag = lag
            best_bpm = 60.0 * frame_rate / lag
    return best_bpm, best_lag


def ellis_beats(onset: np.ndarray, frame_rate: float, bpm: float, tightness: float = 100.0) -> list[float]:
    """Ellis 2007 beat DP. Returns beat times in seconds."""
    period = max(1.0, (60.0 / bpm) * frame_rate)
    n = len(onset)
    pr_start = int(round(-2 * period))
    pr_end = int(round(-period / 2))
    if pr_end >= -1:
        pr_end = -1
    prange = np.arange(pr_start, pr_end + 1, dtype=int)
    txwt = -tightness * (np.log((-prange) / period) ** 2)
    backlink = np.full(n, -1, dtype=int)
    cumscore = np.full(n, -np.inf)
    localscore = onset.astype(np.float64)
    std = float(localscore.std())
    if std > 1e-9:
        localscore = localscore / std

    start_i = int(max(0, -int(prange[0])))
    for i in range(min(start_i, n)):
        cumscore[i] = localscore[i]

    for i in range(start_i, n):
        timerange = i + prange
        valid = (timerange >= 0) & (timerange < i)
        if not np.any(valid):
            cumscore[i] = localscore[i]
            continue
        cand = txwt[valid] + cumscore[timerange[valid]]
        xx = int(np.argmax(cand))
        chosen = int(timerange[valid][xx])
        cumscore[i] = float(localscore[i] + cand[xx])
        backlink[i] = chosen

    end_lo = max(0, n - int(round(period)))
    end = end_lo + int(np.argmax(cumscore[end_lo:]))
    frames: list[int] = []
    seen: set[int] = set()
    while end >= 0 and end not in seen:
        seen.add(end)
        frames.append(end)
        end = int(backlink[end])
    frames.reverse()
    return [frame / frame_rate for frame in frames]


def strong_beats(beat_times: list[float], hf_flux: np.ndarray, frame_rate: float) -> list[float]:
    if not beat_times:
        return []
    idxs = [min(len(hf_flux) - 1, max(0, int(round(t * frame_rate)))) for t in beat_times]
    vals = np.array([hf_flux[i] for i in idxs], dtype=np.float64)
    med = float(np.median(vals))
    threshold = med * 1.35
    return [t for t, v in zip(beat_times, vals) if v >= threshold and v > 0]


def track(y: np.ndarray, sr: int) -> dict:
    flux, hf_flux, frame_rate = _flux(y, sr)
    bpm, _lag = estimate_bpm(flux, frame_rate)
    beats = ellis_beats(flux, frame_rate, bpm)
    strong = strong_beats(beats, hf_flux, frame_rate)
    return {
        "tempo": bpm,
        "beats": beats,
        "strong_beats": strong,
        "frame_rate": frame_rate,
    }
