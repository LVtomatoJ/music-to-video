#!/usr/bin/env python3
"""Python prototype: librosa beat tracking, cairo frame render, ffmpeg MP4.

Pipeline: wav/mp3 -> librosa beat_track -> 6 balls + particles -> raw BGRA
frames on ffmpeg stdin -> H.264 + AAC MP4 with the source audio.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import cairo
import librosa
import numpy as np

FPS = 30
N_BALLS = 6
SEED = 42
N_FFT = 2048
HOP = 512


def xorshift32(state: list[int]) -> float:
    x = state[0] & 0xFFFFFFFF
    x = (x ^ ((x << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
    x = (x ^ (x >> 17)) & 0xFFFFFFFF
    x = (x ^ ((x << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
    state[0] = x
    return x / 4294967296.0


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    h = h % 360.0
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    hp = h / 60.0
    x = c * (1.0 - abs(hp % 2.0 - 1.0))
    if hp < 1:
        r, g, b = c, x, 0.0
    elif hp < 2:
        r, g, b = x, c, 0.0
    elif hp < 3:
        r, g, b = 0.0, c, x
    elif hp < 4:
        r, g, b = 0.0, x, c
    elif hp < 5:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    m = l - c / 2.0
    return r + m, g + m, b + m


def analyze(path: Path) -> dict:
    y, sr = librosa.load(str(path), sr=44100, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, units="frames")
    tempo_f = float(np.atleast_1d(tempo)[0])
    beat_frames = np.asarray(beat_frames, dtype=int)
    times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP).tolist()

    spectrum = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    hf = spectrum[freqs >= 2000.0]
    hf_onset = librosa.onset.onset_strength(S=hf, sr=sr, hop_length=HOP)
    if len(times) == 0 or len(hf_onset) == 0:
        strong: list[float] = []
    else:
        idx = np.clip(beat_frames, 0, len(hf_onset) - 1)
        vals = hf_onset[idx]
        med = float(np.median(vals))
        strong = [t for t, v in zip(times, vals) if v >= med * 1.35 and v > 0]
    return {
        "tempo": tempo_f,
        "beats": times,
        "strong_beats": strong,
        "duration": float(len(y) / sr),
    }


class Ball:
    def __init__(self, x, y, vx, vy, r, hue):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.hue = hue
        self.flash = 0.0


class Particle:
    def __init__(self, x, y, vx, vy, r, hue, life):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.r = r
        self.hue = hue
        self.life = life
        self.max_life = life


def make_scene(width: int, height: int) -> tuple[list[Ball], list[int]]:
    state = [SEED]
    balls = []
    for _ in range(N_BALLS):
        radius = 22.0 + xorshift32(state) * 18.0
        x = radius + xorshift32(state) * (width - 2.0 * radius)
        y = radius + xorshift32(state) * (height - 2.0 * radius)
        ang = xorshift32(state) * math.tau
        speed = 180.0 + xorshift32(state) * 220.0
        hue = xorshift32(state) * 360.0
        balls.append(Ball(x, y, math.cos(ang) * speed, math.sin(ang) * speed, radius, hue))
    return balls, state


def on_beat(ball: Ball) -> None:
    ball.vx = -ball.vx
    ball.vy = -ball.vy
    speed = math.hypot(ball.vx, ball.vy)
    target = min(640.0, max(220.0, speed * 1.12))
    if speed > 1e-6:
        ball.vx *= target / speed
        ball.vy *= target / speed
    ball.hue = (ball.hue + 47.0) % 360.0
    ball.flash = 1.0


def spawn(ball: Ball, state: list[int], particles: list[Particle]) -> None:
    for _ in range(28):
        ang = xorshift32(state) * math.tau
        speed = 220.0 + xorshift32(state) * 520.0
        radius = 6.0 + xorshift32(state) * 8.0
        particles.append(
            Particle(
                ball.x,
                ball.y,
                math.cos(ang) * speed,
                math.sin(ang) * speed - 30.0,
                radius,
                ball.hue,
                0.45,
            )
        )


def step_scene(balls, particles, state, width, height, beat: bool, strong: bool, dt: float) -> float:
    if beat:
        for ball in balls:
            on_beat(ball)
    if strong:
        for ball in balls:
            spawn(ball, state, particles)
    for ball in balls:
        ball.x += ball.vx * dt
        ball.y += ball.vy * dt
        if ball.x < ball.r:
            ball.x = ball.r
            ball.vx = abs(ball.vx)
        elif ball.x > width - ball.r:
            ball.x = width - ball.r
            ball.vx = -abs(ball.vx)
        if ball.y < ball.r:
            ball.y = ball.r
            ball.vy = abs(ball.vy)
        elif ball.y > height - ball.r:
            ball.y = height - ball.r
            ball.vy = -abs(ball.vy)
        ball.flash *= 0.86
    alive = []
    for p in particles:
        p.x += p.vx * dt
        p.y += p.vy * dt
        p.vy += 280.0 * dt
        p.life -= dt
        if p.life > 0:
            alive.append(p)
    particles[:] = alive
    return 1.0 if beat else 0.0


def draw(ctx: cairo.Context, balls, particles, width, height, flash: float) -> None:
    base = (14, 16, 24)
    lift = flash * 36.0
    ctx.set_source_rgb((base[0] + lift) / 255.0, (base[1] + lift) / 255.0, (base[2] + lift) / 255.0)
    ctx.rectangle(0, 0, width, height)
    ctx.fill()
    for p in particles:
        k = max(0.0, p.life / p.max_life)
        r, g, b = hsl_to_rgb(p.hue, 0.80, 0.35 + 0.35 * k)
        ctx.set_source_rgb(r, g, b)
        ctx.arc(p.x, p.y, p.r * (0.35 + 0.65 * k), 0, math.tau)
        ctx.fill()
    for ball in balls:
        r, g, b = hsl_to_rgb(ball.hue, 0.75, 0.55)
        ctx.set_source_rgb(r, g, b)
        ctx.arc(ball.x, ball.y, ball.r, 0, math.tau)
        ctx.fill()
        if ball.flash > 0.4:
            ctx.set_source_rgb(1, 1, 1)
        else:
            sr, sg, sb = hsl_to_rgb(ball.hue, 0.80, 0.75)
            ctx.set_source_rgb(sr, sg, sb)
        ctx.set_line_width(3.0)
        ctx.arc(ball.x, ball.y, ball.r, 0, math.tau)
        ctx.stroke()


def render(path: Path, output: Path, analysis: dict, width: int, height: int) -> None:
    duration = analysis["duration"]
    nframes = int(math.floor(duration * FPS + 1e-6))
    beats = analysis["beats"]
    strong_list = analysis["strong_beats"]
    beat_is_strong = []
    si = 0
    for t in beats:
        while si < len(strong_list) and strong_list[si] < t - 1e-6:
            si += 1
        beat_is_strong.append(si < len(strong_list) and abs(strong_list[si] - t) <= 1e-6)
        if beat_is_strong[-1]:
            si += 1

    balls, state = make_scene(width, height)
    particles: list[Particle] = []
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgra",
            "-s",
            f"{width}x{height}",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    stderr_chunks: list[bytes] = []

    def drain() -> None:
        assert proc.stderr is not None
        stderr_chunks.append(proc.stderr.read())

    thread = threading.Thread(target=drain)
    thread.start()
    bi = 0
    flash = 0.0
    dt = 1.0 / FPS
    try:
        for frame in range(nframes):
            t0 = frame * dt
            t1 = t0 + dt
            beat = False
            strong_now = False
            while bi < len(beats) and beats[bi] < t1:
                if beats[bi] >= t0:
                    beat = True
                    strong_now = strong_now or beat_is_strong[bi]
                bi += 1
            pulse = step_scene(balls, particles, state, width, height, beat, strong_now, dt)
            flash = max(flash * 0.86, pulse)
            draw(ctx, balls, particles, width, height, flash)
            surface.flush()
            proc.stdin.write(surface.get_data())
    finally:
        proc.stdin.close()
        thread.join()
        code = proc.wait()
    if code != 0:
        sys.stderr.write(b"".join(stderr_chunks).decode() + "\n")
        raise SystemExit(f"ffmpeg exited {code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beats-out", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    analysis = analyze(args.input)
    t1 = time.perf_counter()
    if not args.analyze_only:
        render(args.input, args.output, analysis, args.width, args.height)
    t2 = time.perf_counter()
    payload = {
        "stack": "python-librosa-cairo",
        "tempo": analysis["tempo"],
        "beats": analysis["beats"],
        "strong_beats": analysis["strong_beats"],
        "duration": analysis["duration"],
        "timing": {
            "analysis_s": t1 - t0,
            "render_encode_s": 0.0 if args.analyze_only else t2 - t1,
            "total_s": t2 - t0,
        },
    }
    args.beats_out.parent.mkdir(parents=True, exist_ok=True)
    args.beats_out.write_text(json.dumps(payload) + "\n")
    print(json.dumps(payload["timing"]))


if __name__ == "__main__":
    main()
