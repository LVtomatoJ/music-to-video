"""Rain on a moonlit pond. Each onset is a raindrop that lands exactly on its time.

The drop falls as a short streak along the current wind, splashes, and sends
out concentric ripples. Loudness sets the drop size and brightness, the tempo
sets how fast the rings travel and how far apart they sit, and strong beats
bring bigger drops, a crown splash, and a brief brightening of the water.

Everything is closed-form in time. A drop's position, life and ring speed are
fixed when the scene is built, so frame N always looks the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cairo
import librosa
import numpy as np

from music2video.analysis import Analysis, section_at
from music2video.scene import xorshift32

HOP = 512
FALL = 0.30          # seconds a drop is visible before it lands
MAX_LIVE = 22        # concurrent ripple sets before older ones are hurried out
MOON = (0.86, 0.92, 1.0)


def _smooth(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _damp(current: float, target: float, dt: float, rate: float) -> float:
    return target + (current - target) * math.exp(-rate * dt)


@dataclass
class Drop:
    time: float
    u: float            # 0..1 across the pond
    d: float            # 0 far (horizon) .. 1 near (bottom of frame)
    size: float         # 0.5..1.9, follows loudness
    strong: bool
    life: float         # seconds the rings stay visible
    rings: int
    speed: float        # ring speed in surface px per second at scale 1
    gap: float          # seconds between rings
    seed: float
    kind: str = "note"  # note, beat, echo


@dataclass
class RainScene:
    form: str
    palette_name: str
    width: float
    height: float
    duration: float
    horizon: float
    drops: list[Drop]
    beats: list[float]
    strong: list[float]
    energy_t: np.ndarray
    energy_v: np.ndarray
    wind_t: np.ndarray
    wind_v: np.ndarray
    streaks: list[tuple[float, float, float, float]]
    shore: list[tuple[float, float]]
    timing: dict
    sync_report: dict
    analysis: Analysis
    caption_title: str = ""
    caption_artist: str = ""
    time: float = 0.0
    cam_x: float = 0.0
    cam_y: float = 0.0
    cam_zoom: float = 1.0
    _cam_x: float = 0.0
    _cam_y: float = 0.0
    _cam_zoom: float = 1.0
    _first: int = 0

    # --- geometry -------------------------------------------------------
    def project(self, u: float, d: float) -> tuple[float, float, float, float]:
        """Surface point to screen x, y, scale, and ring aspect (y squash)."""
        depth = 0.16 + 0.84 * d
        y = self.horizon + (self.height - self.horizon) * (d ** 1.35) * 1.04
        scale = 0.32 + 1.0 * depth
        # Spread x out in front so the near water is wider than the far shore.
        x = self.width * 0.5 + (u - 0.5) * self.width * (0.78 + 0.5 * d)
        aspect = 0.22 + 0.24 * d
        return x, y, scale, aspect

    def energy(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        return float(np.interp(when, self.energy_t, self.energy_v))

    def wind(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        return float(np.interp(when, self.wind_t, self.wind_v))

    def beat_glow(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        level = 0.0
        for hit in self.strong:
            age = when - hit
            if age < -0.05:
                break
            if age < 1.4:
                if age < 0.0:
                    level = max(level, _smooth((age + 0.05) / 0.05))
                else:
                    level = max(level, math.exp(-age * 2.6))
        return level

    def beat_push(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        level = 0.0
        for hit in self.beats:
            age = when - hit
            if age < 0.0:
                break
            if age < 0.9:
                level = max(level, math.exp(-age * 5.0) * (1.0 - age / 0.9))
        return level

    def live(self, time: float | None = None) -> list[Drop]:
        when = self.time if time is None else time
        out = []
        for drop in self.drops:
            if drop.time - FALL > when:
                break
            if when - drop.time <= drop.life:
                out.append(drop)
        return out

    def update(self, dt: float, time: float, events: list[tuple[float, bool]]) -> None:
        del events
        self.time = time
        push = self.beat_push(time)
        energy = self.energy(time)
        tx = 14.0 * math.sin(time * 0.07) + self.wind(time) * 10.0
        ty = 6.0 * math.cos(time * 0.05)
        tz = 1.0 + 0.018 * energy
        self._cam_x = _damp(self._cam_x, tx, dt, 0.9)
        self._cam_y = _damp(self._cam_y, ty, dt, 0.9)
        self._cam_zoom = _damp(self._cam_zoom, tz, dt, 1.2)
        self.cam_x, self.cam_y = self._cam_x, self._cam_y
        self.cam_zoom = self._cam_zoom + 0.008 * push + 0.010 * self.beat_glow(time)


# --- build ---------------------------------------------------------------

def _onsets(analysis: Analysis) -> tuple[list[float], list[float]]:
    if not analysis.wav_path.is_file():
        return list(analysis.beats), [0.6] * len(analysis.beats)
    y, sr = librosa.load(str(analysis.wav_path), sr=analysis.sample_rate, mono=True)
    if len(y) < 4096:
        return list(analysis.beats), [0.6] * len(analysis.beats)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, units="frames", delta=0.04, wait=4)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    out_t, out_v = [], []
    hi = float(np.percentile(env, 99)) + 1e-9
    rhi = float(np.percentile(rms, 98)) + 1e-9
    for frame, when in zip(frames, times):
        f = int(min(frame, len(env) - 1))
        r = float(rms[min(f + 2, len(rms) - 1)]) / rhi
        strength = 0.55 * min(1.0, float(env[f]) / hi) + 0.45 * min(1.0, r)
        out_t.append(float(when))
        out_v.append(strength)
    return out_t, out_v


def _energy_curve(analysis: Analysis) -> tuple[np.ndarray, np.ndarray]:
    if analysis.rms_times and analysis.rms:
        t = np.asarray(analysis.rms_times, dtype=float)
        v = np.asarray(analysis.rms, dtype=float)
    else:
        t = np.asarray([0.0, max(analysis.duration, 1.0)])
        v = np.asarray([0.4, 0.4])
    v = v / (np.percentile(v, 97) + 1e-9)
    # A gentle 1.2 s smoothing so the rain density breathes, not flickers.
    if len(v) > 3:
        step = max(1e-3, float(np.median(np.diff(t))))
        k = max(1, int(round(1.2 / step)))
        kernel = np.hanning(2 * k + 1)
        kernel /= kernel.sum()
        v = np.convolve(np.pad(v, k, mode="edge"), kernel, mode="valid")
    return t, np.clip(v, 0.0, 1.0)


def _wind_curve(analysis: Analysis, strong: list[float], rng: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Base slant plus gusts on every few strong beats and on the way into loud sections."""
    duration = max(analysis.duration, 1.0)
    t = np.linspace(0.0, duration, int(duration * 20) + 2)
    base = 0.10 + 0.05 * np.sin(t * 0.21)
    gusts: list[float] = []
    last = -99.0
    for index, hit in enumerate(strong):
        if hit - last >= 5.0 and index % 3 == 0:
            gusts.append(hit)
            last = hit
    prev = None
    for section in analysis.sections:
        if prev is not None and section.energy > prev.energy * 1.25 and section.start - last >= 2.5:
            gusts.append(section.start)
            last = section.start
        prev = section
    v = base.copy()
    for hit in gusts:
        sign = 1.0 if xorshift32(rng) > 0.35 else -1.0
        age = t - hit
        shape = np.where(age < 0, 0.0, (1.0 - np.exp(-age * 5.0)) * np.exp(-np.maximum(age, 0) * 0.7))
        v += sign * 0.22 * shape
    return t, v


def _place(drops: list[Drop], when: float, size: float, rng: list[int], energy: float) -> tuple[float, float]:
    """Best of many random candidates: far from recent drops, never on a grid."""
    recent = [drop for drop in drops[-40:] if when - drop.time < 2.6]
    best = (0.5, 0.6)
    best_score = -1e9
    for _ in range(28):
        d = 0.08 + 0.74 * (xorshift32(rng) ** 0.8)
        # Near drops have big rings, so they stay further from the side edges.
        margin = 0.05 + 0.07 * d
        u = margin + (1.0 - 2.0 * margin) * xorshift32(rng)
        score = 0.0
        for drop in recent:
            age = when - drop.time
            weight = math.exp(-age * 1.1)
            du = (u - drop.u) * 1.9
            dd = (d - drop.d)
            dist = math.hypot(du, dd)
            score -= weight / (dist * dist + 0.004)
        # Gentle preference for the moonlit middle so the picture has a center.
        score -= 0.6 * abs(u - 0.55) + 0.35 * abs(d - 0.55)
        score += 0.08 * xorshift32(rng)
        if score > best_score:
            best_score = score
            best = (u, d)
    return best


def build_rain(width: int, height: int, analysis: Analysis, seed: int) -> RainScene:
    rng = [seed * 2654435761 & 0xFFFFFFFF or 7]
    beats = [float(b) for b in analysis.beats]
    strong = [float(b) for b in analysis.strong_beats]
    strong_set = set(round(b, 4) for b in strong)
    period = float(np.median(np.diff(beats))) if len(beats) > 2 else 0.6
    period = min(1.2, max(0.3, period))
    energy_t, energy_v = _energy_curve(analysis)
    on_t, on_v = _onsets(analysis)

    # One event list: onsets with their strength, beats that have no onset nearby.
    events: list[tuple[float, float, bool, str]] = []
    for when, value in zip(on_t, on_v):
        events.append((when, value, False, "note"))
    for beat in beats:
        is_strong = round(beat, 4) in strong_set
        near = [i for i, (w, _, _, _) in enumerate(events) if abs(w - beat) < 0.07]
        if near:
            i = near[0]
            w, v, _, k = events[i]
            # The drop lands on the beat itself, so the splash is the downbeat.
            events[i] = (beat, max(v, 0.5), is_strong, "beat")
        else:
            level = float(np.interp(beat, energy_t, energy_v))
            events.append((beat, 0.35 + 0.4 * level, is_strong, "beat"))
    events.sort(key=lambda item: item[0])
    merged: list[tuple[float, float, bool, str]] = []
    for item in events:
        if merged and item[0] - merged[-1][0] < 0.045:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], item[1]), prev[2] or item[2], prev[3])
            continue
        merged.append(item)

    drops: list[Drop] = []
    speed_base = 96.0 * (0.62 / period) ** 0.5
    gap = min(0.24, max(0.11, period * 0.22))
    for when, value, is_strong, kind in merged:
        energy = float(np.interp(when, energy_t, energy_v))
        size = 0.55 + 0.95 * value + (0.45 if is_strong else 0.0)
        size = min(1.95, size)
        u, d = _place(drops, when, size, rng, energy)
        rings = 3 + (1 if size > 1.15 else 0) + (1 if is_strong else 0)
        life = 2.6 + 1.6 * min(1.0, size / 1.6)
        drops.append(Drop(when, u, d, size, is_strong, life, rings, speed_base * (0.8 + 0.35 * size), gap, xorshift32(rng), kind))
        # A loud drop throws one small secondary droplet that lands a moment later.
        if is_strong and size > 1.3:
            back = 0.16 + 0.08 * xorshift32(rng)
            ang = xorshift32(rng) * math.tau
            du, dd = math.cos(ang) * 0.035, math.sin(ang) * 0.05
            drops.append(Drop(when + back, min(0.96, max(0.04, u + du)), min(0.98, max(0.08, d + dd)), 0.42, False,
                              1.3, 2, speed_base * 0.8, gap, xorshift32(rng), "echo"))
    # Light rain between the notes: tiny faint drops, more of them when the song is louder.
    when = 0.15
    while when < analysis.duration - 0.2:
        energy = float(np.interp(when, energy_t, energy_v))
        rate = 1.6 + 5.0 * energy
        when += -math.log(max(1e-4, xorshift32(rng))) / rate
        u, d = _place(drops, when, 0.3, rng, energy)
        drops.append(Drop(when, u, d, 0.26 + 0.14 * xorshift32(rng), False, 1.5 + 0.6 * xorshift32(rng), 2,
                          speed_base * 0.72, gap * 0.8, xorshift32(rng), "mist"))
        drops.sort(key=lambda drop: drop.time)
    drops.sort(key=lambda drop: drop.time)

    # Hurry older ripples out when the pond gets busy, so dense bars stay clean.
    for index, drop in enumerate(drops):
        later = [other for other in drops[index + 1:index + 120] if other.time < drop.time + drop.life and other.kind != "mist"]
        alive_peak = len(later)
        if alive_peak > MAX_LIVE * 0.6:
            drop.life *= max(0.55, (MAX_LIVE * 0.6) / alive_peak)

    wind_t, wind_v = _wind_curve(analysis, strong, rng)
    streaks = []
    for _ in range(260):
        streaks.append((xorshift32(rng), xorshift32(rng), 0.6 + 0.8 * xorshift32(rng), xorshift32(rng)))
    shore = []
    x = 0.0
    phase = xorshift32(rng) * 10
    while x <= 1.02:
        h = 0.5 + 0.28 * math.sin(x * 7.1 + phase) + 0.18 * math.sin(x * 17.3 + phase * 1.7) + 0.08 * math.sin(x * 41.0)
        shore.append((x, h))
        x += 0.006
    horizon = height * 0.27
    scene = RainScene(
        form="rain", palette_name="rain", width=float(width), height=float(height),
        duration=max(analysis.duration, 1e-3), horizon=horizon, drops=drops, beats=beats, strong=strong,
        energy_t=energy_t, energy_v=energy_v, wind_t=wind_t, wind_v=wind_v, streaks=streaks, shore=shore,
        timing={}, sync_report={}, analysis=analysis,
    )
    main = [drop for drop in drops if drop.kind in ("note", "beat")]
    scene.timing = {"max_timing_error_s": 0.0, "max_position_error_px": 0.0, "hits": len(main)}
    scene.sync_report = {
        "drops": len(main),
        "echo_drops": sum(1 for drop in drops if drop.kind == "echo"),
        "mist_drops": sum(1 for drop in drops if drop.kind == "mist"),
        "strong_drops": sum(1 for drop in drops if drop.strong),
        "peak_live": _peak_live(drops),
        "min_landing_gap_px": round(_min_landing_gap(scene), 1),
    }
    scene._cam_zoom = scene.cam_zoom = 1.0
    return scene


def _peak_live(drops: list[Drop]) -> int:
    peak = 0
    for drop in drops:
        live = sum(1 for other in drops if other.time <= drop.time <= other.time + other.life)
        peak = max(peak, live)
    return peak


def _min_landing_gap(scene: RainScene) -> float:
    """Smallest screen distance between two drops landing within 0.6 s of each other."""
    worst = 1e9
    drops = scene.drops
    for i, a in enumerate(drops):
        ax, ay, _, _ = scene.project(a.u, a.d)
        for b in drops[i + 1:]:
            if b.time - a.time > 0.6:
                break
            if b.kind in ("echo", "mist") or a.kind in ("echo", "mist"):
                continue
            bx, by, _, _ = scene.project(b.u, b.d)
            worst = min(worst, math.hypot(ax - bx, ay - by))
    return worst if worst < 1e9 else 0.0


# --- paint ---------------------------------------------------------------

def paint_rain(ctx, scene: RainScene) -> None:
    w, h = scene.width, scene.height
    t = scene.time
    energy = scene.energy()
    glow = scene.beat_glow()
    live = scene.live()

    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source_rgb(0.01, 0.02, 0.04)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)

    ctx.save()
    ctx.translate(w / 2.0, h / 2.0)
    ctx.scale(scene.cam_zoom, scene.cam_zoom)
    ctx.translate(-w / 2.0 + scene.cam_x, -h / 2.0 + scene.cam_y)
    pad = 80.0
    _paint_sky(ctx, scene, w, h, pad, glow)
    _paint_water(ctx, scene, w, h, pad, energy, glow)
    _paint_swell(ctx, scene, w, pad, energy)
    _paint_moon_path(ctx, scene, live, energy, glow)
    ctx.set_operator(cairo.OPERATOR_ADD)
    for drop in live:
        _paint_ripples(ctx, scene, drop, glow)
    ctx.set_operator(cairo.OPERATOR_OVER)
    for drop in live:
        _paint_splash(ctx, scene, drop)
    _paint_rain_streaks(ctx, scene, w, h, energy)
    for drop in live:
        _paint_falling(ctx, scene, drop)
    ctx.restore()


def _moon_xy(scene: RainScene) -> tuple[float, float]:
    return scene.width * 0.66, scene.horizon * 0.40


def _paint_sky(ctx, scene: RainScene, w, h, pad, glow) -> None:
    hz = scene.horizon
    sky = cairo.LinearGradient(0, -pad, 0, hz)
    sky.add_color_stop_rgb(0.0, 0.020, 0.032, 0.075)
    sky.add_color_stop_rgb(0.65, 0.050, 0.085, 0.150)
    sky.add_color_stop_rgb(1.0, 0.090, 0.140, 0.205)
    ctx.set_source(sky)
    ctx.rectangle(-pad, -pad, w + 2 * pad, hz + pad + 2)
    ctx.fill()
    mx, my = _moon_xy(scene)
    halo = cairo.RadialGradient(mx, my, 0, mx, my, 420)
    halo.add_color_stop_rgba(0.0, 0.70, 0.82, 0.95, 0.30 + 0.10 * glow)
    halo.add_color_stop_rgba(0.12, 0.45, 0.62, 0.82, 0.16 + 0.06 * glow)
    halo.add_color_stop_rgba(0.45, 0.20, 0.32, 0.50, 0.06)
    halo.add_color_stop_rgba(1.0, 0.1, 0.2, 0.3, 0.0)
    ctx.set_source(halo)
    ctx.rectangle(-pad, -pad, w + 2 * pad, hz + pad + 2)
    ctx.fill()
    disc = cairo.RadialGradient(mx - 6, my - 6, 2, mx, my, 34)
    disc.add_color_stop_rgba(0.0, 1.0, 0.99, 0.95, 0.98)
    disc.add_color_stop_rgba(0.8, 0.88, 0.93, 1.0, 0.92)
    disc.add_color_stop_rgba(1.0, 0.80, 0.88, 1.0, 0.0)
    ctx.set_source(disc)
    ctx.arc(mx, my, 34, 0, math.tau)
    ctx.fill()
    # Thin drifting cloud veils across the moon.
    t = scene.time
    for index in range(4):
        cx = (mx - 520 + ((t * (9 + index * 3) + index * 380) % 1100))
        cy = my - 30 + index * 26
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(1.0, 0.16)
        veil = cairo.RadialGradient(0, 0, 0, 0, 0, 260 + index * 40)
        veil.add_color_stop_rgba(0, 0.30, 0.38, 0.50, 0.16)
        veil.add_color_stop_rgba(1, 0.30, 0.38, 0.50, 0.0)
        ctx.set_source(veil)
        ctx.arc(0, 0, 260 + index * 40, 0, math.tau)
        ctx.fill()
        ctx.restore()
    # Far shore: low hills and a tree line, darker than the sky.
    ctx.move_to(-pad, hz + 2)
    for x, hgt in scene.shore:
        ctx.line_to(-pad + x * (w + 2 * pad), hz - 6 - hgt * 34)
    ctx.line_to(w + pad, hz + 2)
    ctx.close_path()
    ctx.set_source_rgb(0.018, 0.030, 0.052)
    ctx.fill()


def _paint_water(ctx, scene: RainScene, w, h, pad, energy, glow) -> None:
    hz = scene.horizon
    water = cairo.LinearGradient(0, hz, 0, h + pad)
    lift = 0.026 * glow + 0.012 * energy
    water.add_color_stop_rgb(0.0, 0.080 + lift, 0.135 + lift, 0.180 + lift)
    water.add_color_stop_rgb(0.18, 0.036 + lift, 0.088 + lift, 0.120 + lift)
    water.add_color_stop_rgb(0.6, 0.016 + lift * 0.5, 0.050 + lift, 0.075 + lift)
    water.add_color_stop_rgb(1.0, 0.006, 0.022, 0.036)
    ctx.set_source(water)
    ctx.rectangle(-pad, hz, w + 2 * pad, h - hz + pad)
    ctx.fill()
    # Shore reflection: the hills mirrored, faint, just under the horizon.
    ctx.move_to(-pad, hz)
    for x, hgt in scene.shore:
        ctx.line_to(-pad + x * (w + 2 * pad), hz + 4 + hgt * 20)
    ctx.line_to(w + pad, hz)
    ctx.close_path()
    ctx.set_source_rgba(0.010, 0.020, 0.036, 0.55)
    ctx.fill()
    # Broad moonlit sheen on the water, breathing with the strong beats.
    mx, _ = _moon_xy(scene)
    ctx.save()
    ctx.translate(mx, hz + (h - hz) * 0.42)
    ctx.scale(1.0, 0.55)
    sheen = cairo.RadialGradient(0, 0, 0, 0, 0, 760)
    a = 0.12 + 0.17 * glow + 0.06 * energy
    sheen.add_color_stop_rgba(0.0, 0.40, 0.58, 0.70, a)
    sheen.add_color_stop_rgba(0.5, 0.20, 0.36, 0.48, a * 0.4)
    sheen.add_color_stop_rgba(1.0, 0.1, 0.2, 0.3, 0.0)
    ctx.set_source(sheen)
    ctx.arc(0, 0, 760, 0, math.tau)
    ctx.fill()
    ctx.restore()


def _displacement(scene: RainScene, live: list[Drop], x: float, y: float) -> float:
    """Horizontal shove of the moon path at (x, y) from the rings passing under it."""
    t = scene.time
    shove = 0.0
    for drop in live:
        age = t - drop.time
        if age <= 0.0:
            continue
        px, py, scale, aspect = scene.project(drop.u, drop.d)
        dx = (x - px) / scale
        dy = (y - py) / (scale * aspect)
        dist = math.hypot(dx, dy)
        front = drop.speed * age
        if dist > front + 30 or dist < 1e-3:
            continue
        fade = max(0.0, 1.0 - age / drop.life)
        wave = math.sin((front - dist) * 0.16) * math.exp(-max(0.0, front - dist) / 70.0)
        shove += wave * drop.size * 9.0 * fade * (dx / dist) * scale
    return max(-26.0, min(26.0, shove))


def _paint_moon_path(ctx, scene: RainScene, live, energy, glow) -> None:
    """The moon's reflection as broken horizontal glints that the ripples push around."""
    mx, _ = _moon_xy(scene)
    hz = scene.horizon
    h = scene.height
    t = scene.time
    y = hz + 3.0
    row = 0
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    while y < h + 60:
        d = min(1.0, (y - hz) / (h - hz))
        spread = 26 + 250 * d
        step = 2.4 + 9.0 * d
        for k in range(3):
            seed = (row * 7.31 + k * 3.17)
            off = math.sin(seed * 12.9898) * 43758.5453
            off = (off - math.floor(off)) * 2 - 1
            cx = mx + off * spread * 0.8 + math.sin(t * 0.9 + seed) * 6 * (0.4 + d)
            cx += _displacement(scene, live, cx, y)
            length = (10 + 70 * d) * (0.5 + 0.5 * abs(math.sin(t * 1.3 + seed * 1.7)))
            fall = 1.0 - min(1.0, abs(cx - mx) / (spread + 1))
            alpha = (0.05 + 0.20 * (1 - d) ** 0.6) * fall * (0.80 + 0.65 * glow + 0.25 * energy)
            if alpha < 0.008:
                continue
            ctx.set_source_rgba(0.62, 0.72, 0.82, alpha)
            ctx.set_line_width(0.9 + 2.4 * d)
            ctx.move_to(cx - length / 2, y)
            ctx.line_to(cx + length / 2, y)
            ctx.stroke()
        y += step
        row += 1
    ctx.set_operator(cairo.OPERATOR_OVER)


def _ring_alpha(drop: Drop, age: float, k: int) -> float:
    local = age - k * drop.gap
    if local <= 0.0:
        return 0.0
    life = drop.life - k * drop.gap * 0.6
    if local >= life:
        return 0.0
    attack = _smooth(local / 0.08)
    fade = (1.0 - local / life) ** 1.6
    return attack * fade * (1.0 - 0.16 * k)


def _paint_ripples(ctx, scene: RainScene, drop: Drop, glow: float) -> None:
    age = scene.time - drop.time
    if age <= 0.0:
        return
    x, y, scale, aspect = scene.project(drop.u, drop.d)
    bright = (0.58 + 0.40 * min(1.0, drop.size / 1.5)) * (1.0 + 0.30 * glow)
    if drop.kind == "mist":
        bright *= 0.55
    for k in range(drop.rings):
        alpha = _ring_alpha(drop, age, k) * bright
        if alpha < 0.01:
            continue
        local = age - k * drop.gap
        r = drop.speed * local * (1.0 - 0.10 * k) + 2.0
        r *= (0.55 + 0.45 * drop.size)
        width = (1.1 + 1.6 * drop.size) * (1.0 + local * 0.6) / (1.0 + 0.4 * k)
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(scale, scale * aspect)
        # Far side of each ring catches the sky and is brighter than the near side.
        grad = cairo.LinearGradient(0, -r, 0, r)
        grad.add_color_stop_rgba(0.0, 0.78, 0.88, 0.98, alpha)
        grad.add_color_stop_rgba(0.5, 0.42, 0.66, 0.78, alpha * 0.55)
        grad.add_color_stop_rgba(1.0, 0.30, 0.55, 0.66, alpha * 0.32)
        ctx.set_source(grad)
        ctx.set_line_width(width)
        ctx.new_path()
        ctx.arc(0, 0, r, 0, math.tau)
        ctx.stroke()
        # A softer, wider swell just inside the crest gives each ring some body.
        if r > width * 2.5:
            soft = cairo.LinearGradient(0, -r, 0, r)
            soft.add_color_stop_rgba(0.0, 0.40, 0.58, 0.72, alpha * 0.20)
            soft.add_color_stop_rgba(1.0, 0.20, 0.40, 0.52, alpha * 0.08)
            ctx.set_source(soft)
            ctx.set_line_width(width * 3.4)
            ctx.new_path()
            ctx.arc(0, 0, max(0.5, r - width * 2.0), 0, math.tau)
            ctx.stroke()
        ctx.restore()


def _paint_splash(ctx, scene: RainScene, drop: Drop) -> None:
    age = scene.time - drop.time
    if age < 0.0 or age > 0.7:
        return
    x, y, scale, aspect = scene.project(drop.u, drop.d)
    s = scale * (0.55 + 0.45 * drop.size)
    # Flash where the drop hits.
    flash = math.exp(-age * 9.0) * (0.35 if drop.kind == "mist" else 1.0)
    if flash > 0.02:
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(1.0, 0.5 + aspect)
        rad = 40 * s * (1.0 + 0.7 * drop.strong)
        g = cairo.RadialGradient(0, 0, 0, 0, 0, rad)
        g.add_color_stop_rgba(0.0, 0.95, 0.98, 1.0, 0.95 * flash)
        g.add_color_stop_rgba(0.18, 0.70, 0.88, 1.0, 0.55 * flash)
        g.add_color_stop_rgba(0.5, 0.45, 0.68, 0.85, 0.18 * flash)
        g.add_color_stop_rgba(1.0, 0.3, 0.5, 0.7, 0.0)
        ctx.set_source(g)
        ctx.arc(0, 0, rad, 0, math.tau)
        ctx.fill()
        ctx.restore()
    # The first wave leaves fast and bright, so the hit reads at a glance.
    if drop.kind != "mist" and age < 0.32:
        u = age / 0.32
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(s, s * aspect)
        ctx.set_source_rgba(0.85, 0.95, 1.0, 0.75 * (1 - u) ** 1.5)
        ctx.set_line_width(2.6 * (1 - u) + 0.6)
        ctx.new_path()
        ctx.arc(0, 0, 6 + 70 * (1 - (1 - u) ** 2.2), 0, math.tau)
        ctx.restore()
        ctx.stroke()
    # Crown droplets on parabolas, more for loud and strong drops.
    count = 0 if drop.kind in ("echo", "mist") else 3 + int(drop.size * 2.2) + (3 if drop.strong else 0)
    gravity = 1500.0
    for i in range(count):
        jitter = math.sin((i + 1) * 12.9898 + drop.seed * 78.233) * 43758.5453
        jitter = jitter - math.floor(jitter)
        ang = drop.seed * math.tau + i * (math.tau / max(1, count)) + 0.9 * (jitter - 0.5)
        spd = (60 + 90 * jitter) * s
        up = (150 + 160 * ((jitter * 7.13 + drop.seed) % 1.0)) * s * (1.2 if drop.strong else 1.0)
        tt = age
        lift = up * tt - 0.5 * gravity * s * tt * tt
        if lift < 0 and tt > 0.05:
            continue
        px = x + math.cos(ang) * spd * tt
        py = y + math.sin(ang) * spd * tt * aspect - lift
        a = (1.0 - tt / 0.7) * (0.45 + 0.4 * jitter)
        ctx.set_source_rgba(0.86, 0.94, 1.0, a)
        ctx.arc(px, py, (0.7 + 0.8 * jitter + 0.4 * drop.size) * scale, 0, math.tau)
        ctx.fill()
    # A short rebound jet for the biggest drops.
    if drop.strong and drop.size > 1.2 and 0.05 < age < 0.42:
        u = (age - 0.05) / 0.37
        hgt = math.sin(u * math.pi) * 34 * s
        jw = 2.4 * s
        g = cairo.LinearGradient(x, y - hgt, x, y)
        g.add_color_stop_rgba(0, 0.95, 0.98, 1.0, 0.9 * (1 - u * 0.5))
        g.add_color_stop_rgba(1, 0.55, 0.75, 0.9, 0.0)
        ctx.set_source(g)
        ctx.move_to(x - jw, y)
        ctx.curve_to(x - jw * 0.6, y - hgt * 0.6, x - jw * 0.4, y - hgt, x, y - hgt - jw)
        ctx.curve_to(x + jw * 0.4, y - hgt, x + jw * 0.6, y - hgt * 0.6, x + jw, y)
        ctx.close_path()
        ctx.fill()
        ctx.set_source_rgba(0.95, 0.98, 1.0, 0.8 * (1 - u))
        ctx.arc(x, y - hgt - jw * 2.2, jw * 0.9, 0, math.tau)
        ctx.fill()


def _paint_falling(ctx, scene: RainScene, drop: Drop) -> None:
    age = scene.time - drop.time
    if age >= 0.0 or age < -FALL or drop.kind == "mist":
        return
    x, y, scale, aspect = scene.project(drop.u, drop.d)
    u = (age + FALL) / FALL          # 0 at appearance, 1 at impact
    height = 520 * scale             # how far above its landing point it starts
    slant = scene.wind(drop.time)
    lift = height * (1.0 - u) ** 1.25
    hx = x + slant * lift
    hy = y - lift
    length = (46 + 30 * drop.size) * scale * (0.6 + 0.4 * u)
    tx = hx + slant * length
    ty = hy - length
    alpha = _smooth(u / 0.35) * (0.55 + 0.3 * min(1.0, drop.size))
    g = cairo.LinearGradient(tx, ty, hx, hy)
    g.add_color_stop_rgba(0, 0.75, 0.86, 1.0, 0.0)
    g.add_color_stop_rgba(1, 0.92, 0.97, 1.0, alpha)
    ctx.set_source(g)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_width((1.3 + 0.9 * drop.size) * scale)
    ctx.move_to(tx, ty)
    ctx.line_to(hx, hy)
    ctx.stroke()
    # Its reflection rises to meet it.
    ry = y + lift * aspect * 0.9
    rl = length * aspect * 0.9
    g2 = cairo.LinearGradient(hx, ry + rl, hx, ry)
    g2.add_color_stop_rgba(0, 0.5, 0.7, 0.85, 0.0)
    g2.add_color_stop_rgba(1, 0.6, 0.8, 0.95, alpha * 0.35)
    ctx.set_source(g2)
    ctx.set_line_width((1.0 + 0.6 * drop.size) * scale)
    ctx.move_to(hx, ry + rl)
    ctx.line_to(hx, ry)
    ctx.stroke()


def _paint_rain_streaks(ctx, scene: RainScene, w, h, energy) -> None:
    """Fine background rain. Density follows loudness, slant follows the wind."""
    t = scene.time
    slant = scene.wind()
    count = int(60 + 200 * energy)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for index, (sx, sy, depth, phase) in enumerate(scene.streaks[:count]):
        speed = 900 + 700 * depth
        span = h + 260
        yy = (sy * span + t * speed) % span - 130
        xx = (sx * (w + 400) - 200) + slant * (yy - h * 0.5) + phase * 3
        length = (22 + 30 * depth) * (0.8 + 0.5 * energy)
        alpha = (0.03 + 0.065 * depth) * (0.75 + 0.7 * energy)
        ctx.set_source_rgba(0.70, 0.80, 0.92, alpha)
        ctx.set_line_width(0.6 + 0.6 * depth)
        ctx.move_to(xx, yy)
        ctx.line_to(xx - slant * length, yy - length)
        ctx.stroke()


def _paint_swell(ctx, scene: RainScene, w, pad, energy) -> None:
    """Long, faint swell lines drifting toward the viewer, so the water is never still."""
    hz = scene.horizon
    h = scene.height
    t = scene.time
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    lines = 34
    for i in range(lines):
        phase = ((i + t * 0.05) / lines) % 1.0
        d = phase ** 1.6
        y = hz + (h - hz) * d * 1.04 + 2
        seed = math.sin(i * 91.7) * 1000.0
        seed -= math.floor(seed)
        alpha = 0.035 * math.sin(phase * math.pi) * (0.7 + 0.5 * energy)
        ctx.set_line_width(0.8 + 2.4 * d)
        segs = 3 + int(seed * 3)
        for k in range(segs):
            cx = -pad + (w + 2 * pad) * ((seed * 3.7 + k / segs + 0.02 * math.sin(t * 0.3 + i)) % 1.0)
            half = (80 + 260 * d) * (0.5 + 0.5 * ((seed * 13.1 + k * 0.37) % 1.0))
            g = cairo.LinearGradient(cx - half, 0, cx + half, 0)
            g.add_color_stop_rgba(0.0, 0.40, 0.60, 0.70, 0.0)
            g.add_color_stop_rgba(0.5, 0.40, 0.60, 0.70, alpha)
            g.add_color_stop_rgba(1.0, 0.40, 0.60, 0.70, 0.0)
            ctx.set_source(g)
            wob = 3.0 * d * math.sin(t * 0.8 + k + i)
            ctx.move_to(cx - half, y + wob)
            ctx.curve_to(cx - half * 0.3, y - 2 * d + wob, cx + half * 0.3, y + 2 * d - wob, cx + half, y - wob)
            ctx.stroke()
    ctx.set_operator(cairo.OPERATOR_OVER)
