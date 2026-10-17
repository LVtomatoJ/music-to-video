"""Beat-synced forms other than the ball map: a growing city, a note waterfall, a sky of constellations.

Each form is closed-form in time. A building finishes on its beat, a note head
meets its key on the onset, and a star reaches full brightness on its note.
The camera only eases toward a slow target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cairo
import numpy as np

from music2video.analysis import Analysis, section_at
from music2video.notes import Tone, extract_tones
from music2video.registry import active_palette
from music2video.scene import xorshift32

NOTE_COLORS = (
    (0.95, 0.45, 0.48),
    (0.96, 0.62, 0.38),
    (0.96, 0.82, 0.42),
    (0.62, 0.86, 0.46),
    (0.38, 0.82, 0.72),
    (0.38, 0.72, 0.95),
    (0.45, 0.55, 0.95),
    (0.62, 0.48, 0.95),
    (0.82, 0.45, 0.90),
    (0.95, 0.45, 0.72),
    (0.90, 0.55, 0.48),
    (0.78, 0.80, 0.86),
)
WHITES = {0, 2, 4, 5, 7, 9, 11}


def _smooth(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _damp(current: float, target: float, dt: float, rate: float) -> float:
    return current + (target - current) * (1.0 - math.exp(-rate * dt))


def _round_rect(ctx, x: float, y: float, w: float, h: float, r: float) -> None:
    if w <= 0.0 or h <= 0.0:
        return
    r = min(r, w * 0.5, h * 0.5)
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2.0, 0.0)
    ctx.arc(x + w - r, y + h - r, r, 0.0, math.pi / 2.0)
    ctx.arc(x + r, y + h - r, r, math.pi / 2.0, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3.0 * math.pi / 2.0)
    ctx.close_path()


def _median_gap(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    gaps = sorted(abs(times[i] - times[i - 1]) for i in range(1, len(times)))
    return gaps[len(gaps) // 2]


def _nearest(times: list[float], when: float) -> float:
    if not times:
        return 0.0
    return min(abs(item - when) for item in times)


@dataclass
class Building:
    born: float
    grow: float
    x: float
    width: float
    target: float
    energy: float
    floors: int
    cols: int
    tower: bool
    roof: int = 0
    tint: int = 0
    sign: int = 0


@dataclass
class WindowLight:
    time: float
    index: int
    col: int
    row: int


@dataclass
class CityScene:
    form: str
    palette_name: str
    width: float
    height: float
    duration: float
    buildings: list[Building]
    lights: list[WindowLight]
    bass_times: list[float]
    beat_times: list[float]
    born_lit: set[tuple[int, int, int]]
    far: list[tuple[float, float, float, int]]
    mid: list[tuple[float, float, float, int]]
    sky_stars: list[tuple[float, float, float, float]]
    timing: dict
    sync_report: dict
    caption_title: str = ""
    caption_artist: str = ""
    time: float = 0.0
    cam_x: float = 0.0
    cam_zoom: float = 1.0
    _cam_x: float = 0.0
    _cam_zoom: float = 1.0

    def building_height(self, building: Building, time: float | None = None) -> float:
        when = self.time if time is None else time
        if when <= building.born - building.grow:
            return 0.0
        u = (when - (building.born - building.grow)) / building.grow
        if u >= 1.0:
            return building.target
        return building.target * _smooth(u)

    def street_glow(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        glow = 0.0
        for hit in self.bass_times:
            age = when - hit
            if 0.0 <= age < 0.48:
                glow = max(glow, _smooth(1.0 - age / 0.48))
        return glow

    def beat_pulse(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        pulse = 0.0
        for hit in self.beat_times:
            age = when - hit
            if 0.0 <= age < 0.18:
                pulse = max(pulse, _smooth(1.0 - age / 0.18))
        return pulse

    def _target(self, time: float) -> tuple[float, float]:
        if not self.buildings:
            return 0.0, 1.0
        span = max(1e-3, self.duration)
        u = _smooth(max(0.0, min(1.0, time / span)))
        first = self.buildings[0]
        start_x = first.x + first.width * 0.5
        end_x = 0.0
        total = (self.buildings[-1].x + self.buildings[-1].width) - self.buildings[0].x
        fit = (self.width * 0.86) / max(total, 1.0)
        end_zoom = min(0.96, fit)
        start_zoom = min(1.48, end_zoom * 1.65)
        zoom = start_zoom + (end_zoom - start_zoom) * u + 0.04 * self.street_glow(time)
        x = start_x + (end_x - start_x) * u
        return x, zoom

    def update(self, dt: float, time: float, events: list[tuple[float, bool]]) -> None:
        del events
        self.time = time
        target_x, target_zoom = self._target(time)
        self._cam_x = _damp(self._cam_x, target_x, dt, 1.6)
        self._cam_zoom = _damp(self._cam_zoom, target_zoom, dt, 1.6)
        self.cam_x = self._cam_x
        self.cam_zoom = self._cam_zoom


def _low_energy(analysis: Analysis, time: float) -> float:
    if not analysis.rms_times or not analysis.bands:
        return 0.0
    index = min(range(len(analysis.rms_times)), key=lambda i: abs(analysis.rms_times[i] - time))
    row = analysis.bands[index]
    return float(row[0]) if row else 0.0


def _build_city(width: int, height: int, analysis: Analysis, seed: int) -> CityScene:
    rng = [seed & 0xFFFFFFFF or 1]
    beats = [float(beat) for beat in analysis.beats] or [0.0]
    count = min(len(beats), 18 if len(beats) >= 16 else max(1, len(beats)))
    if len(beats) >= 24:
        count = min(18, max(12, len(beats) // 3))
    last = max(0, int(round((len(beats) - 1) * 0.72)))
    if count == 1:
        indexes = [0]
    else:
        indexes = []
        for step in range(count):
            indexes.append(min(len(beats) - 1, int(round(step * last / (count - 1)))))
        # Keep the birth times unique and in order.
        deduped = []
        for index in indexes:
            if deduped and index <= deduped[-1]:
                index = min(len(beats) - 1, deduped[-1] + 1)
            deduped.append(index)
        indexes = deduped
        count = len(indexes)
    buildings: list[Building] = []
    cursor = 0.0
    specs = []
    for index in indexes:
        beat = beats[index]
        section = section_at(analysis.sections, beat)
        energy = section.energy
        block_w = 34.0 + xorshift32(rng) * 92.0
        target = (110.0 + 300.0 * energy) * (0.70 + 0.62 * xorshift32(rng))
        if energy >= 0.55:
            target *= 1.08
        target = min(max(target, 96.0), height * 0.58)
        floors = max(3, min(16, int(target / 32.0)))
        cols = max(2, min(6, int(block_w / 18.0)))
        roof = int(xorshift32(rng) * 5)
        tint = int(xorshift32(rng) * 5)
        sign = int(xorshift32(rng) * 3) if xorshift32(rng) > 0.55 else 0
        specs.append((beat, block_w, target, energy, floors, cols, energy >= 0.62 or roof == 3, roof, tint, sign))
    gaps = [14.0 + xorshift32(rng) * 34.0 for _ in specs]
    total = sum(item[1] for item in specs) + sum(gaps[:-1] if gaps else [])
    cursor = -total / 2.0
    for (beat, block_w, target, energy, floors, cols, tower, roof, tint, sign), gap in zip(specs, gaps):
        buildings.append(
            Building(
                born=beat,
                grow=0.34,
                x=cursor,
                width=block_w,
                target=target,
                energy=energy,
                floors=floors,
                cols=cols,
                tower=tower,
                roof=roof,
                tint=tint,
                sign=sign,
            )
        )
        cursor += block_w + gap
    pattern = (0.58, 1.25, 0.72, 1.4, 0.64, 1.08, 0.8, 1.48, 0.7, 1.16)
    for index, building in enumerate(buildings):
        building.target = min(height * 0.68, max(110.0, building.target * pattern[index % len(pattern)]))
        building.floors = max(3, min(16, int(building.target / 30.0)))
        building.cols = max(2, min(6, int(building.width / 18.0)))
    birth = {building.born for building in buildings}
    others = [beat for beat in beats if beat not in birth]
    cells = [(index, col, row) for index, building in enumerate(buildings) for row in range(building.floors) for col in range(building.cols)]
    order = list(cells)
    for i in range(len(order) - 1, 0, -1):
        swap = int(xorshift32(rng) * (i + 1))
        order[i], order[swap] = order[swap], order[i]
    born_lit: set[tuple[int, int, int]] = set()
    lights: list[WindowLight] = []
    for index, building in enumerate(buildings):
        own = [cell for cell in order if cell[0] == index]
        for cell in own[: max(1, len(own) // 3)]:
            born_lit.add(cell)
            lights.append(WindowLight(building.born, *cell))
    pool = [cell for cell in order if cell not in born_lit]
    for step, beat in enumerate(others):
        if pool:
            cell = pool.pop()
        elif cells:
            cell = cells[step % len(cells)]
        else:
            continue
        lights.append(WindowLight(beat, *cell))
    lows = [_low_energy(analysis, beat) for beat in beats]
    if lows and max(lows) > 0.05:
        ordered = sorted(lows)
        cut = max(0.42, ordered[int(0.62 * (len(ordered) - 1))])
        bass = [beat for beat, level in zip(beats, lows) if level >= cut]
    else:
        bass = []
    if not bass:
        bass = list(analysis.strong_beats) or beats[::4]
    far, mid = _skyline_layers(rng)
    sky_stars = []
    for _ in range(70):
        sky_stars.append(
            (
                xorshift32(rng) * width,
                xorshift32(rng) * height * 0.62,
                0.6 + xorshift32(rng) * 1.3,
                xorshift32(rng),
            )
        )
    scene = CityScene(
        form="city",
        palette_name="city",
        width=float(width),
        height=float(height),
        duration=max(analysis.duration, 1e-3),
        buildings=buildings,
        lights=lights,
        bass_times=bass,
        beat_times=list(beats),
        born_lit=born_lit,
        far=far,
        mid=mid,
        sky_stars=sky_stars,
        timing={},
        sync_report={},
    )
    worst = 0.0
    for building in buildings:
        worst = max(worst, abs(scene.building_height(building, building.born) - building.target))
    scene.timing = {"max_timing_error_s": 0.0, "max_position_error_px": worst, "hits": len(beats)}
    scene.sync_report = {
        "buildings": len(buildings),
        "beats": len(beats),
        "bass": len(bass),
        "window_lights": len(lights),
    }
    scene._cam_x, scene._cam_zoom = scene._target(0.0)
    scene.cam_x, scene.cam_zoom = scene._cam_x, scene._cam_zoom
    return scene


_NIGHT_TONES = (
    (0.09, 0.11, 0.20),
    (0.14, 0.10, 0.22),
    (0.07, 0.09, 0.14),
    (0.16, 0.10, 0.16),
    (0.08, 0.13, 0.17),
)


def _skyline_layers(rng: list[int]) -> tuple[list[tuple[float, float, float, int]], list[tuple[float, float, float, int]]]:
    """Far and middle silhouettes. They are scenery, not beat hits."""
    far: list[tuple[float, float, float, int]] = []
    mid: list[tuple[float, float, float, int]] = []
    x = -2400.0
    while x < 2400.0:
        width = 48.0 + xorshift32(rng) * 110.0
        height = 28.0 + xorshift32(rng) * 100.0
        far.append((x, width, height, int(xorshift32(rng) * 3)))
        x += width * (0.55 + xorshift32(rng) * 0.35)
    x = -2200.0
    while x < 2200.0:
        width = 36.0 + xorshift32(rng) * 80.0
        height = 70.0 + xorshift32(rng) * 150.0
        mid.append((x, width, height, int(xorshift32(rng) * 4)))
        x += width * (0.72 + xorshift32(rng) * 0.45)
    return far, mid


def _paint_city(ctx, scene: CityScene) -> None:
    pal = active_palette()
    width, height = scene.width, scene.height
    gradient = cairo.LinearGradient(0, 0, 0, height)
    gradient.add_color_stop_rgb(0, 0.03, 0.04, 0.11)
    gradient.add_color_stop_rgb(0.4, 0.06, 0.045, 0.13)
    gradient.add_color_stop_rgb(0.7, 0.1, 0.06, 0.14)
    gradient.add_color_stop_rgb(1, 0.03, 0.02, 0.04)
    ctx.set_source(gradient)
    ctx.paint()
    for sx, sy, radius, warmth in scene.sky_stars:
        tone = 0.75 + 0.2 * warmth
        ctx.set_source_rgba(tone, tone * (0.92 + 0.05 * (1.0 - warmth)), 0.85 + 0.1 * (1.0 - warmth), 0.35 + 0.4 * radius / 2.0)
        ctx.arc(sx, sy, radius, 0, math.tau)
        ctx.fill()
    ctx.set_source_rgba(0.96, 0.95, 0.9, 0.92)
    ctx.arc(width * 0.78, height * 0.16, 26, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(0.03, 0.025, 0.07, 1)
    ctx.arc(width * 0.78 + 11, height * 0.16 - 6, 22, 0, math.tau)
    ctx.fill()
    ground = height * 0.78
    _paint_layer(ctx, scene, scene.far, ground - 36.0, 0.32, 0.45, (0.035, 0.04, 0.07))
    _paint_layer(ctx, scene, scene.mid, ground - 16.0, 0.62, 0.72, (0.055, 0.06, 0.10))
    ctx.save()
    ctx.translate(width / 2.0, ground)
    far_zoom = 1.0 + (scene.cam_zoom - 1.0) * 0.35
    ctx.scale(far_zoom, far_zoom)
    haze = cairo.LinearGradient(0, -80, 0, 30)
    haze.add_color_stop_rgba(0, 0.45, 0.28, 0.55, 0.0)
    haze.add_color_stop_rgba(0.55, 0.85, 0.55, 0.32, 0.10)
    haze.add_color_stop_rgba(1, 0.4, 0.25, 0.45, 0.0)
    ctx.set_source(haze)
    ctx.rectangle(-width * 2, -80, width * 4, 110)
    ctx.fill()
    ctx.restore()
    ctx.save()
    ctx.translate(width / 2.0, ground)
    ctx.scale(scene.cam_zoom, scene.cam_zoom)
    ctx.translate(-scene.cam_x, 0.0)
    glow = scene.street_glow()
    for building in scene.buildings:
        _paint_building(ctx, scene, building, pal)
    _paint_street(ctx, scene, pal, glow)
    ctx.restore()


def _paint_layer(ctx, scene: CityScene, blocks, ground: float, parallax: float, zoom_mix: float, color) -> None:
    ctx.save()
    ctx.translate(scene.width / 2.0, ground)
    zoom = 1.0 + (scene.cam_zoom - 1.0) * zoom_mix
    ctx.scale(zoom, zoom)
    ctx.translate(-scene.cam_x * parallax, 0.0)
    ctx.set_source_rgba(*color, 1.0)
    for x, width, height, roof in blocks:
        ctx.rectangle(x, -height, width, height)
        ctx.fill()
        if roof == 1 and height > 50:
            ctx.rectangle(x + width * 0.22, -height - height * 0.18, width * 0.56, height * 0.18)
            ctx.fill()
        elif roof == 2:
            ctx.move_to(x, -height)
            ctx.line_to(x + width * 0.5, -height - 16)
            ctx.line_to(x + width, -height)
            ctx.fill()
        if height > 90 and width > 40:
            ctx.set_source_rgba(1.0, 0.78, 0.42, 0.18)
            win_y = -height * 0.55
            ctx.rectangle(x + width * 0.35, win_y, max(4.0, width * 0.08), 5)
            ctx.fill()
            ctx.set_source_rgba(*color, 1.0)
    ctx.restore()


def _building_rgb(building: Building) -> tuple[float, float, float]:
    base = _NIGHT_TONES[building.tint % len(_NIGHT_TONES)]
    lift = 0.04 * building.energy
    return (min(1.0, base[0] + lift), min(1.0, base[1] + lift * 0.4), min(1.0, base[2] + lift))


def _paint_building(ctx, scene: CityScene, building: Building, pal) -> None:
    risen = scene.building_height(building)
    if risen < 1.5:
        return
    body = _building_rgb(building)
    _round_rect(ctx, building.x, -risen, building.width, risen, 3)
    fill = cairo.LinearGradient(building.x, -risen, building.x + building.width, 0)
    fill.add_color_stop_rgb(0, min(1, body[0] + 0.05), min(1, body[1] + 0.04), min(1, body[2] + 0.07))
    fill.add_color_stop_rgb(1, body[0] * 0.62, body[1] * 0.62, body[2] * 0.72)
    ctx.set_source(fill)
    ctx.fill()
    ctx.set_source_rgba(0, 0, 0, 0.32)
    ctx.rectangle(building.x + building.width - min(10.0, building.width * 0.18), -risen, min(10.0, building.width * 0.18), risen)
    ctx.fill()
    crown = risen * 0.16
    ctx.set_source_rgba(body[0] * 0.85, body[1] * 0.85, body[2] * 0.95, 1)
    if building.roof == 1:
        inset = building.width * 0.2
        ctx.rectangle(building.x + inset, -risen - crown, building.width - 2 * inset, crown)
        ctx.fill()
    elif building.roof == 2:
        ctx.move_to(building.x - 2, -risen)
        ctx.line_to(building.x + building.width * 0.5, -risen - crown * 1.4)
        ctx.line_to(building.x + building.width + 2, -risen)
        ctx.fill()
    elif building.roof == 3 or building.tower:
        ctx.move_to(building.x + building.width * 0.28, -risen)
        ctx.line_to(building.x + building.width * 0.5, -risen - crown * 2.1)
        ctx.line_to(building.x + building.width * 0.72, -risen)
        ctx.fill()
    elif building.roof == 4:
        ctx.rectangle(building.x + building.width * 0.12, -risen - crown * 0.45, building.width * 0.76, crown * 0.45)
        ctx.fill()
        ctx.rectangle(building.x + building.width * 0.28, -risen - crown, building.width * 0.44, crown * 0.55)
        ctx.fill()
    ctx.set_source_rgba(*pal.rim, 0.28)
    ctx.set_line_width(1.2)
    ctx.move_to(building.x + 2, -risen)
    ctx.line_to(building.x + building.width - 2, -risen)
    ctx.stroke()
    if building.sign and risen > building.target * 0.72:
        sign_w = building.width * (0.42 if building.sign == 1 else 0.55)
        sign_x = building.x + (building.width - sign_w) * 0.5
        sign_y = -risen * (0.62 if building.roof == 0 else 0.48)
        if building.sign == 1:
            color = (1.0, 0.55, 0.28, 0.85)
        else:
            color = (0.45, 0.85, 1.0, 0.8)
        glow = cairo.LinearGradient(sign_x, sign_y, sign_x + sign_w, sign_y)
        glow.add_color_stop_rgba(0, color[0], color[1], color[2], 0.0)
        glow.add_color_stop_rgba(0.15, *color)
        glow.add_color_stop_rgba(0.85, *color)
        glow.add_color_stop_rgba(1, color[0], color[1], color[2], 0.0)
        ctx.set_source(glow)
        ctx.rectangle(sign_x, sign_y, sign_w, 7)
        ctx.fill()
    _paint_windows(ctx, scene, building, risen, pal)


def _paint_reflections(ctx, scene: CityScene) -> None:
    """Wet asphalt picks up the towers. Drawn after the road so the smear sits on it."""
    for building in scene.buildings:
        risen = scene.building_height(building)
        if risen < 12.0:
            continue
        smear = min(78.0, risen * 0.28)
        cols = max(2, building.cols)
        for col in range(cols):
            x = building.x + building.width * (col + 0.5) / cols - 1.4
            grad = cairo.LinearGradient(0, 8, 0, 8 + smear)
            grad.add_color_stop_rgba(0, 1.0, 0.78, 0.45, 0.22)
            grad.add_color_stop_rgba(1, 1.0, 0.7, 0.4, 0.0)
            ctx.set_source(grad)
            ctx.rectangle(x, 12, 2.8, smear)
            ctx.fill()
        if building.sign:
            color = (1.0, 0.45, 0.22) if building.sign == 1 else (0.4, 0.78, 1.0)
            ctx.set_source_rgba(*color, 0.2)
            ctx.rectangle(building.x + building.width * 0.22, 16, building.width * 0.56, 3.5)
            ctx.fill()


def _paint_street(ctx, scene: CityScene, pal, glow: float) -> None:
    """Asphalt, a tram, and long headlight streaks. Beats flare the road; bass washes the curb.

    Every bright mark is a long horizontal streak. A short lamp blooms into a ball.
    """
    if not scene.buildings:
        return
    left = scene.buildings[0].x - 120.0
    right = scene.buildings[-1].x + scene.buildings[-1].width + 120.0
    span = max(240.0, right - left)
    pulse = scene.beat_pulse()
    road_h = 168.0
    ctx.set_source_rgba(0.012, 0.014, 0.022, 1)
    ctx.rectangle(left - 280, 0, span + 560, road_h)
    ctx.fill()
    wash = cairo.LinearGradient(0, 0, 0, 36)
    wash.add_color_stop_rgba(0, 1.0, 0.86, 0.58, 0.08 + 0.55 * glow + 0.7 * pulse)
    wash.add_color_stop_rgba(1, pal.accent2[0], pal.accent2[1], pal.accent2[2], 0.0)
    ctx.set_source(wash)
    ctx.rectangle(left - 280, 0, span + 560, 36)
    ctx.fill()
    _paint_reflections(ctx, scene)
    if pulse > 0.04:
        ctx.set_source_rgba(1.0, 0.9, 0.7, 0.22 * pulse)
        ctx.rectangle(left - 280, 2, span + 560, 5)
        ctx.fill()
    ctx.set_source_rgba(0.78, 0.82, 0.88, 0.22)
    dash = left
    while dash < right:
        ctx.rectangle(dash, 78, 28, 2.4)
        ctx.fill()
        dash += 64
    _paint_tram(ctx, left, span, scene.time, max(glow, pulse))
    # Two lanes. The near lane is lower, thicker, and faster.
    lanes = ((44.0, 8.0, 1.0), (112.0, 12.0, -1.0))
    for index in range(8):
        lane_y, thick, direction = lanes[index % 2]
        speed = 170.0 + (index % 4) * 55.0 + (40.0 if direction < 0 else 0.0)
        length = 190.0 + 90.0 * max(pulse, 0.45 * glow) + (index % 3) * 24.0
        travel = (scene.time * speed + index * span / 8.0) % (span + length + 80.0)
        _paint_headlight(ctx, left, right, lane_y, thick, direction, length, travel, pulse)


def _paint_tram(ctx, left: float, span: float, time: float, flare: float) -> None:
    tram_w = 340.0
    tram_h = 32.0
    tram_x = left + ((time * 78.0) % (span + tram_w + 160.0)) - tram_w
    y = 14.0
    ctx.set_source_rgba(0.04, 0.045, 0.06, 0.98)
    _round_rect(ctx, tram_x, y, tram_w, tram_h, 5)
    ctx.fill()
    ctx.set_source_rgba(0.55, 0.62, 0.7, 0.35)
    ctx.set_line_width(1.4)
    ctx.move_to(tram_x + tram_w * 0.5, y)
    ctx.line_to(tram_x + tram_w * 0.5, y - 16)
    ctx.stroke()
    ctx.move_to(tram_x + tram_w * 0.38, y - 16)
    ctx.line_to(tram_x + tram_w * 0.62, y - 16)
    ctx.stroke()
    window = cairo.LinearGradient(tram_x, 0, tram_x + tram_w, 0)
    window.add_color_stop_rgba(0.0, 1.0, 0.78, 0.42, 0.0)
    window.add_color_stop_rgba(0.08, 1.0, 0.82, 0.48, 0.28 + 0.45 * flare)
    window.add_color_stop_rgba(0.92, 1.0, 0.82, 0.48, 0.28 + 0.45 * flare)
    window.add_color_stop_rgba(1.0, 1.0, 0.78, 0.42, 0.0)
    ctx.set_source(window)
    ctx.rectangle(tram_x + 18, y + 8, tram_w - 36, 8)
    ctx.fill()
    beam = 180.0 + 140.0 * flare
    head = cairo.LinearGradient(tram_x + tram_w - 8, 0, tram_x + tram_w + beam, 0)
    head.add_color_stop_rgba(0, 0.8, 0.94, 1.0, 0.9)
    head.add_color_stop_rgba(1, 0.8, 0.94, 1.0, 0.0)
    ctx.set_source(head)
    ctx.rectangle(tram_x + tram_w - 8, y + 14, beam, 8)
    ctx.fill()


def _paint_headlight(ctx, left, right, lane, thick, direction, length, travel, pulse) -> None:
    """One long bright streak. Only the last quarter fades, or bloom turns the nose into a dot."""
    alpha = 0.82 + 0.18 * pulse
    if direction > 0:
        nose = left + travel
        streak = cairo.LinearGradient(nose - length, lane, nose, lane)
        streak.add_color_stop_rgba(0.0, 0.95, 0.18, 0.12, 0.0)
        streak.add_color_stop_rgba(0.18, 1.0, 0.28, 0.16, 0.75 * alpha)
        streak.add_color_stop_rgba(0.42, 1.0, 0.9, 0.7, 0.9 * alpha)
        streak.add_color_stop_rgba(1.0, 1.0, 0.98, 0.92, alpha)
        ctx.set_source(streak)
        ctx.rectangle(nose - length, lane, length, thick)
    else:
        nose = right - travel
        streak = cairo.LinearGradient(nose, lane, nose + length, lane)
        streak.add_color_stop_rgba(0.0, 1.0, 0.98, 0.92, alpha)
        streak.add_color_stop_rgba(0.58, 1.0, 0.9, 0.7, 0.9 * alpha)
        streak.add_color_stop_rgba(0.82, 1.0, 0.28, 0.16, 0.75 * alpha)
        streak.add_color_stop_rgba(1.0, 0.95, 0.18, 0.12, 0.0)
        ctx.set_source(streak)
        ctx.rectangle(nose, lane, length, thick)
    ctx.fill()


def _paint_windows(ctx, scene: CityScene, building: Building, risen: float, pal) -> None:
    index = scene.buildings.index(building)
    margin = building.width * 0.16
    usable = max(1.0, building.width - 2.0 * margin)
    win_w = min(10.0, usable / building.cols * 0.55)
    win_h = min(14.0, risen / max(building.floors, 1) * 0.42)
    lit = {}
    for light in scene.lights:
        if light.index != index or scene.time + 1e-6 < light.time:
            continue
        age = scene.time - light.time
        flash = _smooth(1.0 - age / 0.36) if age < 0.36 else 0.0
        previous = lit.get((light.col, light.row), 0.0)
        lit[(light.col, light.row)] = max(previous, 0.72 + 0.28 * flash)
    for row in range(building.floors):
        for col in range(building.cols):
            x = building.x + margin + (col + 0.5) * usable / building.cols - win_w / 2.0
            span = building.target * 0.78
            y = -building.target * 0.9 + (row + 0.5) * span / building.floors
            if y < -risen + 6:
                continue
            level = lit.get((col, row), 0.0)
            if level <= 0.0:
                ctx.set_source_rgba(0.02, 0.03, 0.05, 0.65)
            else:
                ctx.set_source_rgba(pal.accent[0], pal.accent[1], pal.accent[2], min(1.0, level))
            _round_rect(ctx, x, y, win_w, win_h, 1.6)
            ctx.fill()


@dataclass
class KeyNote:
    time: float
    duration: float
    midi: int
    display: int
    source: str


@dataclass
class PianoScene:
    form: str
    palette_name: str
    width: float
    height: float
    duration: float
    notes: list[KeyNote]
    lo: int
    hi: int
    key_w: float
    key_left: float
    key_top: float
    white_h: float
    lead: float
    timing: dict
    sync_report: dict
    analysis: Analysis
    caption_title: str = ""
    caption_artist: str = ""
    time: float = 0.0
    cam_zoom: float = 1.0
    _cam_zoom: float = 1.0

    def key_center(self, midi: int) -> tuple[float, float]:
        shown = self._clamp_midi(midi)
        if shown % 12 in WHITES:
            x = self.key_left + self._whites_before(shown) * self.key_w + self.key_w * 0.5
        else:
            x = self.key_left + self._whites_before(shown) * self.key_w
        return x, self.key_top

    def note_head(self, note: KeyNote, time: float | None = None) -> tuple[float, float]:
        when = self.time if time is None else time
        x, key_y = self.key_center(note.display)
        speed = (key_y * 0.92) / self.lead
        if when >= note.time:
            y = key_y
        else:
            y = key_y - (note.time - when) * speed
        return x, y

    def _clamp_midi(self, midi: int) -> int:
        while midi < self.lo:
            midi += 12
        while midi > self.hi:
            midi -= 12
        return midi

    def _whites_before(self, midi: int) -> int:
        return sum(1 for tone in range(self.lo, midi) if tone % 12 in WHITES)

    def _target_zoom(self, time: float) -> float:
        del time
        # The keyboard stays put. The press, not a zoom, is the motion.
        return 1.0

    def update(self, dt: float, time: float, events: list[tuple[float, bool]]) -> None:
        del events
        self.time = time
        self._cam_zoom = _damp(self._cam_zoom, self._target_zoom(time), dt, 1.8)
        self.cam_zoom = self._cam_zoom


def _build_piano(width: int, height: int, analysis: Analysis, seed: int) -> PianoScene:
    del seed
    tones = extract_tones(analysis)
    if not tones:
        tones = [Tone(0.0, 0.4, 60, "chroma")]
    midis = [tone.midi for tone in tones]
    center = sorted(midis)[len(midis) // 2]
    lo = max(36, min(midis) - 3)
    hi = min(96, max(midis) + 3)
    lo -= lo % 12
    hi += (11 - (hi % 12)) % 12
    if hi - lo > 36:
        lo = center - 17
        hi = center + 18
        lo -= lo % 12
        if hi % 12 != 11:
            hi += (11 - hi % 12) % 12
    notes = [
        KeyNote(tone.time, tone.duration, tone.midi, tone.midi, tone.source)
        for tone in tones
        if 0.0 <= tone.time <= analysis.duration + 1e-3
    ]
    white_count = sum(1 for tone in range(lo, hi + 1) if tone % 12 in WHITES)
    white_count = max(1, white_count)
    # Cap the width so a narrow melody does not become a row of slabs.
    key_w = min(68.0, (width * 0.9) / white_count)
    key_left = (width - key_w * white_count) / 2.0
    scene = PianoScene(
        form="piano",
        palette_name="waterfall",
        width=float(width),
        height=float(height),
        duration=max(analysis.duration, 1e-3),
        notes=notes,
        lo=lo,
        hi=hi,
        key_w=key_w,
        key_left=key_left,
        key_top=height * 0.655,
        white_h=min(height * 0.30, key_w * 5.05),
        lead=1.45,
        timing={},
        sync_report={},
        analysis=analysis,
    )
    worst = 0.0
    for note in notes:
        x, y = scene.note_head(note, note.time)
        kx, ky = scene.key_center(note.display)
        worst = max(worst, math.hypot(x - kx, y - ky))
    beat_gap = _median_gap([])
    if notes and analysis.beats:
        gaps = sorted(_nearest(list(analysis.beats), note.time) for note in notes)
        beat_gap = gaps[len(gaps) // 2]
    scene.timing = {"max_timing_error_s": 0.0, "max_position_error_px": worst, "hits": len(notes)}
    scene.sync_report = {
        "notes": len(notes),
        "pitched": sum(1 for note in notes if note.source == "pitch"),
        "chroma": sum(1 for note in notes if note.source == "chroma"),
        "median_beat_gap_ms": round(beat_gap * 1000.0, 1),
    }
    scene._cam_zoom = scene._target_zoom(0.0)
    scene.cam_zoom = scene._cam_zoom
    return scene


def _paint_piano(ctx, scene: PianoScene) -> None:
    pal = active_palette()
    width, height = scene.width, scene.height
    gradient = cairo.LinearGradient(0, 0, 0, height)
    gradient.add_color_stop_rgb(0, *pal.bg_top)
    gradient.add_color_stop_rgb(1, *pal.bg_bottom)
    ctx.set_source(gradient)
    ctx.paint()
    ctx.save()
    ctx.translate(width / 2.0, scene.key_top)
    ctx.scale(scene.cam_zoom, scene.cam_zoom)
    ctx.translate(-width / 2.0, -scene.key_top)
    # Faint lanes so a falling note has a road down to its key.
    ctx.set_line_width(1)
    for tone in range(scene.lo, scene.hi + 1):
        if tone % 12 not in WHITES:
            continue
        x = scene.key_left + scene._whites_before(tone) * scene.key_w + scene.key_w * 0.5
        ctx.set_source_rgba(*pal.accent2, 0.035)
        ctx.move_to(x, 0)
        ctx.line_to(x, scene.key_top)
        ctx.stroke()
    speed = (scene.key_top * 0.92) / scene.lead
    for note in scene.notes:
        color = NOTE_COLORS[note.midi % 12]
        front_x, front_y = scene.note_head(note)
        if scene.time < note.time:
            tail = note.duration * speed
            top = front_y - tail
            bottom = front_y
        else:
            remain = note.time + note.duration - scene.time
            if remain <= 0.0:
                continue
            top = scene.key_top - remain * speed
            bottom = scene.key_top
        if bottom < -20 or top > height:
            continue
        shown = scene._clamp_midi(note.display)
        black = shown % 12 not in WHITES
        note_w = scene.key_w * (0.46 if black else 0.72)
        rect_x = front_x - note_w / 2.0
        rect_h = max(8.0, bottom - top)
        _round_rect(ctx, rect_x, top, note_w, rect_h, min(8.0, note_w * 0.4))
        fill = cairo.LinearGradient(0, top, 0, bottom)
        fill.add_color_stop_rgba(0, *color, 0.12)
        fill.add_color_stop_rgba(0.72, *color, 0.82)
        fill.add_color_stop_rgba(1, *color, 1.0)
        ctx.set_source(fill)
        ctx.fill()
    _paint_key_columns(ctx, scene)
    _paint_keyboard(ctx, scene, pal)
    _paint_key_sparks(ctx, scene)
    ctx.restore()


def _press_level(scene: PianoScene) -> dict[int, float]:
    pressed: dict[int, float] = {}
    for note in scene.notes:
        if note.time <= scene.time <= note.time + note.duration + 0.22:
            age = scene.time - note.time
            if age <= note.duration:
                level = 1.0
            else:
                level = _smooth(1.0 - (age - note.duration) / 0.22)
            shown = scene._clamp_midi(note.display)
            pressed[shown] = max(pressed.get(shown, 0.0), level)
    return pressed


def _hit_level(note: KeyNote, time: float) -> float:
    age = time - note.time
    if age < 0.0 or age > note.duration + 0.28:
        return 0.0
    if age <= note.duration:
        return 1.0
    return _smooth(1.0 - (age - note.duration) / 0.28)


def _paint_key_columns(ctx, scene: PianoScene) -> None:
    for note in scene.notes:
        level = _hit_level(note, scene.time)
        if level <= 0.0:
            continue
        color = NOTE_COLORS[note.midi % 12]
        x, key_y = scene.key_center(note.display)
        shown = scene._clamp_midi(note.display)
        black = shown % 12 not in WHITES
        column_w = scene.key_w * (0.5 if black else 0.7)
        column_h = 80.0 + 180.0 * level
        column = cairo.LinearGradient(0, key_y - column_h, 0, key_y)
        column.add_color_stop_rgba(0, *color, 0.0)
        column.add_color_stop_rgba(0.45, *color, 0.22 * level)
        column.add_color_stop_rgba(1, *color, 0.85 * level)
        ctx.set_source(column)
        ctx.rectangle(x - column_w / 2.0, key_y - column_h, column_w, column_h)
        ctx.fill()


def _paint_key_sparks(ctx, scene: PianoScene) -> None:
    """Burst drawn after the keys. Streaks, not dots: a short bright dot blooms into a ball."""
    for note in scene.notes:
        age = scene.time - note.time
        if age < 0.0 or age > 0.5:
            continue
        color = NOTE_COLORS[note.midi % 12]
        x, key_y = scene.key_center(note.display)
        fade = 1.0 - age / 0.5
        splash = cairo.LinearGradient(x - 70, key_y, x + 70, key_y)
        splash.add_color_stop_rgba(0.0, *color, 0.0)
        splash.add_color_stop_rgba(0.5, 1.0, 0.98, 0.95, 0.85 * fade)
        splash.add_color_stop_rgba(1.0, *color, 0.0)
        ctx.set_source(splash)
        ctx.rectangle(x - 70, key_y - 8, 140, 10)
        ctx.fill()
        ring = min(1.0, age / 0.36)
        ctx.set_source_rgba(*color, fade * 0.95)
        ctx.set_line_width(7.0 * fade + 2.0)
        ctx.arc(x, key_y + 4.0, 28.0 + ring * 86.0, math.pi * 1.08, math.pi * 1.92)
        ctx.stroke()
        for spark in range(13):
            angle = math.pi * 1.02 + spark * (math.pi * 0.96 / 12.0)
            dist = 46.0 + age * (320.0 + spark * 18.0)
            x2 = x + math.cos(angle) * dist
            y2 = key_y + math.sin(angle) * dist
            x1 = x + math.cos(angle) * (dist - 22.0)
            y1 = key_y + math.sin(angle) * (dist - 22.0)
            ctx.set_source_rgba(1.0, 0.98, 0.95, fade)
            ctx.set_line_width(2.6 * fade + 0.8)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.stroke()
            ctx.set_source_rgba(*color, fade * 0.8)
            ctx.set_line_width(5.0 * fade)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.stroke()


def _paint_keyboard(ctx, scene: PianoScene, pal) -> None:
    del pal
    pressed = _press_level(scene)
    # White keys are tall and narrow. A press drops the key into a dark well.
    for tone in range(scene.lo, scene.hi + 1):
        if tone % 12 not in WHITES:
            continue
        x = scene.key_left + scene._whites_before(tone) * scene.key_w
        level = pressed.get(tone, 0.0)
        sink = 46.0 * level
        gap = 3.2
        ctx.set_source_rgba(0.01, 0.011, 0.016, 1)
        _round_rect(ctx, x + gap * 0.5, scene.key_top, scene.key_w - gap, scene.white_h, 4)
        ctx.fill()
        _round_rect(ctx, x + gap * 0.5 + 0.6, scene.key_top + sink, scene.key_w - gap - 1.2, scene.white_h - sink, 4)
        fill = cairo.LinearGradient(0, scene.key_top + sink, 0, scene.key_top + scene.white_h)
        if level > 0.0:
            color = NOTE_COLORS[tone % 12]
            fill.add_color_stop_rgb(0, 0.35 + 0.6 * color[0], 0.32 + 0.55 * color[1], 0.34 + 0.5 * color[2])
            fill.add_color_stop_rgb(0.12, 0.55, 0.57, 0.62)
            fill.add_color_stop_rgb(1, 0.72, 0.74, 0.78)
        else:
            fill.add_color_stop_rgb(0, 0.99, 0.99, 1.0)
            fill.add_color_stop_rgb(0.08, 0.93, 0.94, 0.96)
            fill.add_color_stop_rgb(1, 0.74, 0.76, 0.80)
        ctx.set_source(fill)
        ctx.fill()
        if level > 0.05:
            ctx.set_source_rgba(1, 0.98, 0.94, 0.55 + 0.4 * level)
            ctx.rectangle(x + gap, scene.key_top + sink, scene.key_w - gap * 2, 3.5)
            ctx.fill()
        ctx.set_source_rgba(0, 0, 0, 0.16)
        ctx.rectangle(x + scene.key_w - gap - 3.0, scene.key_top + sink, 2.4, scene.white_h - sink)
        ctx.fill()
    # Black keys are a little over half a white key wide, and about two thirds as long.
    black_h = scene.white_h * 0.64
    black_w = scene.key_w * 0.58
    for tone in range(scene.lo, scene.hi + 1):
        if tone % 12 in WHITES:
            continue
        x = scene.key_left + scene._whites_before(tone) * scene.key_w - black_w / 2.0
        level = pressed.get(tone, 0.0)
        sink = 24.0 * level
        ctx.set_source_rgba(0.008, 0.008, 0.012, 1)
        _round_rect(ctx, x, scene.key_top, black_w, black_h, 3)
        ctx.fill()
        _round_rect(ctx, x + 0.8, scene.key_top + sink, black_w - 1.6, black_h - sink, 3)
        if level > 0.0:
            color = NOTE_COLORS[tone % 12]
            ctx.set_source_rgb(0.22 + 0.55 * color[0], 0.16 + 0.42 * color[1], 0.18 + 0.4 * color[2])
        else:
            shade = cairo.LinearGradient(0, scene.key_top, 0, scene.key_top + black_h)
            shade.add_color_stop_rgb(0, 0.16, 0.17, 0.2)
            shade.add_color_stop_rgb(0.2, 0.05, 0.052, 0.07)
            shade.add_color_stop_rgb(1, 0.02, 0.02, 0.03)
            ctx.set_source(shade)
        ctx.fill()
        ctx.set_source_rgba(1, 1, 1, 0.22 if level <= 0 else 0.7)
        ctx.rectangle(x + 2.5, scene.key_top + sink + 2, black_w - 5, 2.2)
        ctx.fill()


@dataclass
class Star:
    time: float
    x: float
    y: float
    phase: float
    midi: int
    phrase: int
    scale: float = 1.0
    warmth: float = 0.5


@dataclass
class StarScene:
    form: str
    palette_name: str
    width: float
    height: float
    duration: float
    stars: list[Star]
    links: list[tuple[int, int]]
    dust: list[tuple[float, float, float]]
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

    def brightness(self, star: Star, time: float | None = None) -> float:
        when = self.time if time is None else time
        age = when - star.time
        if age < -0.24:
            return 0.0
        if age <= 0.0:
            return _smooth((age + 0.24) / 0.24)
        twinkle = 0.84 + 0.16 * math.sin(when * 1.5 + star.phase)
        return twinkle

    def _target(self, time: float) -> tuple[float, float, float]:
        # A small drift. Large camera motion would slide constellations into each other.
        return (
            8.0 * math.sin(time * 0.045),
            5.0 * math.cos(time * 0.032),
            1.0,
        )

    def update(self, dt: float, time: float, events: list[tuple[float, bool]]) -> None:
        del events
        self.time = time
        tx, ty, tz = self._target(time)
        self._cam_x = _damp(self._cam_x, tx, dt, 0.8)
        self._cam_y = _damp(self._cam_y, ty, dt, 0.8)
        self._cam_zoom = _damp(self._cam_zoom, tz, dt, 0.8)
        self.cam_x, self.cam_y, self.cam_zoom = self._cam_x, self._cam_y, self._cam_zoom


def _build_stars(width: int, height: int, analysis: Analysis, seed: int) -> StarScene:
    rng = [seed & 0xFFFFFFFF or 1]
    tones = extract_tones(analysis)
    if not tones:
        tones = [Tone(float(beat), 0.4, 60, "chroma") for beat in analysis.beats] or [Tone(0.0, 0.4, 60, "chroma")]
    phrases: list[list[Tone]] = []
    bucket: list[Tone] = []
    for tone in tones:
        section = section_at(analysis.sections, tone.time).label
        if bucket:
            gap = tone.time - bucket[-1].time
            previous = section_at(analysis.sections, bucket[-1].time).label
            # Short groups read as constellations. A whole verse on one line does not.
            if gap > 1.15 or (section != previous and gap > 0.45) or len(bucket) >= 6:
                phrases.append(bucket)
                bucket = []
        bucket.append(tone)
    if bucket:
        phrases.append(bucket)
    stars, links = _layout_constellations(phrases, float(width), float(height), rng)
    dust = []
    occupied = [(star.x, star.y) for star in stars]
    for _ in range(260):
        x = 24.0 + xorshift32(rng) * (width - 48.0)
        y = 20.0 + xorshift32(rng) * (height * 0.92)
        if any(math.hypot(x - sx, y - sy) < 26.0 for sx, sy in occupied):
            continue
        dust.append((x, y, 0.35 + xorshift32(rng) * 1.15, xorshift32(rng), xorshift32(rng) * math.tau))
    scene = StarScene(
        form="stars",
        palette_name="stars",
        width=float(width),
        height=float(height),
        duration=max(analysis.duration, 1e-3),
        stars=stars,
        links=links,
        dust=dust,
        timing={},
        sync_report={},
        analysis=analysis,
    )
    gaps = [_nearest(list(analysis.beats), star.time) for star in stars] if analysis.beats else [0.0]
    gaps.sort()
    scene.timing = {"max_timing_error_s": 0.0, "max_position_error_px": 0.0, "hits": len(stars)}
    scene.sync_report = {
        "stars": len(stars),
        "lines": len(links),
        "phrases": len(phrases),
        "median_beat_gap_ms": round((gaps[len(gaps) // 2] if gaps else 0.0) * 1000.0, 1),
    }
    scene._cam_x, scene._cam_y, scene._cam_zoom = scene._target(0.0)
    scene.cam_x, scene.cam_y, scene.cam_zoom = scene._cam_x, scene._cam_y, scene._cam_zoom
    return scene


def _layout_constellations(phrases: list[list[Tone]], width: float, height: float, rng: list[int]):
    """Each phrase is a small hooked figure in its own part of the sky. Lines stay a simple chain."""
    pending = []
    for index, phrase in enumerate(phrases):
        local = _organic_points(len(phrase), rng)
        radius = max((math.hypot(x, y) for x, y in local), default=0.0)
        pending.append((index, phrase, local, radius))
    pending.sort(key=lambda item: -item[3])
    placed: list[tuple[float, float, float]] = []
    origins: dict[int, tuple[float, float]] = {}
    for index, phrase, local, radius in pending:
        origin = _seat_figure(radius, placed, width, height, rng)
        placed.append((origin[0], origin[1], radius))
        origins[index] = origin
    stars: list[Star] = []
    links: list[tuple[int, int]] = []
    for index, phrase in enumerate(phrases):
        local = next(item[2] for item in pending if item[0] == index)
        ox, oy = origins[index]
        peak = max(tone.midi for tone in phrase)
        first = len(stars)
        for place, tone in enumerate(phrase):
            x, y = local[place]
            scale = 0.48 + xorshift32(rng) * 0.72
            if tone.midi == peak:
                scale += 0.28
            scale = min(1.45, scale)
            warmth = xorshift32(rng)
            stars.append(
                Star(
                    tone.time,
                    ox + x,
                    oy + y,
                    xorshift32(rng) * math.tau,
                    tone.midi,
                    index,
                    scale,
                    warmth,
                )
            )
            if place:
                links.append((first + place - 1, first + place))
    return stars, links


def _seat_figure(radius: float, placed: list[tuple[float, float, float]], width: float, height: float, rng: list[int]) -> tuple[float, float]:
    margin_x = 90.0 + radius
    margin_y = 70.0 + radius
    sky_bottom = height * 0.84
    best = (width * 0.5, height * 0.35)
    best_clear = -1.0
    for attempt in range(48):
        cx = margin_x + xorshift32(rng) * max(1.0, width - 2.0 * margin_x)
        cy = margin_y + xorshift32(rng) * max(1.0, sky_bottom - 2.0 * margin_y)
        clear = _clearance(cx, cy, radius, placed)
        if clear >= 62.0:
            return cx, cy
        if clear > best_clear:
            best, best_clear = (cx, cy), clear
    # A loose scatter, still not a row: walk a jittered lattice and keep the roomiest cell.
    cols = max(2, int(width // 340))
    rows = max(2, int(sky_bottom // 240))
    for row in range(rows):
        for col in range(cols):
            jx = (xorshift32(rng) - 0.5) * 80.0
            jy = (xorshift32(rng) - 0.5) * 50.0
            cx = width * (col + 0.5) / cols + jx
            cy = sky_bottom * (row + 0.45) / rows + jy
            cx = min(width - 60.0, max(60.0, cx))
            cy = min(sky_bottom - 40.0, max(50.0, cy))
            clear = _clearance(cx, cy, radius, placed)
            if clear > best_clear:
                best, best_clear = (cx, cy), clear
    return best


def _clearance(cx: float, cy: float, radius: float, placed: list[tuple[float, float, float]]) -> float:
    if not placed:
        return 1e6
    return min(math.hypot(cx - ox, cy - oy) - radius - other for ox, oy, other in placed)


def _organic_points(count: int, rng: list[int]) -> list[tuple[float, float]]:
    if count <= 1:
        return [(0.0, 0.0)]
    for _ in range(10):
        style = int(xorshift32(rng) * 3)
        if style == 0:
            sweep = 1.15 + xorshift32(rng) * 1.5
            points = _arc_points(count, sweep, rng)
        elif style == 1:
            sweep = 2.7 + xorshift32(rng) * 1.5
            points = _arc_points(count, sweep, rng)
        else:
            points = _walk_points(count, rng)
            if points is None:
                continue
        points = _rotate_points(points, xorshift32(rng) * math.tau)
        if _figure_ok(points) and _bent(points):
            return _center_points(points)
    hooked = _arc_points(count, 2.6, rng)
    return _center_points(_rotate_points(hooked, xorshift32(rng) * math.tau))


def _arc_points(count: int, sweep: float, rng: list[int]) -> list[tuple[float, float]]:
    step = 78.0
    delta = sweep / max(1, count - 1)
    radius = step / max(0.35, 2.0 * math.sin(delta / 2.0))
    start = xorshift32(rng) * math.tau
    return [(math.cos(start + i * delta) * radius, math.sin(start + i * delta) * radius) for i in range(count)]


def _walk_points(count: int, rng: list[int]) -> list[tuple[float, float]] | None:
    points = [(0.0, 0.0)]
    heading = xorshift32(rng) * math.tau
    for _ in range(1, count):
        found = None
        for _try in range(28):
            turn = (xorshift32(rng) - 0.5) * 2.6
            if xorshift32(rng) < 0.3:
                turn += 0.9 if turn >= 0.0 else -0.9
            heading_next = heading + turn
            step = 74.0 + xorshift32(rng) * 24.0
            nxt = (
                points[-1][0] + math.cos(heading_next) * step,
                points[-1][1] + math.sin(heading_next) * step,
            )
            trial = points + [nxt]
            if _figure_ok(trial):
                found = (nxt, heading_next)
                break
        if found is None:
            return None
        points.append(found[0])
        heading = found[1]
    return points


def _rotate_points(points: list[tuple[float, float]], angle: float) -> list[tuple[float, float]]:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return [(x * cos - y * sin, x * sin + y * cos) for x, y in points]


def _center_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)
    return [(x - cx, y - cy) for x, y in points]


def _bent(points: list[tuple[float, float]]) -> bool:
    """A chain of three or more should bend, so it does not read as a ruled line."""
    if len(points) < 3:
        return True
    centered = _center_points(points)
    xs = [point[0] for point in centered]
    ys = [point[1] for point in centered]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    short = max(1.0, min(span_x, span_y))
    return max(span_x, span_y) / short <= 2.5


def _figure_ok(points: list[tuple[float, float]]) -> bool:
    for i, a in enumerate(points):
        for b in points[i + 1 :]:
            if math.hypot(a[0] - b[0], a[1] - b[1]) < 70.0:
                return False
    if len(points) < 4:
        return True
    for i in range(len(points) - 1):
        for j in range(i + 2, len(points) - 1):
            if _pair_cross(points[i], points[i + 1], points[j], points[j + 1]):
                return False
    return True


def _pair_cross(a, b, c, d) -> bool:
    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def _star_tint(warmth: float) -> tuple[float, float, float]:
    return (
        0.76 + 0.24 * warmth,
        0.84 + 0.08 * (1.0 - abs(warmth - 0.35)),
        0.98 - 0.26 * warmth,
    )


def _paint_star_glyph(ctx, x: float, y: float, level: float, warmth: float, phase: float, time: float, scale: float = 1.0) -> None:
    """A bright core with a wide soft glow. Spikes stay short and fade."""
    twinkle = 0.9 + 0.1 * math.sin(time * 2.1 + phase)
    tint = _star_tint(warmth)
    spike = (9.0 + 10.0 * level) * (0.75 + 0.45 * scale) * twinkle
    glow_r = (10.0 + 16.0 * scale) * (0.65 + 0.35 * level)
    glow = cairo.RadialGradient(x, y, 0.4, x, y, glow_r)
    glow.add_color_stop_rgba(0, 1, 1, 1, min(1.0, 0.95 * level * twinkle))
    glow.add_color_stop_rgba(0.22, tint[0], tint[1], tint[2], 0.72 * level * twinkle)
    glow.add_color_stop_rgba(0.55, tint[0], tint[1], tint[2], 0.22 * level)
    glow.add_color_stop_rgba(1, tint[0], tint[1], tint[2], 0.0)
    ctx.set_source(glow)
    ctx.arc(x, y, glow_r, 0, math.tau)
    ctx.fill()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    angles = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    for index, angle in enumerate(angles):
        length = spike * (1.12 if index % 2 else 0.84)
        _fading_spike(ctx, x, y, angle, length, 1.35 * (0.75 + 0.35 * scale), tint, 0.9 * level * twinkle)
    diag = spike * 0.36
    for angle in (math.pi / 4.0, 3.0 * math.pi / 4.0, 5.0 * math.pi / 4.0, 7.0 * math.pi / 4.0):
        _fading_spike(ctx, x, y, angle, diag, 0.9, (1.0, 0.98, 0.96), 0.45 * level * twinkle)
    ctx.set_source_rgba(1, 1, 1, min(1.0, level * twinkle))
    ctx.arc(x, y, 1.6 + scale, 0, math.tau)
    ctx.fill()


def _fading_spike(ctx, x: float, y: float, angle: float, length: float, width: float, color, alpha: float) -> None:
    if length < 1.0 or alpha <= 0.0:
        return
    x2 = x + math.cos(angle) * length
    y2 = y + math.sin(angle) * length
    grad = cairo.LinearGradient(x, y, x2, y2)
    grad.add_color_stop_rgba(0.0, color[0], color[1], color[2], alpha)
    grad.add_color_stop_rgba(0.4, color[0], color[1], color[2], alpha * 0.35)
    grad.add_color_stop_rgba(1.0, color[0], color[1], color[2], 0.0)
    ctx.set_source(grad)
    ctx.set_line_width(width)
    ctx.move_to(x, y)
    ctx.line_to(x2, y2)
    ctx.stroke()


def _blur_rgba(surface, radius: int) -> None:
    """Separable box blur, twice, which reads as a soft gaussian on a glow layer."""
    if radius < 1:
        return
    surface.flush()
    height, width = surface.get_height(), surface.get_width()
    buf = np.ndarray((height, width, 4), dtype=np.uint8, buffer=surface.get_data())
    image = buf.astype(np.float32)
    for _ in range(2):
        window = radius * 2 + 1
        for axis in (1, 0):
            pad = [(0, 0), (0, 0), (0, 0)]
            pad[axis] = (radius, radius)
            padded = np.pad(image, pad, mode="edge")
            cumulative = np.cumsum(padded, axis=axis, dtype=np.float32)
            shape = list(cumulative.shape)
            shape[axis] = 1
            cumulative = np.concatenate([np.zeros(shape, dtype=np.float32), cumulative], axis=axis)
            high = [slice(None), slice(None), slice(None)]
            low = [slice(None), slice(None), slice(None)]
            high[axis] = slice(window, window + image.shape[axis])
            low[axis] = slice(0, image.shape[axis])
            image = (cumulative[tuple(high)] - cumulative[tuple(low)]) / window
    np.clip(image, 0, 255, out=image)
    buf[:] = image.astype(np.uint8)
    surface.mark_dirty()


def _paint_luminous_sky(ctx, width: float, height: float) -> None:
    """Milky way and nebulae added on top of the sky. They can only lighten it.

    The frame is painted with OPERATOR_SOURCE, so a translucent fill would replace
    the sky with a dark premultiplied color. This layer is blurred, then added.
    """
    glow_w = max(8, int(width) // 4)
    glow_h = max(8, int(height) // 4)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, glow_w, glow_h)
    glow = cairo.Context(surface)
    glow.set_operator(cairo.OPERATOR_SOURCE)
    glow.set_source_rgba(0, 0, 0, 0)
    glow.paint()
    glow.set_operator(cairo.OPERATOR_OVER)
    glow.scale(glow_w / width, glow_h / height)
    glow.save()
    glow.translate(width * 0.56, height * 0.42)
    glow.rotate(-0.26)
    band = cairo.LinearGradient(0, -260, 0, 260)
    band.add_color_stop_rgba(0.0, 0.72, 0.68, 0.95, 0.0)
    band.add_color_stop_rgba(0.5, 0.88, 0.84, 1.0, 0.9)
    band.add_color_stop_rgba(1.0, 0.65, 0.6, 0.92, 0.0)
    glow.set_source(band)
    glow.rectangle(-width * 1.4, -280, width * 2.8, 560)
    glow.fill()
    glow.restore()
    for ox, oy, radius, tint, alpha in (
        (width * 0.3, height * 0.34, 420.0, (0.72, 0.62, 1.0), 0.55),
        (width * 0.68, height * 0.5, 480.0, (0.62, 0.74, 1.0), 0.42),
        (width * 0.48, height * 0.24, 300.0, (1.0, 0.86, 0.72), 0.28),
    ):
        blob = cairo.RadialGradient(ox, oy, radius * 0.02, ox, oy, radius)
        blob.add_color_stop_rgba(0.0, tint[0], tint[1], tint[2], alpha)
        blob.add_color_stop_rgba(0.45, tint[0], tint[1], tint[2], alpha * 0.4)
        blob.add_color_stop_rgba(1.0, tint[0], tint[1], tint[2], 0.0)
        glow.set_source(blob)
        glow.rectangle(ox - radius, oy - radius, radius * 2, radius * 2)
        glow.fill()
    _blur_rgba(surface, 6)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    pattern = cairo.SurfacePattern(surface)
    pattern.set_filter(cairo.FILTER_BILINEAR)
    pattern.set_matrix(cairo.Matrix(xx=glow_w / float(width), yy=glow_h / float(height)))
    ctx.set_source(pattern)
    ctx.paint_with_alpha(0.42)
    ctx.restore()


def _paint_stars(ctx, scene: StarScene) -> None:
    width, height = scene.width, scene.height
    gradient = cairo.LinearGradient(0, 0, 0, height)
    gradient.add_color_stop_rgb(0, 0.045, 0.055, 0.16)
    gradient.add_color_stop_rgb(0.34, 0.09, 0.055, 0.18)
    gradient.add_color_stop_rgb(0.66, 0.11, 0.07, 0.17)
    gradient.add_color_stop_rgb(1, 0.05, 0.04, 0.10)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source(gradient)
    ctx.paint()
    # Everything after the opaque sky must blend. SOURCE would punch dark holes.
    ctx.set_operator(cairo.OPERATOR_OVER)
    _paint_luminous_sky(ctx, width, height)
    ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.save()
    ctx.translate(width / 2.0, height / 2.0)
    ctx.scale(scene.cam_zoom, scene.cam_zoom)
    ctx.translate(-width / 2.0 + scene.cam_x, -height / 2.0 + scene.cam_y)
    for x, y, scale, warmth, phase in scene.dust:
        twinkle = 0.55 + 0.45 * math.sin(scene.time * 1.4 + phase)
        tint = _star_tint(warmth)
        alpha = (0.12 + 0.28 * min(1.0, scale)) * twinkle
        ctx.set_source_rgba(tint[0], tint[1], tint[2], alpha)
        ctx.arc(x, y, 0.7 + scale * 0.7, 0, math.tau)
        ctx.fill()
        if scale > 1.05:
            _fading_spike(ctx, x, y, phase, 4.5 + scale * 2.0, 0.7, tint, alpha * 0.8)
            _fading_spike(ctx, x, y, phase + math.pi / 2.0, 5.5 + scale * 2.0, 0.7, tint, alpha * 0.7)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for left, right in scene.links:
        a = scene.stars[left]
        b = scene.stars[right]
        born = max(a.time, b.time)
        age = scene.time - born
        if age < 0.0:
            continue
        appear = _smooth(min(1.0, age / 0.45))
        if age < 5.0:
            alpha = appear * 0.72
        else:
            alpha = max(0.16, 0.72 * _smooth(1.0 - (age - 5.0) / 7.0))
        tint = _star_tint((a.warmth + b.warmth) * 0.5)
        ctx.set_source_rgba(tint[0], tint[1], tint[2], alpha)
        ctx.set_line_width(1.55)
        ctx.move_to(a.x, a.y)
        ctx.line_to(b.x, b.y)
        ctx.stroke()
    for star in scene.stars:
        level = scene.brightness(star)
        if level <= 0.01:
            continue
        _paint_star_glyph(ctx, star.x, star.y, level, star.warmth, star.phase, scene.time, star.scale)
    ctx.restore()


def build_form(kind: str, width: int, height: int, analysis: Analysis, seed: int, theme: str = "neon"):
    del theme
    if kind == "city":
        return _build_city(width, height, analysis, seed)
    if kind == "piano":
        return _build_piano(width, height, analysis, seed)
    if kind == "stars":
        return _build_stars(width, height, analysis, seed)
    raise SystemExit(f"unknown scene {kind}")


def paint_form(ctx, scene) -> None:
    if scene.form == "city":
        _paint_city(ctx, scene)
    elif scene.form == "piano":
        _paint_piano(ctx, scene)
    elif scene.form == "stars":
        _paint_stars(ctx, scene)
    else:
        raise SystemExit(f"unknown form {scene.form}")
