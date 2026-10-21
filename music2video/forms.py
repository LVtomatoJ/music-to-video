"""Beat-synced forms other than the ball map: a growing city, a note waterfall, a sky of constellations.

Each form is closed-form in time. A building finishes on its beat, a note head
meets its key on the onset, and a star reaches full brightness on its note.
The city camera leads the next towers, then eases back to the whole skyline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cairo
import numpy as np

from music2video.analysis import Analysis, section_at
from music2video.notes import Tone, chroma_columns, extract_tones
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


def _pulse(age: float, attack: float, release: float) -> float:
    """Smooth bump. Zero slope at the start and the end, so a beat eases in and out."""
    if age < 0.0 or age >= attack + release:
        return 0.0
    if age < attack:
        return _smooth(age / max(attack, 1e-4))
    return _smooth(1.0 - (age - attack) / max(release, 1e-4))


def _hit_flash(times, now: float, rise: float, release: float) -> float:
    """One at the hit, already rising just before it, then easing out."""
    level = 0.0
    lead = max(rise, 1e-4)
    for hit in times:
        if hit > now + lead:
            break
        age = now - hit
        if age < -lead or age >= release:
            continue
        if age <= 0.0:
            level = max(level, _smooth((age + lead) / lead))
        else:
            level = max(level, _smooth(1.0 - age / release))
    return level


def _times_pulse(times, now: float, attack: float, release: float) -> float:
    level = 0.0
    span = attack + release
    for hit in times:
        if hit > now:
            break
        age = now - hit
        if age >= span:
            continue
        level = max(level, _pulse(age, attack, release))
    return level


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
    sections: tuple = ()
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
            age = when - building.born
            settle = 0.36
            # Exact target on the beat, then a small rise and return. Width stays put.
            if 0.0 < age < settle:
                bump = math.sin(math.pi * age / settle)
                lift = 0.032 * (0.6 + 0.4 * building.energy)
                return building.target * (1.0 + lift * bump)
            return building.target
        return building.target * _smooth(u)

    def street_glow(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        return _times_pulse(self.bass_times, when, 0.05, 0.42)

    def beat_pulse(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        return _times_pulse(self.beat_times, when, 0.04, 0.16)

    def energy_area(self, time: float) -> float:
        """Integral of section energy. Continuous, so faster cars do not jump."""
        area = 0.0
        for section in self.sections:
            start = max(0.0, section.start)
            end = min(time, section.end)
            if end > start:
                area += (end - start) * section.energy
        return area

    def _cover_index(self, time: float) -> int:
        """Rightmost tower that should already be inside the frame.

        Growth is anticipated from the timeline: every tower whose rise begins
        within the lead, plus the one after it. The damped camera then arrives
        before either of them starts.
        """
        if not self.buildings:
            return -1
        ready = time + 2.8
        last = 0
        for index, building in enumerate(self.buildings):
            if building.born - building.grow <= ready:
                last = index
            else:
                break
        return min(len(self.buildings) - 1, last + 1)

    def _target(self, time: float) -> tuple[float, float]:
        if not self.buildings:
            return 0.0, 1.0
        end = self._cover_index(time)
        left = self.buildings[0].x
        right = self.buildings[end].x + self.buildings[end].width
        span = max(1.0, right - left)
        pad = max(48.0, span * 0.05)
        left -= pad
        right += pad
        span = right - left
        # 12% inset in the target. The damped camera tracks this closely, and
        # the finished skyline still clears the 10% margin.
        zoom = min(1.50, (self.width * 0.76) / span)
        return (left + right) * 0.5, max(0.62, zoom)

    def screen_box(self, building: Building, risen: float | None = None) -> tuple[float, float, float, float]:
        """Screen left, top, right, bottom of the tower body at the current camera."""
        if risen is None:
            risen = self.building_height(building)
        ground = self.height * 0.78
        zoom = self.cam_zoom
        left = self.width * 0.5 + (building.x - self.cam_x) * zoom
        right = self.width * 0.5 + (building.x + building.width - self.cam_x) * zoom
        bottom = ground
        top = ground - max(0.0, risen) * zoom
        return left, top, right, bottom

    def update(self, dt: float, time: float, events: list[tuple[float, bool]]) -> None:
        del events
        self.time = time
        target_x, target_zoom = self._target(time)
        self._cam_x = _damp(self._cam_x, target_x, dt, 2.6)
        # Zooming out is the direction that uncovers a new tower. Catch that
        # faster than a pan, and don't punch back in on a bass hit.
        zoom_rate = 3.6 if target_zoom < self._cam_zoom else 1.6
        self._cam_zoom = _damp(self._cam_zoom, target_zoom, dt, zoom_rate)
        energy = section_at(self.sections, time).energy if self.sections else 0.4
        # A short push on the beat. The framing margin is wide enough that this
        # stays inside the picture; it is not a zoom back toward a tight crop.
        push = 0.016 * self.beat_pulse(time) * (0.4 + 0.6 * energy)
        self.cam_x = self._cam_x
        self.cam_zoom = self._cam_zoom * (1.0 + push)


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
        sections=tuple(analysis.sections),
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
    pulse = scene.beat_pulse()
    for index, (sx, sy, radius, warmth) in enumerate(scene.sky_stars):
        tone = 0.75 + 0.2 * warmth
        twinkle = 0.62 + 0.38 * math.sin(scene.time * (0.9 + radius) + sx * 0.013 + index)
        flare = 1.0 + 0.55 * pulse * (0.35 + 0.65 * warmth)
        alpha = (0.28 + 0.42 * radius / 2.0) * twinkle * flare
        ctx.set_source_rgba(tone, tone * (0.92 + 0.05 * (1.0 - warmth)), 0.85 + 0.1 * (1.0 - warmth), min(0.95, alpha))
        ctx.arc(sx, sy, radius * (1.0 + 0.18 * pulse), 0, math.tau)
        ctx.fill()
    moon_x, moon_y = width * 0.78, height * 0.16
    halo = cairo.RadialGradient(moon_x, moon_y, 20, moon_x, moon_y, 58 + 16 * pulse)
    halo.add_color_stop_rgba(0, 0.95, 0.94, 0.88, 0.16 + 0.2 * pulse)
    halo.add_color_stop_rgba(1, 0.85, 0.86, 0.95, 0.0)
    ctx.set_source(halo)
    ctx.arc(moon_x, moon_y, 58 + 16 * pulse, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(0.96, 0.95, 0.9, 0.92)
    ctx.arc(moon_x, moon_y, 26, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(0.03, 0.025, 0.07, 1)
    ctx.arc(moon_x + 11, moon_y - 6, 22, 0, math.tau)
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
        breathe = 0.78 + 0.22 * scene.street_glow()
        if building.sign == 1:
            color = (1.0, 0.55, 0.28, 0.85 * breathe)
        else:
            color = (0.45, 0.85, 1.0, 0.8 * breathe)
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
    _paint_tram(ctx, left, span, scene.time, scene.energy_area(scene.time) * 42.0, max(glow, pulse))
    # Two lanes. The near lane is lower, thicker, and faster.
    lanes = ((44.0, 8.0, 1.0), (112.0, 12.0, -1.0))
    for index in range(8):
        lane_y, thick, direction = lanes[index % 2]
        speed = 170.0 + (index % 4) * 55.0 + (40.0 if direction < 0 else 0.0)
        length = 190.0 + 90.0 * max(pulse, 0.45 * glow) + (index % 3) * 24.0
        # Extra distance is the integral of energy, so a louder section speeds the lane
        # without teleporting a car when the section changes.
        travel = (scene.time * speed + scene.energy_area(scene.time) * (64.0 + (index % 3) * 16.0) + index * span / 8.0) % (
            span + length + 80.0
        )
        _paint_headlight(ctx, left, right, lane_y, thick, direction, length, travel, pulse)


def _paint_tram(ctx, left: float, span: float, time: float, boost: float, flare: float) -> None:
    tram_w = 340.0
    tram_h = 32.0
    tram_x = left + ((time * 78.0 + boost) % (span + tram_w + 160.0)) - tram_w
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
            lag = (building.floors - 1 - row) * 0.03
            ripple = 0.0
            for hit in scene.beat_times:
                ripple = max(ripple, _pulse(scene.time - hit - lag, 0.045, 0.14))
            bass = scene.street_glow()
            if level <= 0.0:
                sheen = max(0.07 * ripple, 0.05 * bass)
                if sheen <= 0.01:
                    ctx.set_source_rgba(0.02, 0.03, 0.05, 0.65)
                else:
                    ctx.set_source_rgba(pal.accent[0], pal.accent[1], pal.accent[2], 0.10 + 0.55 * sheen)
            else:
                ctx.set_source_rgba(
                    pal.accent[0],
                    pal.accent[1],
                    pal.accent[2],
                    min(1.0, level + 0.22 * ripple + 0.16 * bass),
                )
            _round_rect(ctx, x, y, win_w, win_h, 1.6)
            ctx.fill()


@dataclass
class KeyNote:
    time: float
    duration: float
    midi: int
    display: int
    source: str
    span: float = 0.0


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
    cam_x: float = 0.0
    cam_zoom: float = 1.0
    keyboard_dip: float = 0.0
    beat_level: float = 0.0
    _cam_zoom: float = 1.0
    _energy: float = 0.4

    def key_center(self, midi: int) -> tuple[float, float]:
        shown = self._clamp_midi(midi)
        if shown % 12 in WHITES:
            x = self.key_left + self._whites_before(shown) * self.key_w + self.key_w * 0.5
        else:
            x = self.key_left + self._whites_before(shown) * self.key_w
        return x, self.key_top

    def fall_offset(self, remain: float) -> float:
        """How far above the key an event still is. Zero once it has landed.

        Mostly steady, a little slower at the ends, and never flat — a flat
        arrival would stack the next note on top of this one.
        """
        travel = self.key_top * 0.92
        if remain <= 0.0:
            return 0.0
        speed = travel / max(self.lead, 1e-3)
        if remain >= self.lead:
            return travel + (remain - self.lead) * speed * 0.72
        t = remain / self.lead
        eased = (0.72 * t) + (0.28 * _smooth(t))
        return travel * eased

    def note_head(self, note: KeyNote, time: float | None = None) -> tuple[float, float]:
        when = self.time if time is None else time
        x, key_y = self.key_center(note.display)
        return x, key_y - self.fall_offset(note.time - when)

    def note_bounds(self, note: KeyNote, time: float | None = None) -> tuple[float, float]:
        """Top and bottom of the drawn pill. The tail is the release, above the head."""
        when = self.time if time is None else time
        _, key_y = self.key_center(note.display)
        head = key_y - self.fall_offset(note.time - when)
        if note.span <= 0.02:
            return head, head
        tail = key_y - self.fall_offset(note.time + note.span - when)
        return tail, head

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
        # Chorus leans in a little. The keys and the notes share this scale.
        return 1.0 + 0.04 * self._energy

    def update(self, dt: float, time: float, events: list[tuple[float, bool]]) -> None:
        del events
        self.time = time
        raw = section_at(self.analysis.sections, time).energy
        self._energy = _damp(self._energy, raw, dt, 1.15)
        self._cam_zoom = _damp(self._cam_zoom, self._target_zoom(time), dt, 1.25)
        beat = _times_pulse(self.analysis.beats, time, 0.04, 0.15)
        strong = _times_pulse(self.analysis.strong_beats, time, 0.045, 0.18)
        self.beat_level = beat
        self.cam_zoom = self._cam_zoom * (1.0 + 0.012 * beat * (0.35 + 0.65 * self._energy))
        self.cam_x = (1.6 + 6.0 * self._energy) * math.sin(time * 0.62)
        self.keyboard_dip = (3.0 + 4.2 * self._energy) * strong


def _four_octave_span(center: int) -> tuple[int, int]:
    """C-to-B, four octaves, with the melody above a free bass octave."""
    best: tuple[float, int, int] | None = None
    for lo in range(36, 72, 12):
        hi = lo + 47
        if not lo <= center <= hi:
            continue
        score = abs((center - lo) / 47.0 - 0.62)
        if best is None or score < best[0]:
            best = (score, lo, hi)
    if best is None:
        return 48, 95
    return best[1], best[2]


def _keys_for(lo: int, hi: int, pitch_class: int) -> list[int]:
    first = lo + (pitch_class - lo) % 12
    return list(range(first, hi + 1, 12))


def _slot_free(taken: dict[int, list[tuple[float, float]]], midi: int, time: float, duration: float) -> bool:
    gap = 0.05
    start, end = time, time + duration + gap
    for other, length in taken.get(midi, ()):
        if start < other + length + gap and other < end:
            return False
    return True


def _hold(taken: dict[int, list[tuple[float, float]]], midi: int, time: float, duration: float) -> None:
    taken.setdefault(midi, []).append((time, duration))


def _chord_classes(column: list[float], count: int, root_hint: int) -> list[int]:
    if column and max(column) > 1e-6:
        order = sorted(range(12), key=lambda pitch: column[pitch], reverse=True)
    else:
        order = [root_hint % 12, (root_hint + 4) % 12, (root_hint + 7) % 12, (root_hint + 11) % 12]
    picked: list[int] = []
    for pitch in order:
        if pitch in picked:
            continue
        picked.append(pitch)
        if len(picked) >= count:
            break
    extra = 0
    while len(picked) < count:
        candidate = (picked[0] + 4 + 3 * extra) % 12
        extra += 1
        if candidate not in picked:
            picked.append(candidate)
    return picked[:count]


def _bass_times(analysis: Analysis) -> list[float]:
    beats = [float(beat) for beat in analysis.beats]
    if not beats:
        return []
    if analysis.bands and analysis.rms_times:
        lows = []
        for beat in beats:
            index = min(range(len(analysis.rms_times)), key=lambda i: abs(analysis.rms_times[i] - beat))
            row = analysis.bands[index]
            lows.append(float(row[0]) if row else 0.0)
        if lows and max(lows) > 0.05:
            ordered = sorted(lows)
            cut = max(0.42, ordered[int(0.62 * (len(ordered) - 1))])
            found = [beat for beat, level in zip(beats, lows) if level >= cut]
            if found:
                return found
    return [float(beat) for beat in analysis.strong_beats] or beats[::4]


def _voice(lo: int, hi: int, pitch_class: int, time: float, duration: float, taken, prefer: str) -> int | None:
    keys = _keys_for(lo, hi, pitch_class)
    if not keys:
        return None
    if prefer == "bass":
        ordered = keys[:2]
    elif prefer == "high":
        ordered = list(reversed(keys[1:] or keys))
    else:
        mid = keys[1:-1] or keys
        ordered = mid
    for midi in ordered:
        if _slot_free(taken, midi, time, duration):
            return midi
    for midi in keys:
        if _slot_free(taken, midi, time, duration):
            return midi
    return None


def _add_arrangement(notes: list[KeyNote], analysis: Analysis, lo: int, hi: int) -> None:
    """Chord tones on beats, bass on low hits, and a short rising ripple on strong beats.

    Every added onset lands on its own time. A key is used only when the previous
    note on it has already finished, so the same key never stacks.
    """
    beats = [float(beat) for beat in analysis.beats]
    if not beats:
        return
    taken: dict[int, list[tuple[float, float]]] = {}
    for note in notes:
        _hold(taken, note.display, note.time, note.duration)
    columns = chroma_columns(analysis, beats)
    strong = {round(float(hit), 5) for hit in analysis.strong_beats}
    bass_hits = {round(hit, 5) for hit in _bass_times(analysis)}
    melody_at = {}
    for note in notes:
        if note.source == "pitch":
            melody_at.setdefault(round(note.time, 3), note.midi)
    for index, beat in enumerate(beats):
        if beat > analysis.duration + 1e-3:
            continue
        energy = section_at(analysis.sections, beat).energy
        count = 3 if energy < 0.62 else 4
        column = columns[index] if index < len(columns) else [0.0] * 12
        root_hint = melody_at.get(round(beat, 3), notes[0].midi if notes else 60)
        classes = _chord_classes(column, count, root_hint)
        duration = max(0.12, min(0.34, 0.12 + 0.2 * energy))
        for chord_index, pitch_class in enumerate(classes):
            prefer = "mid"
            if chord_index == count - 1 and energy >= 0.62:
                prefer = "high"
            midi = _voice(lo, hi, pitch_class, beat, duration, taken, prefer)
            if midi is None:
                continue
            _hold(taken, midi, beat, duration)
            notes.append(KeyNote(beat, duration, midi, midi, "chroma", duration))
        if round(beat, 5) in bass_hits and classes:
            bass_duration = max(0.16, min(0.42, 0.18 + 0.2 * energy))
            midi = _voice(lo, hi, classes[0], beat, bass_duration, taken, "bass")
            if midi is not None:
                _hold(taken, midi, beat, bass_duration)
                notes.append(KeyNote(beat, bass_duration, midi, midi, "bass", bass_duration))
        if round(beat, 5) in strong:
            steps = 2 if energy < 0.45 else 3 if energy < 0.72 else 4
            for step in range(steps):
                when = beat + 0.085 * (step + 1)
                if when > analysis.duration:
                    break
                pitch_class = classes[step % len(classes)]
                ripple = max(0.12, min(0.22, 0.1 + 0.08 * energy))
                midi = _voice(lo, hi, pitch_class, when, ripple, taken, "high")
                if midi is None:
                    continue
                _hold(taken, midi, when, ripple)
                notes.append(KeyNote(when, ripple, midi, midi, "arp", ripple))
    notes.sort(key=lambda note: (note.time, note.display))


def _build_piano(width: int, height: int, analysis: Analysis, seed: int) -> PianoScene:
    del seed
    tones = extract_tones(analysis)
    if not tones:
        tones = [Tone(0.0, 0.4, 60, "chroma")]
    midis = [tone.midi for tone in tones]
    center = sorted(midis)[len(midis) // 2]
    lo, hi = _four_octave_span(center)
    notes = []
    for tone in tones:
        if not 0.0 <= tone.time <= analysis.duration + 1e-3:
            continue
        shown = tone.midi
        while shown < lo:
            shown += 12
        while shown > hi:
            shown -= 12
        notes.append(KeyNote(tone.time, tone.duration, tone.midi, shown, tone.source, tone.duration))
    _add_arrangement(notes, analysis, lo, hi)
    _open_same_key_gaps(notes)
    white_count = sum(1 for tone in range(lo, hi + 1) if tone % 12 in WHITES)
    white_count = max(1, white_count)
    board = float(width) * 0.875
    key_w = board / white_count
    key_left = (float(width) - board) / 2.0
    white_h = min(float(height) * 0.24, key_w * 4.35)
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
        key_top=float(height) - white_h - float(height) * 0.11,
        white_h=white_h,
        lead=1.68,
        timing={},
        sync_report={},
        analysis=analysis,
    )
    worst = 0.0
    for note in notes:
        x, y = scene.note_head(note, note.time)
        kx, ky = scene.key_center(note.display)
        worst = max(worst, math.hypot(x - kx, y - ky))
    melody = [note for note in notes if note.source == "pitch"]
    beat_gap = _median_gap([])
    if melody and analysis.beats:
        gaps = sorted(_nearest(list(analysis.beats), note.time) for note in melody)
        beat_gap = gaps[len(gaps) // 2]
    scene.timing = {"max_timing_error_s": 0.0, "max_position_error_px": worst, "hits": len(notes)}
    scene.sync_report = {
        "notes": len(notes),
        "pitched": len(melody),
        "chroma": sum(1 for note in notes if note.source == "chroma"),
        "bass": sum(1 for note in notes if note.source == "bass"),
        "arp": sum(1 for note in notes if note.source == "arp"),
        "median_beat_gap_ms": round(beat_gap * 1000.0, 1),
    }
    scene._energy = section_at(analysis.sections, 0.0).energy
    scene._cam_zoom = scene._target_zoom(0.0)
    scene.cam_zoom = scene._cam_zoom
    return scene


def _open_same_key_gaps(notes: list[KeyNote]) -> None:
    """Shorten each tail so the next note on that key has air in front of it.

    The onset stays put. `duration` stays the musical length; `span` is what is drawn.
    """
    groups: dict[int, list[KeyNote]] = {}
    for note in notes:
        groups.setdefault(note.display, []).append(note)
    for group in groups.values():
        group.sort(key=lambda note: (note.time, -note.duration))
        for index, note in enumerate(group[:-1]):
            interval = group[index + 1].time - note.time
            if interval <= 0.03:
                note.span = 0.0
                continue
            gap = min(0.05, interval * 0.36)
            note.span = min(note.duration, max(0.0, interval - gap))
        group[-1].span = group[-1].duration


def _note_rgb(note: KeyNote, energy: float) -> tuple[float, float, float]:
    base = NOTE_COLORS[note.midi % 12]
    lift = 0.05 + 0.1 * energy
    return tuple(min(1.0, channel + (1.0 - channel) * lift) for channel in base)


def _paint_piano(ctx, scene: PianoScene) -> None:
    pal = active_palette()
    width, height = scene.width, scene.height
    gradient = cairo.LinearGradient(0, 0, 0, height)
    gradient.add_color_stop_rgb(0, *pal.bg_top)
    gradient.add_color_stop_rgb(1, *pal.bg_bottom)
    ctx.set_source(gradient)
    ctx.paint()
    # Glows have to composite over the sky. Source mode would punch them out as dark discs.
    ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.save()
    ctx.translate(width / 2.0, scene.key_top)
    ctx.translate(scene.cam_x, scene.keyboard_dip)
    ctx.scale(scene.cam_zoom, scene.cam_zoom)
    ctx.translate(-width / 2.0, -scene.key_top)
    _paint_piano_air(ctx, scene, pal)
    _paint_key_beams(ctx, scene)
    _paint_falling_notes(ctx, scene)
    _paint_keyboard(ctx, scene, pal)
    _paint_key_floor(ctx, scene)
    _paint_key_sparks(ctx, scene)
    ctx.restore()


def _live_colors(scene: PianoScene) -> list[tuple[float, float, float]]:
    colors: list[tuple[float, float, float]] = []
    for note in scene.notes:
        remain = note.time - scene.time
        age = scene.time - note.time
        if -0.05 <= remain <= scene.lead or 0.0 <= age <= note.duration + 0.2:
            colors.append(NOTE_COLORS[note.midi % 12])
            if len(colors) >= 8:
                break
    return colors


def _paint_piano_air(ctx, scene: PianoScene, pal) -> None:
    """Soft color in the space above the keys. The wash follows whatever is in the air."""
    colors = _live_colors(scene) or [pal.accent2]
    red = sum(color[0] for color in colors) / len(colors)
    green = sum(color[1] for color in colors) / len(colors)
    blue = sum(color[2] for color in colors) / len(colors)
    energy = scene._energy
    beat = scene.beat_level
    wash = 0.11 + 0.16 * energy + 0.09 * beat
    glow = cairo.RadialGradient(scene.width * 0.5, scene.key_top * 0.72, 40, scene.width * 0.5, scene.key_top * 0.35, scene.width * 0.72)
    glow.add_color_stop_rgba(0.0, red, green, blue, wash)
    glow.add_color_stop_rgba(0.55, red * 0.7, green * 0.75, blue, wash * 0.45)
    glow.add_color_stop_rgba(1.0, red, green, blue, 0.0)
    ctx.set_source(glow)
    ctx.rectangle(0, 0, scene.width, scene.key_top + 4)
    ctx.fill()
    span = max(1.0, scene.key_w * sum(1 for tone in range(scene.lo, scene.hi + 1) if tone % 12 in WHITES))
    count = 28 + int(22 * energy)
    for index in range(count):
        phase = index * 2.399
        drift = (index * 83.0 + scene.time * (14.0 + (index % 5) * 3.0)) % span
        x = scene.key_left + drift
        travel = scene.key_top * 0.9
        y = travel - ((scene.time * (18.0 + (index % 6) * 4.0) + index * 37.0) % travel)
        twinkle = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(scene.time * 1.6 + phase))
        alpha = (0.06 + 0.1 * energy + 0.05 * beat) * twinkle
        if alpha < 0.03:
            continue
        tone = colors[index % len(colors)]
        ctx.set_source_rgba(tone[0], tone[1], tone[2], alpha)
        ctx.arc(x, y, 1.15 + (index % 3) * 0.35, 0, math.tau)
        ctx.fill()


def _paint_key_beams(ctx, scene: PianoScene) -> None:
    """Colored light rising out of a key that is about to be played or has just been played."""
    best: dict[int, tuple[float, tuple[float, float, float]]] = {}
    for note in scene.notes:
        level = _hit_level(note, scene.time)
        remain = note.time - scene.time
        if 0.0 < remain < scene.lead:
            level = max(level, 0.28 * (1.0 - remain / scene.lead))
        if level <= 0.04:
            continue
        shown = scene._clamp_midi(note.display)
        color = NOTE_COLORS[note.midi % 12]
        previous = best.get(shown)
        if previous is None or level > previous[0]:
            best[shown] = (level, color)
    for midi, (level, color) in best.items():
        x, key_y = scene.key_center(midi)
        black = midi % 12 not in WHITES
        beam_w = scene.key_w * (0.34 if black else 0.5)
        beam = cairo.LinearGradient(0, 24, 0, key_y)
        beam.add_color_stop_rgba(0.0, color[0], color[1], color[2], 0.0)
        beam.add_color_stop_rgba(0.55, color[0], color[1], color[2], 0.1 * level)
        beam.add_color_stop_rgba(1.0, color[0], color[1], color[2], 0.38 * level)
        ctx.set_source(beam)
        ctx.rectangle(x - beam_w / 2.0, 24, beam_w, key_y - 24)
        ctx.fill()


def _paint_falling_notes(ctx, scene: PianoScene) -> None:
    """Pills, back to front, each one kept inside its own slot on the key."""
    visible: list[tuple[float, float, int, KeyNote, float, float]] = []
    for index, note in enumerate(scene.notes):
        if note.span <= 0.02:
            continue
        top, bottom = scene.note_bounds(note)
        if bottom - top < 1.2 or bottom < -24 or top > scene.key_top + 6:
            continue
        visible.append((bottom, note.time, index, note, top, bottom))
    visible.sort()
    for _, _, _, note, top, bottom in visible:
        _paint_note_pill(ctx, scene, note, top, bottom)


def _paint_note_pill(ctx, scene: PianoScene, note: KeyNote, top: float, bottom: float) -> None:
    color = _note_rgb(note, scene._energy)
    x, _ = scene.note_head(note)
    shown = scene._clamp_midi(note.display)
    black = shown % 12 not in WHITES
    # Narrow enough that a black pill and its white neighbours do not touch.
    note_w = scene.key_w * (0.36 if black else 0.52)
    height = bottom - top
    remain = note.time - scene.time
    if remain > 0.0:
        near = _smooth(1.0 - min(1.0, remain / 0.32))
        hot = 0.8 + 0.2 * near
    else:
        hot = 1.0 + 0.28 * _pulse(scene.time - note.time, 0.03, 0.11)
    hot *= 1.0 + 0.14 * scene.beat_level * (0.45 + 0.55 * scene._energy)
    rect_x = x - note_w / 2.0
    radius = note_w * 0.5
    # Soft halo, kept close so it does not spill into the next note or the next key.
    halo = 5.0
    _round_rect(ctx, rect_x - halo * 0.5, top - 2.0, note_w + halo, height + 4.0, radius)
    ctx.set_source_rgba(color[0], color[1], color[2], 0.16 * min(hot, 1.35))
    ctx.fill()
    _round_rect(ctx, rect_x, top, note_w, height, radius)
    fill = cairo.LinearGradient(0, top, 0, bottom)
    fill.add_color_stop_rgba(0.0, color[0], color[1], color[2], 0.28 * hot)
    fill.add_color_stop_rgba(0.45, color[0], color[1], color[2], 0.9 * min(hot, 1.0))
    fill.add_color_stop_rgba(1.0, min(1, color[0] + 0.18), min(1, color[1] + 0.16), min(1, color[2] + 0.12), min(1.0, hot))
    ctx.set_source(fill)
    ctx.fill()
    if height > note_w * 0.9:
        inset = 2.2
        shine_h = min(height * 0.42, note_w * 1.3)
        _round_rect(ctx, rect_x + inset, bottom - shine_h, note_w - inset * 2, shine_h - 1.5, max(1.0, radius - inset))
        ctx.set_source_rgba(1.0, 0.98, 0.96, 0.22 * min(hot, 1.1))
        ctx.fill()
    _paint_note_sparkles(ctx, scene, note, x, bottom, note_w, color, hot)


def _paint_note_sparkles(ctx, scene, note: KeyNote, x: float, head_y: float, note_w: float, color, hot: float) -> None:
    """A few specks drifting off the head. They stay inside the lane and fade quickly."""
    if head_y < -10 or head_y > scene.key_top + 4:
        return
    for index in range(3):
        period = 0.34
        local = (scene.time - note.time + 0.2 + index * 0.11) % period
        life = 0.26
        if local >= life:
            continue
        u = local / life
        fade = _smooth(1.0 - u) * _smooth(min(1.0, local / 0.05)) * min(hot, 1.2)
        if fade < 0.04:
            continue
        side = (index - 1) * note_w * 0.42 * u
        rise = 6.0 + 16.0 * u
        ctx.set_source_rgba(1.0, 0.98, 0.95, 0.75 * fade)
        ctx.arc(x + side, head_y - rise, 1.35, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(*color, 0.45 * fade)
        ctx.arc(x + side, head_y - rise, 2.4, 0, math.tau)
        ctx.fill()


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


def _paint_key_floor(ctx, scene: PianoScene) -> None:
    """A dark lip and a fading reflection so the keyboard sits on a surface."""
    pressed = _press_level(scene)
    top = scene.key_top + scene.white_h
    whites = [tone for tone in range(scene.lo, scene.hi + 1) if tone % 12 in WHITES]
    if not whites:
        return
    board = scene.key_w * len(whites)
    reflect_h = min(scene.height - top - 6.0, scene.white_h * 0.62)
    if reflect_h < 8.0:
        return
    ctx.set_source_rgba(0.0, 0.0, 0.0, 0.45)
    ctx.rectangle(scene.key_left - 6, top, board + 12, 7)
    ctx.fill()
    for tone in whites:
        x = scene.key_left + scene._whites_before(tone) * scene.key_w
        level = pressed.get(tone, 0.0)
        color = NOTE_COLORS[tone % 12] if level > 0.0 else (0.75, 0.77, 0.82)
        alpha = 0.16 + 0.28 * level
        grad = cairo.LinearGradient(0, top, 0, top + reflect_h)
        grad.add_color_stop_rgba(0.0, color[0], color[1], color[2], alpha)
        grad.add_color_stop_rgba(1.0, color[0], color[1], color[2], 0.0)
        ctx.set_source(grad)
        ctx.rectangle(x + 1.4, top + 6, scene.key_w - 2.8, reflect_h - 6)
        ctx.fill()
    black_w = scene.key_w * 0.58
    black_reflect = reflect_h * 0.55
    for tone in range(scene.lo, scene.hi + 1):
        if tone % 12 in WHITES:
            continue
        x = scene.key_left + scene._whites_before(tone) * scene.key_w - black_w / 2.0
        level = pressed.get(tone, 0.0)
        color = NOTE_COLORS[tone % 12] if level > 0.0 else (0.05, 0.05, 0.07)
        grad = cairo.LinearGradient(0, top, 0, top + black_reflect)
        grad.add_color_stop_rgba(0.0, color[0], color[1], color[2], 0.28 + 0.35 * level)
        grad.add_color_stop_rgba(1.0, color[0], color[1], color[2], 0.0)
        ctx.set_source(grad)
        ctx.rectangle(x, top + 6, black_w, black_reflect)
        ctx.fill()


def _paint_key_sparks(ctx, scene: PianoScene) -> None:
    """Impact after the keys: a flash on the key, one shockwave, and streaks on the lead notes."""
    for note in scene.notes:
        age = scene.time - note.time
        if age < 0.0 or age > 0.46:
            continue
        color = _note_rgb(note, scene._energy)
        x, key_y = scene.key_center(note.display)
        fade = _smooth(1.0 - age / 0.46)
        half = scene.key_w * 0.62
        splash = cairo.LinearGradient(x - half, key_y, x + half, key_y)
        splash.add_color_stop_rgba(0.0, *color, 0.0)
        splash.add_color_stop_rgba(0.5, 1.0, 0.98, 0.95, 0.9 * fade)
        splash.add_color_stop_rgba(1.0, *color, 0.0)
        ctx.set_source(splash)
        ctx.rectangle(x - half, key_y - 7, half * 2, 9)
        ctx.fill()
        ring = _smooth(min(1.0, age / 0.32))
        ctx.set_source_rgba(*color, fade * 0.9)
        ctx.set_line_width(3.2 * fade + 1.4)
        ctx.arc(x, key_y + 2.0, 22.0 + ring * 70.0, math.pi * 1.08, math.pi * 1.92)
        ctx.stroke()
        if note.source == "chroma":
            continue
        for spark in range(8):
            angle = math.pi * 1.05 + spark * (math.pi * 0.9 / 7.0)
            dist = 46.0 + age * (280.0 + spark * 16.0)
            x2 = x + math.cos(angle) * dist
            y2 = key_y + math.sin(angle) * dist
            x1 = x + math.cos(angle) * (dist - 18.0)
            y1 = key_y + math.sin(angle) * (dist - 18.0)
            ctx.set_source_rgba(*color, fade * 0.85)
            ctx.set_line_width(3.4 * fade + 0.6)
            ctx.move_to(x1, y1)
            ctx.line_to(x2, y2)
            ctx.stroke()
            ctx.set_source_rgba(1.0, 0.98, 0.95, fade * 0.9)
            ctx.set_line_width(1.4 * fade + 0.4)
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
        ctx.set_source_rgba(1, 1, 1, 0.16 if level <= 0 else 0.34)
        ctx.rectangle(x + gap + 1.2, scene.key_top + sink + 3.0, 1.5, max(4.0, scene.white_h - sink - 14.0))
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
    kind: str = "pitch"


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
    phrase_span: dict | None = None
    phrase_centroid: dict | None = None
    caption_title: str = ""
    caption_artist: str = ""
    time: float = 0.0
    cam_x: float = 0.0
    cam_y: float = 0.0
    cam_zoom: float = 1.0
    _cam_x: float = 0.0
    _cam_y: float = 0.0
    _cam_zoom: float = 1.0
    # Steady clock for drift. The renderer samples a frame on its beat, which
    # would hitch a sine; musical flashes still read `time`.
    _drift: float = -1.0
    _weights: dict | None = None
    # When the next phrase is actually on screen. None until the first query.
    _ready: dict | None = None

    def brightness(self, star: Star, time: float | None = None) -> float:
        when = self.time if time is None else time
        age = when - star.time
        if age < -0.22:
            return 0.0
        if age <= 0.0:
            return _smooth((age + 0.22) / 0.22)
        life = self.life(star.phrase, when)
        if life <= 0.0:
            return 0.0
        flash = math.exp(-age * 3.4)
        mix = self.slow_mix(when)
        depth = 0.08
        rate = 1.5 + 1.8 * self.energy(when)
        clock = when if mix <= 0.0 else self.drift_time(when)
        if mix > 0.0:
            # A slow bar would otherwise sit at one brightness between hits.
            depth = 0.08 + 0.12 * mix
            rate *= 1.0 - 0.28 * mix
        rest = 0.46 + depth * math.sin(clock * rate + star.phase)
        level = rest + (1.0 - rest) * flash + self.beat_lift(star, when)
        return max(0.0, min(1.0, level * life))

    def energy(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        return section_at(self.analysis.sections, when).energy

    def bands(self, time: float | None = None) -> tuple[float, float]:
        """Low and high band energy in 0..1. Both are 0 when the analysis has no bands."""
        when = self.time if time is None else time
        times = self.analysis.rms_times
        rows = self.analysis.bands
        if not times or not rows:
            return 0.0, 0.0
        index = min(range(len(times)), key=lambda i: abs(times[i] - when))
        row = rows[index]
        if not row:
            return 0.0, 0.0
        low = float(sum(row[:2]) / max(1, min(2, len(row))))
        high = float(sum(row[-3:]) / max(1, min(3, len(row))))
        return low, high

    def pulse_gap(self, time: float | None = None) -> float:
        """Median of the beat intervals nearest `time`, in seconds."""
        when = self.time if time is None else time
        beats = self.analysis.beats
        if len(beats) < 2:
            tempo = float(self.analysis.tempo or 0.0)
            return 60.0 / tempo if tempo > 1.0 else 0.5
        gaps = [beats[i] - beats[i - 1] for i in range(1, len(beats))]
        center = 0
        for index, start in enumerate(beats[:-1]):
            if start <= when:
                center = index
            else:
                break
        lo = max(0, center - 1)
        hi = min(len(gaps), center + 2)
        sample = sorted(gaps[lo:hi])
        return max(0.2, sample[len(sample) // 2])

    def slow_mix(self, time: float | None = None) -> float:
        """0 on a fast pulse, 1 when a bar is long enough that a short flash would freeze.

        110 BPM (0.545s) still uses the existing hit, so a 110–123 BPM song stays
        crisp. By about 90 BPM (0.66s) the picture is in the slow drift, and the
        second half of the bar keeps moving. A slower bar stays there.
        """
        gap = self.pulse_gap(time)
        if gap <= 0.55:
            return 0.0
        if gap >= 0.66:
            return 1.0
        return _smooth((gap - 0.55) / (0.66 - 0.55))

    def drift_time(self, when: float) -> float:
        """Clock for continuous motion. A direct query uses `when` unchanged."""
        if self._drift < 0.0 or abs(when - self.time) > 1e-4 or self.slow_mix(when) <= 0.0:
            return when
        return self._drift

    def _event_window(self, time: float, base: float, fraction: float, cap: float) -> float:
        """How long a beat gesture lasts. Fast songs keep `base` exactly."""
        mix = self.slow_mix(time)
        if mix <= 0.0:
            return base
        gap = self.pulse_gap(time)
        slow = min(cap, max(base, gap * fraction))
        # End before the next beat, or the gesture restarts mid-travel and jumps.
        slow = min(slow, max(base, gap * 0.94))
        return base + mix * (slow - base)

    def _envelope(self, times, when: float, rise: float, release: float) -> float:
        mix = self.slow_mix(when)
        if mix > 0.0:
            gap = self.pulse_gap(when)
            # The frame before a beat is up to two frames early, because that
            # frame is drawn on the beat itself. A longer rise is already bright
            # by then, so the hit is not a one-frame slam.
            slow_rise = min(0.26, max(rise, gap * 0.24))
            slow_release = min(0.62, max(release, gap * 0.46))
            rise = rise + mix * (slow_rise - rise)
            release = release + mix * (slow_release - release)
            span = rise + release
            limit = gap * 0.92
            if span > limit > 0.0:
                rise *= limit / span
                release *= limit / span
        return _hit_flash(times, when, rise, release)

    def beat_level(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        # Peak on the beat itself. The frame that contains the hit is the bright one.
        return self._envelope(self.analysis.beats, when, 0.04, 0.22)

    def strong_level(self, time: float | None = None) -> float:
        when = self.time if time is None else time
        return self._envelope(self.analysis.strong_beats, when, 0.04, 0.28)

    def glint_u(self, time: float | None = None) -> float | None:
        """0..1 along the live figure for this beat, or None once the light has passed.

        Fast songs cross in 0.32s, same as before. Slow songs ease across most of the bar.
        """
        when = self.time if time is None else time
        age = _since(self.analysis.beats, when)
        window = self._event_window(when, 0.32, 0.88, 1.25)
        if age is None or age > window or window <= 0.0:
            return None
        u = age / window
        mix = self.slow_mix(when)
        if mix <= 0.0:
            return u
        return u + mix * (_smooth(u) - u)

    def beat_lift(self, star: Star, time: float | None = None) -> float:
        """How much this star brightens for the current beat. The live phrase answers hardest."""
        when = self.time if time is None else time
        beat = self.beat_level(when)
        strong = self.strong_level(when)
        if star.phrase == self.active_phrase(when):
            return 0.4 * beat + 0.22 * strong
        return 0.1 * beat

    def shown(self, star: Star, time: float | None = None) -> bool:
        """True once the glyph is drawn. The beat light and the ripple use this."""
        return self.brightness(star, time) > 0.02

    def lit_edges(self, time: float | None = None) -> list[tuple[Star, Star]]:
        """Edges the beat light may travel.

        Both stars are on screen, and both notes have arrived by this beat.
        The path is fixed for the whole crossing, so it does not run into a
        star that appears later, and it does not jump when one does.
        """
        when = self.time if time is None else time
        phrase = self.active_phrase(when)
        if phrase < 0:
            return []
        age = _since(self.analysis.beats, when)
        if age is None:
            return []
        beat_at = when - age
        edges: list[tuple[Star, Star]] = []
        for a, b in self.links:
            left = self.stars[a]
            right = self.stars[b]
            if left.phrase != phrase or right.phrase != phrase:
                continue
            if left.time > beat_at + 1e-4 or right.time > beat_at + 1e-4:
                continue
            if not self.shown(left, when) or not self.shown(right, when):
                continue
            edges.append((left, right))
        return edges

    def ripple_members(self, time: float | None = None) -> list[Star]:
        """Stars the beat ring is centered on: the live figure, and only the lit ones."""
        when = self.time if time is None else time
        phrase = self.active_phrase(when)
        if phrase < 0 or self.life(phrase, when) <= 0.2:
            return []
        return [star for star in self.stars if star.phrase == phrase and self.shown(star, when)]

    def _successor_ready(self, phrase: int) -> float | None:
        if self._ready is None:
            self._ready = self._build_successor_ready()
        return self._ready.get(phrase)

    def _build_successor_ready(self) -> dict[int, float]:
        """When the next figure has enough stars to read as a constellation."""
        spans = self.phrase_span or {}
        by_phrase: dict[int, list[float]] = {}
        for star in self.stars:
            by_phrase.setdefault(star.phrase, []).append(star.time)
        ordered = sorted((phrase for phrase in spans if by_phrase.get(phrase)), key=lambda phrase: spans[phrase][0])
        ready: dict[int, float] = {}
        for prev, nxt in zip(ordered, ordered[1:]):
            times = sorted(by_phrase[nxt])
            # A one-star figure counts as soon as it appears. A larger one counts
            # once a second star is up, so a chorus does not open on a single dot.
            mark = times[0] if len(times) == 1 else times[1]
            ready[prev] = mark - 0.22
        return ready

    def life(self, phrase: int, time: float | None = None) -> float:
        """Older figures ease out. The latest one stays, and so does the one before a rest.

        A phrase does not fade until the next one is actually on screen. Otherwise
        a long rest leaves the milky way with no constellation, and the chorus
        opens on a single star.
        """
        when = self.time if time is None else time
        spans = self.phrase_span or {}
        span = spans.get(phrase)
        if span is None:
            return 1.0
        _start, end = span
        ready = self._successor_ready(phrase)
        if ready is None:
            return 1.0
        hold = max(end + 6.5, ready)
        if when <= hold:
            return 1.0
        return _smooth(1.0 - min(1.0, (when - hold) / 2.4))

    def latest_phrase(self) -> int:
        spans = self.phrase_span or {}
        if not spans:
            return -1
        return max(spans, key=lambda phrase: spans[phrase][0])

    def active_phrase(self, time: float | None = None) -> int:
        when = self.time if time is None else time
        spans = self.phrase_span or {}
        best = -1
        best_start = -1e9
        for phrase, (start, end) in spans.items():
            if start - 0.2 <= when <= end + 2.0 and start >= best_start:
                best = phrase
                best_start = start
        if best >= 0:
            return best
        for phrase, (start, _end) in spans.items():
            if start <= when and start >= best_start:
                best = phrase
                best_start = start
        return best

    def placed(self, star: Star, time: float | None = None) -> tuple[float, float]:
        """A few pixels of outward breath on the live figure. Positions in the layout stay put.

        On a fast pulse this is the beat swell only. On a slow pulse the live figure
        also floats, and it eases back into the layout instead of snapping.
        """
        when = self.time if time is None else time
        mix = self.slow_mix(when)
        active = star.phrase == self.active_phrase(when)
        weight = 1.0
        if mix > 0.0:
            weights = self._weights or {}
            if star.phrase in weights:
                weight = weights[star.phrase]
            elif not active:
                weight = 0.0
            if weight < 0.02:
                return star.x, star.y
        elif not active:
            return star.x, star.y
        clock = when if mix <= 0.0 else self.drift_time(when)
        centroid = (self.phrase_centroid or {}).get(star.phrase)
        beat = self.beat_level(when)
        energy = self.energy(when)
        if centroid is None:
            bob = (1.2 + 3.2 * beat) * math.sin(clock * 6.0 + star.phase)
            return star.x, star.y + bob * (1.0 if mix <= 0.0 else weight)
        cx, cy = centroid
        dx, dy = star.x - cx, star.y - cy
        dist = math.hypot(dx, dy)
        push = (1.2 + 3.4 * beat) * (0.45 + 0.55 * energy)
        sway = 1.3 * math.sin(clock * 0.9 + star.phase)
        ox = oy = 0.0
        if mix > 0.0:
            slow_sway = (4.2 + 1.4 * energy) * math.sin(clock * 0.62 + star.phase)
            sway = (1.0 - mix) * sway + mix * slow_sway
            # Fast enough to read between hits, slow enough that one frame never jumps.
            float_amp = weight * mix * (18.0 + 12.0 * (1.0 - min(1.0, energy)))
            ox = float_amp * math.sin(clock * 0.62 + star.phrase * 1.3)
            oy = float_amp * 0.72 * math.cos(clock * 0.44 + star.phrase * 0.8)
            ang = weight * mix * 0.07 * math.sin(clock * 0.28 + star.phrase)
            cos = math.cos(ang)
            sin = math.sin(ang)
            dx, dy = dx * cos - dy * sin, dx * sin + dy * cos
            dist = math.hypot(dx, dy)
            push *= weight
            sway *= weight
        if dist < 1.0:
            return star.x + ox, star.y + oy + sway + push * math.sin(star.phase)
        scale = (push + sway) / dist
        return star.x + dx * scale + ox, star.y + dy * scale + oy

    def _target(self, time: float) -> tuple[float, float, float]:
        # Drift stays inside the layout margin. Zoom stays at 1 so edge stars are not pushed out.
        energy = self.energy(time)
        mix = self.slow_mix(time)
        clock = time if mix <= 0.0 else self.drift_time(time)
        amp_x = (8.0 + 6.0 * energy) + mix * (22.0 + 6.0 * energy)
        amp_y = (5.0 + 4.0 * energy) + mix * (14.0 + 4.0 * energy)
        return (
            amp_x * math.sin(clock * (0.05 + 0.20 * mix)),
            amp_y * math.cos(clock * (0.036 + 0.15 * mix)),
            1.0,
        )

    def _advance_drift(self, dt: float, time: float) -> None:
        if self._drift < 0.0:
            self._drift = time
            return
        delta = time - self._drift
        if delta < -1e-3 or delta > 0.45:
            self._drift = time
            return
        self._drift += dt

    def _advance_weights(self, dt: float, time: float) -> None:
        if self.slow_mix(time) <= 0.0:
            return
        if self._weights is None:
            self._weights = {}
        active = self.active_phrase(time)
        for phrase in self.phrase_span or {}:
            target = 1.0 if phrase == active else 0.0
            current = self._weights.get(phrase, target)
            self._weights[phrase] = _damp(current, target, dt, 4.0)

    def update(self, dt: float, time: float, events: list[tuple[float, bool]]) -> None:
        del events
        self._advance_drift(dt, time)
        self.time = time
        self._advance_weights(dt, time)
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
    tones = _accent_tones(analysis, tones)
    phrases = _group_phrases(tones, analysis)
    phrases = _company_for_strays(phrases, analysis)
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
        "phrases": len({star.phrase for star in stars}),
        "median_beat_gap_ms": round((gaps[len(gaps) // 2] if gaps else 0.0) * 1000.0, 1),
    }
    spans: dict[int, tuple[float, float]] = {}
    centroids: dict[int, tuple[float, float]] = {}
    for star in stars:
        start, end = spans.get(star.phrase, (star.time, star.time))
        spans[star.phrase] = (min(start, star.time), max(end, star.time))
    for phrase, start_end in spans.items():
        del start_end
        members = [star for star in stars if star.phrase == phrase]
        centroids[phrase] = (
            sum(star.x for star in members) / len(members),
            sum(star.y for star in members) / len(members),
        )
    scene.phrase_span = spans
    scene.phrase_centroid = centroids
    scene._cam_x, scene._cam_y, scene._cam_zoom = scene._target(0.0)
    scene.cam_x, scene.cam_y, scene.cam_zoom = scene._cam_x, scene._cam_y, scene._cam_zoom
    return scene


def _accent_tones(analysis: Analysis, tones: list[Tone]) -> list[Tone]:
    """Melody draws the figures. A thin melody borrows strong beats so the sky is not one stray dot."""
    ordered = sorted(tones, key=lambda tone: tone.time)
    strong = [float(hit) for hit in analysis.strong_beats] or [float(hit) for hit in list(analysis.beats)[::2]]
    if not ordered or not strong:
        return ordered
    gaps = [right.time - left.time for left, right in zip(ordered, ordered[1:])]
    median = gaps[len(gaps) // 2] if gaps else 9.0
    sparse = median > 1.35 or len(ordered) < 10
    extras: list[Tone] = []
    if sparse:
        for hit in strong:
            if any(abs(hit - tone.time) < 0.18 for tone in ordered):
                continue
            extras.append(Tone(float(hit), 0.24, 70, "beat"))
    else:
        for left, right in zip(ordered, ordered[1:]):
            span = right.time - left.time
            if span < 0.8 or span > 1.9:
                continue
            hits = [hit for hit in strong if left.time + 0.26 < hit < right.time - 0.2]
            if hits:
                extras.append(Tone(float(hits[len(hits) // 2]), 0.2, 70, "beat"))
    return sorted([*ordered, *extras], key=lambda tone: tone.time)


def _company_for_strays(phrases: list[list[Tone]], analysis: Analysis) -> list[list[Tone]]:
    """A lone star with a long wait grows a short hook on the next beats.

    The beats are already in the song. Without them, that one dot is the whole
    sky until the next phrase arrives.
    """
    beats = [float(hit) for hit in analysis.beats]
    filled: list[list[Tone]] = []
    for index, phrase in enumerate(phrases):
        if len(phrase) != 1:
            filled.append(list(phrase))
            continue
        start = phrase[0].time
        nxt = phrases[index + 1][0].time if index + 1 < len(phrases) else start + 30.0
        if nxt - start < 4.0:
            filled.append(list(phrase))
            continue
        extras: list[Tone] = []
        limit = min(start + 2.4, nxt - 0.5)
        for hit in beats:
            if start + 0.25 < hit < limit:
                extras.append(Tone(float(hit), 0.22, 67, "beat"))
            if len(extras) >= 3:
                break
        filled.append([*phrase, *extras])
    return filled


def _group_phrases(tones: list[Tone], analysis: Analysis, gap: float = 2.7, maxlen: int = 6) -> list[list[Tone]]:
    """A phrase is a short bent figure. A rest can sit inside it; a whole verse cannot."""
    del analysis
    phrases: list[list[Tone]] = []
    bucket: list[Tone] = []
    for tone in tones:
        if bucket and (tone.time - bucket[-1].time > gap or len(bucket) >= maxlen):
            phrases.append(bucket)
            bucket = []
        bucket.append(tone)
    if bucket:
        phrases.append(bucket)
    merged: list[list[Tone]] = []
    for phrase in phrases:
        if not merged:
            merged.append(list(phrase))
            continue
        silence = phrase[0].time - merged[-1][-1].time
        room = len(merged[-1]) + len(phrase) <= maxlen
        # A single leftover joins its neighbor so it does not sit in the sky as a stray dot.
        orphan = len(phrase) == 1 or len(merged[-1]) == 1
        if silence <= 3.6 and (room or (orphan and len(merged[-1]) + len(phrase) <= maxlen + 1)):
            merged[-1].extend(phrase)
        else:
            merged.append(list(phrase))
    return merged


STAR_MARGIN = 108.0
STAR_GAP = 64.0


def _min_pair(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 1e6
    best = 1e6
    for index, (ax, ay) in enumerate(points):
        for bx, by in points[index + 1 :]:
            best = min(best, math.hypot(ax - bx, ay - by))
    return best


def _layout_constellations(phrases: list[list[Tone]], width: float, height: float, rng: list[int]):
    """Each phrase is a small hooked figure in its own part of the sky. Lines stay a simple chain."""
    cap = 0.22 * min(width, height)
    pending = []
    for index, phrase in enumerate(phrases):
        local = _organic_points(len(phrase), rng)
        radius = max((math.hypot(x, y) for x, y in local), default=0.0)
        if radius > cap and radius > 1.0:
            shrink = cap / radius
            closest = _min_pair(local)
            if closest * shrink < STAR_GAP:
                shrink = STAR_GAP / closest
            local = [(x * shrink, y * shrink) for x, y in local]
            radius *= shrink
        pending.append((index, phrase, local, radius))
    # Wide frames give the largest figure first pick of the sky, as before.
    # A tall frame seats in time order so the opening phrase can take the lower
    # band instead of losing it to whichever figure happens to be biggest.
    portrait = height > width
    if portrait:
        pending.sort(key=lambda item: item[0])
    else:
        pending.sort(key=lambda item: -item[3])
    placed: list[tuple[float, float, float]] = []
    occupied: list[tuple[float, float]] = []
    seated: dict[int, tuple[tuple[float, float], list[tuple[float, float]]]] = {}
    for index, phrase, local, radius in pending:
        prefer_y = _portrait_band(index, height) if portrait else None
        found = _seat_phrase(local, radius, placed, occupied, width, height, rng, prefer_y)
        if found is None:
            continue
        origin, points, used_radius = found
        placed.append((origin[0], origin[1], used_radius))
        occupied.extend((origin[0] + x, origin[1] + y) for x, y in points)
        seated[index] = (origin, points)
    stars: list[Star] = []
    links: list[tuple[int, int]] = []
    for index, phrase in enumerate(phrases):
        if index not in seated:
            continue
        (ox, oy), local = seated[index]
        peak = max(tone.midi for tone in phrase)
        first = len(stars)
        for place, tone in enumerate(phrase):
            x, y = local[place]
            scale = 0.48 + xorshift32(rng) * 0.72
            if tone.midi == peak and tone.source != "beat":
                scale += 0.28
            if tone.source == "beat":
                scale *= 0.62
            scale = min(1.35, scale)
            warmth = 0.12 + xorshift32(rng) * 0.2 if tone.source == "beat" else xorshift32(rng)
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
                    tone.source,
                )
            )
            if place:
                links.append((first + place - 1, first + place))
    return stars, links


def _portrait_band(index: int, height: float) -> float:
    """Where a phrase wants to sit in a tall frame.

    Every band is in the lower half, so a long rest still has a figure down in
    the frame instead of a constellation over an empty well. Adjacent phrases
    sit on opposite sides and on bands far enough apart that the circles clear.
    """
    frac = (0.56, 0.74, 0.88)[index % 3]
    return frac * height


def _seat_phrase(
    local: list[tuple[float, float]],
    radius: float,
    placed: list[tuple[float, float, float]],
    occupied: list[tuple[float, float]],
    width: float,
    height: float,
    rng: list[int],
    prefer_y: float | None = None,
) -> tuple[tuple[float, float], list[tuple[float, float]], float] | None:
    """Shrink a figure until it sits inside the frame and clear of the others. Otherwise leave it out."""
    closest = _min_pair(local)
    for shrink in (1.0, 0.94):
        if closest * shrink < STAR_GAP:
            continue
        points = [(x * shrink, y * shrink) for x, y in local]
        used = radius * shrink
        origin = _seat_figure(used, placed, width, height, rng, points, occupied, prefer_y)
        if origin is None:
            continue
        return origin, points, used
    return None


def _seat_figure(
    radius: float,
    placed: list[tuple[float, float, float]],
    width: float,
    height: float,
    rng: list[int],
    points: list[tuple[float, float]] | None = None,
    occupied: list[tuple[float, float]] | None = None,
    prefer_y: float | None = None,
) -> tuple[float, float] | None:
    points = points or [(0.0, 0.0)]
    occupied = occupied or []
    margin = STAR_MARGIN
    span_x = max(abs(x) for x, _ in points)
    span_y = max(abs(y) for _, y in points)
    margin_x = margin + span_x
    margin_y = margin + span_y
    if width <= margin_x * 2 or height <= margin_y * 2:
        return None
    best = None
    best_clear = -1.0
    best_score = -1e9
    order = len(placed)
    target_x = width * (0.3 if order % 2 == 0 else 0.7)
    target_y = min(height - margin_y, height * (0.22 + 0.18 * order))

    def consider(cx: float, cy: float) -> bool:
        nonlocal best, best_clear, best_score
        if not _points_inside(cx, cy, points, width, height, margin):
            return False
        # A tall frame keeps every figure in the lower half. One that does not
        # fit is left out, instead of being parked over an empty well.
        if prefer_y is not None and cy < height * 0.50:
            return False
        clear = _clearance(cx, cy, radius, placed)
        if clear < 28.0 or not _points_apart(cx, cy, points, occupied, STAR_GAP):
            return False
        if prefer_y is None:
            # Among clear seats, walk down the sky so a tall frame does not leave the bottom empty.
            score = clear - 0.2 * math.hypot(cx - target_x, cy - target_y)
        else:
            # The band is the point. Clearance is already a gate, so it only breaks ties.
            score = -abs(cy - prefer_y) - 0.12 * abs(cx - target_x) + 0.03 * min(clear, 400.0)
        if score > best_score:
            best, best_score, best_clear = (cx, cy), score, clear
        return False

    if prefer_y is not None:
        for x_frac in (0.34, 0.50, 0.66):
            for y_off in (0.0, -0.05, 0.05, -0.10, 0.10, -0.16, 0.16):
                consider(width * x_frac, prefer_y + height * y_off)
    for _attempt in range(72):
        cx = margin_x + xorshift32(rng) * (width - 2.0 * margin_x)
        cy = margin_y + xorshift32(rng) * (height - 2.0 * margin_y)
        if consider(cx, cy):
            return best
    cols = max(2, int(width // 280))
    rows = max(2, int(height // 220))
    for row in range(rows):
        for col in range(cols):
            jx = (xorshift32(rng) - 0.5) * 60.0
            jy = (xorshift32(rng) - 0.5) * 40.0
            cx = width * (col + 0.5) / cols + jx
            cy = height * (row + 0.5) / rows + jy
            cx = min(width - margin_x, max(margin_x, cx))
            cy = min(height - margin_y, max(margin_y, cy))
            if consider(cx, cy):
                return best
    if best is not None and best_clear >= 28.0:
        return best
    return None


def _points_inside(cx: float, cy: float, points, width: float, height: float, margin: float) -> bool:
    for x, y in points:
        px, py = cx + x, cy + y
        if px < margin or py < margin or px > width - margin or py > height - margin:
            return False
    return True


def _points_apart(cx: float, cy: float, points, occupied: list[tuple[float, float]], gap: float) -> bool:
    for x, y in points:
        px, py = cx + x, cy + y
        for ox, oy in occupied:
            if math.hypot(px - ox, py - oy) < gap:
                return False
    return True


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


def _paint_star_glyph(
    ctx, x: float, y: float, level: float, warmth: float, phase: float, time: float, scale: float = 1.0, pop: float = 0.0
) -> None:
    """A soft core at rest. Spikes and the halo open on the beat, then ease back."""
    twinkle = 0.92 + 0.08 * math.sin(time * 2.4 + phase)
    tint = _star_tint(warmth)
    opened = 1.0 + pop
    spike = (3.4 + 14.0 * level) * (0.8 + 0.26 * scale) * twinkle * opened
    glow_r = (8.0 + 16.0 * level) * (0.75 + 0.3 * scale) * (1.0 + 0.85 * pop)
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


def _paint_luminous_sky(ctx, scene: StarScene) -> None:
    """Milky way and nebulae. They only lighten the sky, and they swell with the bass and the beat."""
    width, height = scene.width, scene.height
    energy = scene.energy()
    low, _high = scene.bands()
    beat = scene.beat_level()
    strong = scene.strong_level()
    lift = 0.7 + 0.85 * energy + 0.55 * low + 0.7 * beat + 0.35 * strong
    mix = scene.slow_mix()
    clock = scene.time if mix <= 0.0 else scene.drift_time(scene.time)
    drift = math.sin(clock * (0.08 + 0.12 * mix)) * (26.0 + 22.0 * mix) + math.sin(clock * (0.21 + 0.08 * mix)) * (
        8.0 + 8.0 * mix
    )
    sway = math.cos(clock * (0.06 + 0.10 * mix)) * (14.0 + 12.0 * mix)
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.save()
    ctx.translate(width * 0.56 + drift, height * 0.42 + sway * 0.4)
    ctx.rotate(-0.26 + 0.03 * math.sin(clock * (0.05 + 0.04 * mix)))
    band_h = (200.0 + 70.0 * low + 24.0 * beat) * (0.85 if height > width else 1.0)
    band = cairo.LinearGradient(0, -band_h, 0, band_h)
    mid = min(0.2, 0.055 + 0.06 * energy + 0.05 * low + 0.07 * beat)
    band.add_color_stop_rgba(0.0, 0.72, 0.68, 0.95, 0.0)
    band.add_color_stop_rgba(0.5, 0.9, 0.86, 1.0, mid)
    band.add_color_stop_rgba(1.0, 0.65, 0.6, 0.92, 0.0)
    ctx.set_source(band)
    ctx.rectangle(-width * 1.3, -band_h, width * 2.6, band_h * 2.0)
    ctx.fill()
    ctx.restore()
    for ox, oy, radius, tint, alpha in (
        (0.28, 0.38, 0.34, (0.55, 0.5, 0.95), 0.045),
        (0.7, 0.55, 0.38, (0.45, 0.62, 0.98), 0.04),
        (0.48, 0.28, 0.26, (0.7, 0.62, 0.95), 0.028),
    ):
        x = ox * width + drift * 0.45
        y = oy * height + sway * 0.3
        reach = radius * min(width, height) * (0.92 + 0.16 * low + 0.08 * beat)
        blob = cairo.RadialGradient(x, y, reach * 0.02, x, y, reach)
        strength = alpha * lift
        blob.add_color_stop_rgba(0.0, tint[0], tint[1], tint[2], strength)
        blob.add_color_stop_rgba(0.5, tint[0], tint[1], tint[2], strength * 0.35)
        blob.add_color_stop_rgba(1.0, tint[0], tint[1], tint[2], 0.0)
        ctx.set_source(blob)
        ctx.arc(x, y, reach, 0, math.tau)
        ctx.fill()
    ctx.restore()


def _paint_stars(ctx, scene: StarScene) -> None:
    width, height = scene.width, scene.height
    energy = scene.energy()
    beat = scene.beat_level()
    gradient = cairo.LinearGradient(0, 0, 0, height)
    gradient.add_color_stop_rgb(0, 0.04 + 0.03 * energy, 0.05, 0.15 + 0.05 * energy + 0.03 * beat)
    gradient.add_color_stop_rgb(0.4, 0.07 + 0.04 * energy, 0.05, 0.16 + 0.04 * beat)
    gradient.add_color_stop_rgb(1, 0.035, 0.03, 0.08)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source(gradient)
    ctx.paint()
    # Everything after the opaque sky must blend. SOURCE would punch dark holes.
    ctx.set_operator(cairo.OPERATOR_OVER)
    _paint_luminous_sky(ctx, scene)
    ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.save()
    ctx.translate(width / 2.0, height / 2.0)
    ctx.scale(scene.cam_zoom, scene.cam_zoom)
    ctx.translate(-width / 2.0 + scene.cam_x, -height / 2.0 + scene.cam_y)
    _low, high = scene.bands()
    mix = scene.slow_mix()
    clock = scene.time if mix <= 0.0 else scene.drift_time(scene.time)
    for x, y, scale, warmth, phase in scene.dust:
        speed = 0.8 + 5.5 * high
        amp_x, amp_y = 8.0, 5.0
        freq_x, freq_y = 0.16, 0.11
        if mix > 0.0:
            speed = speed + mix * (0.9 + 0.8 * (1.0 - min(1.0, high)))
            amp_x += 26.0 * mix
            amp_y += 16.0 * mix
            freq_x += 0.20 * mix
            freq_y += 0.14 * mix
        twinkle = 0.55 + 0.45 * math.sin(clock * speed + phase)
        # Only the brighter specks answer the beat, so the field sparkles instead of flashing white.
        spark = beat * max(0.0, math.sin(phase)) * (1.0 if scale > 0.85 else 0.25)
        tint = _star_tint(warmth)
        alpha = (0.1 + 0.22 * high) * (0.45 + 0.55 * twinkle) + 0.42 * spark
        drift_x = amp_x * math.sin(clock * freq_x + phase)
        drift_y = amp_y * math.cos(clock * freq_y + warmth * 4.0)
        ctx.set_source_rgba(tint[0], tint[1], tint[2], min(0.75, alpha))
        ctx.arc(x + drift_x, y + drift_y, (0.6 + scale * 0.5) * (1.0 + 0.8 * spark), 0, math.tau)
        ctx.fill()
    _paint_constellation_lines(ctx, scene)
    _paint_phrase_ripple(ctx, scene)
    _paint_beat_glint(ctx, scene)
    for star in scene.stars:
        x, y = scene.placed(star)
        age = scene.time - star.time
        if -0.5 < age < -0.22:
            hint = _smooth((age + 0.5) / 0.28) * 0.4
            ctx.set_source_rgba(0.8, 0.86, 1.0, hint)
            ctx.arc(x, y, 1.6 + star.scale, 0, math.tau)
            ctx.fill()
            continue
        level = scene.brightness(star)
        if level <= 0.02:
            continue
        if 0.0 <= age < 0.4:
            fade = 1.0 - age / 0.4
            ring = (6.0 + age * 36.0) * (0.75 + 0.25 * star.scale)
            ctx.set_source_rgba(1.0, 0.97, 0.92, 0.5 * fade)
            ctx.set_line_width(1.15)
            ctx.arc(x, y, ring, 0, math.tau)
            ctx.stroke()
        pop = scene.beat_level() if star.phrase == scene.active_phrase() else 0.18 * scene.beat_level()
        if star.phrase == scene.active_phrase():
            pop = min(1.0, pop + 0.4 * scene.strong_level())
        glyph_time = scene.time if scene.slow_mix() <= 0.0 else scene.drift_time(scene.time)
        _paint_star_glyph(ctx, x, y, level, star.warmth, star.phase, glyph_time, star.scale, pop)
    ctx.restore()


def _paint_constellation_lines(ctx, scene: StarScene) -> None:
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    beat = scene.beat_level()
    active = scene.active_phrase()
    mix = scene.slow_mix()
    clock = scene.time if mix <= 0.0 else scene.drift_time(scene.time)
    for left, right in scene.links:
        a = scene.stars[left]
        b = scene.stars[right]
        life = min(scene.life(a.phrase), scene.life(b.phrase))
        if life < 0.04:
            continue
        earlier, later = (a, b) if a.time <= b.time else (b, a)
        age = scene.time - later.time
        if age < -0.12:
            continue
        grow = 1.0 if age >= 0.32 else _smooth((age + 0.12) / 0.44)
        ax, ay = scene.placed(earlier)
        bx, by = scene.placed(later)
        x2 = ax + (bx - ax) * grow
        y2 = ay + (by - ay) * grow
        hot = beat if a.phrase == active else 0.0
        breath = 0.0 if mix <= 0.0 else mix * 0.10 * (0.5 + 0.5 * math.sin(clock * 0.7 + a.phase))
        alpha = (0.22 + 0.42 * grow + 0.5 * hot + breath) * life
        tint = _star_tint((a.warmth + b.warmth) * 0.5)
        ctx.set_source_rgba(tint[0], tint[1], tint[2], min(0.55, alpha * 0.45))
        ctx.set_line_width(2.6 + 2.4 * hot)
        ctx.move_to(ax, ay)
        ctx.line_to(x2, y2)
        ctx.stroke()
        ctx.set_source_rgba(min(1.0, tint[0] + 0.2), min(1.0, tint[1] + 0.15), 1.0, min(0.95, alpha))
        ctx.set_line_width(0.8 + 1.5 * hot)
        ctx.move_to(ax, ay)
        ctx.line_to(x2, y2)
        ctx.stroke()


def _glint_point(scene: StarScene, chain, u: float) -> tuple[float, float]:
    slot = u * len(chain)
    index = min(len(chain) - 1, int(slot))
    local = slot - index
    earlier, later = chain[index]
    if earlier.time > later.time:
        earlier, later = later, earlier
    ax, ay = scene.placed(earlier)
    bx, by = scene.placed(later)
    return ax + (bx - ax) * local, ay + (by - ay) * local


def _paint_beat_glint(ctx, scene: StarScene) -> None:
    """One soft light runs the live constellation once per beat.

    Only edges whose stars are already lit. The path is the figure as of this
    beat, so the light does not travel into stars that have not appeared.
    """
    chain = scene.lit_edges()
    if not chain:
        return
    u = scene.glint_u()
    if u is None:
        return
    mix = scene.slow_mix()
    # A short trail keeps a slow crossing from reading as a dot that hops.
    if mix <= 0.0:
        samples = ((u, 1.0),)
    else:
        samples = ((max(0.0, u - 0.10), 0.22), (max(0.0, u - 0.05), 0.45), (u, 1.0))
    for sample, gain in samples:
        x, y = _glint_point(scene, chain, sample)
        fade = math.sin(math.pi * min(1.0, sample if mix <= 0.0 else max(sample, u))) * gain
        radius = 18.0 if gain > 0.9 else 11.0
        glow = cairo.RadialGradient(x, y, 0.6, x, y, radius)
        glow.add_color_stop_rgba(0.0, 1.0, 1.0, 1.0, 0.95 * fade)
        glow.add_color_stop_rgba(0.45, 0.8, 0.88, 1.0, 0.35 * fade)
        glow.add_color_stop_rgba(1.0, 0.7, 0.8, 1.0, 0.0)
        ctx.set_source(glow)
        ctx.arc(x, y, radius, 0, math.tau)
        ctx.fill()


def _paint_phrase_ripple(ctx, scene: StarScene) -> None:
    """A soft ring opens from the live figure on each beat and dies before the next constellation.

    The center is the stars already on screen. Counting the ones still to come
    pulls the ring off the figure and into empty sky.
    """
    members = scene.ripple_members()
    if len(members) < 1:
        return
    age = _since(scene.analysis.beats, scene.time)
    window = scene._event_window(scene.time, 0.42, 0.78, 1.10)
    if age is None or age > window or window <= 0.0:
        return
    points = [scene.placed(star) for star in members]
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    reach = max(math.hypot(point[0] - cx, point[1] - cy) for point in points)
    u = age / window
    mix = scene.slow_mix()
    if mix > 0.0:
        u = u + mix * (_smooth(u) - u)
    radius = 8.0 + u * min(100.0, reach + 16.0)
    alpha = 0.42 * (1.0 - u) * (0.65 + 0.35 * scene.strong_level())
    ctx.set_line_width(1.4)
    ctx.set_source_rgba(0.85, 0.9, 1.0, alpha)
    ctx.arc(cx, cy, radius, 0, math.tau)
    ctx.stroke()


def _since(times: list[float], now: float) -> float | None:
    last = None
    for hit in times:
        if hit > now:
            break
        last = hit
    if last is None:
        return None
    return now - last


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
