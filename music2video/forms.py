"""Beat-synced forms other than the ball map: a growing city, a note waterfall, a sky of constellations.

Each form is closed-form in time. A building finishes on its beat, a note head
meets its key on the onset, and a star reaches full brightness on its note.
The camera only eases toward a slow target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cairo

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
        block_w = 54.0 + xorshift32(rng) * 48.0
        target = (150.0 + 280.0 * energy) * (0.82 + 0.22 * xorshift32(rng))
        if energy >= 0.55:
            target *= 1.18
        target = min(target, height * 0.62)
        floors = max(4, min(14, int(target / 38.0)))
        cols = max(2, min(5, int(block_w / 20.0)))
        specs.append((beat, block_w, target, energy, floors, cols, energy >= 0.62))
    total = sum(item[1] for item in specs) + 26.0 * (len(specs) - 1)
    cursor = -total / 2.0
    for beat, block_w, target, energy, floors, cols, tower in specs:
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
            )
        )
        cursor += block_w + 26.0
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


def _paint_city(ctx, scene: CityScene) -> None:
    pal = active_palette()
    width, height = scene.width, scene.height
    sky = ctx
    gradient = cairo.LinearGradient(0, 0, 0, height)
    gradient.add_color_stop_rgb(0, *pal.bg_top)
    gradient.add_color_stop_rgb(0.72, *pal.bg_bottom)
    gradient.add_color_stop_rgb(1, pal.bg_bottom[0] * 0.45, pal.bg_bottom[1] * 0.4, pal.bg_bottom[2] * 0.55)
    sky.set_source(gradient)
    sky.paint()
    # A moon and a few fixed stars sit behind the skyline.
    sky.set_source_rgba(0.95, 0.94, 0.88, 0.9)
    sky.arc(width * 0.78, height * 0.16, 28, 0, math.tau)
    sky.fill()
    sky.set_source_rgba(*pal.bg_top, 1)
    sky.arc(width * 0.78 + 12, height * 0.16 - 6, 24, 0, math.tau)
    sky.fill()
    sky.save()
    ground = height * 0.86
    sky.translate(width / 2.0, ground)
    sky.scale(scene.cam_zoom, scene.cam_zoom)
    sky.translate(-scene.cam_x, 0.0)
    glow = scene.street_glow()
    for building in scene.buildings:
        risen = scene.building_height(building)
        if risen < 1.5:
            continue
        body = (
            pal.ink[0] + 0.04 * building.energy,
            pal.ink[1] + 0.03,
            pal.ink[2] + 0.06 + 0.05 * building.energy,
        )
        _round_rect(sky, building.x, -risen, building.width, risen, 5)
        fill = cairo.LinearGradient(building.x, -risen, building.x + building.width, 0)
        fill.add_color_stop_rgb(0, min(1, body[0] + 0.05), min(1, body[1] + 0.04), min(1, body[2] + 0.08))
        fill.add_color_stop_rgb(1, body[0] * 0.72, body[1] * 0.72, body[2] * 0.8)
        sky.set_source(fill)
        sky.fill()
        # The side face keeps the block from looking like a flat poster.
        sky.set_source_rgba(0, 0, 0, 0.28)
        sky.rectangle(building.x + building.width - 8, -risen, 8, risen)
        sky.fill()
        sky.set_source_rgba(*pal.rim, 0.35)
        sky.set_line_width(1.4)
        sky.move_to(building.x + 2, -risen)
        sky.line_to(building.x + building.width - 2, -risen)
        sky.stroke()
        if building.tower and risen > building.target * 0.92:
            sky.set_source_rgba(*pal.accent2, 0.9)
            mast_x = building.x + building.width * 0.55
            sky.set_line_width(2)
            sky.move_to(mast_x, -risen)
            sky.line_to(mast_x, -risen - 28)
            sky.stroke()
            sky.move_to(mast_x - 4, -risen - 28)
            sky.line_to(mast_x + 4, -risen - 28)
            sky.stroke()
        _paint_windows(sky, scene, building, risen, pal)
    _paint_street(sky, scene, pal, glow)
    sky.restore()


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
            if gap > 1.15 or (section != previous and gap > 0.45) or len(bucket) >= 5:
                phrases.append(bucket)
                bucket = []
        bucket.append(tone)
    if bucket:
        phrases.append(bucket)
    stars, links = _layout_constellations(phrases, float(width), float(height), rng)
    dust = []
    for _ in range(48):
        dust.append((xorshift32(rng) * width, xorshift32(rng) * height * 0.85, 0.6 + xorshift32(rng) * 1.4))
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
    """One separated patch per phrase. Stars run left to right inside it, so lines cannot cross."""
    count = max(1, len(phrases))
    longest = max(len(phrase) for phrase in phrases)
    margin_x = width * 0.04
    margin_y = height * 0.05
    usable_w = width - 2.0 * margin_x
    usable_h = height * 0.88 - margin_y
    min_gap = 100.0
    min_step = 84.0
    chosen: tuple | None = None
    for cols in range(1, count + 1):
        rows = math.ceil(count / cols)
        cell_w = usable_w / cols
        cell_h = usable_h / rows
        inner_w = cell_w - min_gap
        inner_h = cell_h - min_gap
        step = inner_w / max(1, longest - 1)
        if longest == 1:
            step = min_step
        if inner_w < min_step or inner_h < 48.0 or step < min_step:
            continue
        rank = (step, inner_h, -rows)
        if chosen is None or rank > chosen[0]:
            chosen = (rank, cols, rows, cell_w, cell_h)
    if chosen is None:
        cols = max(1, min(count, 4))
        rows = math.ceil(count / cols)
        cell_w = usable_w / cols
        cell_h = usable_h / rows
        min_gap = 64.0
        min_step = 64.0
    else:
        _, cols, rows, cell_w, cell_h = chosen
    stars: list[Star] = []
    links: list[tuple[int, int]] = []
    for index, phrase in enumerate(phrases):
        col = index % cols
        row = index // cols
        left = margin_x + col * cell_w + min_gap * 0.5
        right = margin_x + (col + 1) * cell_w - min_gap * 0.5
        top = margin_y + row * cell_h + min_gap * 0.5
        bottom = margin_y + (row + 1) * cell_h - min_gap * 0.5
        if right - left < 36:
            right = left + 36
        if bottom - top < 28:
            bottom = top + 28
        first = len(stars)
        steps = len(phrase)
        mid_y = (top + bottom) / 2.0
        amp = min(36.0, (bottom - top) * 0.38)
        if steps <= 1:
            slack_x = (right - left) / 2.0
            slack_y = (bottom - top) / 2.0
            shift_x = math.sin(index * 1.7) * min(16.0, slack_x)
            shift_y = math.cos(index * 1.3) * min(12.0, slack_y)
            stars.append(Star(phrase[0].time, (left + right) / 2.0 + shift_x, mid_y + shift_y, xorshift32(rng) * math.tau, phrase[0].midi, index))
            continue
        fit = (right - left) / (steps - 1)
        step = min(120.0, max(min_step, fit))
        if step > fit:
            step = fit
        span = step * (steps - 1)
        # Shift only into leftover space, so neighboring figures cannot be pushed together.
        slack_x = max(0.0, (right - left) - span) / 2.0
        slack_y = max(0.0, (bottom - top) / 2.0 - amp)
        shift_x = math.sin(index * 1.7) * min(16.0, slack_x)
        shift_y = math.cos(index * 1.3) * min(12.0, slack_y)
        x0 = (left + right) / 2.0 - span / 2.0 + shift_x
        mid_y += shift_y
        for place, tone in enumerate(phrase):
            x = x0 + place * step
            y = mid_y + math.sin(place * 1.15 + index * 0.85) * amp
            stars.append(Star(tone.time, x, y, xorshift32(rng) * math.tau, tone.midi, index))
            if place:
                links.append((first + place - 1, first + place))
    return stars, links


def _paint_star_glyph(ctx, x: float, y: float, level: float, color, phase: float, time: float, scale: float = 1.0) -> None:
    """A diffraction cross. The core stays smaller than the spikes so bloom cannot turn it into a ball."""
    twinkle = 0.82 + 0.18 * math.sin(time * 2.4 + phase)
    spike = (30.0 + 12.0 * level) * twinkle * scale
    tint = (
        0.82 + 0.18 * color[0],
        0.8 + 0.18 * color[1],
        0.74 + 0.18 * color[2],
    )
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_width(4.2 * scale)
    ctx.set_source_rgba(*tint, min(1.0, 0.95 * level * twinkle))
    ctx.move_to(x - spike, y)
    ctx.line_to(x + spike, y)
    ctx.stroke()
    ctx.move_to(x, y - spike * 1.08)
    ctx.line_to(x, y + spike * 1.02)
    ctx.stroke()
    diagonal = spike * 0.38
    ctx.set_source_rgba(1, 0.98, 0.94, 0.7 * level * twinkle)
    ctx.set_line_width(1.8 * scale)
    ctx.move_to(x - diagonal, y - diagonal)
    ctx.line_to(x + diagonal, y + diagonal)
    ctx.stroke()
    ctx.move_to(x - diagonal, y + diagonal)
    ctx.line_to(x + diagonal, y - diagonal)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, min(1.0, level))
    ctx.arc(x, y, 1.7 * scale, 0, math.tau)
    ctx.fill()


def _paint_stars(ctx, scene: StarScene) -> None:
    pal = active_palette()
    width, height = scene.width, scene.height
    gradient = cairo.LinearGradient(0, 0, 0, height)
    gradient.add_color_stop_rgb(0, *pal.bg_top)
    gradient.add_color_stop_rgb(0.55, 0.03, 0.02, 0.07)
    gradient.add_color_stop_rgb(1, *pal.bg_bottom)
    ctx.set_source(gradient)
    ctx.paint()
    # A soft band, so the sky has depth before any note arrives.
    band = cairo.LinearGradient(0, height * 0.35, 0, height * 0.62)
    band.add_color_stop_rgba(0, *pal.accent2, 0)
    band.add_color_stop_rgba(0.5, *pal.accent2, 0.05)
    band.add_color_stop_rgba(1, *pal.accent, 0)
    ctx.set_source(band)
    ctx.paint()
    ctx.save()
    ctx.translate(width / 2.0, height / 2.0)
    ctx.scale(scene.cam_zoom, scene.cam_zoom)
    ctx.translate(-width / 2.0 + scene.cam_x, -height / 2.0 + scene.cam_y)
    for x, y, radius in scene.dust:
        ctx.set_source_rgba(0.85, 0.88, 0.95, 0.18 + 0.08 * math.sin(scene.time * 1.2 + radius))
        ctx.arc(x, y, 1.1, 0, math.tau)
        ctx.fill()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for left, right in scene.links:
        a = scene.stars[left]
        b = scene.stars[right]
        born = max(a.time, b.time)
        age = scene.time - born
        if age < 0.0:
            continue
        # Fresh phrases stay drawn. Older ones recede so the sky does not become one web.
        appear = _smooth(min(1.0, age / 0.45))
        if age < 4.5:
            alpha = appear * 0.7
        else:
            alpha = max(0.14, 0.7 * _smooth(1.0 - (age - 4.5) / 6.0))
        ctx.set_source_rgba(0.82, 0.86, 0.94, alpha)
        ctx.set_line_width(1.35)
        ctx.move_to(a.x, a.y)
        ctx.line_to(b.x, b.y)
        ctx.stroke()
    for star in scene.stars:
        level = scene.brightness(star)
        if level <= 0.01:
            continue
        _paint_star_glyph(ctx, star.x, star.y, level, NOTE_COLORS[star.midi % 12], star.phase, scene.time)
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
