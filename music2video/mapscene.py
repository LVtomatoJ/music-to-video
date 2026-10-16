"""One ball, one generated map: a surface is placed so the ball hits it on each beat."""

from __future__ import annotations

import math
from dataclasses import dataclass

from music2video.analysis import Analysis, section_at
from music2video.registry import BALL_R, EFFECTS, Obstacle, obstacle_penetration
from music2video.scene import xorshift32
from music2video.themes import THEMES, Theme, pick_theme

SPEED = 680.0
SPREAD = 0.58
CLEARANCE = 72.0
# A contact kiss may sink this far into the surface it is hitting. Any other overlap counts.
SOLID_EPS = 1.5


@dataclass
class Segment:
    t0: float
    t1: float
    x0: float
    y0: float
    x1: float
    y1: float
    bow: float = 0.0


@dataclass
class Spark:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    hue: float
    life: float
    max_life: float
    shape: str
    spin: float
    spin_v: float


@dataclass
class Ring:
    x: float
    y: float
    radius: float
    growth: float
    width: float
    hue: float
    life: float
    max_life: float


def _unit(x: float, y: float) -> tuple[float, float]:
    mag = math.hypot(x, y)
    if mag < 1e-8:
        return 1.0, 0.0
    return x / mag, y / mag


def _segment_distance(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> float:
    """Minimum distance between segments ab and cd."""
    d1x, d1y = b[0] - a[0], b[1] - a[1]
    d2x, d2y = d[0] - c[0], d[1] - c[1]
    rx, ry = a[0] - c[0], a[1] - c[1]
    aa = d1x * d1x + d1y * d1y
    ee = d2x * d2x + d2y * d2y
    ff = d2x * rx + d2y * ry
    bb = d1x * d2x + d1y * d2y
    cc = d1x * rx + d1y * ry
    denom = aa * ee - bb * bb
    if denom > 1e-8 and aa > 1e-8:
        s = min(1.0, max(0.0, (bb * ff - cc * ee) / denom))
    else:
        s = 0.0
    if ee > 1e-8:
        t = (bb * s + ff) / ee
    else:
        t = 0.0
    if t < 0.0:
        t = 0.0
        s = 0.0 if aa <= 1e-8 else min(1.0, max(0.0, -cc / aa))
    elif t > 1.0:
        t = 1.0
        s = 0.0 if aa <= 1e-8 else min(1.0, max(0.0, (bb - cc) / aa))
    cx1, cy1 = a[0] + d1x * s, a[1] + d1y * s
    cx2, cy2 = c[0] + d2x * t, c[1] + d2y * t
    return math.hypot(cx1 - cx2, cy1 - cy2)


def _intersects(a, b, c, d) -> bool:
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def straddles(p, q, r, s):
        c1 = cross(p, q, r)
        c2 = cross(p, q, s)
        return c1 * c2 < 0

    return straddles(a, b, c, d) and straddles(c, d, a, b)


class MapScene:
    def __init__(self) -> None:
        self.width = 1920.0
        self.height = 1080.0
        self.obstacles: list[Obstacle] = []
        self.segments: list[Segment] = []
        self.sparks: list[Spark] = []
        self.rings: list[Ring] = []
        self.trail: list[tuple[float, float]] = []
        self.stars: list[tuple[float, float]] = []
        self.rng = [1]
        self.time = 0.0
        self.pos = (0.0, 0.0)
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.cam_zoom = 1.0
        self.caption_title = ""
        self.caption_artist = ""
        self.shake = 0.0
        self.squash = 0.0
        self.squash_nx = 1.0
        self.squash_ny = 0.0
        self.chroma = 0.0
        self.morph = 0.0
        self.hue_jump = 0.0
        self.theme_hue = 190.0
        self.bg_flash = 0.0
        self.squash_v = 0.0
        self.hold = 0.0
        self.spin = 0.0
        self.spin_v = 0.0
        self.split_life = 0.0
        self.split_max = 0.5
        self.split_count = 0
        self.section_energy = 0.4
        self.skin: Theme | None = None
        self.ball_r = BALL_R
        self.analysis: Analysis | None = None
        self.timing: dict = {}
        self.duration = 0.0

    @classmethod
    def create(cls, width: int, height: int, analysis: Analysis, seed: int = 42, theme: str = "neon") -> MapScene:
        scene = cls()
        scene.width = float(width)
        scene.height = float(height)
        scene.analysis = analysis
        scene.duration = analysis.duration
        scene.rng = [seed & 0xFFFFFFFF or 1]
        name = pick_theme(analysis) if theme == "auto" else theme
        if name not in THEMES:
            raise SystemExit(f"unknown theme {theme}")
        scene.skin = THEMES[name]
        scene.theme_hue = scene.skin.hue
        flags = []
        si = 0
        strong_list = list(analysis.strong_beats)
        for beat in analysis.beats:
            while si < len(strong_list) and strong_list[si] < beat - 1e-6:
                si += 1
            is_strong = si < len(strong_list) and abs(strong_list[si] - beat) <= 1e-4
            flags.append(is_strong)
            if is_strong:
                si += 1
        scene._build(list(analysis.beats), flags)
        scene.stars = _stars(scene.rng, scene.segments)
        scene.timing = scene.measure()
        scene.pos = scene.position_at(0.0)
        scene.cam_x, scene.cam_y = scene.pos
        return scene

    def _build(self, beats: list[float], flags: list[bool]) -> None:
        beats = [beat for beat in beats if beat >= -1e-4]
        deduped: list[tuple[float, bool]] = []
        for beat, flag in zip(beats, flags):
            if deduped and abs(beat - deduped[-1][0]) < 1e-3:
                continue
            deduped.append((beat, flag))
        if not deduped:
            self.segments = [Segment(0.0, max(self.duration, 0.1), 0.0, 0.0, SPEED * self.duration, 0.0)]
            return
        lane = (xorshift32(self.rng) * 2.0 - 1.0) * math.pi
        side = 1.0 if xorshift32(self.rng) < 0.5 else -1.0
        heading = lane + side * SPREAD
        # Square spiral: two equal sides, then the run grows, so the return
        # lane sits outside the path already drawn.
        row_len = 5
        row_left = row_len
        turns = 0
        pos = (0.0, 0.0)
        t_prev = 0.0
        # Approach the first beat from off-screen time zero when the beat is not at 0.
        if deduped[0][0] > 1e-3:
            ivx, ivy = _unit(math.cos(heading), math.sin(heading))
            dist = SPEED * deduped[0][0]
            pos = (-ivx * dist, -ivy * dist)
        counts = {name: 0 for name in self.skin.objects}
        last_kind = ""
        last_phrase = ""
        strong_ordinal = 0
        gaps = [deduped[i][0] - deduped[i - 1][0] for i in range(1, len(deduped))]
        median_dt = sorted(gaps)[len(gaps) // 2] if gaps else 0.5
        polys: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        for index, (beat, is_strong) in enumerate(deduped):
            section = section_at(self.analysis.sections if self.analysis else (), beat)
            phrase, speed, spread, bow_frac, extend = _phrase_motion(section.label, index, is_strong)
            # A stretch starts a longer hall before the next square-spiral corner.
            if extend and phrase != last_phrase:
                row_left += 2
            last_phrase = phrase
            ivx, ivy = _unit(math.cos(heading), math.sin(heading))
            dt = max(1e-3, beat - t_prev)
            length = speed * dt
            # A fast grid cannot throw a chord across the spiral. A slow gap
            # keeps the longer flight, which is the visible stretch.
            if median_dt < 0.55:
                length = min(length, 340.0 * math.sqrt(dt / median_dt))
            hit = (pos[0] + ivx * length, pos[1] + ivy * length)
            next_lane = lane
            next_side = -side
            next_spread = spread
            row_left -= 1
            if row_left <= 0:
                next_lane = lane + math.pi / 2
                turns += 1
                if turns % 2 == 0:
                    row_len += 3
                row_left = row_len
                next_spread = 0.22 if phrase == "cluster" else 0.12
                next_side = side
            ovx, ovy = _unit(math.cos(next_lane + next_side * next_spread), math.sin(next_lane + next_side * next_spread))
            # Steer around flight lines and around walls already on the map.
            # A yaw turns the arrival and the departure together. If that still
            # meets an old body, a wider or reversed bounce is tried before a
            # shorter chord.
            hit, ivx, ivy, ovx, ovy = _avoid_overlap(
                pos, hit, (ivx, ivy), (ovx, ovy), polys, length, self.obstacles, self.segments, t_prev, beat
            )
            kind = _choose_kind(self.skin.objects, is_strong, counts, last_kind, self.rng)
            counts[kind] += 1
            last_kind = kind
            flourish = False
            if is_strong:
                strong_ordinal += 1
                flourish = strong_ordinal % 4 == 0
            variant = _choose_variant(self.skin, kind, section.label, flourish, self.rng)
            scale = min(1.0, length / 280.0)
            size, peg_r, spin, _ignored = _object_dims(kind, is_strong, section.energy, scale, self.rng)
            size = min(420.0, size * _VARIANT_SCALE.get(variant, 1.0) * _SECTION_SIZE.get(section.label, 1.0))
            if kind == "peg":
                peg_r = size
            bow = 0.0
            if phrase == "curve":
                bow = _fit_bow(
                    pos, hit, min(18.0, bow_frac * length), polys, self.obstacles, self.segments, t_prev, beat
                )
            # The departure just chosen is the final normal of the surface we left.
            if self.obstacles and self.segments:
                _write_normal(
                    self.obstacles[-1],
                    _segment_tangent(self.segments[-1], True),
                    _tangent_of(pos, hit, bow, False),
                )
            end_tx, end_ty = _tangent_of(pos, hit, bow, True)
            nx, ny = _normal_from(end_tx, end_ty, ovx, ovy) or _unit(ivx - ovx, ivy - ovy)
            tx, ty = -ny, nx
            kind_hue = self.skin.hue_step * self.skin.objects.index(kind)
            light = self.skin.object_light * _SECTION_LIGHT.get(section.label, 1.0)
            self.obstacles.append(
                Obstacle(
                    kind=kind,
                    hit_time=beat,
                    x=hit[0],
                    y=hit[1],
                    nx=nx,
                    ny=ny,
                    tx=tx,
                    ty=ty,
                    strong=is_strong,
                    energy=section.energy,
                    section=section.label,
                    size=size,
                    spin=spin,
                    peg_r=peg_r,
                    kind_hue=kind_hue,
                    sat=self.skin.object_sat,
                    light=light,
                    variant=variant,
                    flourish=flourish,
                )
            )
            self.segments.append(Segment(t_prev, beat, pos[0], pos[1], hit[0], hit[1], bow))
            polys.append((pos, hit, bow))
            pos = hit
            t_prev = beat
            lane = next_lane
            side = next_side
            heading = math.atan2(ovy, ovx)
        if self.duration > t_prev + 1e-3:
            ivx, ivy = _unit(math.cos(heading), math.sin(heading))
            dist = SPEED * (self.duration - t_prev)
            end = (pos[0] + ivx * dist, pos[1] + ivy * dist)
            end, ivx, ivy, _ox, _oy = _avoid_overlap(
                pos,
                end,
                (ivx, ivy),
                (ivx, ivy),
                polys,
                dist,
                self.obstacles,
                self.segments,
                t_prev,
                self.duration,
                allow_bend=False,
            )
            self.segments.append(Segment(t_prev, self.duration, pos[0], pos[1], end[0], end[1]))
        # Rebuild every normal from the two segments that actually meet, then
        # shorten any body that still crosses the flown path. Routing runs first;
        # shrinking is only the residue a turn could not get around.
        _align_bounce_normals(self.obstacles, self.segments)
        _seat_obstacles(self)

    def position_at(self, time: float) -> tuple[float, float]:
        if not self.segments:
            return (0.0, 0.0)
        if time <= self.segments[0].t0:
            return self.segments[0].x0, self.segments[0].y0
        for seg in self.segments:
            if time <= seg.t1:
                span = seg.t1 - seg.t0
                u = 0.0 if span <= 1e-9 else (time - seg.t0) / span
                return _point_on(seg, u)
        last = self.segments[-1]
        return last.x1, last.y1

    def velocity_at(self, time: float) -> tuple[float, float]:
        for seg in self.segments:
            if time <= seg.t1:
                span = seg.t1 - seg.t0
                u = 0.0 if span <= 1e-9 else (time - seg.t0) / span
                return _velocity_on(seg, u)
        if not self.segments:
            return SPEED, 0.0
        last = self.segments[-1]
        return _velocity_on(last, 1.0)

    def update(self, dt: float, time: float, events: list[tuple[float, bool]]) -> None:
        self._decay(dt)
        for beat_time, _strong in events:
            self._trigger(beat_time)
        self.time = time
        self.pos = self.position_at(time)
        self.trail.append(self.pos)
        trail_keep = self.skin.feel.trail_length if self.skin else 16
        if len(self.trail) > trail_keep:
            del self.trail[:-trail_keep]
        section = section_at(self.analysis.sections if self.analysis else (), time)
        self.section_energy = section.energy
        vx, vy = self.velocity_at(time)
        # Damped follow with a short lookahead. The blend is slow enough that a
        # bounce does not snap the frame, and the zoom only breathes with energy.
        look = 0.16
        target_x = self.pos[0] + vx * look
        target_y = self.pos[1] + vy * look
        blend = 1.0 - math.exp(-3.6 * dt)
        self.cam_x += (target_x - self.cam_x) * blend
        self.cam_y += (target_y - self.cam_y) * blend
        target_zoom = 1.0 + 0.055 * self.section_energy + 0.028 * self.sample_rms(time)
        zblend = 1.0 - math.exp(-2.0 * dt)
        self.cam_zoom += (target_zoom - self.cam_zoom) * zblend
        if self.skin is not None:
            self.theme_hue = self.skin.hue
        self._integrate(dt)
        self.bg_flash = max(self.bg_flash, 0.0)

    def _decay(self, dt: float) -> None:
        fade = math.exp(-8.0 * dt)
        self.shake *= math.exp(-14.0 * dt)
        if self.split_life > 0.0:
            self.split_life = max(0.0, self.split_life - dt)
        self._spring(dt)
        self.chroma *= math.exp(-5.0 * dt)
        self.morph *= math.exp(-3.5 * dt)
        self.hue_jump *= math.exp(-4.5 * dt)
        self.bg_flash *= fade
        # Hit light lasts about 300ms and eases out (the draw samples smoothstep).
        step = dt / 0.30
        for obs in self.obstacles:
            if obs.flash > 0.0:
                obs.flash = max(0.0, obs.flash - step)

    def _spring(self, dt: float) -> None:
        """Deformation spring. The center stays on the planned path.

        The step is the closed-form damped oscillator. A forward Euler step
        of the stiff themes (neon, arcade) flipped the squash every frame,
        and the sprite then snapped between the wall normal and the flight
        tangent between hits.
        """
        if self.skin is None:
            self.squash *= math.exp(-7.0 * dt)
            return
        feel = self.skin.feel
        if self.hold > 0.0:
            self.hold = max(0.0, self.hold - dt)
            self.squash_v = 0.0
        elif dt > 0.0:
            self.squash, self.squash_v = _damped_step(
                self.squash, self.squash_v, feel.stiffness, feel.damping, max(0.05, feel.mass), dt
            )
            self.squash = max(-0.55, min(1.35, self.squash))
        self.spin += self.spin_v * dt
        self.spin_v *= math.exp(-5.5 * dt)

    def _trigger(self, beat_time: float) -> None:
        obs = None
        for candidate in self.obstacles:
            if abs(candidate.hit_time - beat_time) <= 1e-3:
                obs = candidate
                break
        if obs is None or obs.lit:
            return
        obs.lit = True
        obs.flash = 1.0
        feel = self.skin.feel
        section_gain = _SECTION_GAIN.get(obs.section, 1.0)
        weight = 0.7 + 0.3 * obs.energy + (0.15 if obs.strong else 0.0)
        intensity = min(1.6, self.skin.intensity * weight * section_gain)
        if obs.flourish:
            intensity = min(1.9, intensity * 1.65)
        self.bg_flash = (feel.pause + 0.35) * section_gain * (1.35 if obs.flourish else 1.0)
        self.squash = min(1.2, feel.squash * (0.72 + 0.38 * section_gain) * (0.85 + 0.15 * weight))
        self.squash_v = 0.0
        self.hold = feel.pause * (1.2 if obs.strong else 1.0)
        self.squash_nx, self.squash_ny = obs.nx, obs.ny
        if obs.strong and feel.shake > 0:
            self.bump_shake(feel.shake * intensity)
        if feel.tumble:
            direction = 1.0 if xorshift32(self.rng) < 0.5 else -1.0
            self.spin_v += feel.tumble * direction * (1.0 if obs.strong else 0.45)
        for name in self.skin.effects:
            EFFECTS[name](self, obs, intensity, self.rng)
        if obs.flourish:
            _flourish(self, obs, intensity)

    def add_ring(self, x: float, y: float, growth: float, hue: float, width: float, life: float = 0.38) -> None:
        self.rings.append(Ring(x, y, 12.0, growth, width, hue, life, life))

    def add_spark(self, x, y, vx, vy, radius, hue, life) -> None:
        self.sparks.append(Spark(x, y, vx, vy, radius, hue, life, life, "spark", 0.0, 0.0))

    def add_fragment(self, x, y, vx, vy, radius, hue, life, spin) -> None:
        self.sparks.append(Spark(x, y, vx, vy, radius, hue, life, life, "fragment", spin, (spin - 1.5) * 6))

    def bump_shake(self, amount: float) -> None:
        self.shake = max(self.shake, amount)

    def _integrate(self, dt: float) -> None:
        alive: list[Spark] = []
        for spark in self.sparks:
            spark.x += spark.vx * dt
            spark.y += spark.vy * dt
            spark.vx *= 0.985
            spark.vy = spark.vy * 0.985 + (210 if spark.shape == "fragment" else 90) * dt
            spark.life -= dt
            spark.spin += spark.spin_v * dt
            if spark.life > 0:
                alive.append(spark)
        if len(alive) > 420:
            alive = alive[-420:]
        self.sparks = alive
        rings: list[Ring] = []
        for ring in self.rings:
            ring.radius += ring.growth * dt
            ring.life -= dt
            if ring.life > 0:
                rings.append(ring)
        self.rings = rings

    def sample_bands(self, time: float) -> list[float]:
        if self.analysis is None or not self.analysis.bands:
            return [0.0] * 8
        times = self.analysis.rms_times
        if not times:
            return [0.0] * 8
        if time <= times[0]:
            return list(self.analysis.bands[0])
        if time >= times[-1]:
            return list(self.analysis.bands[-1])
        # linear scan is fine at ~40 Hz feature rate
        for index in range(1, len(times)):
            if times[index] >= time:
                span = times[index] - times[index - 1]
                u = 0.0 if span <= 1e-9 else (time - times[index - 1]) / span
                a = self.analysis.bands[index - 1]
                b = self.analysis.bands[index]
                return [a[k] + (b[k] - a[k]) * u for k in range(len(a))]
        return list(self.analysis.bands[-1])

    def sample_rms(self, time: float) -> float:
        if self.analysis is None or not self.analysis.rms:
            return 0.0
        times = self.analysis.rms_times
        if time <= times[0]:
            return self.analysis.rms[0]
        if time >= times[-1]:
            return self.analysis.rms[-1]
        for index in range(1, len(times)):
            if times[index] >= time:
                span = times[index] - times[index - 1]
                u = 0.0 if span <= 1e-9 else (time - times[index - 1]) / span
                return self.analysis.rms[index - 1] + (self.analysis.rms[index] - self.analysis.rms[index - 1]) * u
        return self.analysis.rms[-1]

    def measure(self) -> dict:
        max_pos = 0.0
        max_time = 0.0
        min_clear = 1e9
        flown = self.segments[: len(self.obstacles)]
        for index, obs in enumerate(self.obstacles):
            px, py = self.position_at(obs.hit_time)
            max_pos = max(max_pos, math.hypot(px - obs.x, py - obs.y))
            best_t = obs.hit_time
            best_err = _gap(self, obs, obs.hit_time)
            for offset in (-0.002, -0.001, -0.0005, 0.0005, 0.001, 0.002):
                err = _gap(self, obs, obs.hit_time + offset)
                if err + 1e-4 < best_err:
                    best_err = err
                    best_t = obs.hit_time + offset
            max_time = max(max_time, abs(best_t - obs.hit_time))
            for other in range(index - 1):
                min_clear = min(min_clear, _pieces_clearance(flown[index], flown[other]))
        if min_clear > 1e8:
            min_clear = 0.0
        kinds = sorted({obs.kind for obs in self.obstacles})
        return {
            "hits": len(self.obstacles),
            "max_position_error_px": max_pos,
            "max_timing_error_s": max_time,
            "min_clearance_px": min_clear,
            "kinds": kinds,
        }

    def shake_offset(self) -> tuple[float, float]:
        if self.shake < 0.2:
            return 0.0, 0.0
        return math.sin(self.time * 18.0) * self.shake, math.cos(self.time * 15.0) * self.shake * 0.55


def max_flyer_penetration(scene: MapScene, fps: float = 30.0) -> float:
    """Largest flyer-disk overlap with any obstacle that is not the current contact.

    Samples are every video frame, the exact beat times, and a point every few
    pixels along the path. Within half a frame of an obstacle's own beat, the
    first SOLID_EPS pixels are the contact kiss and do not count.
    """
    worst = 0.0
    half = 0.5 / fps
    for px, py, time in _flight_samples(scene, fps):
        for obs in scene.obstacles:
            pen = obstacle_penetration(obs, px, py, BALL_R, time)
            if pen <= 0.0:
                continue
            if abs(time - obs.hit_time) <= half + 1e-4:
                pen -= SOLID_EPS
            if pen > worst:
                worst = pen
    return worst


def _flight_samples(scene: MapScene, fps: float):
    nframes = int(math.floor(scene.duration * fps + 1e-6))
    for frame in range(max(nframes, 1)):
        time = frame / fps
        x, y = scene.position_at(time)
        yield x, y, time
    for obs in scene.obstacles:
        x, y = scene.position_at(obs.hit_time)
        yield x, y, obs.hit_time
    for seg in scene.segments:
        length = math.hypot(seg.x1 - seg.x0, seg.y1 - seg.y0)
        steps = max(1, int(math.ceil(length / 8.0)))
        span = seg.t1 - seg.t0
        for step in range(steps + 1):
            u = step / steps
            x, y = _point_on(seg, u)
            yield x, y, seg.t0 + span * u


def _gap(scene: MapScene, obs: Obstacle, time: float) -> float:
    px, py = scene.position_at(time)
    if obs.kind == "peg":
        cx = obs.x + obs.nx * (BALL_R + obs.peg_r)
        cy = obs.y + obs.ny * (BALL_R + obs.peg_r)
        return abs(math.hypot(px - cx, py - cy) - (BALL_R + obs.peg_r))
    cx, cy = obs.x + obs.nx * BALL_R, obs.y + obs.ny * BALL_R
    signed = (px - cx) * obs.nx + (py - cy) * obs.ny
    return abs(signed + BALL_R)


def _damped_step(x: float, v: float, stiffness: float, damping: float, mass: float, dt: float) -> tuple[float, float]:
    """Exact step of x'' + (c/m) x' + (k/m) x = 0."""
    omega = math.sqrt(stiffness / mass)
    if omega < 1e-8:
        return x, 0.0
    zeta = damping / (2.0 * math.sqrt(stiffness * mass))
    if zeta > 1.0001:
        disc = math.sqrt(zeta * zeta - 1.0)
        r1 = omega * (-zeta + disc)
        r2 = omega * (-zeta - disc)
        growth_a = math.exp(r1 * dt)
        growth_b = math.exp(r2 * dt)
        coeff_a = (v - r2 * x) / (r1 - r2)
        coeff_b = x - coeff_a
        position = coeff_a * growth_a + coeff_b * growth_b
        velocity = coeff_a * r1 * growth_a + coeff_b * r2 * growth_b
    elif zeta < 0.999:
        damped = omega * math.sqrt(max(0.0, 1.0 - zeta * zeta))
        decay = math.exp(-zeta * omega * dt)
        cosine, sine = math.cos(damped * dt), math.sin(damped * dt)
        coeff_b = (v + zeta * omega * x) / damped
        position = decay * (x * cosine + coeff_b * sine)
        velocity = -zeta * omega * position + decay * damped * (-x * sine + coeff_b * cosine)
    else:
        decay = math.exp(-omega * dt)
        coeff_b = v + omega * x
        position = (x + coeff_b * dt) * decay
        velocity = coeff_b * decay - omega * position
    return position, velocity


def _reflect(ix: float, iy: float, nx: float, ny: float) -> tuple[float, float]:
    dot = ix * nx + iy * ny
    return _unit(ix - 2.0 * dot * nx, iy - 2.0 * dot * ny)


def _tangent_of(a, b, bow: float, at_end: bool) -> tuple[float, float]:
    return _segment_tangent(Segment(0.0, 1.0, a[0], a[1], b[0], b[1], bow), at_end)


def _normal_from(ix: float, iy: float, ox: float, oy: float):
    dx, dy = ix - ox, iy - oy
    if math.hypot(dx, dy) < 1e-4:
        return None
    return _unit(dx, dy)


def _write_normal(obs: Obstacle, arrival: tuple[float, float], departure: tuple[float, float]) -> None:
    normal = _normal_from(arrival[0], arrival[1], departure[0], departure[1])
    if normal is None:
        return
    obs.nx, obs.ny = normal
    obs.tx, obs.ty = -normal[1], normal[0]


def _iter_points(a, b, bow: float, t0: float, t1: float, step: float = 8.0):
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    steps = max(1, int(math.ceil(length / step)))
    span = t1 - t0
    for index in range(steps + 1):
        u = index / steps
        x, y = _bulge_point(a, b, bow, u)
        yield x, y, t0 + span * u


def _flown_points(segments: list[Segment]) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for seg in segments:
        points.extend(_iter_points((seg.x0, seg.y0), (seg.x1, seg.y1), seg.bow, seg.t0, seg.t1))
    return points


def _count_pen(obs: Obstacle, px: float, py: float, time: float, normal, half: float) -> float:
    pen = obstacle_penetration(obs, px, py, BALL_R, time, normal)
    if pen <= 0.0:
        return 0.0
    if abs(time - obs.hit_time) <= half + 1e-4:
        pen -= SOLID_EPS
    return pen if pen > 0.0 else 0.0


def _leg_solid_pen(a, b, bow, t0, t1, obstacles, segments, flown) -> float:
    """Overlap of this chord, and of the path already flown, with solid obstacles.

    The surface just left turns with this departure. A yaw that swings that
    body into the path already drawn is rejected along with a chord that
    enters any earlier body.
    """
    if not obstacles:
        return 0.0
    half = 0.5 / 30.0
    overrides = {}
    if segments:
        normal = _normal_from(*_segment_tangent(segments[-1], True), *_tangent_of(a, b, bow, False))
        if normal is None:
            return 1e6
        overrides[len(obstacles) - 1] = normal
    worst = 0.0
    for px, py, time in _iter_points(a, b, bow, t0, t1):
        for index, obs in enumerate(obstacles):
            pen = _count_pen(obs, px, py, time, overrides.get(index), half)
            if pen > worst:
                worst = pen
    if flown and overrides:
        obs = obstacles[-1]
        normal = overrides[len(obstacles) - 1]
        for px, py, time in flown:
            pen = _count_pen(obs, px, py, time, normal, half)
            if pen > worst:
                worst = pen
    return worst


_SIMPLER = {
    "double": ("long", "short"),
    "corner": ("long", "short"),
    "long": ("short",),
    "arch": ("long", "short"),
    "stack": ("block", "crack"),
    "block": ("crack",),
    "pattern": ("chord", "single"),
    "chord": ("single",),
    "pair": ("single",),
}


def _relax_obstacle(obs: Obstacle, pen_fn) -> bool:
    """Shrink this body until pen_fn is clear. Keeps the variant when a shorter one fits."""
    if pen_fn() <= 1e-6:
        return False
    original_variant = obs.variant
    original_size = max(obs.size, 1.0)
    original_peg = obs.peg_r
    original_spin = obs.spin
    variants = [original_variant]
    for name in _SIMPLER.get(original_variant, ()):
        if name not in variants:
            variants.append(name)
    sizes = []
    size = original_size
    while size > 18.0:
        size = max(18.0, size * 0.72)
        sizes.append(size)
    changed = False
    for variant in variants:
        try_sizes = list(sizes)
        if variant != original_variant:
            try_sizes = [original_size, *sizes]
        for next_size in try_sizes:
            obs.variant = variant
            obs.size = next_size
            if obs.kind == "peg":
                obs.peg_r = max(10.0, original_peg * next_size / original_size)
            changed = True
            if pen_fn() <= 1e-6:
                return True
    if obs.kind == "spinner":
        for factor in (0.45, 0.2, 0.0):
            obs.variant = "short"
            obs.size = 18.0
            obs.spin = original_spin * factor
            changed = True
            if pen_fn() <= 1e-6:
                return True
    return changed


def _shrink_blockers(pos, candidate, t0, t1, obstacles, segments, flown) -> bool:
    """Shorten bodies this chord still enters, so a later pass can route around them."""
    if not obstacles:
        return False
    half = 0.5 / 30.0
    points = list(_iter_points(pos, candidate, 0.0, t0, t1))
    normal = None
    if segments:
        normal = _normal_from(*_segment_tangent(segments[-1], True), *_tangent_of(pos, candidate, 0.0, False))
    last = len(obstacles) - 1
    changed = False
    for index, obs in enumerate(obstacles):
        use = normal if index == last else None

        def pen_fn(obs=obs, use=use) -> float:
            worst = 0.0
            for px, py, time in points:
                worst = max(worst, _count_pen(obs, px, py, time, use, half))
            if use is not None:
                for px, py, time in flown:
                    worst = max(worst, _count_pen(obs, px, py, time, use, half))
            return worst

        if _relax_obstacle(obs, pen_fn):
            changed = True
    return changed


def _pen_on(obs: Obstacle, samples, half: float) -> float:
    worst = 0.0
    for px, py, time in samples:
        pen = _count_pen(obs, px, py, time, None, half)
        if pen > worst:
            worst = pen
    return worst


def _seat_obstacles(scene: MapScene) -> None:
    """Shrink a body that the routed path still enters. The path stays put."""
    if not scene.obstacles or not scene.segments:
        return
    samples = list(_flight_samples(scene, 30.0))
    half = 0.5 / 30.0
    for obs in scene.obstacles:
        if _pen_on(obs, samples, half) <= 1e-6:
            continue
        original_variant = obs.variant
        original_size = obs.size
        original_peg = obs.peg_r
        original_spin = obs.spin
        variants = [original_variant]
        for name in _SIMPLER.get(original_variant, ()):
            if name not in variants:
                variants.append(name)
        sizes = [original_size]
        size = original_size
        while size > 18.0:
            size = max(18.0, size * 0.72)
            sizes.append(size)
        best = (_pen_on(obs, samples, half), original_variant, original_size, original_peg, original_spin)
        cleared = False
        for variant in variants:
            for size in sizes:
                if variant == original_variant and abs(size - original_size) < 1e-6:
                    continue
                obs.variant = variant
                obs.size = size
                if obs.kind == "peg":
                    obs.peg_r = max(10.0, original_peg * (size / max(original_size, 1.0)))
                pen = _pen_on(obs, samples, half)
                if pen <= 1e-6:
                    cleared = True
                    break
                if best is None or pen < best[0]:
                    best = (pen, variant, size, obs.peg_r, obs.spin)
            if cleared:
                break
        if cleared:
            continue
        if obs.kind == "spinner":
            for factor in (0.45, 0.2, 0.0):
                obs.variant = "short"
                obs.size = 18.0
                obs.spin = original_spin * factor
                if _pen_on(obs, samples, half) <= 1e-6:
                    cleared = True
                    break
        if cleared:
            continue
        if best is None:
            continue
        _pen, variant, size, peg_r, spin = best
        obs.variant = variant
        obs.size = size
        obs.peg_r = peg_r
        obs.spin = spin


def _avoid_overlap(pos, hit, incoming, outgoing, polys, length, obstacles=None, segments=None, t0=0.0, t1=1.0, allow_bend=True):
    """Find a chord that misses flight lines and obstacle bodies.

    The pair of directions yaws together, so a clear corridor keeps its bounce.
    If every yaw still meets a wall, the departure is opened or reversed, and
    then the chord is shortened. The least overlap is kept only when nothing
    is clear.
    """
    obstacles = list(obstacles or [])
    segments = list(segments or [])
    ix, iy = incoming
    ox, oy = outgoing
    flown = _flown_points(segments)
    angles = [0]
    for degrees in (12, 22, 34, 48, 66, 84, 110, 140, 168):
        angles.extend((degrees, -degrees))

    def search(origin_in, origin_out, chords, yaw_angles):
        best = None
        for chord in chords:
            if chord < 24.0:
                continue
            for degrees in yaw_angles:
                rad = math.radians(degrees)
                cosine, sine = math.cos(rad), math.sin(rad)
                ux = origin_in[0] * cosine - origin_in[1] * sine
                uy = origin_in[0] * sine + origin_in[1] * cosine
                vx = origin_out[0] * cosine - origin_out[1] * sine
                vy = origin_out[0] * sine + origin_out[1] * cosine
                candidate = (pos[0] + ux * chord, pos[1] + uy * chord)
                if _blocked(pos, candidate, polys):
                    continue
                pen = _leg_solid_pen(pos, candidate, 0.0, t0, t1, obstacles, segments, flown)
                if pen <= 1e-6:
                    return (candidate, ux, uy, vx, vy), None
                if best is None or pen < best[0]:
                    best = (pen, candidate, ux, uy, vx, vy)
        return None, best

    def probe():
        found, best = search((ix, iy), (ox, oy), (length, length * 0.78, length * 0.58, length * 0.4, length * 0.28), angles)
        if found is not None:
            return found, best
        # The planned bounce cannot pass. Open the turn, then try the mirror.
        for extra in (32, -32, 58, -58, 90, -90, 130, -130, 175, -175):
            rad = math.radians(extra)
            cosine, sine = math.cos(rad), math.sin(rad)
            ex = ox * cosine - oy * sine
            ey = ox * sine + oy * cosine
            found, alt = search((ix, iy), (ex, ey), (length, length * 0.62, length * 0.4), (0, 24, -24, 48, -48, 90, -90))
            if found is not None:
                return found, best
            if alt is not None and (best is None or alt[0] < best[0]):
                best = alt
        if allow_bend and math.hypot(ix - ox, iy - oy) > 1e-3:
            nx, ny = _unit(ix - ox, iy - oy)
            tx, ty = -iy, ix
            for scale in (0.45, 0.9, -0.45, -0.9, 1.4, -1.4, 2.0, -2.0):
                ux, uy = _unit(ix + tx * scale, iy + ty * scale)
                candidate = (pos[0] + ux * length, pos[1] + uy * length)
                if _blocked(pos, candidate, polys):
                    continue
                vx, vy = _reflect(ux, uy, nx, ny)
                pen = _leg_solid_pen(pos, candidate, 0.0, t0, t1, obstacles, segments, flown)
                if pen <= 1e-6:
                    return (candidate, ux, uy, vx, vy), best
                if best is None or pen < best[0]:
                    best = (pen, candidate, ux, uy, vx, vy)
        return None, best

    found, best = probe()
    if found is not None:
        return found
    # No clear corridor. Shorten the walls this chord hits and look again.
    for _ in range(3):
        if best is None or not obstacles:
            break
        if not _shrink_blockers(pos, best[1], t0, t1, obstacles, segments, flown):
            break
        found, retry = probe()
        if found is not None:
            return found
        if retry is not None and (best is None or retry[0] < best[0]):
            best = retry
        else:
            break
    if best is not None:
        return best[1], best[2], best[3], best[4], best[5]
    # Every solid-aware chord crossed the drawn path. Take a free heading anyway
    # so a later beat is not boxed into that crossing.
    for origin in ((ix, iy), (-ix, -iy)):
        for chord in (length, length * 0.55, max(24.0, length * 0.3)):
            for degrees in angles:
                rad = math.radians(degrees)
                cosine, sine = math.cos(rad), math.sin(rad)
                ux = origin[0] * cosine - origin[1] * sine
                uy = origin[0] * sine + origin[1] * cosine
                candidate = (pos[0] + ux * chord, pos[1] + uy * chord)
                if _blocked(pos, candidate, polys):
                    continue
                vx = ox * cosine - oy * sine
                vy = ox * sine + oy * cosine
                return candidate, ux, uy, vx, vy
    return hit, ix, iy, ox, oy


def _align_bounce_normals(obstacles, segments) -> None:
    """Make each surface the reflection normal of the flown corner.

    A gentle bow changes the tangent at the joint, so the normal follows
    those tangents rather than the straight chord.
    """
    for index, obs in enumerate(obstacles):
        if index + 1 >= len(segments):
            continue
        ix, iy = _segment_tangent(segments[index], at_end=True)
        ox, oy = _segment_tangent(segments[index + 1], at_end=False)
        dx, dy = ix - ox, iy - oy
        if math.hypot(dx, dy) < 1e-6:
            continue
        nx, ny = _unit(dx, dy)
        obs.nx, obs.ny = nx, ny
        obs.tx, obs.ty = -ny, nx


def _blocked(a, b, polys) -> bool:
    # The newest segment shares its end with this departure. The corner may
    # leave that line, but the new end is where the next leg starts, so it has
    # to sit a full clearance away or the following beat is boxed in.
    for index, (c, d, bow) in enumerate(polys):
        recent = index == len(polys) - 1
        for p, q in _poly_pieces(c, d, bow):
            if recent:
                if _folds_back(a, b, p, q) or _segment_distance(b, b, p, q) < CLEARANCE:
                    return True
                continue
            if _intersects(a, b, p, q) or _segment_distance(a, b, p, q) < CLEARANCE:
                return True
    return False


def _folds_back(a, b, p, q) -> bool:
    """True when the new chord returns onto the segment it just left."""
    if _intersects(a, b, p, q) and math.hypot(b[0] - a[0], b[1] - a[1]) > 24:
        # A shared start is not an intersection under the strict cross test.
        return True
    for step in range(2, 7):
        u = step / 6.0
        point = (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
        if math.hypot(point[0] - a[0], point[1] - a[1]) < 80:
            continue
        if _segment_distance(point, point, p, q) < 18:
            return True
    return False


_SECTION_SIZE = {"calm": 0.74, "mid": 1.0, "intense": 1.36}
_SECTION_LIGHT = {"calm": 0.78, "mid": 1.0, "intense": 1.08}
_SECTION_GAIN = {"calm": 0.42, "mid": 1.0, "intense": 1.5}
_VARIANT_SCALE = {
    "short": 0.58,
    "long": 1.7,
    "corner": 1.2,
    "double": 1.15,
    "arch": 1.35,
    "single": 0.85,
    "pair": 1.0,
    "chord": 1.05,
    "pattern": 1.1,
    "crack": 1.25,
    "block": 1.0,
    "stack": 1.2,
}
_QUIET_VARIANTS = {"short", "single", "crack"}
_RICH_VARIANTS = {"long", "double", "corner", "stack", "chord", "pattern", "arch", "pair"}


def _phrase_motion(label: str, index: int, strong: bool) -> tuple[str, float, float, float, bool]:
    """Speed, bounce spread, and bow for this beat.

    Calm stretches travel farther between hits. Intense clusters pack the
    same family of obstacles into a shorter run. Curves stay a small bow
    on the chord so the tangent never reverses.
    """
    if label == "intense" and index % 10 != 0:
        return "cluster", 480.0, 0.8 if strong else 0.64, 0.0, False
    if label == "calm":
        if (index // 4) % 2 == 0:
            return "stretch", 860.0, 0.22, 0.0, True
        return "corridor", 680.0, 0.36, 0.0, False
    if index % 3 != 2:
        return "curve", 640.0, 0.5, 0.12, False
    return "corridor", 720.0, 0.32, 0.0, False


def _variants_for(theme: Theme, kind: str) -> tuple[str, ...]:
    for name, options in theme.variants:
        if name == kind:
            return options
    return ("plain",)


def _choose_variant(theme: Theme, kind: str, label: str, flourish: bool, rng: list[int]) -> str:
    options = list(_variants_for(theme, kind))
    if flourish:
        rich = [name for name in options if name in _RICH_VARIANTS]
        if rich:
            return rich[int(xorshift32(rng) * len(rich)) % len(rich)]
    if label == "calm":
        quiet = [name for name in options if name in _QUIET_VARIANTS]
        if quiet and xorshift32(rng) < 0.72:
            return quiet[int(xorshift32(rng) * len(quiet)) % len(quiet)]
    if label == "intense":
        rich = [name for name in options if name in _RICH_VARIANTS]
        if rich and xorshift32(rng) < 0.7:
            return rich[int(xorshift32(rng) * len(rich)) % len(rich)]
    return options[int(xorshift32(rng) * len(options)) % len(options)]


def _flourish(scene: MapScene, obs: Obstacle, intensity: float) -> None:
    kind = scene.skin.flourish if scene.skin is not None else ""
    if kind == "ring":
        scene.add_ring(obs.x, obs.y, 48 + 30 * intensity, obs.kind_hue, 20 + 6 * intensity, life=0.55)
    elif kind == "split":
        scene.split_max = 0.52
        scene.split_life = scene.split_max
        scene.split_count = 3 if obs.section == "intense" else 2
    elif kind == "break":
        for _ in range(16):
            ang = xorshift32(scene.rng) * math.tau
            speed = 140 + xorshift32(scene.rng) * 320
            scene.add_fragment(
                obs.x,
                obs.y,
                math.cos(ang) * speed + obs.nx * 80,
                math.sin(ang) * speed + obs.ny * 80,
                6 + xorshift32(scene.rng) * 8,
                obs.kind_hue,
                0.7,
                xorshift32(scene.rng) * math.tau,
            )
        scene.bump_shake(max(scene.shake, 12 + 10 * intensity))
    elif kind == "arpeggio":
        scene.add_ring(obs.x, obs.y, 40, obs.kind_hue, 8, life=0.45)
    elif kind == "burst":
        scene.add_ring(obs.x, obs.y, 80, obs.kind_hue, 6, life=0.32)


def _point_on(seg: Segment, u: float) -> tuple[float, float]:
    return _bulge_point((seg.x0, seg.y0), (seg.x1, seg.y1), seg.bow, u)


def _velocity_on(seg: Segment, u: float) -> tuple[float, float]:
    span = max(1e-6, seg.t1 - seg.t0)
    dx, dy = seg.x1 - seg.x0, seg.y1 - seg.y0
    lateral = 4.0 * (1.0 - 2.0 * u) * seg.bow
    if abs(lateral) < 1e-8:
        return dx / span, dy / span
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length
    return (dx + px * lateral) / span, (dy + py * lateral) / span


def _segment_tangent(seg: Segment, at_end: bool) -> tuple[float, float]:
    return _unit(*_velocity_on(seg, 1.0 if at_end else 0.0))


def _bulge_point(a, b, bow: float, u: float) -> tuple[float, float]:
    x = a[0] + (b[0] - a[0]) * u
    y = a[1] + (b[1] - a[1]) * u
    bulge = 4.0 * u * (1.0 - u) * bow
    if abs(bulge) < 1e-8:
        return x, y
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    return x + (-dy / length) * bulge, y + (dx / length) * bulge


def _poly_pieces(a, b, bow: float):
    if abs(bow) < 1e-3:
        return [(a, b)]
    points = [_bulge_point(a, b, bow, u) for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return list(zip(points, points[1:]))


def _pieces_clearance(left: Segment, right: Segment) -> float:
    best = 1e9
    for a, b in _poly_pieces((left.x0, left.y0), (left.x1, left.y1), left.bow):
        for c, d in _poly_pieces((right.x0, right.y0), (right.x1, right.y1), right.bow):
            if _intersects(a, b, c, d):
                return 0.0
            best = min(best, _segment_distance(a, b, c, d))
    return best


def _fit_bow(a, b, magnitude: float, polys, obstacles=None, segments=None, t0=0.0, t1=1.0) -> float:
    if magnitude < 1.0:
        return 0.0
    obstacles = list(obstacles or [])
    segments = list(segments or [])
    flown = _flown_points(segments)
    for sign in (1.0, -1.0):
        bow = magnitude * sign
        if _curve_blocked(a, b, bow, polys):
            continue
        if _leg_solid_pen(a, b, bow, t0, t1, obstacles, segments, flown) > 1e-6:
            continue
        return bow
    return 0.0


def _curve_blocked(a, b, bow: float, polys) -> bool:
    # Skip the piece that leaves the shared start; it sits on the previous corner.
    pieces = _poly_pieces(a, b, bow)
    if len(pieces) > 1:
        pieces = pieces[1:]
    for c, d, other in polys:
        for p, q in pieces:
            for r, s in _poly_pieces(c, d, other):
                if _intersects(p, q, r, s) or _segment_distance(p, q, r, s) < CLEARANCE:
                    return True
    return False


def _choose_kind(pool: tuple[str, ...], strong: bool, counts: dict[str, int], last: str, rng: list[int]) -> str:
    """Stay inside the theme. Strong hits lean toward the heavier object in that set."""
    options = list(pool)
    if strong:
        heavy = [name for name in options if name in {"breakable", "pad", "spinner", "wall"}]
        if heavy and xorshift32(rng) < 0.7:
            options = heavy
    least = min(options, key=lambda name: (counts[name], name == last))
    if counts[least] == 0 or xorshift32(rng) < 0.6:
        return least
    rest = [name for name in options if name != last] or options
    return rest[int(xorshift32(rng) * len(rest)) % len(rest)]


def _object_dims(kind: str, strong: bool, energy: float, scale: float, rng: list[int]) -> tuple[float, float, float, float]:
    bases = {
        "wall": 150.0,
        "platform": 230.0,
        "peg": 30.0,
        "pad": 120.0,
        "breakable": 74.0,
        "spinner": 128.0,
        "piano": 26.0,
    }
    hues = {
        "wall": 0.0,
        "platform": 28.0,
        "peg": 55.0,
        "pad": 200.0,
        "breakable": 330.0,
        "spinner": 150.0,
        "piano": 265.0,
    }
    boost = 1.0 + 0.22 * energy + (0.28 if strong else 0.0)
    size = bases[kind] * boost * max(0.55, scale)
    peg_r = size if kind == "peg" else 0.0
    if kind == "peg":
        size = peg_r
    spin = 0.0
    if kind == "spinner":
        spin = (1.6 + xorshift32(rng) * 1.4) * (1 if xorshift32(rng) < 0.5 else -1)
    return size, peg_r, spin, hues[kind]


def _stars(rng: list[int], segments: list[Segment]) -> list[tuple[float, float]]:
    if not segments:
        return []
    xs = [value for seg in segments for value in (seg.x0, seg.x1)]
    ys = [value for seg in segments for value in (seg.y0, seg.y1)]
    min_x, max_x = min(xs) - 900, max(xs) + 900
    min_y, max_y = min(ys) - 900, max(ys) + 900
    stars = []
    step = 240.0
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            if xorshift32(rng) > 0.45:
                stars.append((x + (xorshift32(rng) - 0.5) * 80, y + (xorshift32(rng) - 0.5) * 80))
            y += step
        x += step
    return stars

