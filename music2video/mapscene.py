"""One ball, one generated map: a surface is placed so the ball hits it on each beat."""

from __future__ import annotations

import math
from dataclasses import dataclass

from music2video.analysis import Analysis, section_at
from music2video.registry import BALL_R, EFFECTS, Obstacle
from music2video.scene import xorshift32
from music2video.themes import THEMES, Theme, pick_theme

SPEED = 680.0
SPREAD = 0.58
CLEARANCE = 72.0


@dataclass
class Segment:
    t0: float
    t1: float
    x0: float
    y0: float
    x1: float
    y1: float


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
        polys: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for index, (beat, is_strong) in enumerate(deduped):
            ivx, ivy = _unit(math.cos(heading), math.sin(heading))
            dt = max(1e-3, beat - t_prev)
            length = SPEED * dt
            hit = (pos[0] + ivx * length, pos[1] + ivy * length)
            next_lane = lane
            next_side = -side
            next_spread = SPREAD + (0.32 if is_strong else 0.0)
            row_left -= 1
            if row_left <= 0:
                next_lane = lane + math.pi / 2
                turns += 1
                if turns % 2 == 0:
                    row_len += 3
                row_left = row_len
                next_spread = 0.12
                next_side = side
            ovx, ovy = _unit(math.cos(next_lane + next_side * next_spread), math.sin(next_lane + next_side * next_spread))
            # Steer around a clash by rotating the whole bounce. The departure
            # stays the reflection of the arrival, so the next segment cannot
            # leave at an unrelated angle.
            hit, ivx, ivy, ovx, ovy = _avoid_overlap(pos, hit, (ivx, ivy), (ovx, ovy), polys, length)
            nx, ny = _unit(ivx - ovx, ivy - ovy)
            tx, ty = -ny, nx
            section = section_at(self.analysis.sections if self.analysis else (), beat)
            kind = _choose_kind(self.skin.objects, is_strong, counts, last_kind, self.rng)
            counts[kind] += 1
            last_kind = kind
            scale = min(1.0, length / 280.0)
            size, peg_r, spin, _ignored = _object_dims(kind, is_strong, section.energy, scale, self.rng)
            kind_hue = self.skin.hue_step * self.skin.objects.index(kind)
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
                    light=self.skin.object_light,
                )
            )
            self.segments.append(Segment(t_prev, beat, pos[0], pos[1], hit[0], hit[1]))
            polys.append((pos, hit))
            pos = hit
            t_prev = beat
            lane = next_lane
            side = next_side
            heading = math.atan2(ovy, ovx)
        if self.duration > t_prev + 1e-3:
            ivx, ivy = _unit(math.cos(heading), math.sin(heading))
            dist = SPEED * (self.duration - t_prev)
            self.segments.append(
                Segment(t_prev, self.duration, pos[0], pos[1], pos[0] + ivx * dist, pos[1] + ivy * dist)
            )
        # The next corner may still yaw the departure. Rebuild every normal
        # from the two segments that actually meet, which is the direction
        # that reflects the arrival into that departure.
        _align_bounce_normals(self.obstacles, self.segments)

    def position_at(self, time: float) -> tuple[float, float]:
        if not self.segments:
            return (0.0, 0.0)
        if time <= self.segments[0].t0:
            return self.segments[0].x0, self.segments[0].y0
        for seg in self.segments:
            if time <= seg.t1:
                span = seg.t1 - seg.t0
                u = 0.0 if span <= 1e-9 else (time - seg.t0) / span
                return seg.x0 + (seg.x1 - seg.x0) * u, seg.y0 + (seg.y1 - seg.y0) * u
        last = self.segments[-1]
        return last.x1, last.y1

    def velocity_at(self, time: float) -> tuple[float, float]:
        for seg in self.segments:
            if time <= seg.t1:
                span = max(1e-6, seg.t1 - seg.t0)
                return (seg.x1 - seg.x0) / span, (seg.y1 - seg.y0) / span
        if not self.segments:
            return SPEED, 0.0
        last = self.segments[-1]
        span = max(1e-6, last.t1 - last.t0)
        return (last.x1 - last.x0) / span, (last.y1 - last.y0) / span

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
        vx, vy = self.velocity_at(time)
        # Ease the camera. A hard velocity jump at a bounce must not yank the frame.
        target_x = self.pos[0] + vx * 0.02
        target_y = self.pos[1] + vy * 0.02
        blend = 1.0 - math.exp(-10.0 * dt)
        self.cam_x += (target_x - self.cam_x) * blend
        self.cam_y += (target_y - self.cam_y) * blend
        if self.skin is not None:
            self.theme_hue = self.skin.hue
        self._integrate(dt)
        self.bg_flash = max(self.bg_flash, 0.0)

    def _decay(self, dt: float) -> None:
        fade = math.exp(-8.0 * dt)
        self.shake *= math.exp(-14.0 * dt)
        self._spring(dt)
        self.chroma *= math.exp(-5.0 * dt)
        self.morph *= math.exp(-3.5 * dt)
        self.hue_jump *= math.exp(-4.5 * dt)
        self.bg_flash *= fade
        for obs in self.obstacles:
            obs.flash *= fade

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
        self.bg_flash = feel.pause + 0.35
        weight = 0.7 + 0.3 * obs.energy + (0.15 if obs.strong else 0.0)
        intensity = min(1.0, self.skin.intensity * weight)
        self.squash = feel.squash * (0.85 + 0.15 * weight)
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

    def add_ring(self, x: float, y: float, growth: float, hue: float, width: float) -> None:
        self.rings.append(Ring(x, y, 12.0, growth, width, hue, 0.38, 0.38))

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
            spark.vx *= 0.98
            spark.vy = spark.vy * 0.98 + (240 if spark.shape == "fragment" else 20) * dt
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
        polys = [(seg.x0, seg.y0, seg.x1, seg.y1) for seg in self.segments[: len(self.obstacles)]]
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
            a = (polys[index][0], polys[index][1])
            b = (polys[index][2], polys[index][3])
            for other in range(index - 1):
                c = (polys[other][0], polys[other][1])
                d = (polys[other][2], polys[other][3])
                if _intersects(a, b, c, d):
                    min_clear = min(min_clear, 0.0)
                else:
                    min_clear = min(min_clear, _segment_distance(a, b, c, d))
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


def _avoid_overlap(pos, hit, incoming, outgoing, polys, length):
    """Yaw the whole corner until this arrival is clear.

    Incoming and outgoing rotate together, so the bounce angle is unchanged.
    A one-sided bend used to mirror across the old normal and cancel the
    turn, which made the next segment wander.
    """
    ix, iy = incoming
    ox, oy = outgoing
    if not _blocked(pos, hit, polys):
        return hit, ix, iy, ox, oy
    for degrees in (12, -12, 22, -22, 34, -34, 48, -48, 66, -66, 84, -84):
        rad = math.radians(degrees)
        cosine, sine = math.cos(rad), math.sin(rad)
        ux, uy = ix * cosine - iy * sine, ix * sine + iy * cosine
        vx, vy = ox * cosine - oy * sine, ox * sine + oy * cosine
        candidate = (pos[0] + ux * length, pos[1] + uy * length)
        if not _blocked(pos, candidate, polys):
            return candidate, ux, uy, vx, vy
    # Last resort: bend the arrival and reflect it, still a real bounce.
    nx, ny = _unit(ix - ox, iy - oy)
    tx, ty = -iy, ix
    for scale in (0.45, 0.9, -0.45, -0.9, 1.4, -1.4, 2.0, -2.0):
        ux, uy = _unit(ix + tx * scale, iy + ty * scale)
        candidate = (pos[0] + ux * length, pos[1] + uy * length)
        if not _blocked(pos, candidate, polys):
            vx, vy = _reflect(ux, uy, nx, ny)
            return candidate, ux, uy, vx, vy
    return hit, ix, iy, ox, oy


def _align_bounce_normals(obstacles, segments) -> None:
    """Make each surface the reflection normal of the flown corner."""
    for index, obs in enumerate(obstacles):
        if index + 1 >= len(segments):
            continue
        arrived = segments[index]
        left = segments[index + 1]
        ix, iy = _unit(arrived.x1 - arrived.x0, arrived.y1 - arrived.y0)
        ox, oy = _unit(left.x1 - left.x0, left.y1 - left.y0)
        dx, dy = ix - ox, iy - oy
        if math.hypot(dx, dy) < 1e-6:
            continue
        nx, ny = _unit(dx, dy)
        obs.nx, obs.ny = nx, ny
        obs.tx, obs.ty = -ny, nx


def _blocked(a, b, polys) -> bool:
    # The newest segment shares an endpoint with the candidate, so it is not a clash.
    older = polys[:-1]
    for c, d in older:
        if _intersects(a, b, c, d) or _segment_distance(a, b, c, d) < CLEARANCE:
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

