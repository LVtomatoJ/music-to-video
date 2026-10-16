"""Pluggable collision objects and hit effects.

Add an object with `register_object` (a draw function plus the effect names it
fires). Add an effect with `register_effect`; the callback receives the scene,
which only needs `add_ring`, `add_spark`, `add_fragment`, and the juice fields
`shake`, `squash`, `chroma`, and `morph`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cairo

from music2video.scene import hsl_to_rgb
from music2video.themes import PALETTES, palette_of

BALL_R = 28.0


@dataclass
class Obstacle:
    kind: str
    hit_time: float
    x: float
    y: float
    nx: float
    ny: float
    tx: float
    ty: float
    strong: bool
    energy: float
    section: str
    size: float
    spin: float
    peg_r: float
    kind_hue: float
    sat: float = 0.72
    light: float = 0.42
    variant: str = "plain"
    flourish: bool = False
    lit: bool = False
    broken: bool = False
    flash: float = 0.0


@dataclass(frozen=True)
class ObjectKind:
    name: str
    sections: tuple[str, ...]
    effects: tuple[str, ...]
    draw: Callable[[object, Obstacle, float, float, float], None]


OBJECTS: dict[str, ObjectKind] = {}
EFFECTS: dict[str, Callable] = {}


def register_object(kind: ObjectKind) -> ObjectKind:
    OBJECTS[kind.name] = kind
    return kind


def register_effect(name: str):
    def decorate(fn: Callable) -> Callable:
        EFFECTS[name] = fn
        return fn

    return decorate


def contact(obs: Obstacle) -> tuple[float, float]:
    return obs.x + obs.nx * BALL_R, obs.y + obs.ny * BALL_R


def arch_offset(length: float, thick: float) -> float:
    """Shift so the stroked arch's near side sits on the contact plane."""
    rad = thick / 2.0
    min_x = min(_arch_point(length, step / 32.0)[0] for step in range(33))
    return rad - min_x


def _arch_point(length: float, u: float) -> tuple[float, float]:
    p0 = (length * 0.18, -length / 2.0)
    c0 = (-length * 0.06, -length * 0.18)
    c1 = (-length * 0.06, length * 0.18)
    p1 = (length * 0.18, length / 2.0)
    omu = 1.0 - u
    x = omu**3 * p0[0] + 3 * omu**2 * u * c0[0] + 3 * omu * u**2 * c1[0] + u**3 * p1[0]
    y = omu**3 * p0[1] + 3 * omu**2 * u * c0[1] + 3 * omu * u**2 * c1[1] + u**3 * p1[1]
    return x, y


def _rect_penetration(lx: float, ly: float, x: float, y: float, w: float, h: float, radius: float) -> float:
    """How far a disk at (lx, ly) enters an axis-aligned rectangle."""
    if w < 0.0:
        x, w = x + w, -w
    if h < 0.0:
        y, h = y + h, -h
    nearest_x = min(max(lx, x), x + w)
    nearest_y = min(max(ly, y), y + h)
    if lx < x or lx > x + w or ly < y or ly > y + h:
        return radius - math.hypot(lx - nearest_x, ly - nearest_y)
    inset = min(lx - x, x + w - lx, ly - y, y + h - ly)
    return radius + inset


def _circle_penetration(px: float, py: float, cx: float, cy: float, cr: float, radius: float) -> float:
    return radius + cr - math.hypot(px - cx, py - cy)


def _capsule_penetration(
    lx: float, ly: float, x0: float, y0: float, x1: float, y1: float, cr: float, radius: float
) -> float:
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-8:
        return _circle_penetration(lx, ly, x0, y0, cr, radius)
    t = min(1.0, max(0.0, ((lx - x0) * dx + (ly - y0) * dy) / length_sq))
    return _circle_penetration(lx, ly, x0 + dx * t, y0 + dy * t, cr, radius)


def _solid_reach(obs: Obstacle) -> float:
    """Overestimate of the distance from the ball center to any solid point."""
    span = max(abs(obs.size), abs(obs.peg_r), 1.0)
    if obs.kind == "piano":
        return span * 8.0 + 80.0
    if obs.kind == "peg":
        return BALL_R + span * 3.5 + 8.0
    if obs.kind == "spinner":
        return span * 2.0 + 48.0
    return span * 0.9 + 80.0


def obstacle_penetration(
    obs: Obstacle,
    px: float,
    py: float,
    radius: float,
    time: float,
    normal: tuple[float, float] | None = None,
) -> float:
    """Signed overlap of a disk with the obstacle's drawn solid.

    Positive means the disk enters the body. The contact kiss of a face that
    starts on the contact plane is zero when the center sits one radius outside.
    A broken block keeps its unbroken shape: the flyer still cannot fly the hole.
    """
    nx, ny = obs.nx, obs.ny
    if normal is not None:
        nx, ny = normal
    tx, ty = -ny, nx
    if math.hypot(px - obs.x, py - obs.y) > _solid_reach(obs) + radius:
        return -1.0
    if obs.kind == "peg":
        return _peg_penetration(obs, px, py, nx, ny, tx, ty, radius)
    if obs.kind == "spinner":
        return _spinner_penetration(obs, px, py, nx, ny, tx, ty, radius, time)
    ox = obs.x + nx * BALL_R
    oy = obs.y + ny * BALL_R
    dx, dy = px - ox, py - oy
    lx = dx * nx + dy * ny
    ly = dx * tx + dy * ty
    if obs.kind == "pad":
        thick = 28.0 + (10.0 if obs.strong else 0.0)
        half = obs.size / 2.0
        return _capsule_penetration(lx, ly, thick * 0.5, -half, thick * 0.5, half, thick / 2.0, radius)
    if obs.kind == "platform" and obs.variant == "arch":
        return _arch_penetration(obs.size, 16.0 if obs.variant == "short" else 22.0, lx, ly, radius)
    worst = -1e9
    for x, y, w, h in _solid_rects(obs):
        worst = max(worst, _rect_penetration(lx, ly, x, y, w, h, radius))
    return worst


def _solid_rects(obs: Obstacle) -> list[tuple[float, float, float, float]]:
    length = obs.size
    variant = obs.variant
    if obs.kind == "wall":
        thick = 14.0 if variant == "short" else 18.0
        rects = [(0.0, -length / 2.0, thick, length)]
        if variant == "double":
            rects.append((thick + 16.0, -length * 0.42, thick * 0.75, length * 0.84))
        if variant == "corner":
            arm = max(54.0, length * 0.42)
            rects.append((0.0, length / 2.0 - thick, arm, thick))
        return rects
    if obs.kind == "platform":
        thick = 16.0 if variant == "short" else 22.0
        rects = [(0.0, -length / 2.0, thick, length)]
        for foot in (-length * 0.32, length * 0.32):
            rects.append((thick * 0.35, foot - 5.0, thick * 1.3, 10.0))
        return rects
    if obs.kind == "breakable":
        side = length
        if variant == "crack":
            return [(0.0, -side, 12.0, side * 2.0)]
        if variant == "stack":
            block_h = side * 0.42
            return [
                (0.0, slot * (block_h + 8.0) - block_h / 2.0, side * 0.9, block_h) for slot in (-1, 0, 1)
            ]
        return [(0.0, -side / 2.0, side, side)]
    if obs.kind == "piano":
        if variant == "single":
            indices = (0,)
        elif variant == "chord":
            indices = (-1, 0, 1)
        else:
            indices = tuple(range(-3, 4))
        key_w = max(18.0, length if variant != "single" else length * 1.35)
        key_h = key_w * (3.1 if variant == "single" else 2.6)
        rects = []
        for index in indices:
            local_y = index * (key_w + 3.0)
            black = variant == "pattern" and index % 2 != 0
            height = key_h * (0.62 if black else 1.0)
            rects.append((0.0, local_y - key_w / 2.0, height, key_w))
        return rects
    return []


def _arch_penetration(length: float, thick: float, lx: float, ly: float, radius: float) -> float:
    shift = arch_offset(length, thick)
    rad = thick / 2.0
    steps = max(16, int(length / 4.0))
    worst = -1e9
    for step in range(steps + 1):
        x, y = _arch_point(length, step / steps)
        worst = max(worst, _circle_penetration(lx, ly, x + shift, y, rad, radius))
    return worst


def _peg_penetration(
    obs: Obstacle, px: float, py: float, nx: float, ny: float, tx: float, ty: float, radius: float
) -> float:
    cx = obs.x + nx * (BALL_R + obs.peg_r)
    cy = obs.y + ny * (BALL_R + obs.peg_r)
    worst = _circle_penetration(px, py, cx, cy, obs.peg_r, radius)
    if obs.variant == "pair":
        offset = obs.peg_r * 2.4
        worst = max(
            worst,
            _circle_penetration(px, py, cx + tx * offset, cy + ty * offset, obs.peg_r * 0.78, radius),
        )
    return worst


def _spinner_penetration(
    obs: Obstacle,
    px: float,
    py: float,
    nx: float,
    ny: float,
    tx: float,
    ty: float,
    radius: float,
    time: float,
) -> float:
    length = obs.size * (0.7 if obs.variant == "short" else 1.35 if obs.variant == "long" else 1.0)
    thick = 22.0 if obs.variant == "short" else 14.0
    angle = math.atan2(ty, tx) + obs.spin * (time - obs.hit_time)
    cosine, sine = math.cos(angle), math.sin(angle)
    ox = obs.x + nx * BALL_R
    oy = obs.y + ny * BALL_R
    dx, dy = px - ox, py - oy
    lx = dx * cosine + dy * sine
    ly = dx * (-sine) + dy * cosine
    worst = _rect_penetration(lx, ly, -length / 2.0, -thick, length, thick, radius)
    cap = thick / 2.0
    worst = max(worst, _circle_penetration(lx, ly, -length / 2.0, -thick / 2.0, cap, radius))
    worst = max(worst, _circle_penetration(lx, ly, length / 2.0, -thick / 2.0, cap, radius))
    return worst


def _place(ctx, obs: Obstacle) -> None:
    cx, cy = contact(obs)
    ctx.translate(cx, cy)
    ctx.rotate(math.atan2(obs.ny, obs.nx))


_PALETTE = None


def bind_palette(theme) -> None:
    global _PALETTE
    _PALETTE = palette_of(theme)


def bind_palette_name(name: str) -> None:
    global _PALETTE
    _PALETTE = PALETTES.get(name, PALETTES["neon"])


def active_palette():
    return _PALETTE if _PALETTE is not None else palette_of(None)


def _ease_flash(flash: float) -> float:
    u = max(0.0, min(1.0, flash))
    return u * u * (3.0 - 2.0 * u)


def _mix(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _color(hue: float, sat: float, light: float, flash: float) -> tuple[float, float, float]:
    return hsl_to_rgb(hue, sat, min(0.94, light + 0.28 * _ease_flash(flash)))


def _body(obs: Obstacle, hue: float, lit: bool) -> tuple[float, float, float]:
    light = obs.light if lit else obs.light * 0.72
    return _color(hue, obs.sat, light, obs.flash)


def _ink(obs: Obstacle, hue: float) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
    """Body, rim, shadow, and the eased hit glow in the active palette."""
    glow = _ease_flash(obs.flash)
    pal = active_palette()
    shift = ((hue % 48.0) / 48.0) * 0.22
    color = _mix(pal.ink, pal.accent2, shift)
    if obs.lit:
        color = _mix(color, pal.accent, 0.18 + 0.45 * glow)
    else:
        color = _mix(color, pal.bg_bottom, 0.28)
    return color, pal.rim, pal.shadow, glow


def _round_rect(ctx, x: float, y: float, w: float, h: float, radius: float) -> None:
    radius = max(0.0, min(radius, abs(w) * 0.5, abs(h) * 0.5))
    if radius < 0.6:
        ctx.rectangle(x, y, w, h)
        return
    ctx.new_sub_path()
    ctx.arc(x + w - radius, y + radius, radius, -math.pi / 2, 0)
    ctx.arc(x + w - radius, y + h - radius, radius, 0, math.pi / 2)
    ctx.arc(x + radius, y + h - radius, radius, math.pi / 2, math.pi)
    ctx.arc(x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def _paint_slab(ctx, x, y, w, h, color, rim, shadow, alpha, glow, radius) -> None:
    """Rounded body, soft shadow, inner gradient, and a thin contact highlight."""
    if w <= 0.5 or h <= 0.5 or alpha <= 0.01:
        return
    ctx.save()
    for offset, shade in ((8.0, 0.10), (3.5, 0.16)):
        ctx.set_source_rgba(*shadow, alpha * shade)
        _round_rect(ctx, x + offset * 0.45, y + offset * 0.7, w, h, radius)
        ctx.fill()
    if glow > 0.05:
        ctx.set_source_rgba(*rim, alpha * 0.16 * glow)
        _round_rect(ctx, x - 3.0, y - 3.0, w + 6.0, h + 6.0, radius + 3.0)
        ctx.fill()
    grad = cairo.LinearGradient(x, y, x + w, y + h * 0.15)
    lit = tuple(min(1.0, channel + 0.14 + 0.18 * glow) for channel in color)
    deep = tuple(channel * 0.58 for channel in color)
    grad.add_color_stop_rgba(0.0, *lit, alpha)
    grad.add_color_stop_rgba(1.0, *deep, alpha)
    ctx.set_source(grad)
    _round_rect(ctx, x, y, w, h, radius)
    ctx.fill()
    ctx.set_source_rgba(*rim, alpha * (0.22 + 0.5 * glow))
    ctx.set_line_width(1.25)
    inset = 1.3
    _round_rect(ctx, x + inset, y + inset, max(1.0, w - inset * 2), max(1.0, h - inset * 2), max(0.0, radius - inset))
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, alpha * (0.38 + 0.5 * glow))
    ctx.set_line_width(1.7)
    ctx.move_to(x + 1.1, y + radius)
    ctx.line_to(x + 1.1, y + h - radius)
    ctx.stroke()
    ctx.restore()


def _paint_disk(ctx, cx, cy, radius, color, rim, shadow, alpha, glow) -> None:
    ctx.set_source_rgba(*shadow, alpha * 0.22)
    ctx.arc(cx + radius * 0.12, cy + radius * 0.16, radius, 0, math.tau)
    ctx.fill()
    if glow > 0.04:
        halo = cairo.RadialGradient(cx, cy, radius * 0.4, cx, cy, radius * (1.55 + 0.35 * glow))
        halo.add_color_stop_rgba(0, *rim, alpha * 0.28 * glow)
        halo.add_color_stop_rgba(1, *rim, 0)
        ctx.set_source(halo)
        ctx.arc(cx, cy, radius * (1.55 + 0.35 * glow), 0, math.tau)
        ctx.fill()
    grad = cairo.RadialGradient(cx - radius * 0.32, cy - radius * 0.38, radius * 0.08, cx, cy, radius)
    grad.add_color_stop_rgba(0.0, 1, 1, 1, alpha)
    grad.add_color_stop_rgba(0.28, *tuple(min(1.0, c + 0.12) for c in color), alpha)
    grad.add_color_stop_rgba(1.0, *tuple(c * 0.55 for c in color), alpha)
    ctx.set_source(grad)
    ctx.arc(cx, cy, radius, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(1, 1, 1, alpha * (0.75 + 0.2 * glow))
    ctx.arc(cx - radius * 0.30, cy - radius * 0.34, radius * 0.16, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(*rim, alpha * (0.35 + 0.4 * glow))
    ctx.set_line_width(1.4)
    ctx.arc(cx, cy, radius - 0.6, 0, math.tau)
    ctx.stroke()


def draw_wall(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    del time
    if alpha <= 0.02:
        return
    length = obs.size
    variant = obs.variant
    thick = 14 if variant == "short" else 18
    color, rim, shadow, glow = _ink(obs, hue)
    ctx.save()
    _place(ctx, obs)
    radius = min(7.0, thick * 0.42)
    _paint_slab(ctx, 0, -length / 2, thick, length, color, rim, shadow, alpha, glow, radius)
    if variant == "double":
        _paint_slab(
            ctx,
            thick + 16,
            -length * 0.42,
            thick * 0.75,
            length * 0.84,
            color,
            rim,
            shadow,
            alpha * 0.95,
            glow,
            radius,
        )
    if variant == "corner":
        arm = max(54.0, length * 0.42)
        _paint_slab(ctx, 0, length / 2 - thick, arm, thick, color, rim, shadow, alpha, glow, radius)
    ctx.restore()


def _arch_path(ctx, length: float, shift: float) -> None:
    ctx.move_to(length * 0.18 + shift, -length / 2)
    ctx.curve_to(
        -length * 0.06 + shift,
        -length * 0.18,
        -length * 0.06 + shift,
        length * 0.18,
        length * 0.18 + shift,
        length / 2,
    )


def draw_platform(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    del time
    if alpha <= 0.02:
        return
    length = obs.size
    thick = 16 if obs.variant == "short" else 22
    color, rim, shadow, glow = _ink(obs, hue)
    ctx.save()
    _place(ctx, obs)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    if obs.variant == "arch":
        # Same centerline as the collision stroke. A soft shadow and a rim sit around it.
        shift = arch_offset(length, thick)
        ctx.set_line_width(thick + 14)
        ctx.set_source_rgba(*shadow, alpha * 0.16)
        _arch_path(ctx, length, shift + 3)
        ctx.stroke()
        ctx.set_line_width(thick)
        ctx.set_source_rgba(*color, alpha)
        _arch_path(ctx, length, shift)
        ctx.stroke()
        ctx.set_line_width(2.0)
        ctx.set_source_rgba(*rim, alpha * (0.4 + 0.45 * glow))
        _arch_path(ctx, length, shift)
        ctx.stroke()
    else:
        radius = min(8.0, thick * 0.4)
        _paint_slab(ctx, 0, -length / 2, thick, length, color, rim, shadow, alpha, glow, radius)
        for foot in (-length * 0.32, length * 0.32):
            _paint_slab(ctx, thick * 0.35, foot - 5, thick * 1.3, 10, color, rim, shadow, alpha * 0.9, glow, 4.0)
    ctx.restore()


def draw_peg(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    del time
    if alpha <= 0.02:
        return
    cx = obs.x + obs.nx * (BALL_R + obs.peg_r)
    cy = obs.y + obs.ny * (BALL_R + obs.peg_r)
    color, rim, shadow, glow = _ink(obs, hue)
    disks = [(cx, cy, obs.peg_r)]
    if obs.variant == "pair":
        disks.append((cx + obs.tx * obs.peg_r * 2.4, cy + obs.ty * obs.peg_r * 2.4, obs.peg_r * 0.78))
    for disk_x, disk_y, radius in disks:
        _paint_disk(ctx, disk_x, disk_y, radius, color, rim, shadow, alpha, glow)


def draw_pad(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    del time
    if alpha <= 0.02:
        return
    length = obs.size
    thick = 28 + (10 if obs.strong else 0)
    color, rim, shadow, glow = _ink(obs, hue)
    ctx.save()
    _place(ctx, obs)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    # Shadow, body, then a contact highlight. The body centerline stays at half thickness.
    ctx.set_line_width(thick + 10)
    ctx.set_source_rgba(*shadow, alpha * 0.16)
    ctx.move_to(thick * 0.5 + 3, -length / 2)
    ctx.line_to(thick * 0.5 + 3, length / 2)
    ctx.stroke()
    ctx.set_line_width(thick)
    ctx.set_source_rgba(*color, alpha)
    ctx.move_to(thick * 0.5, -length / 2)
    ctx.line_to(thick * 0.5, length / 2)
    ctx.stroke()
    ctx.set_line_width(2.0)
    ctx.set_source_rgba(*rim, alpha * (0.45 + 0.4 * glow))
    ctx.move_to(2.2, -length * 0.42)
    ctx.line_to(2.2, length * 0.42)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, alpha * (0.35 + 0.4 * glow))
    ctx.move_to(thick * 0.42, -length * 0.16)
    ctx.line_to(thick * 0.72, 0)
    ctx.line_to(thick * 0.42, length * 0.16)
    ctx.stroke()
    ctx.restore()


def draw_breakable(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    del time
    if alpha <= 0.02:
        return
    side = obs.size
    variant = obs.variant
    color, rim, shadow, glow = _ink(obs, hue)
    body_alpha = alpha * (0.62 if obs.broken else 1.0)
    ctx.save()
    _place(ctx, obs)
    if variant == "crack":
        _paint_slab(ctx, 0, -side, 12, side * 2, color, rim, shadow, body_alpha, glow, 5.0)
        ctx.set_source_rgba(*active_palette().accent2, body_alpha * 0.8)
        ctx.set_line_width(1.4)
        ctx.move_to(6, -side * 0.7)
        ctx.line_to(8, -side * 0.2)
        ctx.line_to(4, side * 0.15)
        ctx.line_to(7, side * 0.7)
        ctx.stroke()
    elif variant == "stack":
        block_h = side * 0.42
        for slot in (-1, 0, 1):
            _paint_slab(
                ctx,
                0,
                slot * (block_h + 8) - block_h / 2,
                side * 0.9,
                block_h,
                color,
                rim,
                shadow,
                body_alpha,
                glow if slot == 0 else glow * 0.4,
                5.0,
            )
    else:
        _paint_slab(ctx, 0, -side / 2, side, side, color, rim, shadow, body_alpha, glow, 8.0)
        ctx.set_source_rgba(*active_palette().accent2, body_alpha * 0.55)
        ctx.set_line_width(1.3)
        ctx.move_to(side * 0.2, -side * 0.15)
        ctx.line_to(side * 0.72, -side * 0.28)
        ctx.move_to(side * 0.28, side * 0.05)
        ctx.line_to(side * 0.7, side * 0.22)
        ctx.stroke()
    ctx.restore()


def draw_spinner(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    if alpha <= 0.02:
        return
    length = obs.size * (0.7 if obs.variant == "short" else 1.35 if obs.variant == "long" else 1.0)
    thick = 22 if obs.variant == "short" else 14
    cx, cy = contact(obs)
    # Tangent at the beat, then spin. The bar sits on the far side of the contact.
    angle = math.atan2(obs.ty, obs.tx) + obs.spin * (time - obs.hit_time)
    color, rim, shadow, glow = _ink(obs, hue)
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(angle)
    _paint_slab(ctx, -length / 2, -thick, length, thick, color, rim, shadow, alpha, glow, min(7.0, thick * 0.45))
    _paint_disk(ctx, 0, -thick / 2, 5.5, color, rim, shadow, alpha, glow)
    ctx.restore()


def draw_piano(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    if alpha <= 0.02:
        return
    if obs.variant == "single":
        indices = (0,)
    elif obs.variant == "chord":
        indices = (-1, 0, 1)
    else:
        indices = tuple(range(-3, 4))
    key_w = max(18.0, obs.size if obs.variant != "single" else obs.size * 1.35)
    key_h = key_w * (3.1 if obs.variant == "single" else 2.6)
    gap = 3.0
    age = time - obs.hit_time if obs.lit else -1.0
    color, rim, shadow, glow = _ink(obs, hue)
    ctx.save()
    _place(ctx, obs)
    for order, index in enumerate(indices):
        local_y = index * (key_w + gap)
        black = obs.variant == "pattern" and index % 2 != 0
        stagger = order * 0.06 if obs.flourish else 0.0
        key_lit = obs.lit and age + 1e-6 >= stagger
        height = key_h * (0.62 if black else 1.0)
        if black:
            key_color = (0.12, 0.11, 0.13)
            key_rim = (0.35, 0.33, 0.36)
        elif key_lit and (index == 0 or obs.flourish):
            key_color = _mix(color, active_palette().accent2, 0.35 + 0.4 * glow)
            key_rim = rim
        else:
            key_color = color
            key_rim = rim
        _paint_slab(
            ctx,
            0,
            local_y - key_w / 2,
            height,
            key_w - 1.2,
            key_color,
            key_rim,
            shadow,
            alpha,
            glow if key_lit else 0.0,
            min(4.0, key_w * 0.22),
        )
    ctx.restore()


register_object(ObjectKind("wall", ("calm", "mid", "intense"), ("shockwave", "sparks", "glow", "squash", "morph", "shake"), draw_wall))
register_object(ObjectKind("platform", ("calm", "mid"), ("shockwave", "sparks", "glow", "squash", "morph"), draw_platform))
register_object(ObjectKind("peg", ("calm", "mid", "intense"), ("shockwave", "sparks", "glow", "squash", "morph", "shake"), draw_peg))
register_object(ObjectKind("pad", ("mid", "intense"), ("shockwave", "sparks", "glow", "squash", "morph", "chromatic", "shake"), draw_pad))
register_object(ObjectKind("breakable", ("intense", "mid"), ("shockwave", "sparks", "shatter", "glow", "squash", "morph", "chromatic", "shake"), draw_breakable))
register_object(ObjectKind("spinner", ("mid", "intense"), ("shockwave", "sparks", "glow", "squash", "morph", "shake"), draw_spinner))
register_object(ObjectKind("piano", ("calm", "mid"), ("glow", "sparks", "shockwave", "squash", "morph"), draw_piano))


@register_effect("shockwave")
def fx_shockwave(scene, obs: Obstacle, intensity: float, rng: list[int]) -> None:
    del rng
    scene.add_ring(obs.x, obs.y, 20 + 70 * intensity, obs.kind_hue, 4 + 4 * intensity)


@register_effect("sparks")
def fx_sparks(scene, obs: Obstacle, intensity: float, rng: list[int]) -> None:
    from music2video.scene import xorshift32

    count = int(10 + 28 * intensity)
    for _ in range(count):
        ang = xorshift32(rng) * math.tau
        speed = 80 + xorshift32(rng) * (260 + 280 * intensity)
        scene.add_spark(
            obs.x,
            obs.y,
            math.cos(ang) * speed + obs.nx * 40,
            math.sin(ang) * speed + obs.ny * 40,
            2.2 + xorshift32(rng) * 3.2,
            obs.kind_hue + xorshift32(rng) * 30,
            0.28 + 0.22 * intensity,
        )


@register_effect("shatter")
def fx_shatter(scene, obs: Obstacle, intensity: float, rng: list[int]) -> None:
    from music2video.scene import xorshift32

    obs.broken = True
    for _ in range(int(8 + 10 * intensity)):
        ang = xorshift32(rng) * math.tau
        speed = 60 + xorshift32(rng) * 280 * intensity
        scene.add_fragment(
            obs.x + obs.nx * (BALL_R + 10),
            obs.y + obs.ny * (BALL_R + 10),
            math.cos(ang) * speed + obs.nx * 120,
            math.sin(ang) * speed + obs.ny * 80,
            4 + xorshift32(rng) * 7,
            obs.kind_hue,
            0.45 + 0.25 * intensity,
            xorshift32(rng) * math.tau,
        )


@register_effect("glow")
def fx_glow(scene, obs: Obstacle, intensity: float, rng: list[int]) -> None:
    del rng
    scene.add_ring(obs.x, obs.y, 22 + 48 * intensity, obs.kind_hue, 6 + 16 * intensity)


@register_effect("chromatic")
def fx_chromatic(scene, obs: Obstacle, intensity: float, rng: list[int]) -> None:
    del obs, rng
    scene.chroma = max(scene.chroma, 0.55 + 0.45 * intensity)


@register_effect("squash")
def fx_squash(scene, obs: Obstacle, intensity: float, rng: list[int]) -> None:
    del rng
    scene.squash = max(scene.squash, 0.45 + 0.55 * intensity)
    scene.squash_nx = obs.nx
    scene.squash_ny = obs.ny


@register_effect("morph")
def fx_morph(scene, obs: Obstacle, intensity: float, rng: list[int]) -> None:
    del rng
    scene.morph = max(scene.morph, (0.55 + 0.45 * intensity) if obs.strong or obs.kind in {"pad", "breakable", "peg"} else 0.35)
    scene.hue_jump = max(scene.hue_jump, 28 + 40 * intensity)


@register_effect("shake")
def fx_shake(scene, obs: Obstacle, intensity: float, rng: list[int]) -> None:
    del rng
    if obs.strong:
        scene.shake = max(scene.shake, 7 + 16 * intensity)
