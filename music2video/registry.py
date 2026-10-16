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

from music2video.scene import hsl_to_rgb

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


def _place(ctx, obs: Obstacle) -> None:
    cx, cy = contact(obs)
    ctx.translate(cx, cy)
    ctx.rotate(math.atan2(obs.ny, obs.nx))


def _color(hue: float, sat: float, light: float, flash: float) -> tuple[float, float, float]:
    return hsl_to_rgb(hue, sat, min(0.94, light + 0.28 * flash))


def _body(obs: Obstacle, hue: float, lit: bool) -> tuple[float, float, float]:
    light = obs.light if lit else obs.light * 0.72
    return _color(hue, obs.sat, light, obs.flash)


def draw_wall(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    del time
    if alpha <= 0.02:
        return
    length = obs.size
    variant = obs.variant
    thick = 14 if variant == "short" else 18
    ctx.save()
    _place(ctx, obs)
    red, green, blue = _body(obs, hue, obs.lit)
    ctx.set_source_rgba(red, green, blue, alpha)
    ctx.rectangle(0, -length / 2, thick, length)
    ctx.fill()
    if variant == "double":
        ctx.rectangle(thick + 16, -length * 0.42, thick * 0.75, length * 0.84)
        ctx.fill()
    if variant == "corner":
        arm = max(54.0, length * 0.42)
        ctx.rectangle(0, length / 2 - thick, arm, thick)
        ctx.fill()
    ctx.set_source_rgba(1, 1, 1, alpha * (0.85 if obs.lit else 0.25))
    ctx.set_line_width(3)
    ctx.move_to(0, -length / 2)
    ctx.line_to(0, length / 2)
    ctx.stroke()
    if variant == "double":
        ctx.move_to(thick + 16, -length * 0.42)
        ctx.line_to(thick + 16, length * 0.42)
        ctx.stroke()
    if variant == "corner":
        ctx.move_to(0, length / 2 - thick / 2)
        ctx.line_to(max(54.0, length * 0.42), length / 2 - thick / 2)
        ctx.stroke()
    ctx.restore()


def draw_platform(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    del time
    if alpha <= 0.02:
        return
    length = obs.size
    thick = 16 if obs.variant == "short" else 22
    ctx.save()
    _place(ctx, obs)
    red, green, blue = _body(obs, hue, obs.lit)
    ctx.set_source_rgba(red, green, blue, alpha)
    if obs.variant == "arch":
        # Crown sits on the contact point. The body bows back into the platform.
        ctx.set_line_width(thick)
        ctx.set_line_cap(1)
        ctx.move_to(length * 0.18, -length / 2)
        ctx.curve_to(-length * 0.06, -length * 0.18, -length * 0.06, length * 0.18, length * 0.18, length / 2)
        ctx.stroke()
    else:
        ctx.rectangle(0, -length / 2, thick, length)
        ctx.fill()
        ctx.set_source_rgba(1, 1, 1, alpha * (0.9 if obs.lit else 0.35))
        ctx.rectangle(0, -length / 2, 4, length)
        ctx.fill()
        ctx.set_source_rgba(red, green, blue, alpha * 0.85)
        for foot in (-length * 0.32, length * 0.32):
            ctx.rectangle(thick * 0.35, foot - 5, thick * 1.3, 10)
            ctx.fill()
    ctx.restore()


def draw_peg(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    del time
    if alpha <= 0.02:
        return
    cx = obs.x + obs.nx * (BALL_R + obs.peg_r)
    cy = obs.y + obs.ny * (BALL_R + obs.peg_r)
    red, green, blue = _body(obs, hue, obs.lit)
    ctx.set_source_rgba(red, green, blue, alpha)
    disks = [(cx, cy, obs.peg_r)]
    if obs.variant == "pair":
        disks.append((cx + obs.tx * obs.peg_r * 2.4, cy + obs.ty * obs.peg_r * 2.4, obs.peg_r * 0.78))
    for disk_x, disk_y, radius in disks:
        ctx.set_source_rgba(red, green, blue, alpha)
        ctx.arc(disk_x, disk_y, radius, 0, math.tau)
        ctx.fill()
        ctx.set_source_rgba(0.05, 0.06, 0.09, alpha)
        ctx.arc(disk_x, disk_y, radius * 0.38, 0, math.tau)
        ctx.fill()
        if obs.lit:
            ctx.set_source_rgba(1, 1, 1, alpha * 0.8 * max(obs.flash, 0.25))
            ctx.set_line_width(3)
            ctx.arc(disk_x, disk_y, radius + 3, 0, math.tau)
            ctx.stroke()


def draw_pad(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    del time
    if alpha <= 0.02:
        return
    length = obs.size
    thick = 28 + (10 if obs.strong else 0)
    ctx.save()
    _place(ctx, obs)
    red, green, blue = _body(obs, hue, obs.lit)
    ctx.set_source_rgba(red, green, blue, alpha)
    ctx.set_line_width(thick)
    ctx.set_line_cap(1)
    ctx.move_to(thick * 0.45, -length / 2)
    ctx.line_to(thick * 0.45, length / 2)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, alpha * 0.7)
    ctx.set_line_width(4)
    ctx.move_to(6, -length * 0.22)
    ctx.line_to(thick * 0.55, 0)
    ctx.line_to(6, length * 0.22)
    ctx.stroke()
    ctx.restore()


def draw_breakable(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    del time
    if alpha <= 0.02:
        return
    side = obs.size
    variant = obs.variant
    ctx.save()
    _place(ctx, obs)
    red, green, blue = _body(obs, hue, obs.lit or obs.broken)
    if obs.broken:
        ctx.set_source_rgba(red, green, blue, alpha * 0.35)
        ctx.arc(side * 0.35, 0, side * (0.55 if obs.flourish else 0.4), 0, math.tau)
        ctx.fill()
    elif variant == "crack":
        ctx.set_source_rgba(red, green, blue, alpha)
        ctx.rectangle(0, -side, 12, side * 2)
        ctx.fill()
        ctx.set_source_rgba(0.15, 0.05, 0.05, alpha * 0.85)
        ctx.set_line_width(2)
        ctx.move_to(6, -side)
        ctx.line_to(9, -side * 0.35)
        ctx.line_to(3, side * 0.05)
        ctx.line_to(8, side)
        ctx.stroke()
    elif variant == "stack":
        ctx.set_source_rgba(red, green, blue, alpha)
        block_h = side * 0.42
        for slot in (-1, 0, 1):
            ctx.rectangle(0, slot * (block_h + 8) - block_h / 2, side * 0.9, block_h)
            ctx.fill()
        if obs.lit:
            ctx.set_source_rgba(1, 1, 1, alpha * 0.35)
            ctx.rectangle(0, -block_h / 2, side * 0.9, block_h)
            ctx.fill()
    else:
        ctx.set_source_rgba(red, green, blue, alpha)
        ctx.rectangle(0, -side / 2, side, side)
        ctx.fill()
        ctx.set_source_rgba(1, 1, 1, alpha * 0.35)
        ctx.set_line_width(2)
        ctx.move_to(0, 0)
        ctx.line_to(side, -side / 2)
        ctx.move_to(side * 0.3, -side / 2)
        ctx.line_to(side * 0.7, side / 2)
        ctx.stroke()
    ctx.restore()


def draw_spinner(ctx, obs: Obstacle, time: float, alpha: float, hue: float) -> None:
    if alpha <= 0.02:
        return
    length = obs.size * (0.7 if obs.variant == "short" else 1.35 if obs.variant == "long" else 1.0)
    thick = 22 if obs.variant == "short" else 14
    cx, cy = contact(obs)
    # Tangent at the beat, then spin. Local +y points back at the ball, so the
    # bar sits on the far side of the contact point.
    angle = math.atan2(obs.ty, obs.tx) + obs.spin * (time - obs.hit_time)
    ctx.save()
    ctx.translate(cx, cy)
    ctx.rotate(angle)
    red, green, blue = _body(obs, hue, obs.lit)
    ctx.set_source_rgba(red, green, blue, alpha)
    ctx.rectangle(-length / 2, -thick, length, thick)
    ctx.fill()
    ctx.set_source_rgba(1, 1, 1, alpha * 0.85)
    ctx.arc(0, 0, 6, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(red, green, blue, alpha)
    ctx.arc(-length / 2, -thick / 2, thick / 2, 0, math.tau)
    ctx.arc(length / 2, -thick / 2, thick / 2, 0, math.tau)
    ctx.fill()
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
    ctx.save()
    _place(ctx, obs)
    for order, index in enumerate(indices):
        local_y = index * (key_w + gap)
        black = obs.variant == "pattern" and index % 2 != 0
        stagger = order * 0.06 if obs.flourish else 0.0
        key_lit = obs.lit and age + 1e-6 >= stagger
        height = key_h * (0.62 if black else 1.0)
        if key_lit and (index == 0 or obs.flourish):
            red, green, blue = _body(obs, hue, True)
        elif black:
            red, green, blue = (0.16, 0.16, 0.18)
        else:
            light = 0.92 if index != 0 else 0.74
            red, green, blue = (light, light, light * 0.96)
        ctx.set_source_rgba(red, green, blue, alpha)
        ctx.rectangle(0, local_y - key_w / 2, height, key_w)
        ctx.fill()
        ctx.set_source_rgba(0.08, 0.08, 0.1, alpha * 0.7)
        ctx.set_line_width(2)
        ctx.rectangle(0, local_y - key_w / 2, height, key_w)
        ctx.stroke()
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
