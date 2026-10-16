"""Scene models. `balls` is the original multi-ball field; `map` is the default journey."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


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


@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    hue: float
    flash: float = 0.0


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    hue: float
    life: float
    max_life: float


class Effect(Protocol):
    def on_beat(self, scene: Scene, strong: bool) -> None:
        """React to one beat that falls inside the current frame."""


@dataclass
class BounceEffect:
    """Reverse and boost every ball, so beats read as a bounce."""

    def on_beat(self, scene: Scene, strong: bool) -> None:
        del strong
        for ball in scene.balls:
            ball.vx = -ball.vx
            ball.vy = -ball.vy
            speed = math.hypot(ball.vx, ball.vy)
            target = min(640.0, max(220.0, speed * 1.12))
            if speed > 1e-6:
                ball.vx *= target / speed
                ball.vy *= target / speed


@dataclass
class ColorEffect:
    """Shift hue and arm the white ring flash."""

    def on_beat(self, scene: Scene, strong: bool) -> None:
        del strong
        for ball in scene.balls:
            ball.hue = (ball.hue + 47.0) % 360.0
            ball.flash = 1.0


@dataclass
class ParticleBurstEffect:
    """Emit a burst from each ball on strong beats only."""

    count: int = 28

    def on_beat(self, scene: Scene, strong: bool) -> None:
        if not strong:
            return
        for ball in scene.balls:
            for _ in range(self.count):
                angle = xorshift32(scene.rng) * math.tau
                speed = 220.0 + xorshift32(scene.rng) * 520.0
                radius = 6.0 + xorshift32(scene.rng) * 8.0
                scene.particles.append(
                    Particle(
                        x=ball.x,
                        y=ball.y,
                        vx=math.cos(angle) * speed,
                        vy=math.sin(angle) * speed - 30.0,
                        radius=radius,
                        hue=ball.hue,
                        life=0.45,
                        max_life=0.45,
                    )
                )


@dataclass
class Scene:
    width: float
    height: float
    balls: list[Ball]
    particles: list[Particle] = field(default_factory=list)
    rng: list[int] = field(default_factory=lambda: [1])
    bg_flash: float = 0.0
    effects: list[Effect] = field(default_factory=list)

    @classmethod
    def create(cls, width: int, height: int, seed: int = 42, ball_count: int = 6) -> Scene:
        rng = [seed & 0xFFFFFFFF or 1]
        balls: list[Ball] = []
        for _ in range(ball_count):
            radius = 22.0 + xorshift32(rng) * 18.0
            balls.append(
                Ball(
                    x=radius + xorshift32(rng) * (width - 2.0 * radius),
                    y=radius + xorshift32(rng) * (height - 2.0 * radius),
                    vx=0.0,
                    vy=0.0,
                    radius=radius,
                    hue=0.0,
                )
            )
            angle = xorshift32(rng) * math.tau
            speed = 180.0 + xorshift32(rng) * 220.0
            balls[-1].vx = math.cos(angle) * speed
            balls[-1].vy = math.sin(angle) * speed
            balls[-1].hue = xorshift32(rng) * 360.0
        return cls(
            width=float(width),
            height=float(height),
            balls=balls,
            rng=rng,
            effects=[BounceEffect(), ColorEffect(), ParticleBurstEffect()],
        )

    def update(self, dt: float, time: float, events: list[tuple[float, bool]]) -> None:
        del time
        beat = bool(events)
        strong = any(flag for _, flag in events)
        if beat:
            for effect in self.effects:
                effect.on_beat(self, strong)
        self.bg_flash = max(self.bg_flash * 0.86, 1.0 if beat else 0.0)
        self._integrate(dt)

    def _integrate(self, dt: float) -> None:
        for ball in self.balls:
            ball.x += ball.vx * dt
            ball.y += ball.vy * dt
            if ball.x < ball.radius:
                ball.x = ball.radius
                ball.vx = abs(ball.vx)
            elif ball.x > self.width - ball.radius:
                ball.x = self.width - ball.radius
                ball.vx = -abs(ball.vx)
            if ball.y < ball.radius:
                ball.y = ball.radius
                ball.vy = abs(ball.vy)
            elif ball.y > self.height - ball.radius:
                ball.y = self.height - ball.radius
                ball.vy = -abs(ball.vy)
            ball.flash *= 0.86
        alive: list[Particle] = []
        for particle in self.particles:
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.vy += 280.0 * dt
            particle.life -= dt
            if particle.life > 0:
                alive.append(particle)
        self.particles = alive


def build_scene(kind: str, width: int, height: int, analysis, seed: int, theme: str = "neon"):
    if kind == "balls":
        return Scene.create(width, height, seed=seed)
    if kind == "map":
        from music2video.mapscene import MapScene

        return MapScene.create(width, height, analysis, seed, theme)
    raise SystemExit(f"unknown scene {kind}")
