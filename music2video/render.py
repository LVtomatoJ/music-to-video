"""Draw a scene with cairo. Returns the BGRA frame ffmpeg expects."""

from __future__ import annotations

import math

import cairo

from music2video.registry import BALL_R, OBJECTS
from music2video.scene import hsl_to_rgb, Scene


class CairoRenderer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.ctx = cairo.Context(self.surface)

    def draw(self, scene) -> bytes:
        if hasattr(scene, "obstacles"):
            self._draw_map(scene)
        else:
            self._draw_balls(scene)
        # Both scenes paint an opaque background first, so the premultiplied
        # cairo buffer matches the straight BGRA ffmpeg reads.
        self.surface.flush()
        return bytes(self.surface.get_data())

    def _draw_balls(self, scene: Scene) -> None:
        ctx = self.ctx
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        lift = scene.bg_flash * 36.0
        ctx.set_source_rgb((14.0 + lift) / 255.0, (16.0 + lift) / 255.0, (24.0 + lift) / 255.0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        for particle in scene.particles:
            k = max(0.0, particle.life / particle.max_life)
            red, green, blue = hsl_to_rgb(particle.hue, 0.80, 0.35 + 0.35 * k)
            ctx.set_source_rgb(red, green, blue)
            ctx.arc(particle.x, particle.y, particle.radius * (0.35 + 0.65 * k), 0, math.tau)
            ctx.fill()
        for ball in scene.balls:
            red, green, blue = hsl_to_rgb(ball.hue, 0.75, 0.55)
            ctx.set_source_rgb(red, green, blue)
            ctx.arc(ball.x, ball.y, ball.radius, 0, math.tau)
            ctx.fill()
            if ball.flash > 0.4:
                ctx.set_source_rgb(1, 1, 1)
            else:
                red, green, blue = hsl_to_rgb(ball.hue, 0.80, 0.75)
                ctx.set_source_rgb(red, green, blue)
            ctx.set_line_width(3.0)
            ctx.arc(ball.x, ball.y, ball.radius, 0, math.tau)
            ctx.stroke()

    def _draw_map(self, scene) -> None:
        ctx = self.ctx
        shake_x, shake_y = scene.shake_offset()
        bg = _bg_for(scene)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_rgb(*bg)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        self._parallax(scene, shake_x, shake_y, bg)
        if scene.skin is None or scene.skin.show_spectrum:
            self._spectrum(scene)
        ctx.save()
        ctx.translate(self.width / 2 + shake_x, self.height / 2 + shake_y)
        ctx.translate(-scene.cam_x, -scene.cam_y)
        self._world(scene)
        ctx.restore()
        flash = scene.skin.flash if scene.skin is not None else 0.16
        if scene.bg_flash > 0.04 and flash > 0:
            ctx.set_source_rgba(1, 1, 1, flash * min(1.0, scene.bg_flash))
            ctx.paint()
        if scene.chroma > 0.08:
            # A thin screen-space fringe, stronger on heavy hits.
            ctx.set_source_rgba(1, 0.2, 0.35, 0.18 * scene.chroma)
            ctx.rectangle(0, 0, 10 + 18 * scene.chroma, self.height)
            ctx.fill()
            ctx.set_source_rgba(0.2, 0.55, 1, 0.18 * scene.chroma)
            ctx.rectangle(self.width - (10 + 18 * scene.chroma), 0, 10 + 18 * scene.chroma, self.height)
            ctx.fill()

    def _parallax(self, scene, shake_x: float, shake_y: float, bg: tuple[float, float, float]) -> None:
        ctx = self.ctx
        skin = scene.skin
        style = skin.background if skin is not None else "neon"
        hue = scene.theme_hue
        if style in {"arcade", "pastel"}:
            red, green, blue = hsl_to_rgb(hue, 0.35 if style == "pastel" else 0.7, 0.7)
            ctx.save()
            ctx.translate(self.width / 2 + shake_x * 0.4, self.height / 2 + shake_y * 0.4)
            ctx.translate(-scene.cam_x * 0.28, -scene.cam_y * 0.28)
            radius = 7.0 if style == "pastel" else 3.2
            alpha = 0.16 if style == "pastel" else 0.45
            for index, (sx, sy) in enumerate(scene.stars):
                if style == "arcade":
                    red, green, blue = hsl_to_rgb(hue + (index % 5) * 36, 0.75, 0.62)
                ctx.set_source_rgba(red, green, blue, alpha)
                ctx.arc(sx, sy, radius, 0, math.tau)
                ctx.fill()
            ctx.restore()
        if style != "shatter":
            ctx.save()
            ctx.translate(self.width / 2 + shake_x * 0.7, self.height / 2 + shake_y * 0.7)
            ctx.translate(-scene.cam_x * 0.55, -scene.cam_y * 0.55)
            step = 140.0 if style == "piano" else 220.0
            left = scene.cam_x * 0.55 - self.width
            right = scene.cam_x * 0.55 + self.width
            top = scene.cam_y * 0.55 - self.height
            bottom = scene.cam_y * 0.55 + self.height
            line = 0.16 if style == "neon" else 0.05
            ctx.set_source_rgba(*hsl_to_rgb(hue, 0.6, 0.7), line)
            ctx.set_line_width(1.4 if style == "neon" else 1)
            if style == "piano":
                y = math.floor(top / step) * step
                while y < bottom:
                    ctx.move_to(left, y)
                    ctx.line_to(right, y)
                    y += step
            else:
                x = math.floor(left / step) * step
                while x < right:
                    ctx.move_to(x, top)
                    ctx.line_to(x, bottom)
                    x += step
                if style != "pastel":
                    y = math.floor(top / step) * step
                    while y < bottom:
                        ctx.move_to(left, y)
                        ctx.line_to(right, y)
                        y += step
            ctx.stroke()
            ctx.restore()
        pulse = scene.sample_rms(scene.time)
        spread = 0.22 if style == "pastel" else 0.5
        grad = cairo.RadialGradient(
            self.width / 2 + shake_x,
            self.height / 2 + shake_y,
            30,
            self.width / 2,
            self.height / 2,
            max(self.width, self.height) * 0.72,
        )
        lift = hsl_to_rgb(hue, skin.bg_sat if skin is not None else 0.5, 0.18 + 0.08 * pulse)
        grad.add_color_stop_rgba(0, lift[0], lift[1], lift[2], spread)
        grad.add_color_stop_rgba(1, bg[0], bg[1], bg[2], 0.0)
        ctx.set_source(grad)
        ctx.paint()

    def _spectrum(self, scene) -> None:
        bands = scene.sample_bands(scene.time)
        if not bands:
            return
        ctx = self.ctx
        count = len(bands)
        gap = 10
        bar_w = min(36, (self.width * 0.46 - gap * (count - 1)) / count)
        total = count * bar_w + (count - 1) * gap
        x0 = (self.width - total) / 2
        base = self.height - 28
        rms = scene.sample_rms(scene.time)
        for index, level in enumerate(bands):
            height = 14 + float(level) * (110 + 90 * rms)
            red, green, blue = hsl_to_rgb(scene.theme_hue + index * 12, 0.7, 0.58)
            ctx.set_source_rgba(red, green, blue, 0.35 + 0.45 * float(level))
            ctx.rectangle(x0 + index * (bar_w + gap), base - height, bar_w, height)
            ctx.fill()

    def _world(self, scene) -> None:
        ctx = self.ctx
        time = scene.time
        for obs in scene.obstacles:
            alpha = _obstacle_alpha(obs, time)
            if alpha <= 0.02:
                continue
            hue = (scene.theme_hue + obs.kind_hue) % 360
            OBJECTS[obs.kind].draw(ctx, obs, time, alpha, hue)
        _draw_trail(ctx, scene)
        for ring in scene.rings:
            k = max(0.0, ring.life / ring.max_life)
            red, green, blue = hsl_to_rgb((ring.hue + scene.theme_hue) % 360, 0.7, 0.62)
            ctx.set_source_rgba(red, green, blue, 0.55 * k)
            ctx.set_line_width(ring.width * k + 1)
            ctx.arc(ring.x, ring.y, ring.radius, 0, math.tau)
            ctx.stroke()
        for spark in scene.sparks:
            k = max(0.0, spark.life / spark.max_life)
            red, green, blue = hsl_to_rgb((spark.hue + scene.theme_hue) % 360, 0.8, 0.45 + 0.3 * k)
            ctx.set_source_rgba(red, green, blue, 0.3 + 0.7 * k)
            if spark.shape == "fragment":
                ctx.save()
                ctx.translate(spark.x, spark.y)
                ctx.rotate(spark.spin)
                side = spark.radius * (0.4 + 0.6 * k)
                ctx.rectangle(-side, -side * 0.45, side * 2, side * 0.9)
                ctx.fill()
                ctx.restore()
            else:
                ctx.arc(spark.x, spark.y, spark.radius * (0.4 + 0.6 * k), 0, math.tau)
                ctx.fill()
        self._projectile(scene)

    def _projectile(self, scene) -> None:
        ctx = self.ctx
        feel = scene.skin.feel if scene.skin is not None else None
        squash_gain = 0.55
        stretch = 0.5
        if feel is not None:
            squash_gain = 0.35 + 0.55 * feel.squash
            stretch = feel.stretch
        squash = scene.squash
        sx = max(0.55, min(1.25, 1.0 - squash_gain * squash))
        sy = max(0.7, min(1.45, 1.0 + stretch * squash * 0.45))
        vx, vy = scene.velocity_at(scene.time)
        # Face the segment tangent the whole way between hits. The squash
        # spring used to swap this for the wall normal and shove the sprite
        # off the line every time it crossed a threshold.
        angle = math.atan2(vy, vx)
        kind = scene.skin.projectile if scene.skin is not None else "bubble"
        ctx.save()
        ctx.translate(*scene.pos)
        ctx.rotate(angle)
        ctx.scale(sx, sy)
        _PROJECTILES[kind](ctx, scene)
        ctx.restore()


def _bg_for(scene) -> tuple[float, float, float]:
    rms = scene.sample_rms(scene.time)
    skin = scene.skin
    if skin is None:
        return _bg(scene.theme_hue, rms, scene.bg_flash)
    red, green, blue = hsl_to_rgb(skin.hue, skin.bg_sat, skin.bg_light + 0.045 * rms)
    lift = skin.flash * min(1.0, scene.bg_flash)
    return min(1.0, red + lift), min(1.0, green + lift), min(1.0, blue + lift)


def _draw_trail(ctx, scene) -> None:
    feel = scene.skin.feel if scene.skin is not None else None
    style = feel.trail if feel is not None else "ribbon"
    if style == "none" or len(scene.trail) < 2:
        return
    hue = scene.theme_hue
    red, green, blue = hsl_to_rgb(hue, 0.55, 0.72)
    ctx.set_line_cap(1)
    ctx.set_line_join(1)
    # One fading stroke along the path. Filled copies of the flyer read as flicker.
    points = scene.trail
    for index in range(1, len(points)):
        k = index / (len(points) - 1)
        ctx.set_source_rgba(red, green, blue, 0.05 + 0.4 * k)
        ctx.set_line_width(2.0 + 3.0 * k)
        ctx.move_to(*points[index - 1])
        ctx.line_to(*points[index])
        ctx.stroke()


def _paint_bubble(ctx, scene) -> None:
    radius = BALL_R
    red, green, blue = hsl_to_rgb(scene.theme_hue, 0.25, 0.82)
    ctx.set_source_rgba(red, green, blue, 0.38)
    ctx.arc(0, 0, radius, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(1, 1, 1, 0.85)
    ctx.set_line_width(3)
    ctx.arc(0, 0, radius, 0, math.tau)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, 0.7)
    ctx.arc(-radius * 0.28, -radius * 0.32, radius * 0.22, 0, math.tau)
    ctx.fill()


def _paint_pinball(ctx, scene) -> None:
    del scene
    ctx.set_source_rgba(0.72, 0.74, 0.78, 1)
    ctx.arc(0, 0, BALL_R * 0.92, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(0.95, 0.96, 0.98, 0.95)
    ctx.arc(-6, -7, BALL_R * 0.55, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(0.25, 0.27, 0.32, 1)
    ctx.set_line_width(3)
    ctx.arc(0, 0, BALL_R * 0.92, 0, math.tau)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, 0.95)
    ctx.arc(-8, -9, 5, 0, math.tau)
    ctx.fill()


def _paint_cube(ctx, scene) -> None:
    ctx.rotate(scene.spin)
    side = BALL_R * 1.55
    red, green, blue = hsl_to_rgb(scene.theme_hue, 0.12, 0.78)
    ctx.set_source_rgba(red, green, blue, 1)
    ctx.rectangle(-side / 2, -side / 2, side, side)
    ctx.fill()
    ctx.set_source_rgba(0.15, 0.05, 0.05, 0.9)
    ctx.set_line_width(4)
    ctx.rectangle(-side / 2, -side / 2, side, side)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, 0.55)
    ctx.rectangle(-side / 2, -side / 2, side, side * 0.22)
    ctx.fill()


def _paint_arrow(ctx, scene) -> None:
    red, green, blue = hsl_to_rgb(scene.theme_hue, 0.9, 0.58)
    ctx.set_source_rgba(red, green, blue, 1)
    ctx.move_to(BALL_R, 0)
    ctx.line_to(-BALL_R * 0.15, -BALL_R * 0.55)
    ctx.line_to(-BALL_R * 0.55, 0)
    ctx.line_to(-BALL_R * 0.15, BALL_R * 0.55)
    ctx.close_path()
    ctx.fill()
    ctx.set_source_rgba(1, 1, 1, 0.9)
    ctx.set_line_width(2.5)
    ctx.move_to(BALL_R, 0)
    ctx.line_to(-BALL_R * 0.15, -BALL_R * 0.55)
    ctx.line_to(-BALL_R * 0.55, 0)
    ctx.line_to(-BALL_R * 0.15, BALL_R * 0.55)
    ctx.close_path()
    ctx.stroke()


def _paint_note(ctx, scene) -> None:
    red, green, blue = hsl_to_rgb(scene.theme_hue, 0.55, 0.42)
    ctx.set_source_rgba(red, green, blue, 1)
    ctx.save()
    ctx.translate(-4, 6)
    ctx.rotate(-0.5)
    ctx.scale(1.15, 0.82)
    ctx.arc(0, 0, BALL_R * 0.42, 0, math.tau)
    ctx.fill()
    ctx.restore()
    ctx.set_line_width(4)
    ctx.move_to(10, 2)
    ctx.line_to(10, -BALL_R * 0.95)
    ctx.stroke()
    ctx.set_line_width(3)
    ctx.move_to(10, -BALL_R * 0.95)
    ctx.curve_to(22, -BALL_R * 0.7, 24, -BALL_R * 0.2, 12, -4)
    ctx.stroke()


_PROJECTILES = {
    "bubble": _paint_bubble,
    "pinball": _paint_pinball,
    "cube": _paint_cube,
    "arrow": _paint_arrow,
    "note": _paint_note,
}


def _bg(hue: float, rms: float, flash: float) -> tuple[float, float, float]:
    red, green, blue = hsl_to_rgb(hue, 0.55, 0.09 + 0.07 * rms)
    lift = 0.08 * flash
    return min(1.0, red + lift), min(1.0, green + lift), min(1.0, blue + lift)


def _obstacle_alpha(obs, time: float) -> float:
    if obs.lit or time >= obs.hit_time:
        return 1.0
    lead = obs.hit_time - time
    if lead > 1.15:
        return 0.0
    return (1.15 - lead) / 1.15 * 0.9
