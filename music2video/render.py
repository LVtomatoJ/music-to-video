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
        bg = _bg(scene.theme_hue, scene.sample_rms(scene.time), scene.bg_flash)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_rgb(*bg)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        self._parallax(scene, shake_x, shake_y, bg)
        self._spectrum(scene)
        ctx.save()
        ctx.translate(self.width / 2 + shake_x, self.height / 2 + shake_y)
        ctx.translate(-scene.cam_x, -scene.cam_y)
        self._world(scene)
        ctx.restore()
        if scene.bg_flash > 0.04:
            ctx.set_source_rgba(1, 1, 1, 0.16 * scene.bg_flash)
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
        hue = scene.theme_hue
        red, green, blue = hsl_to_rgb(hue, 0.45, 0.55)
        ctx.save()
        ctx.translate(self.width / 2 + shake_x * 0.4, self.height / 2 + shake_y * 0.4)
        ctx.translate(-scene.cam_x * 0.28, -scene.cam_y * 0.28)
        for sx, sy in scene.stars:
            ctx.set_source_rgba(red, green, blue, 0.28)
            ctx.arc(sx, sy, 2.2, 0, math.tau)
            ctx.fill()
        ctx.restore()
        # Slow grid, still in screen space so it reads as depth behind the map.
        ctx.save()
        ctx.translate(self.width / 2 + shake_x * 0.7, self.height / 2 + shake_y * 0.7)
        ctx.translate(-scene.cam_x * 0.55, -scene.cam_y * 0.55)
        step = 280.0
        left = scene.cam_x * 0.55 - self.width
        right = scene.cam_x * 0.55 + self.width
        top = scene.cam_y * 0.55 - self.height
        bottom = scene.cam_y * 0.55 + self.height
        ctx.set_source_rgba(1, 1, 1, 0.04)
        ctx.set_line_width(1)
        x = math.floor(left / step) * step
        while x < right:
            ctx.move_to(x, top)
            ctx.line_to(x, bottom)
            x += step
        y = math.floor(top / step) * step
        while y < bottom:
            ctx.move_to(left, y)
            ctx.line_to(right, y)
            y += step
        ctx.stroke()
        ctx.restore()
        pulse = scene.sample_rms(scene.time)
        grad = cairo.RadialGradient(
            self.width / 2 + shake_x,
            self.height / 2 + shake_y,
            30,
            self.width / 2,
            self.height / 2,
            max(self.width, self.height) * 0.72,
        )
        lift = hsl_to_rgb(hue, 0.5, 0.16 + 0.1 * pulse)
        grad.add_color_stop_rgba(0, lift[0], lift[1], lift[2], 0.55)
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
        if len(scene.trail) >= 2:
            for index, (x, y) in enumerate(scene.trail):
                k = (index + 1) / len(scene.trail)
                red, green, blue = hsl_to_rgb(scene.theme_hue, 0.7, 0.6)
                ctx.set_source_rgba(red, green, blue, 0.05 + 0.28 * k)
                ctx.arc(x, y, BALL_R * (0.25 + 0.55 * k), 0, math.tau)
                ctx.fill()
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
        self._ball(scene)

    def _ball(self, scene) -> None:
        ctx = self.ctx
        x, y = scene.pos
        hue = (scene.theme_hue + scene.hue_jump) % 360
        nx, ny = scene.squash_nx, scene.squash_ny
        angle = math.atan2(ny, nx)
        squash = scene.squash
        sx = max(0.4, 1.0 - 0.55 * squash)
        sy = 1.0 + 0.5 * squash
        morph = max(0.0, min(1.0, scene.morph))
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(angle)
        # Squash pulls the leading face off the surface; slide the center
        # forward so that face still meets the object on the beat.
        ctx.translate(BALL_R * (1.0 - sx), 0)
        ctx.scale(sx, sy)
        if scene.chroma > 0.12:
            for shift, color in ((-16, (1, 0.25, 0.3)), (16, (0.25, 0.55, 1))):
                ctx.set_source_rgba(color[0], color[1], color[2], 0.45 * scene.chroma)
                ctx.arc(shift, 0, BALL_R, 0, math.tau)
                ctx.fill()
        red, green, blue = hsl_to_rgb(hue, 0.78, 0.58)
        ctx.set_source_rgba(red, green, blue, 1)
        if morph < 0.85:
            ctx.arc(0, 0, BALL_R, 0, math.tau)
            ctx.fill()
        if morph > 0.12:
            ctx.set_source_rgba(red, green, blue, min(1.0, morph + 0.15))
            ctx.new_path()
            ctx.move_to(0, -BALL_R * 1.05)
            ctx.line_to(BALL_R * 0.95, 0)
            ctx.line_to(0, BALL_R * 1.05)
            ctx.line_to(-BALL_R * 0.95, 0)
            ctx.close_path()
            ctx.fill()
        ctx.set_source_rgb(1, 1, 1)
        ctx.set_line_width(3)
        if morph > 0.55:
            ctx.move_to(0, -BALL_R * 1.05)
            ctx.line_to(BALL_R * 0.95, 0)
            ctx.line_to(0, BALL_R * 1.05)
            ctx.line_to(-BALL_R * 0.95, 0)
            ctx.close_path()
        else:
            ctx.arc(0, 0, BALL_R, 0, math.tau)
        ctx.stroke()
        ctx.restore()


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
