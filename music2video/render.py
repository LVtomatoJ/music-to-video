"""Draw a scene with cairo. Returns the BGRA frame ffmpeg expects."""

from __future__ import annotations

import math

import cairo

from music2video.scene import hsl_to_rgb, Scene


class CairoRenderer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.ctx = cairo.Context(self.surface)

    def draw(self, scene: Scene) -> memoryview:
        ctx = self.ctx
        lift = scene.bg_flash * 36.0
        ctx.set_source_rgb((14.0 + lift) / 255.0, (16.0 + lift) / 255.0, (24.0 + lift) / 255.0)
        ctx.rectangle(0, 0, self.width, self.height)
        ctx.fill()
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
        self.surface.flush()
        return self.surface.get_data()
