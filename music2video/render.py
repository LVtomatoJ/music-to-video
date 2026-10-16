"""Draw a scene with cairo. Returns the BGRA frame ffmpeg expects."""

from __future__ import annotations

import math

import cairo
import numpy as np

from music2video.registry import BALL_R, OBJECTS, active_palette, bind_palette
from music2video.scene import hsl_to_rgb, Scene


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur. `img` is HxWxC float32."""
    if radius < 1:
        return img
    window = radius * 2 + 1

    def _axis(src: np.ndarray, axis: int) -> np.ndarray:
        pad = [(0, 0)] * src.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(src, pad, mode="edge")
        cumulative = np.cumsum(padded, axis=axis, dtype=np.float32)
        shape = list(cumulative.shape)
        shape[axis] = 1
        cumulative = np.concatenate([np.zeros(shape, dtype=np.float32), cumulative], axis=axis)
        high = [slice(None)] * cumulative.ndim
        low = [slice(None)] * cumulative.ndim
        high[axis] = slice(window, window + src.shape[axis])
        low[axis] = slice(0, src.shape[axis])
        return (cumulative[tuple(high)] - cumulative[tuple(low)]) / window

    return _axis(_axis(img, 1), 0)


def _smooth(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _mix(a: tuple[float, float, float], b: tuple[float, float, float], t: float) -> tuple[float, float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


class CairoRenderer:
    def __init__(self, width: int, height: int, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.ctx = cairo.Context(self.surface)
        self.ctx.set_antialias(cairo.ANTIALIAS_BEST)
        # The map is drawn at 2x and filtered down so curves and rims stay smooth.
        self.hi = cairo.ImageSurface(cairo.FORMAT_ARGB32, width * 2, height * 2)
        self.hi_ctx = cairo.Context(self.hi)
        self.hi_ctx.set_antialias(cairo.ANTIALIAS_BEST)
        self.bloom_w = max(2, width // 4)
        self.bloom_h = max(2, height // 4)
        self.bloom_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.bloom_w, self.bloom_h)

    def draw(self, scene) -> bytes:
        if hasattr(scene, "obstacles"):
            bind_palette(getattr(scene, "skin", None))
            output = self.ctx
            self.ctx = self.hi_ctx
            self.ctx.identity_matrix()
            self.ctx.scale(2, 2)
            self.ctx.set_operator(cairo.OPERATOR_SOURCE)
            self._draw_map(scene)
            self.ctx = output
            self._downsample()
            self._bloom()
            self._caption(scene)
            self._vignette()
            self._grain(scene)
        else:
            self.ctx.identity_matrix()
            self._draw_balls(scene)
            self._vignette()
        # Opaque frames: the premultiplied cairo buffer matches ffmpeg's BGRA.
        self.surface.flush()
        return bytes(self.surface.get_data())

    def _downsample(self) -> None:
        ctx = self.ctx
        ctx.identity_matrix()
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        pattern = cairo.SurfacePattern(self.hi)
        pattern.set_filter(cairo.FILTER_GOOD)
        pattern.set_matrix(cairo.Matrix(xx=2, yy=2))
        ctx.set_source(pattern)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)

    def _bloom(self) -> None:
        bctx = cairo.Context(self.bloom_surface)
        bctx.set_operator(cairo.OPERATOR_SOURCE)
        bctx.scale(self.bloom_w / self.width, self.bloom_h / self.height)
        bctx.set_source_surface(self.surface, 0, 0)
        bctx.paint()
        del bctx
        self.bloom_surface.flush()
        buf = np.ndarray((self.bloom_h, self.bloom_w, 4), dtype=np.uint8, buffer=self.bloom_surface.get_data())
        rgb = np.clip(buf[:, :, :3].astype(np.float32) - 24.0, 0.0, 255.0)
        rgb = _box_blur(rgb, 5)
        rgb = _box_blur(rgb, 3)
        buf[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        buf[:, :, 3] = 255
        self.bloom_surface.mark_dirty()
        ctx = self.ctx
        ctx.save()
        ctx.identity_matrix()
        ctx.set_operator(cairo.OPERATOR_ADD)
        pattern = cairo.SurfacePattern(self.bloom_surface)
        pattern.set_filter(cairo.FILTER_BILINEAR)
        pattern.set_matrix(cairo.Matrix(xx=self.bloom_w / self.width, yy=self.bloom_h / self.height))
        ctx.set_source(pattern)
        # Light skins bloom less, or the cream field turns milky.
        lift = sum(active_palette().bg_top)
        ctx.paint_with_alpha(0.16 if lift > 1.7 else 0.30)
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.restore()

    def _vignette(self) -> None:
        pal = active_palette()
        ctx = self.ctx
        ctx.save()
        ctx.identity_matrix()
        cx, cy = self.width / 2, self.height / 2
        inner = min(self.width, self.height) * 0.28
        outer = math.hypot(self.width, self.height) * 0.58
        grad = cairo.RadialGradient(cx, cy, inner, cx, cy, outer)
        grad.add_color_stop_rgba(0, *pal.accent, 0.05)
        grad.add_color_stop_rgba(0.42, 0, 0, 0, 0)
        grad.add_color_stop_rgba(1, 0.02, 0.012, 0.028, pal.vignette)
        ctx.set_source(grad)
        ctx.paint()
        ctx.restore()

    def _grain(self, scene) -> None:
        self.surface.flush()
        buf = np.ndarray((self.height, self.width, 4), dtype=np.uint8, buffer=self.surface.get_data())
        seed = (int(getattr(scene, "time", 0.0) * 1000.0) * 1103515245 + 12345) & 0xFFFFFFFF
        rng = np.random.default_rng(seed or 1)
        # Half-resolution grain, repeated. Full-frame noise was a large share of the frame.
        small_h = max(1, self.height // 2)
        small_w = max(1, self.width // 2)
        small = rng.integers(-4, 5, size=(small_h, small_w), dtype=np.int16)
        noise = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        noise = noise[: self.height, : self.width]
        rgb = buf[:, :, :3].astype(np.int16)
        rgb += noise[:, :, None]
        np.clip(rgb, 0, 255, out=rgb)
        buf[:, :, :3] = rgb.astype(np.uint8)
        self.surface.mark_dirty()

    def _caption(self, scene) -> None:
        title = getattr(scene, "caption_title", "") or ""
        artist = getattr(scene, "caption_artist", "") or ""
        if not title and not artist:
            return
        t = getattr(scene, "time", 0.0)
        if t < 0.6:
            u = t / 0.6
        elif t < 4.4:
            u = 1.0
        elif t < 5.4:
            u = max(0.0, 1.0 - (t - 4.4))
        else:
            return
        alpha = u * u * (3.0 - 2.0 * u)
        if alpha <= 0.02:
            return
        pal = active_palette()
        ctx = self.ctx
        ctx.save()
        ctx.identity_matrix()
        scale = self.height / 1080.0
        x = 54 * scale
        y = 72 * scale
        max_w = self.width * 0.48
        if title:
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            size = 30 * scale
            ctx.set_font_size(size)
            while size > 15 * scale and ctx.text_extents(title).width > max_w:
                size -= 1
                ctx.set_font_size(size)
            ctx.set_source_rgba(0, 0, 0, 0.4 * alpha)
            ctx.move_to(x + 1.4, y + 1.6)
            ctx.show_text(title)
            ctx.set_source_rgba(*pal.rim, 0.94 * alpha)
            ctx.move_to(x, y)
            ctx.show_text(title)
            y += size * 1.05
        if artist:
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(16 * scale)
            ctx.set_source_rgba(*pal.accent, 0.82 * alpha)
            ctx.move_to(x, y)
            ctx.show_text(artist)
        ctx.restore()

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
        pal = active_palette()
        flash = max(0.0, min(1.0, scene.bg_flash))
        top = _mix(pal.bg_top, pal.accent, 0.10 * flash)
        bottom = _mix(pal.bg_bottom, pal.accent2, 0.05 * flash)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        grad = cairo.LinearGradient(0, 0, 0, self.height)
        grad.add_color_stop_rgb(0, *top)
        grad.add_color_stop_rgb(1, *bottom)
        ctx.set_source(grad)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        self._parallax(scene, shake_x, shake_y)
        if scene.skin is None or scene.skin.show_spectrum:
            self._spectrum(scene)
        ctx.save()
        zoom = getattr(scene, "cam_zoom", 1.0)
        ctx.translate(self.width / 2 + shake_x, self.height / 2 + shake_y)
        ctx.scale(zoom, zoom)
        ctx.translate(-scene.cam_x, -scene.cam_y)
        self._world(scene)
        ctx.restore()
        skin_flash = scene.skin.flash if scene.skin is not None else 0.16
        veil = min(0.14, 0.08 * flash) * skin_flash
        if veil > 0.004:
            ctx.set_source_rgba(*pal.accent, veil)
            ctx.paint()
        if scene.chroma > 0.08:
            fringe = 8 + 14 * scene.chroma
            edge = cairo.LinearGradient(0, 0, fringe, 0)
            edge.add_color_stop_rgba(0, *pal.accent2, 0.22 * scene.chroma)
            edge.add_color_stop_rgba(1, *pal.accent2, 0)
            ctx.set_source(edge)
            ctx.rectangle(0, 0, fringe, self.height)
            ctx.fill()
            edge = cairo.LinearGradient(self.width, 0, self.width - fringe, 0)
            edge.add_color_stop_rgba(0, *pal.accent, 0.22 * scene.chroma)
            edge.add_color_stop_rgba(1, *pal.accent, 0)
            ctx.set_source(edge)
            ctx.rectangle(self.width - fringe, 0, fringe, self.height)
            ctx.fill()

    def _parallax(self, scene, shake_x: float, shake_y: float) -> None:
        ctx = self.ctx
        skin = scene.skin
        style = skin.background if skin is not None else "neon"
        pal = active_palette()
        energy = max(0.0, min(1.0, getattr(scene, "section_energy", 0.4)))
        breath = 0.5 + 0.5 * math.sin(scene.time * (0.55 + 0.9 * energy))
        zoom = getattr(scene, "cam_zoom", 1.0)
        depth_zoom = 1.0 + (zoom - 1.0) * 0.35
        ctx.save()
        ctx.translate(self.width / 2 + shake_x * 0.35, self.height / 2 + shake_y * 0.35)
        ctx.scale(depth_zoom, depth_zoom)
        ctx.translate(-scene.cam_x * 0.28, -scene.cam_y * 0.28)
        window_x = self.width * 0.7
        window_y = self.height * 0.7
        drawn = 0
        for index, (sx, sy) in enumerate(scene.stars):
            if index % 5 != 0:
                continue
            px = (sx - scene.cam_x * 0.28) * depth_zoom
            py = (sy - scene.cam_y * 0.28) * depth_zoom
            if abs(px) > window_x or abs(py) > window_y:
                continue
            color = pal.accent if index % 10 == 0 else pal.accent2
            radius = 90 + (index % 4) * 36
            alpha = (0.045 if style == "pastel" else 0.07) * (0.65 + energy) * (0.8 + 0.3 * breath)
            orb = cairo.RadialGradient(sx, sy, radius * 0.1, sx, sy, radius)
            orb.add_color_stop_rgba(0, *color, alpha)
            orb.add_color_stop_rgba(1, *color, 0)
            ctx.set_source(orb)
            ctx.arc(sx, sy, radius, 0, math.tau)
            ctx.fill()
            drawn += 1
            if drawn >= 10:
                break
        ctx.restore()
        if style == "piano":
            ctx.save()
            ctx.translate(self.width / 2 + shake_x * 0.55, self.height / 2 + shake_y * 0.55)
            ctx.translate(-scene.cam_x * 0.42, -scene.cam_y * 0.42)
            step = 160.0
            left = scene.cam_x * 0.42 - self.width
            right = scene.cam_x * 0.42 + self.width
            top = scene.cam_y * 0.42 - self.height
            bottom = scene.cam_y * 0.42 + self.height
            ctx.set_source_rgba(*pal.rim, 0.04 * (0.45 + energy))
            ctx.set_line_width(1.0)
            y = math.floor(top / step) * step
            while y < bottom:
                ctx.move_to(left, y)
                ctx.line_to(right, y)
                y += step
            ctx.stroke()
            ctx.restore()
        # The center lift lives in the 1x vignette. A full-frame radial at 2x
        # dominated the frame time without changing the picture much.

    def _spectrum(self, scene) -> None:
        bands = scene.sample_bands(scene.time)
        if not bands:
            return
        ctx = self.ctx
        pal = active_palette()
        count = len(bands)
        gap = 18
        bar_w = 4.0
        total = count * bar_w + (count - 1) * gap
        x0 = (self.width - total) / 2
        base = self.height - 40
        rms = scene.sample_rms(scene.time)
        for index, level in enumerate(bands):
            height = 6 + float(level) * (34 + 22 * rms)
            color = pal.accent if index % 2 == 0 else pal.accent2
            ctx.set_source_rgba(*color, 0.08 + 0.14 * float(level))
            _capsule(ctx, x0 + index * (bar_w + gap), base - height, bar_w, height)
            ctx.fill()

    def _world(self, scene) -> None:
        ctx = self.ctx
        time = scene.time
        for obs in scene.obstacles:
            alpha = _obstacle_alpha(obs, time) * _presence(obs, scene)
            if alpha <= 0.02:
                continue
            hue = (scene.theme_hue + obs.kind_hue) % 360
            OBJECTS[obs.kind].draw(ctx, obs, time, alpha, hue)
        _draw_trail(ctx, scene, self.fps)
        _draw_split(ctx, scene)
        pal = active_palette()
        for ring in scene.rings:
            k = max(0.0, ring.life / ring.max_life)
            fade = k * k * (3.0 - 2.0 * k)
            if fade < 0.02:
                continue
            red, green, blue = hsl_to_rgb((ring.hue + scene.theme_hue) % 360, 0.65, 0.66)
            color = _mix((red, green, blue), pal.accent, 0.45)
            inner = max(1.0, ring.radius * 0.78)
            outer = ring.radius + ring.width * (0.35 + 0.65 * fade)
            band = cairo.RadialGradient(ring.x, ring.y, inner, ring.x, ring.y, outer)
            band.add_color_stop_rgba(0.0, *color, 0)
            band.add_color_stop_rgba(0.55, *color, 0.42 * fade)
            band.add_color_stop_rgba(1.0, *pal.accent2, 0)
            ctx.set_source(band)
            ctx.arc(ring.x, ring.y, outer, 0, math.tau)
            ctx.fill()
        for spark in scene.sparks:
            k = max(0.0, spark.life / spark.max_life)
            fall = k ** 1.6
            if fall < 0.02:
                continue
            red, green, blue = hsl_to_rgb((spark.hue + scene.theme_hue) % 360, 0.72, 0.62)
            color = _mix((red, green, blue), pal.rim, 0.25)
            ctx.set_source_rgba(*color, 0.85 * fall)
            if spark.shape == "fragment":
                ctx.save()
                ctx.translate(spark.x, spark.y)
                ctx.rotate(spark.spin)
                side = spark.radius * (0.28 + 0.72 * fall)
                _round_rect(ctx, -side, -side * 0.42, side * 2, side * 0.84, side * 0.28)
                ctx.fill()
                ctx.restore()
            else:
                radius = spark.radius * (0.25 + 0.75 * fall)
                glow = cairo.RadialGradient(spark.x, spark.y, 0, spark.x, spark.y, radius * 1.8)
                glow.add_color_stop_rgba(0, *color, 0.9 * fall)
                glow.add_color_stop_rgba(1, *color, 0)
                ctx.set_source(glow)
                ctx.arc(spark.x, spark.y, radius * 1.8, 0, math.tau)
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
        # Face the segment tangent the whole way between hits. Scale stays
        # centered on the contact so the sprite does not leave the path.
        kind = scene.skin.projectile if scene.skin is not None else "bubble"
        # A circle that is scaled differently on each axis becomes an oval.
        # Bubbles and pinballs stay round; the arrow, note, and block may still
        # squash along the flight, which is the shape of those skins.
        if kind in {"bubble", "pinball"}:
            puff = max(-0.25, min(1.0, squash))
            sx = sy = max(0.97, min(1.07, 1.0 + 0.06 * puff))
            round_body = True
        else:
            sx = max(0.55, min(1.25, 1.0 - squash_gain * squash))
            sy = max(0.7, min(1.45, 1.0 + stretch * squash * 0.45))
            round_body = False
        paint = _PROJECTILES[kind]
        frame = 1.0 / self.fps
        if not round_body:
            for ghost in range(4, 0, -1):
                when = scene.time - frame * (ghost / 4.0)
                gx, gy = scene.position_at(when)
                gvx, gvy = scene.velocity_at(when)
                ctx.save()
                ctx.translate(gx, gy)
                ctx.rotate(math.atan2(gvy, gvx))
                paint(ctx, scene, 0.05 + 0.035 * (4 - ghost))
                ctx.restore()
        vx, vy = scene.velocity_at(scene.time)
        pulse = max(0.0, min(1.0, squash))
        ctx.save()
        ctx.translate(*scene.pos)
        _paint_halo(ctx, scene, 1.0 + (0.85 * pulse if round_body else 0.0))
        ctx.rotate(math.atan2(vy, vx))
        ctx.scale(sx, sy)
        paint(ctx, scene, 1.0)
        ctx.restore()


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


def _capsule(ctx, x: float, y: float, w: float, h: float) -> None:
    radius = min(w, h) * 0.5
    _round_rect(ctx, x, y, w, h, radius)


def _draw_split(ctx, scene) -> None:
    """Pastel flourish: two or three bubbles leave the flyer and come back.

    The flyer itself stays on the path. The satellites only exist during the
    flourish, and their offset returns to zero so they merge instead of lingering.
    """
    if scene.split_life <= 0.0 or scene.split_count < 2 or scene.split_max <= 0.0:
        return
    progress = 1.0 - scene.split_life / scene.split_max
    spread = math.sin(math.pi * max(0.0, min(1.0, progress)))
    if spread < 0.04:
        return
    vx, vy = scene.velocity_at(scene.time)
    length = math.hypot(vx, vy) or 1.0
    px, py = -vy / length, vx / length
    pal = active_palette()
    for index in range(scene.split_count):
        slot = index - (scene.split_count - 1) / 2.0
        offset = slot * 34.0 * spread
        cx = scene.pos[0] + px * offset
        cy = scene.pos[1] + py * offset
        radius = 8 + 5 * spread
        body = cairo.RadialGradient(cx - radius * 0.3, cy - radius * 0.3, radius * 0.1, cx, cy, radius)
        body.add_color_stop_rgba(0, 1, 1, 1, 0.55 * spread)
        body.add_color_stop_rgba(0.45, *pal.ink, 0.35 * spread)
        body.add_color_stop_rgba(1, *pal.accent, 0.05)
        ctx.set_source(body)
        ctx.arc(cx, cy, radius, 0, math.tau)
        ctx.fill()


def _draw_trail(ctx, scene, fps: int) -> None:
    feel = scene.skin.feel if scene.skin is not None else None
    style = feel.trail if feel is not None else "ribbon"
    length = feel.trail_length if feel is not None else 16
    if style == "none" or length < 2:
        return
    steps = max(10, length * 2)
    span = length / max(1, fps)
    points = [scene.position_at(scene.time - span * (1.0 - index / steps)) for index in range(steps + 1)]
    # Drop samples that have not moved, so a held contact does not puff into a blob.
    kept = [points[0]]
    for point in points[1:]:
        if math.hypot(point[0] - kept[-1][0], point[1] - kept[-1][1]) > 0.8:
            kept.append(point)
    if len(kept) < 2:
        return
    pal = active_palette()
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    last = len(kept) - 1
    for index, (x, y) in enumerate(kept):
        if index == 0:
            dx, dy = kept[1][0] - x, kept[1][1] - y
        elif index == last:
            dx, dy = x - kept[index - 1][0], y - kept[index - 1][1]
        else:
            dx = kept[index + 1][0] - kept[index - 1][0]
            dy = kept[index + 1][1] - kept[index - 1][1]
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm
        k = index / last
        half = 0.6 + 7.2 * (k ** 1.45)
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))
    ctx.move_to(*left[0])
    for point in left[1:]:
        ctx.line_to(*point)
    for point in reversed(right):
        ctx.line_to(*point)
    ctx.close_path()
    grad = cairo.LinearGradient(kept[0][0], kept[0][1], kept[-1][0], kept[-1][1])
    grad.add_color_stop_rgba(0.0, *pal.accent2, 0.0)
    grad.add_color_stop_rgba(0.4, *pal.accent2, 0.16)
    grad.add_color_stop_rgba(1.0, *pal.accent, 0.48)
    ctx.set_source(grad)
    ctx.fill()
    hx, hy = kept[-1]
    head = 0.6 + 7.2
    cap = cairo.RadialGradient(hx, hy, head * 0.2, hx, hy, head)
    cap.add_color_stop_rgba(0, *pal.accent, 0.4)
    cap.add_color_stop_rgba(1, *pal.accent, 0)
    ctx.set_source(cap)
    ctx.arc(hx, hy, head, 0, math.tau)
    ctx.fill()


def _paint_halo(ctx, scene, strength: float = 1.0) -> None:
    pal = active_palette()
    radius = BALL_R * (1.85 + 0.35 * max(0.0, strength - 1.0))
    halo = cairo.RadialGradient(0, 0, BALL_R * 0.2, 0, 0, radius)
    halo.add_color_stop_rgba(0, *pal.accent, 0.22 * strength)
    halo.add_color_stop_rgba(0.45, *pal.accent2, 0.07 * strength)
    halo.add_color_stop_rgba(1, *pal.accent, 0)
    ctx.set_source(halo)
    ctx.arc(0, 0, radius, 0, math.tau)
    ctx.fill()


def _paint_bubble(ctx, scene, alpha: float = 1.0) -> None:
    pal = active_palette()
    radius = BALL_R
    pulse = max(0.0, min(1.0, getattr(scene, "squash", 0.0)))
    ease = pulse * pulse * (3.0 - 2.0 * pulse)
    if ease > 0.04 and alpha > 0.5:
        ring = radius * (1.28 + 0.85 * (1.0 - ease))
        band = cairo.RadialGradient(0, 0, ring * 0.78, 0, 0, ring)
        band.add_color_stop_rgba(0.0, *pal.accent, 0.0)
        band.add_color_stop_rgba(0.62, *pal.accent, 0.34 * ease * alpha)
        band.add_color_stop_rgba(1.0, *pal.accent2, 0.0)
        ctx.set_source(band)
        ctx.arc(0, 0, ring, 0, math.tau)
        ctx.fill()
    body = cairo.RadialGradient(-radius * 0.32, -radius * 0.36, radius * 0.08, 0, 0, radius)
    body.add_color_stop_rgba(0.0, 1, 1, 1, (0.95 + 0.05 * ease) * alpha)
    body.add_color_stop_rgba(0.32, *pal.ink, 0.62 * alpha)
    body.add_color_stop_rgba(0.72, *pal.accent, (0.34 + 0.2 * ease) * alpha)
    body.add_color_stop_rgba(1.0, *pal.accent2, 0.12 * alpha)
    ctx.set_source(body)
    ctx.arc(0, 0, radius, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(*pal.rim, (0.72 + 0.2 * ease) * alpha)
    ctx.set_line_width(1.25)
    ctx.arc(0, 0, radius - 0.9, 0, math.tau)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, (0.8 + 0.2 * ease) * alpha)
    ctx.arc(-radius * 0.30, -radius * 0.34, radius * 0.15, 0, math.tau)
    ctx.fill()


def _paint_pinball(ctx, scene, alpha: float = 1.0) -> None:
    del scene
    pal = active_palette()
    radius = BALL_R * 0.92
    body = cairo.RadialGradient(-radius * 0.35, -radius * 0.4, radius * 0.05, radius * 0.1, radius * 0.15, radius)
    body.add_color_stop_rgba(0.0, 1, 1, 1, alpha)
    body.add_color_stop_rgba(0.28, 0.92, 0.93, 0.95, alpha)
    body.add_color_stop_rgba(0.7, *pal.accent, 0.85 * alpha)
    body.add_color_stop_rgba(1.0, 0.28, 0.30, 0.34, alpha)
    ctx.set_source(body)
    ctx.arc(0, 0, radius, 0, math.tau)
    ctx.fill()
    ctx.set_source_rgba(*pal.rim, 0.55 * alpha)
    ctx.set_line_width(1.2)
    ctx.arc(0, 0, radius - 0.7, 0, math.tau)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, 0.9 * alpha)
    ctx.arc(-radius * 0.32, -radius * 0.36, radius * 0.16, 0, math.tau)
    ctx.fill()


def _paint_cube(ctx, scene, alpha: float = 1.0) -> None:
    pal = active_palette()
    ctx.rotate(scene.spin)
    side = BALL_R * 1.42
    body = cairo.LinearGradient(-side / 2, -side / 2, side / 2, side / 2)
    body.add_color_stop_rgba(0.0, *pal.rim, alpha)
    body.add_color_stop_rgba(0.45, *pal.ink, alpha)
    body.add_color_stop_rgba(1.0, *pal.accent2, 0.92 * alpha)
    ctx.set_source(body)
    _round_rect(ctx, -side / 2, -side / 2, side, side, 7)
    ctx.fill()
    ctx.set_source_rgba(*pal.rim, 0.55 * alpha)
    ctx.set_line_width(1.2)
    _round_rect(ctx, -side / 2 + 1.2, -side / 2 + 1.2, side - 2.4, side - 2.4, 6)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, 0.55 * alpha)
    ctx.save()
    ctx.translate(-side * 0.12, -side * 0.18)
    ctx.scale(1.8, 0.45)
    ctx.arc(0, 0, 5, 0, math.tau)
    ctx.fill()
    ctx.restore()


def _chevron(ctx) -> None:
    ctx.move_to(BALL_R * 1.05, 0)
    ctx.line_to(-BALL_R * 0.12, -BALL_R * 0.56)
    ctx.line_to(-BALL_R * 0.50, 0)
    ctx.line_to(-BALL_R * 0.12, BALL_R * 0.56)
    ctx.close_path()


def _paint_arrow(ctx, scene, alpha: float = 1.0) -> None:
    del scene
    pal = active_palette()
    _chevron(ctx)
    body = cairo.LinearGradient(-BALL_R * 0.5, 0, BALL_R, 0)
    body.add_color_stop_rgba(0.0, *pal.accent2, alpha)
    body.add_color_stop_rgba(0.58, *pal.accent, alpha)
    body.add_color_stop_rgba(1.0, *pal.rim, alpha)
    ctx.set_source(body)
    ctx.fill_preserve()
    ctx.set_source_rgba(*pal.rim, 0.7 * alpha)
    ctx.set_line_width(1.15)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, 0.62 * alpha)
    ctx.save()
    ctx.translate(BALL_R * 0.18, -BALL_R * 0.08)
    ctx.scale(1.7, 0.38)
    ctx.arc(0, 0, 4.0, 0, math.tau)
    ctx.fill()
    ctx.restore()


def _paint_note(ctx, scene, alpha: float = 1.0) -> None:
    del scene
    pal = active_palette()
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_source_rgba(*pal.ink, alpha)
    ctx.set_line_width(3.2)
    ctx.move_to(10, 2)
    ctx.line_to(10, -BALL_R * 0.95)
    ctx.stroke()
    ctx.set_line_width(2.6)
    ctx.move_to(10, -BALL_R * 0.95)
    ctx.curve_to(24, -BALL_R * 0.72, 26, -BALL_R * 0.18, 12, -2)
    ctx.stroke()
    ctx.save()
    ctx.translate(-5, 7)
    ctx.rotate(-0.55)
    ctx.scale(1.2, 0.78)
    head = cairo.RadialGradient(-4, -3, 1, 0, 0, BALL_R * 0.42)
    head.add_color_stop_rgba(0, *pal.rim, alpha)
    head.add_color_stop_rgba(1, *pal.accent2, alpha)
    ctx.set_source(head)
    ctx.arc(0, 0, BALL_R * 0.42, 0, math.tau)
    ctx.fill()
    ctx.restore()


_PROJECTILES = {
    "bubble": _paint_bubble,
    "pinball": _paint_pinball,
    "cube": _paint_cube,
    "arrow": _paint_arrow,
    "note": _paint_note,
}


def _obstacle_alpha(obs, time: float) -> float:
    if time >= obs.hit_time:
        return 1.0
    lead = obs.hit_time - time
    if lead > 0.85:
        return 0.0
    u = 1.0 - lead / 0.85
    return u * u * (3.0 - 2.0 * u) * 0.94


def _presence(obs, scene) -> float:
    """Fade a bar once it leaves the frame, and thin the oldest if the view is full.

    The contact the flyer is hitting stays opaque. Extra old bars dissolve
    instead of stacking into a thicket.
    """
    zoom = max(0.5, getattr(scene, "cam_zoom", 1.0) or 1.0)
    half_w = scene.width * 0.5 / zoom
    half_h = scene.height * 0.5 / zoom
    edge = max(abs(obs.x - scene.cam_x) / half_w, abs(obs.y - scene.cam_y) / half_h)
    if edge > 1.22:
        return 0.0
    if edge < 0.8:
        edge_fade = 1.0
    else:
        edge_fade = _smooth((1.22 - edge) / 0.42)
    visible = []
    for other in scene.obstacles:
        if other.hit_time > scene.time + 0.85:
            continue
        if abs(other.x - scene.cam_x) > half_w or abs(other.y - scene.cam_y) > half_h:
            continue
        visible.append(other)
    cap = 5
    if len(visible) <= cap:
        return edge_fade
    visible.sort(key=lambda item: item.hit_time, reverse=True)
    try:
        rank = visible.index(obs)
    except ValueError:
        return edge_fade
    if rank < cap:
        return edge_fade
    fade = 1.0 - (rank - cap + 1) / (len(visible) - cap + 1)
    return edge_fade * max(0.0, fade) * 0.22
