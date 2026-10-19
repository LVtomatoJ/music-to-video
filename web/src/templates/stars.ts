import type { MusicData, Section, ToneEvent } from "../music";
import { sectionAt } from "../music";
import { damp, sampleCurve, smoothstep } from "../math";
import { defineTemplate, type ParamSpec } from "../template";
import { BAKE_DT, TAU, as2d, frameCount, rgb, rgba } from "./shared";

interface Star {
  time: number;
  x: number;
  y: number;
  phase: number;
  midi: number;
  phrase: number;
  scale: number;
  warmth: number;
}

interface StarState {
  width: number;
  height: number;
  unit: number;
  stars: Star[];
  links: [number, number][];
  dust: { x: number; y: number; scale: number; warmth: number; phase: number }[];
  camX: Float32Array;
  camY: Float32Array;
}

const starsParams = {
  seed: { type: "number", label: "种子", min: 1, max: 999, step: 1, default: 42, scope: "plan" },
} satisfies Record<string, ParamSpec>;

export const starsTemplate = defineTemplate<StarState, typeof starsParams>({
  meta: {
    id: "stars",
    name: "星座",
    author: "music-to-video",
    version: "1.0.0",
    description: "一句旋律弯成一个星座。星星在音头上亮起，旧的星座暗下去，天空不会排成格子。",
  },
  tier: "2d",
  params: starsParams,
  setup(ctx, music) {
    const view = as2d(ctx);
    return buildStars(view.width, view.height, music, view.rng);
  },
  render(ctx, t, state) {
    const view = as2d(ctx);
    paintStars(view.ctx, state, t, view);
  },
});

function buildStars(width: number, height: number, music: MusicData, rng: () => number): StarState {
  const unit = width / 1920;
  const tones = music.extensions?.tones?.length
    ? music.extensions.tones
    : music.beats.map((time) => ({ time, duration: 0.4, midi: 60, source: "chroma" as const }));
  const phrases = splitPhrases(tones, music.sections);
  const { stars, links } = layout(phrases, width, height, unit, rng);
  const occupied = stars.map((star) => [star.x, star.y] as [number, number]);
  const dust: StarState["dust"] = [];
  for (let i = 0; i < 220; i++) {
    const x = 24 + rng() * (width - 48);
    const y = 20 + rng() * height * 0.9;
    if (occupied.some(([sx, sy]) => Math.hypot(x - sx, y - sy) < 26 * unit)) continue;
    dust.push({ x, y, scale: 0.35 + rng() * 1.15, warmth: rng(), phase: rng() * TAU });
  }
  const camera = bakeDrift(music.duration);
  return { width, height, unit, stars, links, dust, camX: camera.x, camY: camera.y };
}

function splitPhrases(tones: ToneEvent[], sections: Section[]): ToneEvent[][] {
  const phrases: ToneEvent[][] = [];
  let bucket: ToneEvent[] = [];
  for (const tone of tones) {
    const section = sectionAt({ sections } as MusicData, tone.time).label;
    if (bucket.length) {
      const gap = tone.time - bucket[bucket.length - 1].time;
      const previous = sectionAt({ sections } as MusicData, bucket[bucket.length - 1].time).label;
      if (gap > 1.15 || (section !== previous && gap > 0.45) || bucket.length >= 5) {
        phrases.push(bucket);
        bucket = [];
      }
    }
    bucket.push(tone);
  }
  if (bucket.length) phrases.push(bucket);
  return phrases;
}

interface Seat {
  x: number;
  y: number;
  radius: number;
  start: number;
  end: number;
}

function layout(phrases: ToneEvent[][], width: number, height: number, unit: number, rng: () => number) {
  const pending = phrases.map((phrase, index) => {
    const local = organicPoints(phrase.length, unit, rng);
    const radius = Math.max(...local.map(([x, y]) => Math.hypot(x, y)), 0);
    return { index, phrase, local, radius };
  });
  const placed: Seat[] = [];
  const origins = new Map<number, [number, number]>();
  for (const item of pending) {
    const start = item.phrase[0].time - 0.3;
    const end = item.phrase[item.phrase.length - 1].time + 9;
    const origin = seatFigure(item.radius, placed, width, height, unit, rng, start, end);
    if (!origin) continue;
    placed.push({ x: origin[0], y: origin[1], radius: item.radius, start, end });
    origins.set(item.index, origin);
  }
  const stars: Star[] = [];
  const links: [number, number][] = [];
  for (const item of pending) {
    const origin = origins.get(item.index);
    if (!origin) continue;
    const peak = Math.max(...item.phrase.map((tone) => tone.midi));
    const first = stars.length;
    item.phrase.forEach((tone, place) => {
      const [x, y] = item.local[place];
      let scale = 0.55 + rng() * 0.55;
      if (tone.midi === peak) scale += 0.28;
      stars.push({
        time: tone.time,
        x: origin[0] + x,
        y: origin[1] + y,
        phase: rng() * TAU,
        midi: tone.midi,
        phrase: item.index,
        scale: Math.min(1.35, scale),
        warmth: rng(),
      });
      if (place) links.push([first + place - 1, first + place]);
    });
  }
  return { stars, links };
}

function seatFigure(
  radius: number,
  placed: Seat[],
  width: number,
  height: number,
  unit: number,
  rng: () => number,
  start: number,
  end: number,
): [number, number] | null {
  const marginX = 70 * unit + radius;
  const marginY = 56 * unit + radius;
  const skyBottom = height * 0.86;
  let best: [number, number] | null = null;
  let bestClear = -1;
  for (let attempt = 0; attempt < 56; attempt++) {
    const cx = marginX + rng() * Math.max(1, width - 2 * marginX);
    const cy = marginY + rng() * Math.max(1, skyBottom - 2 * marginY);
    const clear = clearance(cx, cy, radius, placed, start, end);
    if (clear >= 28 * unit) return [cx, cy];
    if (clear > bestClear) {
      best = [cx, cy];
      bestClear = clear;
    }
  }
  if (best && bestClear >= 8 * unit) return best;
  return null;
}

function clearance(cx: number, cy: number, radius: number, placed: Seat[], start: number, end: number): number {
  let clear = 1e6;
  for (const other of placed) {
    if (end < other.start || other.end < start) continue;
    clear = Math.min(clear, Math.hypot(cx - other.x, cy - other.y) - radius - other.radius);
  }
  return clear;
}

function organicPoints(count: number, unit: number, rng: () => number): [number, number][] {
  if (count <= 1) return [[0, 0]];
  for (let attempt = 0; attempt < 10; attempt++) {
    const style = Math.floor(rng() * 3);
    let points: [number, number][] | null;
    if (style === 0) points = arcPoints(count, 1.15 + rng() * 1.5, unit, rng);
    else if (style === 1) points = arcPoints(count, 2.7 + rng() * 1.5, unit, rng);
    else points = walkPoints(count, unit, rng);
    if (!points) continue;
    points = rotate(points, rng() * TAU);
    if (figureOk(points, unit) && bent(points)) return center(points);
  }
  return center(rotate(arcPoints(count, 2.6, unit, rng), rng() * TAU));
}

function arcPoints(count: number, sweep: number, unit: number, rng: () => number): [number, number][] {
  const step = 78 * unit;
  const delta = sweep / Math.max(1, count - 1);
  const radius = step / Math.max(0.35, 2 * Math.sin(delta / 2));
  const start = rng() * TAU;
  return Array.from({ length: count }, (_, i) => {
    const angle = start + i * delta;
    return [Math.cos(angle) * radius, Math.sin(angle) * radius] as [number, number];
  });
}

function walkPoints(count: number, unit: number, rng: () => number): [number, number][] | null {
  const points: [number, number][] = [[0, 0]];
  let heading = rng() * TAU;
  for (let i = 1; i < count; i++) {
    let found: [[number, number], number] | null = null;
    for (let attempt = 0; attempt < 28; attempt++) {
      let turn = (rng() - 0.5) * 2.6;
      if (rng() < 0.3) turn += turn >= 0 ? 0.9 : -0.9;
      const nextHeading = heading + turn;
      const step = (74 + rng() * 24) * unit;
      const next: [number, number] = [
        points[points.length - 1][0] + Math.cos(nextHeading) * step,
        points[points.length - 1][1] + Math.sin(nextHeading) * step,
      ];
      if (figureOk([...points, next], unit)) {
        found = [next, nextHeading];
        break;
      }
    }
    if (!found) return null;
    points.push(found[0]);
    heading = found[1];
  }
  return points;
}

function rotate(points: [number, number][], angle: number): [number, number][] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return points.map(([x, y]) => [x * cos - y * sin, x * sin + y * cos]);
}

function center(points: [number, number][]): [number, number][] {
  const cx = points.reduce((sum, point) => sum + point[0], 0) / points.length;
  const cy = points.reduce((sum, point) => sum + point[1], 0) / points.length;
  return points.map(([x, y]) => [x - cx, y - cy]);
}

function bent(points: [number, number][]): boolean {
  if (points.length < 3) return true;
  const centered = center(points);
  const xs = centered.map((point) => point[0]);
  const ys = centered.map((point) => point[1]);
  const spanX = Math.max(...xs) - Math.min(...xs);
  const spanY = Math.max(...ys) - Math.min(...ys);
  const short = Math.max(1, Math.min(spanX, spanY));
  return Math.max(spanX, spanY) / short <= 2.5;
}

function figureOk(points: [number, number][], unit: number): boolean {
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      if (Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]) < 64 * unit) return false;
    }
  }
  if (points.length < 4) return true;
  for (let i = 0; i < points.length - 1; i++) {
    for (let j = i + 2; j < points.length - 1; j++) {
      if (cross(points[i], points[i + 1], points[j], points[j + 1])) return false;
    }
  }
  return true;
}

function cross(a: [number, number], b: [number, number], c: [number, number], d: [number, number]): boolean {
  const orient = (p: [number, number], q: [number, number], r: [number, number]) =>
    (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  return orient(a, b, c) * orient(a, b, d) < 0 && orient(c, d, a) * orient(c, d, b) < 0;
}

function bakeDrift(duration: number) {
  const n = frameCount(duration);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < n; i++) {
    const time = i * BAKE_DT;
    const tx = 8 * Math.sin(time * 0.045);
    const ty = 5 * Math.cos(time * 0.032);
    if (i > 0) {
      cx = damp(cx, tx, BAKE_DT, 0.8);
      cy = damp(cy, ty, BAKE_DT, 0.8);
    } else {
      cx = tx;
      cy = ty;
    }
    x[i] = cx;
    y[i] = cy;
  }
  return { x, y };
}

function tint(warmth: number): [number, number, number] {
  return [0.76 + 0.24 * warmth, 0.84 + 0.08 * (1 - Math.abs(warmth - 0.35)), 0.98 - 0.26 * warmth];
}

function brightness(star: Star, time: number): number {
  const age = time - star.time;
  if (age < -0.24) return 0;
  if (age <= 0) return smoothstep((age + 0.24) / 0.24);
  const twinkle = 0.84 + 0.16 * Math.sin(time * 1.5 + star.phase);
  const flare = 1 + 0.55 * Math.exp(-age * 3.2);
  const fade = age < 12 ? 1 : smoothstep(1 - Math.min(1, (age - 12) / 6));
  return Math.min(1.45, twinkle * flare * fade);
}

function paintStars(
  ctx: CanvasRenderingContext2D,
  state: StarState,
  time: number,
  view: { post: { glow: number; grain: number; vignette: number } },
): void {
  const camX = sampleCurve(state.camX, BAKE_DT, time);
  const camY = sampleCurve(state.camY, BAKE_DT, time);
  const beat = state.stars.reduce((level, star) => Math.max(level, Math.max(0, 1 - Math.abs(time - star.time) / 0.18)), 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  const sky = ctx.createLinearGradient(0, 0, 0, state.height);
  sky.addColorStop(0, rgb(0.045, 0.055, 0.16));
  sky.addColorStop(0.34, rgb(0.09, 0.055, 0.18));
  sky.addColorStop(0.66, rgb(0.11, 0.07, 0.17));
  sky.addColorStop(1, rgb(0.05, 0.04, 0.1));
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, state.width, state.height);
  paintNebula(ctx, state, time);
  ctx.save();
  ctx.translate(state.width / 2, state.height / 2);
  ctx.translate(-state.width / 2 + camX, -state.height / 2 + camY);
  for (const speck of state.dust) {
    const twinkle = 0.55 + 0.45 * Math.sin(time * 1.4 + speck.phase);
    const color = tint(speck.warmth);
    const alpha = (0.05 + 0.12 * Math.min(1, speck.scale)) * twinkle;
    ctx.fillStyle = rgba(color[0], color[1], color[2], alpha);
    ctx.beginPath();
    ctx.arc(speck.x, speck.y, (0.7 + speck.scale * 0.7) * state.unit, 0, TAU);
    ctx.fill();
  }
  ctx.lineCap = "round";
  for (const [left, right] of state.links) {
    const a = state.stars[left];
    const b = state.stars[right];
    const age = time - Math.max(a.time, b.time);
    if (age < -0.7) continue;
    const appear = age < 0 ? 0.35 * smoothstep((age + 0.7) / 0.7) : smoothstep(Math.min(1, age / 0.45));
    const alpha = age < 5 ? appear * 0.72 : Math.max(0.05, 0.72 * smoothstep(1 - (age - 5) / 6));
    const color = tint((a.warmth + b.warmth) * 0.5);
    ctx.strokeStyle = rgba(color[0], color[1], color[2], alpha);
    ctx.lineWidth = 1.55 * state.unit;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
  for (const star of state.stars) {
    const age = time - star.time;
    if (age < -0.9) continue;
    if (age < -0.24) {
      const hint = smoothstep((age + 0.9) / 0.66) * 0.35;
      ctx.fillStyle = rgba(0.85, 0.9, 1, hint);
      ctx.beginPath();
      ctx.arc(star.x, star.y, (2.2 + star.scale) * state.unit, 0, TAU);
      ctx.fill();
      continue;
    }
    const level = brightness(star, time);
    if (level <= 0.02) continue;
    if (age >= 0 && age < 0.48) {
      const fade = 1 - age / 0.48;
      ctx.strokeStyle = rgba(1, 0.98, 0.94, 0.7 * fade);
      ctx.lineWidth = 1.6 * state.unit;
      ctx.beginPath();
      ctx.arc(star.x, star.y, (6 + age * 78) * state.unit * (0.7 + star.scale), 0, TAU);
      ctx.stroke();
    }
    paintGlyph(ctx, star.x, star.y, level, star.warmth, star.phase, time, star.scale, state.unit);
  }
  ctx.restore();
  view.post.glow = 0.7 + 0.4 * beat;
  view.post.grain = 0.8;
  view.post.vignette = 1;
}

function paintNebula(ctx: CanvasRenderingContext2D, state: StarState, time: number): void {
  const drift = Math.sin(time * 0.07) * 18 * state.unit;
  ctx.save();
  ctx.translate(state.width * 0.56 + drift, state.height * 0.42);
  ctx.rotate(-0.26);
  const band = ctx.createLinearGradient(0, -260 * state.unit, 0, 260 * state.unit);
  band.addColorStop(0, rgba(0.72, 0.68, 0.95, 0));
  band.addColorStop(0.5, rgba(0.88, 0.84, 1, 0.055));
  band.addColorStop(1, rgba(0.65, 0.6, 0.92, 0));
  ctx.fillStyle = band;
  ctx.fillRect(-state.width * 1.2, -220 * state.unit, state.width * 2.4, 440 * state.unit);
  ctx.restore();
  const blobs: [number, number, number, [number, number, number], number][] = [
    [0.3, 0.34, 420, [0.72, 0.62, 1], 0.07],
    [0.68, 0.5, 480, [0.62, 0.74, 1], 0.05],
    [0.48, 0.24, 300, [1, 0.86, 0.72], 0.04],
  ];
  for (const [ox, oy, radius, color, alpha] of blobs) {
    const x = ox * state.width + drift * 0.4;
    const y = oy * state.height;
    const blob = ctx.createRadialGradient(x, y, radius * state.unit * 0.05, x, y, radius * state.unit);
    blob.addColorStop(0, rgba(color[0], color[1], color[2], alpha));
    blob.addColorStop(0.45, rgba(color[0], color[1], color[2], alpha * 0.4));
    blob.addColorStop(1, rgba(color[0], color[1], color[2], 0));
    ctx.fillStyle = blob;
    ctx.beginPath();
    ctx.arc(x, y, radius * state.unit, 0, TAU);
    ctx.fill();
  }
}

function paintGlyph(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  level: number,
  warmth: number,
  phase: number,
  time: number,
  scale: number,
  unit: number,
): void {
  const twinkle = 0.9 + 0.1 * Math.sin(time * 2.1 + phase);
  const color = tint(warmth);
  const hot = Math.min(level, 1.55);
  const spike = (12 + 16 * hot) * (0.75 + 0.45 * scale) * twinkle * unit;
  const glowR = (16 + 26 * scale) * (0.5 + 0.5 * hot) * unit;
  const glow = ctx.createRadialGradient(x, y, 0.4, x, y, glowR);
  glow.addColorStop(0, rgba(1, 1, 1, Math.min(1, 0.95 * level * twinkle)));
  glow.addColorStop(0.22, rgba(color[0], color[1], color[2], 0.72 * level * twinkle));
  glow.addColorStop(0.55, rgba(color[0], color[1], color[2], 0.22 * level));
  glow.addColorStop(1, rgba(color[0], color[1], color[2], 0));
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, glowR, 0, TAU);
  ctx.fill();
  ctx.lineCap = "round";
  [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2].forEach((angle, index) => {
    spikeLine(ctx, x, y, angle, spike * (index % 2 ? 1.12 : 0.84), 1.35 * (0.75 + 0.35 * scale) * unit, color, 0.9 * Math.min(1, level) * twinkle);
  });
  ctx.fillStyle = rgba(1, 1, 1, Math.min(1, level * twinkle));
  ctx.beginPath();
  ctx.arc(x, y, (1.6 + scale) * unit, 0, TAU);
  ctx.fill();
}

function spikeLine(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  angle: number,
  length: number,
  width: number,
  color: [number, number, number],
  alpha: number,
): void {
  if (length < 1 || alpha <= 0) return;
  const x2 = x + Math.cos(angle) * length;
  const y2 = y + Math.sin(angle) * length;
  const grad = ctx.createLinearGradient(x, y, x2, y2);
  grad.addColorStop(0, rgba(color[0], color[1], color[2], alpha));
  grad.addColorStop(0.4, rgba(color[0], color[1], color[2], alpha * 0.35));
  grad.addColorStop(1, rgba(color[0], color[1], color[2], 0));
  ctx.strokeStyle = grad;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}
