import type { MusicData } from "../music";
import { hslToRgb, roundRect, timesPulse } from "../math";
import { defineTemplate, type ParamSpec } from "../template";
import { BAKE_DT, TAU, as2d, frameCount, rgb, rgba } from "./shared";

interface Spark {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  hue: number;
  born: number;
}

interface Impact {
  x: number;
  y: number;
  time: number;
  hue: number;
  kind: "beat" | "wall";
}

interface BallsState {
  width: number;
  height: number;
  scale: number;
  inset: number;
  count: number;
  radii: Float32Array;
  hue0: Float32Array;
  xs: Float32Array;
  ys: Float32Array;
  frames: number;
  beats: number[];
  sparks: Spark[];
  impacts: Impact[];
  neon: boolean;
}

const ballsParams = {
  seed: { type: "number", label: "种子", min: 1, max: 999, step: 1, default: 42, scope: "plan" },
  count: { type: "number", label: "球数", min: 3, max: 7, step: 1, default: 5, scope: "plan" },
  skin: {
    type: "enum",
    label: "皮肤",
    default: "neon",
    scope: "plan",
    options: [
      { value: "neon", label: "霓虹" },
      { value: "pastel", label: "粉彩" },
    ],
  },
} satisfies Record<string, ParamSpec>;

export const ballsTemplate = defineTemplate<BallsState, typeof ballsParams>({
  meta: {
    id: "balls",
    name: "多球",
    author: "music-to-video",
    version: "1.0.0",
    description: "几颗正圆在围栏里跟着拍子撞墙。每一拍都有闪光、涟漪和拖尾。",
  },
  tier: "2d",
  params: ballsParams,
  setup(ctx, music, params) {
    const view = as2d(ctx);
    return buildBalls(view.width, view.height, music, Math.round(params.count), params.skin, view.rng);
  },
  render(ctx, t, state) {
    const view = as2d(ctx);
    paintBalls(view.ctx, state, t, view);
  },
});

function buildBalls(
  width: number,
  height: number,
  music: MusicData,
  count: number,
  skin: string,
  rng: () => number,
): BallsState {
  const scale = width / 1600;
  const inset = 78 * scale;
  const balls: { x: number; y: number; vx: number; vy: number; radius: number; hue: number }[] = [];
  for (let i = 0; i < count; i++) {
    const radius = (34 + rng() * 14) * scale;
    let x = inset + radius;
    let y = inset + radius;
    let placed = false;
    for (let attempt = 0; attempt < 48 && !placed; attempt++) {
      x = inset + radius + rng() * (width - 2 * (inset + radius));
      y = inset + radius + rng() * (height - 2 * (inset + radius));
      placed = balls.every((other) => Math.hypot(other.x - x, other.y - y) >= other.radius + radius + 70 * scale);
    }
    const angle = rng() * TAU;
    const speed = (240 + rng() * 280) * scale;
    balls.push({ x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, radius, hue: rng() * 360 });
  }
  separate(balls, width, height, inset, scale, 10);
  const hue0 = Float32Array.from(balls.map((ball) => ball.hue));
  const frames = frameCount(music.duration);
  const xs = new Float32Array(frames * count);
  const ys = new Float32Array(frames * count);
  const beats = music.beats.slice();
  const strong = new Set((music.extensions?.strongBeats?.length ? music.extensions.strongBeats : music.downbeats).map((beat) => beat.toFixed(4)));
  const sparks: Spark[] = [];
  const impacts: Impact[] = [];
  let beatIndex = 0;
  for (let frame = 0; frame < frames; frame++) {
    const time = frame * BAKE_DT;
    const prev = time - BAKE_DT;
    while (beatIndex < beats.length && beats[beatIndex] <= time && beats[beatIndex] > prev) {
      const beat = beats[beatIndex];
      const isStrong = strong.has(beat.toFixed(4));
      for (const ball of balls) {
        ball.vx = -ball.vx;
        ball.vy = -ball.vy;
        const speed = Math.hypot(ball.vx, ball.vy);
        const target = Math.min(720 * scale, Math.max(240 * scale, speed * 1.12));
        if (speed > 1e-6) {
          ball.vx *= target / speed;
          ball.vy *= target / speed;
        }
        ball.hue = (ball.hue + 47) % 360;
        impacts.push({ x: ball.x, y: ball.y, time: beat, hue: ball.hue, kind: "beat" });
        const burst = isStrong ? 14 : 8;
        for (let n = 0; n < burst; n++) {
          const sparkAngle = rng() * TAU;
          const sparkSpeed = (160 + rng() * (isStrong ? 380 : 220)) * scale;
          sparks.push({
            x: ball.x,
            y: ball.y,
            vx: Math.cos(sparkAngle) * sparkSpeed,
            vy: Math.sin(sparkAngle) * sparkSpeed - 16 * scale,
            radius: (2.6 + rng() * 2.4) * scale,
            hue: ball.hue,
            born: beat,
          });
        }
      }
      beatIndex++;
    }
    for (const ball of balls) {
      ball.x += ball.vx * BAKE_DT;
      ball.y += ball.vy * BAKE_DT;
      const minX = inset + ball.radius;
      const maxX = width - inset - ball.radius;
      const minY = inset + ball.radius;
      const maxY = height - inset - ball.radius;
      if (ball.x < minX) {
        ball.x = minX;
        ball.vx = Math.abs(ball.vx);
        impacts.push({ x: minX - ball.radius, y: ball.y, time, hue: ball.hue, kind: "wall" });
      } else if (ball.x > maxX) {
        ball.x = maxX;
        ball.vx = -Math.abs(ball.vx);
        impacts.push({ x: maxX + ball.radius, y: ball.y, time, hue: ball.hue, kind: "wall" });
      }
      if (ball.y < minY) {
        ball.y = minY;
        ball.vy = Math.abs(ball.vy);
        impacts.push({ x: ball.x, y: minY - ball.radius, time, hue: ball.hue, kind: "wall" });
      } else if (ball.y > maxY) {
        ball.y = maxY;
        ball.vy = -Math.abs(ball.vy);
        impacts.push({ x: ball.x, y: maxY + ball.radius, time, hue: ball.hue, kind: "wall" });
      }
    }
    separate(balls, width, height, inset, scale, 3);
    balls.forEach((ball, index) => {
      xs[frame * count + index] = ball.x;
      ys[frame * count + index] = ball.y;
    });
  }
  return {
    width,
    height,
    scale,
    count,
    radii: Float32Array.from(balls.map((ball) => ball.radius)),
    hue0,
    xs,
    ys,
    frames,
    beats,
    sparks,
    impacts,
    inset,
    neon: skin !== "pastel",
  };
}

function separate(
  balls: { x: number; y: number; vx: number; vy: number; radius: number }[],
  width: number,
  height: number,
  inset: number,
  scale: number,
  passes: number,
): void {
  const pad = 58 * scale;
  for (let pass = 0; pass < passes; pass++) {
    for (let i = 0; i < balls.length; i++) {
      for (let j = i + 1; j < balls.length; j++) {
        const a = balls[i];
        const b = balls[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dist = Math.hypot(dx, dy);
        const min = a.radius + b.radius + pad;
        if (dist >= min) continue;
        if (dist < 1e-4) {
          dx = 1;
          dy = 0;
          dist = 1;
        }
        const push = (min - dist) / 2;
        const nx = dx / dist;
        const ny = dy / dist;
        a.x -= nx * push;
        a.y -= ny * push;
        b.x += nx * push;
        b.y += ny * push;
        const rel = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
        if (rel < 0) {
          a.vx += nx * rel * 0.5;
          a.vy += ny * rel * 0.5;
          b.vx -= nx * rel * 0.5;
          b.vy -= ny * rel * 0.5;
        }
      }
    }
    for (const ball of balls) {
      ball.x = Math.max(inset + ball.radius, Math.min(width - inset - ball.radius, ball.x));
      ball.y = Math.max(inset + ball.radius, Math.min(height - inset - ball.radius, ball.y));
    }
  }
}

function at(samples: Float32Array, count: number, index: number, time: number): number {
  const frame = time / BAKE_DT;
  const i = Math.max(0, Math.min(samples.length / count - 1, frame));
  const lo = Math.floor(i);
  const hi = Math.min(lo + 1, samples.length / count - 1);
  const f = i - lo;
  return samples[lo * count + index] * (1 - f) + samples[hi * count + index] * f;
}

function paintBalls(
  ctx: CanvasRenderingContext2D,
  state: BallsState,
  time: number,
  view: { post: { glow: number; grain: number; vignette: number } },
): void {
  const pulse = timesPulse(state.beats, time, 0.04, 0.18);
  const beatsBefore = beatsUntil(state.beats, time);
  const { width, height, scale, inset } = state;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  const sky = ctx.createLinearGradient(0, 0, 0, height);
  if (state.neon) {
    sky.addColorStop(0, rgb(0.05 + pulse * 0.04, 0.07 + pulse * 0.03, 0.14 + pulse * 0.05));
    sky.addColorStop(1, rgb(0.015, 0.018, 0.04));
  } else {
    sky.addColorStop(0, rgb(0.97, 0.94, 0.92));
    sky.addColorStop(1, rgb(0.86, 0.78, 0.84));
  }
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, width, height);
  for (const [ox, oy, color] of [
    [0.22, 0.28, state.neon ? [0.2, 0.85, 0.9] : [0.9, 0.45, 0.55]],
    [0.78, 0.72, state.neon ? [0.95, 0.3, 0.55] : [0.4, 0.7, 0.62]],
  ] as const) {
    const x = ox * width + Math.sin(time * 0.35 + ox) * 18 * scale;
    const y = oy * height;
    const orb = ctx.createRadialGradient(x, y, 0, x, y, 280 * scale);
    orb.addColorStop(0, rgba(color[0], color[1], color[2], (state.neon ? 0.16 : 0.2) + 0.08 * pulse));
    orb.addColorStop(1, rgba(color[0], color[1], color[2], 0));
    ctx.fillStyle = orb;
    ctx.beginPath();
    ctx.arc(x, y, 280 * scale, 0, TAU);
    ctx.fill();
  }
  const floor = ctx.createRadialGradient(width / 2, height / 2, 40, width / 2, height / 2, Math.min(width, height) * 0.48);
  floor.addColorStop(0, rgba(1, 1, 1, state.neon ? 0.035 + 0.04 * pulse : 0.08));
  floor.addColorStop(1, rgba(1, 1, 1, 0));
  ctx.fillStyle = floor;
  ctx.beginPath();
  ctx.ellipse(width / 2, height * 0.56, width * 0.34, height * 0.16, 0, 0, TAU);
  ctx.fill();
  ctx.lineWidth = (7 + 3 * pulse) * scale;
  ctx.strokeStyle = state.neon ? rgba(0.55, 0.95, 0.95, 0.35 + 0.45 * pulse) : rgba(0.55, 0.28, 0.4, 0.35 + 0.3 * pulse);
  roundRect(ctx, inset * 0.55, inset * 0.55, width - inset * 1.1, height - inset * 1.1, 26 * scale);
  ctx.stroke();
  const gravity = 220 * scale;
  for (const spark of state.sparks) {
    const age = time - spark.born;
    if (age < 0 || age > 0.42) continue;
    const life = 1 - age / 0.42;
    const [r, g, b] = hslToRgb(spark.hue, state.neon ? 0.82 : 0.45, state.neon ? 0.62 : 0.55);
    ctx.fillStyle = rgba(r, g, b, 0.85 * life);
    ctx.beginPath();
    ctx.arc(spark.x + spark.vx * age, spark.y + spark.vy * age + 0.5 * gravity * age * age, spark.radius * (0.4 + 0.6 * life), 0, TAU);
    ctx.fill();
  }
  for (let i = state.impacts.length - 1; i >= 0; i--) {
    const hit = state.impacts[i];
    const age = time - hit.time;
    if (age < 0) continue;
    if (age > 0.5) break;
    const [r, g, b] = hslToRgb(hit.hue, state.neon ? 0.7 : 0.4, 0.7);
    const fade = 1 - age / 0.5;
    ctx.strokeStyle = rgba(r, g, b, (hit.kind === "beat" ? 0.85 : 0.55) * fade);
    ctx.lineWidth = (hit.kind === "beat" ? 3.2 : 2.2) * scale * fade + 0.6;
    ctx.beginPath();
    ctx.arc(hit.x, hit.y, (hit.kind === "beat" ? 18 : 10) * scale + age * (hit.kind === "beat" ? 150 : 90) * scale, 0, TAU);
    ctx.stroke();
  }
  for (let index = 0; index < state.count; index++) {
    const x = at(state.xs, state.count, index, time);
    const y = at(state.ys, state.count, index, time);
    const radius = state.radii[index] * (1 + 0.16 * pulse);
    const hue = (state.hue0[index] + 47 * beatsBefore) % 360;
    const [r, g, b] = ballFill(hue, state.neon);
    for (let ghost = 6; ghost >= 1; ghost--) {
      const gx = at(state.xs, state.count, index, time - ghost * 0.028);
      const gy = at(state.ys, state.count, index, time - ghost * 0.028);
      ctx.fillStyle = rgba(r, g, b, 0.16 * (1 - ghost / 7));
      ctx.beginPath();
      ctx.arc(gx, gy, radius * (1 - ghost * 0.06), 0, TAU);
      ctx.fill();
    }
    const glow = ctx.createRadialGradient(x, y, radius * 0.2, x, y, radius * 2.1);
    glow.addColorStop(0, rgba(r, g, b, 0.45 + 0.25 * pulse));
    glow.addColorStop(1, rgba(r, g, b, 0));
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, radius * 2.1, 0, TAU);
    ctx.fill();
    ctx.fillStyle = rgb(r, g, b);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, TAU);
    ctx.fill();
    ctx.fillStyle = rgba(1, 1, 1, state.neon ? 0.8 : 0.6);
    ctx.beginPath();
    ctx.arc(x - radius * 0.32, y - radius * 0.34, radius * 0.16, 0, TAU);
    ctx.fill();
    const flash = Math.exp(-Math.max(0, time - (state.beats[beatsBefore - 1] ?? -1)) * 8);
    ctx.strokeStyle = flash > 0.35 ? rgba(1, 1, 1, flash) : rgba(1, 1, 1, 0.75);
    ctx.lineWidth = (2.4 + 1.6 * flash) * scale;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, TAU);
    ctx.stroke();
  }
  view.post.glow = state.neon ? 0.62 + 0.45 * pulse : 0.32;
  view.post.grain = state.neon ? 0.55 : 0.28;
  view.post.vignette = state.neon ? 0.7 : 0.35;
}

function ballFill(hue: number, neon: boolean): [number, number, number] {
  return hslToRgb(hue, neon ? 0.78 : 0.42, neon ? 0.56 : 0.62);
}

function beatsUntil(beats: number[], time: number): number {
  let lo = 0;
  let hi = beats.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (beats[mid] <= time) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
