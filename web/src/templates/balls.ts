import type { MusicData } from "../music";
import { hslToRgb, timesPulse } from "../math";
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

interface BallsState {
  width: number;
  height: number;
  scale: number;
  count: number;
  radii: Float32Array;
  hue0: Float32Array;
  xs: Float32Array;
  ys: Float32Array;
  frames: number;
  beats: number[];
  sparks: Spark[];
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
    description: "几颗正圆在拍点上折返加速。它们彼此让开，重拍上溅出一小圈火花。",
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
  const balls: { x: number; y: number; vx: number; vy: number; radius: number; hue: number }[] = [];
  for (let i = 0; i < count; i++) {
    const radius = (30 + rng() * 16) * scale;
    let x = radius;
    let y = radius;
    let placed = false;
    for (let attempt = 0; attempt < 48 && !placed; attempt++) {
      x = radius + rng() * (width - 2 * radius);
      y = radius + rng() * (height - 2 * radius);
      placed = balls.every((other) => Math.hypot(other.x - x, other.y - y) >= other.radius + radius + 14 * scale);
    }
    const angle = rng() * TAU;
    const speed = (240 + rng() * 280) * scale;
    balls.push({ x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, radius, hue: rng() * 360 });
  }
  separate(balls, width, height, scale, 8);
  const hue0 = Float32Array.from(balls.map((ball) => ball.hue));
  const frames = frameCount(music.duration);
  const xs = new Float32Array(frames * count);
  const ys = new Float32Array(frames * count);
  const beats = music.beats.slice();
  const strong = new Set((music.extensions?.strongBeats?.length ? music.extensions.strongBeats : music.downbeats).map((beat) => beat.toFixed(4)));
  const sparks: Spark[] = [];
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
        if (isStrong) {
          for (let n = 0; n < 10; n++) {
            const sparkAngle = rng() * TAU;
            const sparkSpeed = (180 + rng() * 420) * scale;
            sparks.push({
              x: ball.x,
              y: ball.y,
              vx: Math.cos(sparkAngle) * sparkSpeed,
              vy: Math.sin(sparkAngle) * sparkSpeed - 20 * scale,
              radius: (2.4 + rng() * 2.2) * scale,
              hue: ball.hue,
              born: beat,
            });
          }
        }
      }
      beatIndex++;
    }
    for (const ball of balls) {
      ball.x += ball.vx * BAKE_DT;
      ball.y += ball.vy * BAKE_DT;
      if (ball.x < ball.radius) {
        ball.x = ball.radius;
        ball.vx = Math.abs(ball.vx);
      } else if (ball.x > width - ball.radius) {
        ball.x = width - ball.radius;
        ball.vx = -Math.abs(ball.vx);
      }
      if (ball.y < ball.radius) {
        ball.y = ball.radius;
        ball.vy = Math.abs(ball.vy);
      } else if (ball.y > height - ball.radius) {
        ball.y = height - ball.radius;
        ball.vy = -Math.abs(ball.vy);
      }
    }
    separate(balls, width, height, scale, 2);
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
    neon: skin !== "pastel",
  };
}

function separate(
  balls: { x: number; y: number; vx: number; vy: number; radius: number }[],
  width: number,
  height: number,
  scale: number,
  passes: number,
): void {
  const pad = 6 * scale;
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
      ball.x = Math.max(ball.radius, Math.min(width - ball.radius, ball.x));
      ball.y = Math.max(ball.radius, Math.min(height - ball.radius, ball.y));
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
  const pulse = timesPulse(state.beats, time, 0.04, 0.16);
  const beatsBefore = beatsUntil(state.beats, time);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  if (state.neon) {
    const lift = pulse * 0.05;
    ctx.fillStyle = rgb(0.055 + lift, 0.063 + lift, 0.094 + lift);
  } else {
    ctx.fillStyle = rgb(0.96 - pulse * 0.03, 0.93 - pulse * 0.02, 0.9);
  }
  ctx.fillRect(0, 0, state.width, state.height);
  const gravity = 280 * state.scale;
  for (const spark of state.sparks) {
    const age = time - spark.born;
    if (age < 0 || age > 0.45) continue;
    const life = 1 - age / 0.45;
    const [r, g, b] = hslToRgb(spark.hue, state.neon ? 0.8 : 0.45, state.neon ? 0.35 + 0.35 * life : 0.55);
    ctx.fillStyle = rgba(r, g, b, 0.9 * life);
    ctx.beginPath();
    ctx.arc(spark.x + spark.vx * age, spark.y + spark.vy * age + 0.5 * gravity * age * age, spark.radius * (0.45 + 0.55 * life), 0, TAU);
    ctx.fill();
  }
  for (let index = 0; index < state.count; index++) {
    const x = at(state.xs, state.count, index, time);
    const y = at(state.ys, state.count, index, time);
    const radius = state.radii[index] * (1 + 0.08 * pulse);
    const hue = (state.hue0[index] + 47 * beatsBefore) % 360;
    for (let ghost = 3; ghost >= 1; ghost--) {
      const gx = at(state.xs, state.count, index, time - ghost * 0.03);
      const gy = at(state.ys, state.count, index, time - ghost * 0.03);
      ctx.fillStyle = rgba(...ballFill(hue, state.neon), 0.12 * (1 - ghost / 4));
      ctx.beginPath();
      ctx.arc(gx, gy, radius * (1 - ghost * 0.08), 0, TAU);
      ctx.fill();
    }
    const [r, g, b] = ballFill(hue, state.neon);
    ctx.fillStyle = rgb(r, g, b);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, TAU);
    ctx.fill();
    ctx.fillStyle = rgba(1, 1, 1, state.neon ? 0.72 : 0.55);
    ctx.beginPath();
    ctx.arc(x - radius * 0.32, y - radius * 0.34, radius * 0.16, 0, TAU);
    ctx.fill();
    const flash = Math.exp(-Math.max(0, time - (state.beats[beatsBefore - 1] ?? -1)) * 9);
    ctx.strokeStyle = flash > 0.45 ? rgba(1, 1, 1, flash) : rgba(r, g, b, 0.9);
    ctx.lineWidth = 3 * state.scale;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, TAU);
    ctx.stroke();
  }
  view.post.glow = state.neon ? 0.55 + 0.35 * pulse : 0.28;
  view.post.grain = state.neon ? 0.7 : 0.3;
  view.post.vignette = state.neon ? 0.85 : 0.4;
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
