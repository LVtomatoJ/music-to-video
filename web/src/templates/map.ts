import type { MusicData } from "../music";
import { sectionAt } from "../music";
import { clamp, damp, roundRect, sampleCurve, smoothstep, timesPulse } from "../math";
import { defineTemplate, type ParamSpec } from "../template";
import { BAKE_DT, TAU, as2d, frameCount, rgb, rgba } from "./shared";
import { pointOn, solveMap, type Obstacle, type Segment } from "./map-solve";

interface Skin {
  bg0: [number, number, number];
  bg1: [number, number, number];
  ball: [number, number, number];
  peg: [number, number, number];
  bar: [number, number, number];
  ink: [number, number, number];
}

interface MapState {
  width: number;
  height: number;
  obstacles: Obstacle[];
  segments: Segment[];
  beats: number[];
  skin: Skin;
  light: boolean;
  ballR: number;
  camX: Float32Array;
  camY: Float32Array;
  camZ: Float32Array;
}

const SKINS: Record<string, Skin> = {
  neon: {
    bg0: [0.043, 0.067, 0.133],
    bg1: [0.012, 0.016, 0.043],
    ball: [0.361, 0.902, 0.886],
    peg: [0.965, 0.286, 0.541],
    bar: [0.78, 1, 0.965],
    ink: [0.78, 1, 0.965],
  },
  pastel: {
    bg0: [0.965, 0.937, 0.91],
    bg1: [0.855, 0.765, 0.835],
    ball: [0.761, 0.337, 0.478],
    peg: [0.365, 0.612, 0.569],
    bar: [0.62, 0.31, 0.42],
    ink: [0.45, 0.28, 0.36],
  },
  arcade: {
    bg0: [0.11, 0.051, 0.133],
    bg1: [0.035, 0.012, 0.055],
    ball: [1, 0.784, 0.341],
    peg: [1, 0.302, 0.545],
    bar: [1, 0.91, 0.655],
    ink: [1, 0.91, 0.655],
  },
};

const mapParams = {
  seed: { type: "number", label: "种子", min: 1, max: 999, step: 1, default: 42, scope: "plan" },
  skin: {
    type: "enum",
    label: "皮肤",
    default: "neon",
    scope: "plan",
    options: [
      { value: "neon", label: "霓虹" },
      { value: "pastel", label: "粉彩" },
      { value: "arcade", label: "街机" },
    ],
  },
} satisfies Record<string, ParamSpec>;

export const mapTemplate = defineTemplate<MapState, typeof mapParams>({
  meta: {
    id: "map",
    name: "地图",
    author: "music-to-video",
    version: "1.0.0",
    description: "球在每个拍点撞上一面墙。墙按真实反射预先摆好，轨迹随拍距弯折，镜头跟着走。",
  },
  tier: "2d",
  params: mapParams,
  setup(ctx, music, params) {
    const view = as2d(ctx);
    return buildMap(view.width, view.height, music, params.seed, params.skin);
  },
  render(ctx, t, state) {
    const view = as2d(ctx);
    paintMap(view.ctx, state, t, view);
  },
});

function buildMap(width: number, height: number, music: MusicData, seed: number, skinName: string): MapState {
  const solved = solveMap(music, seed, skinName);
  const skin = SKINS[skinName] ?? SKINS.neon;
  const beats = music.beats.length ? music.beats.slice() : [0];
  const camera = bakeCamera(width, height, solved.segments, solved.obstacles, solved.ballR, music);
  return {
    width,
    height,
    obstacles: solved.obstacles,
    segments: solved.segments,
    beats,
    skin,
    light: skinName === "pastel",
    ballR: solved.ballR,
    camX: camera.x,
    camY: camera.y,
    camZ: camera.z,
  };
}

export function auditMap(state: MapState): {
  hits: number;
  maxHitError: number;
  lengthMean: number;
  lengthStd: number;
  lengthMin: number;
  lengthMax: number;
  turnCount: number;
  rightAngles: number;
  minSeparation: number;
  closePairs: number;
  kinds: Record<string, number>;
} {
  let maxHitError = 0;
  const lengths: number[] = [];
  const turns: number[] = [];
  const kinds: Record<string, number> = {};
  for (const obs of state.obstacles) {
    const point = pointOn(state.segments, obs.hitTime);
    maxHitError = Math.max(maxHitError, Math.hypot(point[0] - obs.x, point[1] - obs.y));
    kinds[obs.kind] = (kinds[obs.kind] ?? 0) + 1;
  }
  for (let index = 0; index < state.segments.length; index++) {
    const segment = state.segments[index];
    lengths.push(Math.hypot(segment.x1 - segment.x0, segment.y1 - segment.y0));
    if (index === 0) continue;
    const previous = state.segments[index - 1];
    const before = Math.atan2(previous.y1 - previous.y0, previous.x1 - previous.x0);
    const after = Math.atan2(segment.y1 - segment.y0, segment.x1 - segment.x0);
    let delta = after - before;
    while (delta > Math.PI) delta -= TAU;
    while (delta < -Math.PI) delta += TAU;
    turns.push(delta);
  }
  let minSeparation = Infinity;
  let closePairs = 0;
  for (let i = 0; i < state.obstacles.length; i++) {
    for (let j = i + 1; j < state.obstacles.length; j++) {
      const gap = Math.hypot(state.obstacles[i].x - state.obstacles[j].x, state.obstacles[i].y - state.obstacles[j].y);
      if (gap < minSeparation) minSeparation = gap;
      if (gap < 70) closePairs += 1;
    }
  }
  const mean = lengths.length ? lengths.reduce((sum, value) => sum + value, 0) / lengths.length : 0;
  const variance = lengths.length ? lengths.reduce((sum, value) => sum + (value - mean) ** 2, 0) / lengths.length : 0;
  const rightAngles = turns.filter((turn) => Math.abs(Math.abs(turn) - Math.PI / 2) < 0.09).length;
  return {
    hits: state.obstacles.length,
    maxHitError,
    lengthMean: mean,
    lengthStd: Math.sqrt(variance),
    lengthMin: lengths.length ? Math.min(...lengths) : 0,
    lengthMax: lengths.length ? Math.max(...lengths) : 0,
    turnCount: turns.length,
    rightAngles,
    minSeparation: Number.isFinite(minSeparation) ? minSeparation : 0,
    closePairs,
    kinds,
  };
}

function velocity(segments: Segment[], time: number): [number, number] {
  const dt = BAKE_DT;
  const before = pointOn(segments, time - dt);
  const after = pointOn(segments, time + dt);
  return [(after[0] - before[0]) / (2 * dt), (after[1] - before[1]) / (2 * dt)];
}

function cameraTarget(
  width: number,
  height: number,
  segments: Segment[],
  obstacles: Obstacle[],
  ballR: number,
  time: number,
  energy: number,
): [number, number, number] {
  const ball = pointOn(segments, time);
  const [vx, vy] = velocity(segments, time);
  const aheadX = ball[0] + vx * 0.18;
  const aheadY = ball[1] + vy * 0.18;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  const consider = (x: number, y: number, pad: number) => {
    minX = Math.min(minX, x - pad);
    maxX = Math.max(maxX, x + pad);
    minY = Math.min(minY, y - pad);
    maxY = Math.max(maxY, y + pad);
  };
  const start = Math.max(segments[0]?.t0 ?? 0, time - 3.3);
  const end = time + 0.42;
  for (let cursor = start; cursor <= end; cursor += 0.14) {
    const point = pointOn(segments, cursor);
    consider(point[0], point[1], ballR);
  }
  consider(ball[0], ball[1], ballR * 2.4);
  consider(aheadX, aheadY, ballR);
  for (const obstacle of obstacles) {
    if (obstacle.hitTime < start || obstacle.hitTime > end) continue;
    consider(obstacle.x, obstacle.y, Math.min(obstacle.size, 160) * 0.28 + ballR);
  }
  const spanX = Math.max(maxX - minX, 760);
  const spanY = Math.max(maxY - minY, 480);
  const fitted = Math.min((width * 0.84) / spanX, (height * 0.8) / spanY);
  const floor = 18 / ballR;
  const breath = 1 + 0.045 * energy;
  if (fitted < floor) return [aheadX, aheadY, floor * breath];
  const zoom = clamp(fitted, floor, 1.15) * breath;
  return [(minX + maxX) * 0.31 + aheadX * 0.69, (minY + maxY) * 0.31 + aheadY * 0.69, zoom];
}

function bakeCamera(
  width: number,
  height: number,
  segments: Segment[],
  obstacles: Obstacle[],
  ballR: number,
  music: MusicData,
): { x: Float32Array; y: Float32Array; z: Float32Array } {
  const frames = frameCount(music.duration);
  const x = new Float32Array(frames);
  const y = new Float32Array(frames);
  const z = new Float32Array(frames);
  let cx = 0;
  let cy = 0;
  let cz = 1;
  for (let frame = 0; frame < frames; frame++) {
    const time = frame * BAKE_DT;
    const energy = sectionAt(music, time).energy;
    const [tx, ty, tz] = cameraTarget(width, height, segments, obstacles, ballR, time, energy);
    if (frame === 0) {
      cx = tx;
      cy = ty;
      cz = tz;
    } else {
      cx = damp(cx, tx, BAKE_DT, 3.6);
      cy = damp(cy, ty, BAKE_DT, 3.6);
      cz = damp(cz, tz, BAKE_DT, 2);
    }
    x[frame] = cx;
    y[frame] = cy;
    z[frame] = cz;
  }
  return { x, y, z };
}

function paintMap(
  ctx: CanvasRenderingContext2D,
  state: MapState,
  time: number,
  view: { post: { glow: number; grain: number; vignette: number } },
): void {
  const camX = sampleCurve(state.camX, BAKE_DT, time);
  const camY = sampleCurve(state.camY, BAKE_DT, time);
  const zoom = sampleCurve(state.camZ, BAKE_DT, time);
  const pulse = timesPulse(state.beats, time, 0.035, 0.14);
  const ball = pointOn(state.segments, time);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  const sky = ctx.createLinearGradient(0, 0, 0, state.height);
  sky.addColorStop(0, rgb(...state.skin.bg0));
  sky.addColorStop(1, rgb(...state.skin.bg1));
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, state.width, state.height);
  for (const [ox, oy, tint] of [
    [0.28, 0.32, state.skin.ball],
    [0.74, 0.58, state.skin.bar],
  ] as const) {
    const x = ox * state.width;
    const y = oy * state.height;
    const orb = ctx.createRadialGradient(x, y, 0, x, y, state.width * 0.26);
    orb.addColorStop(0, rgba(tint[0], tint[1], tint[2], state.light ? 0.14 : 0.08 + 0.04 * pulse));
    orb.addColorStop(1, rgba(tint[0], tint[1], tint[2], 0));
    ctx.fillStyle = orb;
    ctx.beginPath();
    ctx.arc(x, y, state.width * 0.26, 0, TAU);
    ctx.fill();
  }
  ctx.save();
  ctx.translate(state.width / 2, state.height / 2);
  ctx.scale(zoom, zoom);
  ctx.translate(-camX, -camY);
  paintCourse(ctx, state, time);
  for (const obstacle of state.obstacles) paintObstacle(ctx, state, obstacle, time - obstacle.hitTime);
  paintRings(ctx, state, time);
  paintBall(ctx, state, ball[0], ball[1], pulse);
  ctx.restore();
  view.post.glow = state.light ? 0.35 : 0.85 + 0.2 * pulse;
  view.post.grain = state.light ? 0.35 : 0.75;
  view.post.vignette = state.light ? 0.45 : 0.9;
}

function paintCourse(ctx: CanvasRenderingContext2D, state: MapState, time: number): void {
  const window = 5.4;
  const start = Math.max(state.segments[0]?.t0 ?? 0, time - window);
  const points: { x: number; y: number; t: number }[] = [];
  for (let cursor = start; cursor < time; cursor += 0.045) {
    const point = pointOn(state.segments, cursor);
    points.push({ x: point[0], y: point[1], t: cursor });
  }
  const live = pointOn(state.segments, time);
  points.push({ x: live[0], y: live[1], t: time });
  if (points.length < 2) return;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (let index = 1; index < points.length; index++) {
    const a = points[index - 1];
    const b = points[index];
    const fade = clamp(1 - (time - b.t) / window, 0, 1) ** 1.35;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], (state.light ? 0.28 : 0.5) * fade);
    ctx.lineWidth = 22;
    ctx.stroke();
    ctx.strokeStyle = rgba(1, 1, 1, (state.light ? 0.4 : 0.92) * fade);
    ctx.lineWidth = 3.4;
    ctx.stroke();
  }
}

function presence(age: number): number {
  if (age < -0.45 || age > 9.2) return 0;
  const born = age < 0 ? smoothstep((age + 0.45) / 0.45) : 1;
  const fade = age > 6.4 ? smoothstep(1 - (age - 6.4) / 2.8) : 1;
  return born * fade;
}

function paintObstacle(ctx: CanvasRenderingContext2D, state: MapState, obstacle: Obstacle, age: number): void {
  const alpha = presence(age);
  if (alpha <= 0.02) return;
  const flash = age >= 0 && age < 0.3 ? 1 - age / 0.3 : 0;
  const color = mixWhite(kindColor(state, obstacle.kind), 0.55 * flash);
  if (obstacle.kind === "peg") {
    paintPeg(ctx, state, obstacle, color, alpha, flash);
    return;
  }
  if (obstacle.kind === "spinner") {
    paintSpinner(ctx, state, obstacle, color, alpha, age);
    return;
  }
  ctx.save();
  place(ctx, obstacle, state.ballR);
  if (obstacle.kind === "platform" && obstacle.variant === "arch") paintArch(ctx, obstacle, color, alpha);
  else if (obstacle.kind === "pad") paintPad(ctx, obstacle, color, alpha, flash);
  else paintWall(ctx, obstacle, color, alpha, flash);
  ctx.restore();
}

function kindColor(state: MapState, kind: string): [number, number, number] {
  if (kind === "peg" || kind === "spinner") return state.skin.peg;
  return state.skin.bar;
}

function mixWhite(color: [number, number, number], amount: number): [number, number, number] {
  return [
    color[0] + (1 - color[0]) * amount,
    color[1] + (1 - color[1]) * amount,
    color[2] + (1 - color[2]) * amount,
  ];
}

function place(ctx: CanvasRenderingContext2D, obstacle: Obstacle, ballR: number): void {
  ctx.translate(obstacle.x + obstacle.nx * ballR, obstacle.y + obstacle.ny * ballR);
  ctx.rotate(Math.atan2(obstacle.ny, obstacle.nx));
}

function paintWall(
  ctx: CanvasRenderingContext2D,
  obstacle: Obstacle,
  color: [number, number, number],
  alpha: number,
  flash: number,
): void {
  const length = obstacle.size;
  const thick = obstacle.variant === "short" ? 14 : 18;
  const radius = Math.min(7, thick * 0.42);
  slab(ctx, 0, -length / 2, thick, length, color, alpha, radius, flash);
  if (obstacle.variant === "double") {
    slab(ctx, thick + 16, -length * 0.42, thick * 0.75, length * 0.84, color, alpha * 0.95, radius, flash);
  }
  if (obstacle.variant === "corner") {
    const arm = Math.max(54, length * 0.42);
    slab(ctx, 0, length / 2 - thick, arm, thick, color, alpha, radius, flash);
  }
}

function paintArch(
  ctx: CanvasRenderingContext2D,
  obstacle: Obstacle,
  color: [number, number, number],
  alpha: number,
): void {
  const length = obstacle.size;
  const thick = obstacle.variant === "short" ? 16 : 22;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(length * 0.18, -length / 2);
  ctx.quadraticCurveTo(-length * 0.08, 0, length * 0.18, length / 2);
  ctx.strokeStyle = rgba(color[0], color[1], color[2], alpha);
  ctx.lineWidth = thick;
  ctx.stroke();
  ctx.strokeStyle = rgba(1, 1, 1, alpha * 0.45);
  ctx.lineWidth = 2;
  ctx.stroke();
}

function paintPad(
  ctx: CanvasRenderingContext2D,
  obstacle: Obstacle,
  color: [number, number, number],
  alpha: number,
  flash: number,
): void {
  const length = obstacle.size;
  const thick = 28 + (obstacle.strong ? 10 : 0);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(thick * 0.5, -length / 2);
  ctx.lineTo(thick * 0.5, length / 2);
  ctx.strokeStyle = rgba(color[0], color[1], color[2], alpha);
  ctx.lineWidth = thick;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(2.2, -length * 0.42);
  ctx.lineTo(2.2, length * 0.42);
  ctx.strokeStyle = rgba(1, 1, 1, alpha * (0.35 + 0.4 * flash));
  ctx.lineWidth = 2;
  ctx.stroke();
}

function paintPeg(
  ctx: CanvasRenderingContext2D,
  state: MapState,
  obstacle: Obstacle,
  color: [number, number, number],
  alpha: number,
  flash: number,
): void {
  const cx = obstacle.x + obstacle.nx * (state.ballR + obstacle.pegR);
  const cy = obstacle.y + obstacle.ny * (state.ballR + obstacle.pegR);
  const disks = [[cx, cy, obstacle.pegR]];
  if (obstacle.variant === "pair") {
    disks.push([
      cx + obstacle.tx * obstacle.pegR * 2.4,
      cy + obstacle.ty * obstacle.pegR * 2.4,
      obstacle.pegR * 0.78,
    ]);
  }
  for (const [x, y, radius] of disks) {
    const halo = ctx.createRadialGradient(x, y, radius * 0.2, x, y, radius * 2.2);
    halo.addColorStop(0, rgba(color[0], color[1], color[2], 0.45 * alpha));
    halo.addColorStop(1, rgba(color[0], color[1], color[2], 0));
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(x, y, radius * 2.2, 0, TAU);
    ctx.fill();
    ctx.fillStyle = rgba(color[0], color[1], color[2], alpha);
    ctx.beginPath();
    ctx.arc(x, y, radius * (1 + 0.08 * flash), 0, TAU);
    ctx.fill();
  }
}

function paintSpinner(
  ctx: CanvasRenderingContext2D,
  state: MapState,
  obstacle: Obstacle,
  color: [number, number, number],
  alpha: number,
  age: number,
): void {
  const length = obstacle.size * (obstacle.variant === "short" ? 0.7 : obstacle.variant === "long" ? 1.35 : 1);
  const thick = obstacle.variant === "short" ? 22 : 14;
  const spin = Math.atan2(obstacle.ty, obstacle.tx) + 2.4 * Math.max(0, age);
  ctx.save();
  ctx.translate(obstacle.x + obstacle.nx * state.ballR, obstacle.y + obstacle.ny * state.ballR);
  ctx.rotate(spin);
  slab(ctx, -length / 2, -thick, length, thick, color, alpha, Math.min(7, thick * 0.45), 0);
  ctx.restore();
}

function slab(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  color: [number, number, number],
  alpha: number,
  radius: number,
  flash: number,
): void {
  if (w <= 0.5 || h <= 0.5) return;
  ctx.fillStyle = rgba(color[0] * 0.35, color[1] * 0.35, color[2] * 0.35, alpha * 0.28);
  roundRect(ctx, x + 3, y + 5, w, h, radius);
  ctx.fill();
  const grad = ctx.createLinearGradient(x, y, x + w, y + h * 0.2);
  grad.addColorStop(0, rgba(Math.min(1, color[0] + 0.16), Math.min(1, color[1] + 0.16), Math.min(1, color[2] + 0.16), alpha));
  grad.addColorStop(1, rgba(color[0] * 0.58, color[1] * 0.58, color[2] * 0.58, alpha));
  ctx.fillStyle = grad;
  roundRect(ctx, x, y, w, h, radius);
  ctx.fill();
  ctx.strokeStyle = rgba(1, 1, 1, alpha * (0.28 + 0.55 * flash));
  ctx.lineWidth = 1.4;
  ctx.stroke();
}

function paintRings(ctx: CanvasRenderingContext2D, state: MapState, time: number): void {
  for (let index = state.obstacles.length - 1; index >= 0; index--) {
    const obstacle = state.obstacles[index];
    const age = time - obstacle.hitTime;
    if (age < 0) continue;
    if (age > 0.34) break;
    const fade = 1 - age / 0.34;
    ctx.strokeStyle = rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], 0.8 * fade);
    ctx.lineWidth = 2.2 * fade + 0.6;
    ctx.beginPath();
    ctx.arc(obstacle.x, obstacle.y, state.ballR + age * 150, 0, TAU);
    ctx.stroke();
  }
}

function paintBall(ctx: CanvasRenderingContext2D, state: MapState, x: number, y: number, pulse: number): void {
  const radius = state.ballR;
  const halo = ctx.createRadialGradient(x, y, radius * 0.4, x, y, radius * 3.1);
  halo.addColorStop(0, rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], 0.5 + 0.2 * pulse));
  halo.addColorStop(1, rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], 0));
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(x, y, radius * 3.1, 0, TAU);
  ctx.fill();
  ctx.fillStyle = rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], 1);
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, TAU);
  ctx.fill();
  ctx.fillStyle = rgba(1, 1, 1, state.light ? 0.55 : 0.8);
  ctx.beginPath();
  ctx.arc(x - radius * 0.32, y - radius * 0.34, radius * 0.16, 0, TAU);
  ctx.fill();
  ctx.strokeStyle = rgba(1, 1, 1, 0.4 + 0.35 * pulse);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, TAU);
  ctx.stroke();
}
