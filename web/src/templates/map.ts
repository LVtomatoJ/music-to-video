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
  let solved: ReturnType<typeof solveMap>;
  try {
    solved = solveMap(music, seed, skinName);
  } catch {
    solved = solveMap({ ...music, beats: [], duration: Math.max(music.duration, 0.5) }, seed, skinName);
  }
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
  const aheadX = ball[0] + vx * 0.16;
  const aheadY = ball[1] + vy * 0.16;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  const consider = (x: number, y: number, pad: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    minX = Math.min(minX, x - pad);
    maxX = Math.max(maxX, x + pad);
    minY = Math.min(minY, y - pad);
    maxY = Math.max(maxY, y + pad);
  };
  const start = Math.max(segments[0]?.t0 ?? 0, time - 11);
  const end = time + 0.55;
  for (let cursor = start; cursor <= end; cursor += 0.18) {
    const point = pointOn(segments, cursor);
    consider(point[0], point[1], ballR * 2);
  }
  consider(ball[0], ball[1], 80);
  consider(aheadX, aheadY, 40);
  for (const obstacle of obstacles) {
    if (obstacle.hitTime < start || obstacle.hitTime > end) continue;
    consider(obstacle.x, obstacle.y, 90);
  }
  if (!Number.isFinite(minX) || !Number.isFinite(ball[0]) || !Number.isFinite(ball[1])) {
    return [Number.isFinite(ball[0]) ? ball[0] : 0, Number.isFinite(ball[1]) ? ball[1] : 0, 0.62];
  }
  const spanX = Math.max(maxX - minX, 980);
  const spanY = Math.max(maxY - minY, 640);
  const fitted = Math.min((width * 0.88) / spanX, (height * 0.84) / spanY);
  const zoom = clamp(fitted, 0.46, 0.78) * (1 + 0.03 * (Number.isFinite(energy) ? energy : 0));
  const cx = (minX + maxX) * 0.62 + (Number.isFinite(aheadX) ? aheadX : ball[0]) * 0.38;
  const cy = (minY + maxY) * 0.62 + (Number.isFinite(aheadY) ? aheadY : ball[1]) * 0.38;
  return [Number.isFinite(cx) ? cx : ball[0], Number.isFinite(cy) ? cy : ball[1], Number.isFinite(zoom) ? zoom : 0.62];
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
    const nextX = Number.isFinite(tx) ? tx : cx;
    const nextY = Number.isFinite(ty) ? ty : cy;
    const nextZ = Number.isFinite(tz) ? tz : cz;
    if (frame === 0) {
      cx = nextX;
      cy = nextY;
      cz = nextZ;
    } else {
      cx = damp(cx, nextX, BAKE_DT, 3.6);
      cy = damp(cy, nextY, BAKE_DT, 3.6);
      cz = damp(cz, nextZ, BAKE_DT, 2);
    }
    x[frame] = cx;
    y[frame] = cy;
    z[frame] = cz;
  }
  return { x, y, z };
}

/** Keeps the ball inside the frame even when the baked camera or the course is bad. */
export function framedView(state: MapState, time: number): {
  camX: number;
  camY: number;
  zoom: number;
  ballX: number;
  ballY: number;
} {
  const raw = pointOn(state.segments, time);
  const worldX = Number.isFinite(raw[0]) ? raw[0] : 0;
  const worldY = Number.isFinite(raw[1]) ? raw[1] : 0;
  let camX = Number.isFinite(sampleCurve(state.camX, BAKE_DT, time)) ? sampleCurve(state.camX, BAKE_DT, time) : worldX;
  let camY = Number.isFinite(sampleCurve(state.camY, BAKE_DT, time)) ? sampleCurve(state.camY, BAKE_DT, time) : worldY;
  let zoom = sampleCurve(state.camZ, BAKE_DT, time);
  if (!Number.isFinite(zoom)) zoom = 0.62;
  zoom = clamp(zoom, 0.35, 1.4);
  let ballX = (worldX - camX) * zoom + state.width / 2;
  let ballY = (worldY - camY) * zoom + state.height / 2;
  const padX = state.width * 0.16;
  const padY = state.height * 0.16;
  if (ballX < padX || ballX > state.width - padX || ballY < padY || ballY > state.height - padY) {
    camX = worldX;
    camY = worldY;
    ballX = state.width / 2;
    ballY = state.height / 2;
  }
  return { camX, camY, zoom, ballX, ballY };
}

export function sampleBallScreens(
  music: MusicData,
  seed: number,
  skin: string,
  width: number,
  height: number,
  times: number[],
): { time: number; x: number; y: number; onScreen: boolean }[] {
  const state = buildMap(width, height, music, seed, skin);
  return times.map((time) => {
    const view = framedView(state, time);
    const onScreen =
      Number.isFinite(view.ballX) &&
      Number.isFinite(view.ballY) &&
      view.ballX >= 20 &&
      view.ballX <= width - 20 &&
      view.ballY >= 20 &&
      view.ballY <= height - 20;
    return { time, x: view.ballX, y: view.ballY, onScreen };
  });
}

function paintMap(
  ctx: CanvasRenderingContext2D,
  state: MapState,
  time: number,
  view: { post: { glow: number; grain: number; vignette: number } },
): void {
  const framed = framedView(state, time);
  const camX = framed.camX;
  const camY = framed.camY;
  const zoom = framed.zoom;
  const pulse = timesPulse(state.beats, time, 0.035, 0.14);
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
  const project = (x: number, y: number): [number, number] => [
    (x - camX) * zoom + state.width / 2,
    (y - camY) * zoom + state.height / 2,
  ];
  paintCourse(ctx, state, time, project, zoom);
  for (const obstacle of state.obstacles) paintObstacle(ctx, state, obstacle, time - obstacle.hitTime, project, zoom);
  paintRings(ctx, state, time, project, zoom);
  paintBall(ctx, state, framed.ballX, framed.ballY, state.ballR * zoom, pulse);
  view.post.glow = state.light ? 0.35 : 0.85 + 0.2 * pulse;
  view.post.grain = state.light ? 0.35 : 0.75;
  view.post.vignette = state.light ? 0.45 : 0.9;
}

function paintCourse(
  ctx: CanvasRenderingContext2D,
  state: MapState,
  time: number,
  project: (x: number, y: number) => [number, number],
  zoom: number,
): void {
  const span = 14;
  const points = coursePoints(state, time, span).map((point) => {
    const [x, y] = project(point.x, point.y);
    return { x, y, t: point.t };
  });
  if (points.length < 2) return;
  paintRibbon(ctx, points, time, span, 18 * zoom, state.skin.ball, state.light ? 0.28 : 0.45);
  paintRibbon(ctx, points, time, span, 3.6 * zoom, [1, 1, 1], state.light ? 0.62 : 0.95);
}

function coursePoints(state: MapState, time: number, span: number): { x: number; y: number; t: number }[] {
  const start = Math.max(state.segments[0]?.t0 ?? 0, time - span);
  const points: { x: number; y: number; t: number }[] = [];
  const push = (t: number) => {
    if (t < start - 1e-3 || t > time + 1e-3) return;
    const point = pointOn(state.segments, t);
    const last = points[points.length - 1];
    if (last && Math.abs(last.t - t) < 1e-3) return;
    points.push({ x: point[0], y: point[1], t });
  };
  for (const segment of state.segments) {
    if (segment.t1 < start || segment.t0 > time) continue;
    const t0 = Math.max(segment.t0, start);
    const t1 = Math.min(segment.t1, time);
    const steps = Math.max(1, Math.round((t1 - t0) / 0.04));
    for (let step = 0; step <= steps; step++) push(t0 + ((t1 - t0) * step) / steps);
  }
  push(time);
  return points;
}

function paintRibbon(
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number; t: number }[],
  time: number,
  span: number,
  width: number,
  color: [number, number, number],
  alpha: number,
): void {
  const half = width / 2;
  const left: { x: number; y: number }[] = [];
  const right: { x: number; y: number }[] = [];
  for (let index = 0; index < points.length; index++) {
    const before = points[Math.max(0, index - 1)];
    const after = points[Math.min(points.length - 1, index + 1)];
    let dx = after.x - before.x;
    let dy = after.y - before.y;
    const length = Math.hypot(dx, dy) || 1;
    dx /= length;
    dy /= length;
    left.push({ x: points[index].x - dy * half, y: points[index].y + dx * half });
    right.push({ x: points[index].x + dy * half, y: points[index].y - dx * half });
  }
  for (let index = 1; index < points.length; index++) {
    const fade = clamp(1 - (time - points[index].t) / span, 0, 1) ** 1.15;
    if (fade < 0.03) continue;
    ctx.fillStyle = rgba(color[0], color[1], color[2], alpha * fade);
    ctx.beginPath();
    ctx.moveTo(left[index - 1].x, left[index - 1].y);
    ctx.lineTo(left[index].x, left[index].y);
    ctx.lineTo(right[index].x, right[index].y);
    ctx.lineTo(right[index - 1].x, right[index - 1].y);
    ctx.closePath();
    ctx.fill();
  }
}

function presence(age: number): number {
  if (age < -0.4 || age > 24) return 0;
  const born = age < 0 ? smoothstep((age + 0.4) / 0.4) : 1;
  const fade = age > 9 ? clamp(1 - (age - 9) / 15, 0, 1) : 1;
  return born * fade;
}

function paintObstacle(
  ctx: CanvasRenderingContext2D,
  state: MapState,
  obstacle: Obstacle,
  age: number,
  project: (x: number, y: number) => [number, number],
  zoom: number,
): void {
  const alpha = presence(age);
  if (alpha <= 0.02) return;
  const flash = age >= 0 && age < 0.32 ? 1 - age / 0.32 : 0;
  const color = mixWhite(kindColor(state, obstacle.kind), 0.7 * flash);
  if (obstacle.kind === "peg") {
    paintPeg(ctx, state, obstacle, color, alpha, flash, project, zoom);
    return;
  }
  if (obstacle.kind === "spinner") {
    paintSpinner(ctx, state, obstacle, color, alpha, age, project, zoom);
    return;
  }
  ctx.save();
  place(ctx, obstacle, state.ballR, project, zoom);
  if (obstacle.kind === "platform" && obstacle.variant === "arch") paintArch(ctx, obstacle, color, alpha);
  else if (obstacle.kind === "pad") paintPad(ctx, obstacle, color, alpha, flash);
  else paintWall(ctx, state, obstacle, color, alpha, flash);
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

function place(
  ctx: CanvasRenderingContext2D,
  obstacle: Obstacle,
  ballR: number,
  project: (x: number, y: number) => [number, number],
  zoom: number,
): void {
  const [x, y] = project(obstacle.x + obstacle.nx * ballR, obstacle.y + obstacle.ny * ballR);
  ctx.translate(x, y);
  ctx.rotate(Math.atan2(obstacle.ny, obstacle.nx));
  ctx.scale(zoom, zoom);
}

function paintWall(
  ctx: CanvasRenderingContext2D,
  state: MapState,
  obstacle: Obstacle,
  color: [number, number, number],
  alpha: number,
  flash: number,
): void {
  const length = wallLength(obstacle, state.obstacles);
  const thick = 20;
  const radius = 6;
  slab(ctx, 0, -length / 2, thick, length, color, alpha, radius, flash);
  if (flash > 0.05) {
    ctx.fillStyle = rgba(1, 1, 1, 0.55 * flash * alpha);
    roundRect(ctx, -3, -length / 2, thick + 8, length, radius);
    ctx.fill();
  }
}

/** One bar, centred on the contact, kept clear of the neighbouring hit. */
function wallLength(obstacle: Obstacle, obstacles: Obstacle[]): number {
  let room = 300;
  for (const other of obstacles) {
    if (other === obstacle) continue;
    const dx = other.x - obstacle.x;
    const dy = other.y - obstacle.y;
    const along = Math.abs(dx * obstacle.tx + dy * obstacle.ty);
    const across = Math.abs(dx * obstacle.nx + dy * obstacle.ny);
    if (across > 110) continue;
    room = Math.min(room, Math.max(108, along - 40));
  }
  const wanted = obstacle.size < 110 ? 168 : Math.min(240, Math.max(168, obstacle.size));
  return Math.min(wanted, room);
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
  project: (x: number, y: number) => [number, number],
  zoom: number,
): void {
  const [cx, cy] = project(
    obstacle.x + obstacle.nx * (state.ballR + obstacle.pegR),
    obstacle.y + obstacle.ny * (state.ballR + obstacle.pegR),
  );
  const disks = [[cx, cy, obstacle.pegR * zoom]];
  if (obstacle.variant === "pair") {
    const [px, py] = project(
      obstacle.x + obstacle.nx * (state.ballR + obstacle.pegR) + obstacle.tx * obstacle.pegR * 2.4,
      obstacle.y + obstacle.ny * (state.ballR + obstacle.pegR) + obstacle.ty * obstacle.pegR * 2.4,
    );
    disks.push([px, py, obstacle.pegR * 0.78 * zoom]);
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
  project: (x: number, y: number) => [number, number],
  zoom: number,
): void {
  const length = obstacle.size * (obstacle.variant === "short" ? 0.7 : obstacle.variant === "long" ? 1.35 : 1);
  const thick = obstacle.variant === "short" ? 22 : 14;
  const spin = Math.atan2(obstacle.ty, obstacle.tx) + 2.4 * Math.max(0, age);
  const [x, y] = project(obstacle.x + obstacle.nx * state.ballR, obstacle.y + obstacle.ny * state.ballR);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(spin);
  ctx.scale(zoom, zoom);
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

function paintRings(
  ctx: CanvasRenderingContext2D,
  state: MapState,
  time: number,
  project: (x: number, y: number) => [number, number],
  zoom: number,
): void {
  for (let index = state.obstacles.length - 1; index >= 0; index--) {
    const obstacle = state.obstacles[index];
    const age = time - obstacle.hitTime;
    if (age < 0) continue;
    if (age > 0.34) break;
    const fade = 1 - age / 0.34;
    const [x, y] = project(obstacle.x, obstacle.y);
    ctx.strokeStyle = rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], 0.8 * fade);
    ctx.lineWidth = (2.2 * fade + 0.8) * zoom;
    ctx.beginPath();
    ctx.arc(x, y, (state.ballR + age * 150) * zoom, 0, TAU);
    ctx.stroke();
  }
}

function paintBall(ctx: CanvasRenderingContext2D, state: MapState, x: number, y: number, radius: number, pulse: number): void {
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
