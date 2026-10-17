import type { MusicData } from "../music";
import { sectionAt } from "../music";
import { clamp, damp, roundRect, sampleCurve, smoothstep, timesPulse } from "../math";
import { defineTemplate, type ParamSpec } from "../template";
import { BAKE_DT, TAU, as2d, frameCount, rgb, rgba } from "./shared";

interface Hit {
  time: number;
  x: number;
  y: number;
  heading: number;
  kind: "peg" | "bar";
  side: number;
}

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
  scale: number;
  hits: Hit[];
  beats: number[];
  skin: Skin;
  light: boolean;
  ballR: number;
  pegR: number;
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
    description: "球沿着节拍在钉和短墙之间弹开。路上留下一条发光的轨迹，镜头跟着走。",
  },
  tier: "2d",
  params: mapParams,
  setup(ctx, music, params) {
    const view = as2d(ctx);
    return buildMap(view.width, view.height, music, params.skin, view.rng);
  },
  render(ctx, t, state) {
    const view = as2d(ctx);
    paintMap(view.ctx, state, t, view);
  },
});

function buildMap(width: number, height: number, music: MusicData, skinName: string, rng: () => number): MapState {
  const scale = width / 1600;
  const beats = music.beats.length ? music.beats.slice() : [0.4];
  const hits: Hit[] = [];
  let x = 0;
  let y = 0;
  let lane = (rng() * 2 - 1) * Math.PI;
  let side = rng() < 0.5 ? 1 : -1;
  let heading = lane + side * 0.55;
  let rowLen = 2;
  let rowLeft = rowLen;
  let turns = 0;
  let stepOut = false;
  beats.forEach((time, index) => {
    const energy = sectionAt(music, time).energy;
    let step = (energy >= 0.62 ? 168 : 214) * scale;
    if (stepOut) {
      step *= 1.45;
      stepOut = false;
    }
    const hitX = x + Math.cos(heading) * step;
    const hitY = y + Math.sin(heading) * step;
    rowLeft -= 1;
    const wantTurn = rowLeft <= 0;
    let nextLane = lane;
    let nextSide = -side;
    let nextSpread = energy >= 0.62 ? 0.5 : 0.36;
    if (wantTurn) {
      nextLane = lane + Math.PI / 2;
      turns += 1;
      if (turns % 2 === 0) rowLen = Math.min(4, rowLen + 1);
      rowLeft = rowLen;
      nextSide = side;
      nextSpread = 0.16;
      stepOut = true;
    }
    const departure = nextLane + nextSide * nextSpread;
    hits.push({
      time,
      x: hitX,
      y: hitY,
      heading: departure,
      kind: wantTurn || index % 4 === 0 ? "bar" : "peg",
      side: nextSide,
    });
    x = hitX;
    y = hitY;
    lane = nextLane;
    side = nextSide;
    heading = departure;
  });
  const skin = SKINS[skinName] ?? SKINS.neon;
  const ballR = 26 * scale;
  const camera = bakeCamera(width, height, hits, beats, music.duration, scale, ballR);
  return {
    width,
    height,
    scale,
    hits,
    beats,
    skin,
    light: skinName === "pastel",
    ballR,
    pegR: 20 * scale,
    camX: camera.x,
    camY: camera.y,
    camZ: camera.z,
  };
}

function position(hits: Hit[], time: number): { x: number; y: number; heading: number } {
  if (!hits.length) return { x: 0, y: 0, heading: 0 };
  if (time <= hits[0].time) return hits[0];
  const last = hits[hits.length - 1];
  if (time >= last.time) return last;
  let lo = 0;
  let hi = hits.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (hits[mid].time <= time) lo = mid;
    else hi = mid;
  }
  const a = hits[lo];
  const b = hits[hi];
  const u = clamp((time - a.time) / Math.max(1e-4, b.time - a.time), 0, 1);
  return {
    x: a.x + (b.x - a.x) * u,
    y: a.y + (b.y - a.y) * u,
    heading: Math.atan2(b.y - a.y, b.x - a.x),
  };
}

function cameraTarget(width: number, height: number, hits: Hit[], time: number, scale: number): [number, number, number] {
  const ball = position(hits, time);
  const ahead = position(hits, time + 0.55);
  let minX = Math.min(ball.x, ahead.x);
  let maxX = Math.max(ball.x, ahead.x);
  let minY = Math.min(ball.y, ahead.y);
  let maxY = Math.max(ball.y, ahead.y);
  let count = 0;
  for (const hit of hits) {
    if (hit.time < time - 12 || hit.time > time + 0.8) continue;
    minX = Math.min(minX, hit.x);
    maxX = Math.max(maxX, hit.x);
    minY = Math.min(minY, hit.y);
    maxY = Math.max(maxY, hit.y);
    count += 1;
  }
  const span = Math.max(maxX - minX, maxY - minY, 480 * scale);
  const zoom = clamp((Math.min(width, height) * 0.84) / (span * 1.18), 0.4, 1.05);
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  const lead = count > 4 ? 0.22 : 0.4;
  return [midX * (1 - lead) + ahead.x * lead, midY * (1 - lead) + ahead.y * lead, zoom];
}

function bakeCamera(width: number, height: number, hits: Hit[], beats: number[], duration: number, scale: number, ballR: number) {
  void ballR;
  const n = frameCount(duration);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const z = new Float32Array(n);
  let cx = 0;
  let cy = 0;
  let cz = 1;
  for (let i = 0; i < n; i++) {
    const time = i * BAKE_DT;
    const [tx, ty, tz] = cameraTarget(width, height, hits, time, scale);
    if (i === 0) {
      cx = tx;
      cy = ty;
      cz = tz;
    } else {
      cx = damp(cx, tx, BAKE_DT, 3.4);
      cy = damp(cy, ty, BAKE_DT, 3.4);
      cz = damp(cz, tz, BAKE_DT, tz < cz ? 3.2 : 1.8);
    }
    const push = 0.02 * timesPulse(beats, time, 0.04, 0.12);
    x[i] = cx;
    y[i] = cy;
    z[i] = cz * (1 + push);
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
  const ball = position(state.hits, time);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  const sky = ctx.createLinearGradient(0, 0, 0, state.height);
  sky.addColorStop(0, rgb(...state.skin.bg0));
  sky.addColorStop(1, rgb(...state.skin.bg1));
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, state.width, state.height);
  for (const [ox, oy, tint] of [
    [0.3, 0.35, state.skin.ball],
    [0.72, 0.62, state.skin.peg],
  ] as const) {
    const x = ox * state.width + Math.sin(time * 0.2 + ox) * 12;
    const y = oy * state.height;
    const orb = ctx.createRadialGradient(x, y, 0, x, y, state.width * 0.28);
    orb.addColorStop(0, rgba(tint[0], tint[1], tint[2], state.light ? 0.16 : 0.1 + 0.05 * pulse));
    orb.addColorStop(1, rgba(tint[0], tint[1], tint[2], 0));
    ctx.fillStyle = orb;
    ctx.beginPath();
    ctx.arc(x, y, state.width * 0.28, 0, TAU);
    ctx.fill();
  }
  ctx.save();
  ctx.translate(state.width / 2, state.height / 2);
  ctx.scale(zoom, zoom);
  ctx.translate(-camX, -camY);
  paintPath(ctx, state, time);
  for (const hit of state.hits) {
    const age = time - hit.time;
    if (age < -0.45 || age > 12.4) continue;
    paintObstacle(ctx, state, hit, age, pulse);
  }
  paintTrail(ctx, state, time);
  paintRings(ctx, state, time);
  paintBall(ctx, state, ball.x, ball.y, pulse);
  ctx.restore();
  view.post.glow = state.light ? 0.35 : 0.85 + 0.25 * pulse;
  view.post.grain = state.light ? 0.35 : 0.75;
  view.post.vignette = state.light ? 0.45 : 0.9;
}

function paintPath(ctx: CanvasRenderingContext2D, state: MapState, time: number): void {
  const points: { x: number; y: number; time: number }[] = [];
  for (const hit of state.hits) {
    if (hit.time < time - 12) continue;
    if (hit.time > time) break;
    points.push(hit);
  }
  const live = position(state.hits, time);
  points.push({ x: live.x, y: live.y, time });
  if (points.length < 2) return;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (let index = 1; index < points.length; index++) {
    const a = points[index - 1];
    const b = points[index];
    const age = time - b.time;
    const fade = clamp(1 - age / 12, 0.14, 1);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], (state.light ? 0.28 : 0.55) * fade);
    ctx.lineWidth = 22 * state.scale;
    ctx.stroke();
    ctx.strokeStyle = rgba(1, 1, 1, (state.light ? 0.35 : 0.92) * fade);
    ctx.lineWidth = 4.2 * state.scale;
    ctx.stroke();
  }
}

function paintObstacle(ctx: CanvasRenderingContext2D, state: MapState, hit: Hit, age: number, pulse: number): void {
  const normalX = -Math.sin(hit.heading) * hit.side;
  const normalY = Math.cos(hit.heading) * hit.side;
  const gap = state.ballR + (hit.kind === "peg" ? state.pegR : 8 * state.scale) + 16 * state.scale;
  const x = hit.x + normalX * gap;
  const y = hit.y + normalY * gap;
  const born = age < 0 ? 0.45 * smoothstep(1 + age / 0.35) : 1;
  const fade = age > 10.2 ? smoothstep(1 - (age - 10.2) / 2.2) : 1;
  const flash = age >= 0 && age < 0.28 ? 1 - age / 0.28 : 0;
  const alpha = born * fade;
  if (alpha <= 0.02) return;
  if (hit.kind === "peg") {
    const radius = state.pegR * (1 + 0.14 * flash);
    const halo = ctx.createRadialGradient(x, y, radius * 0.2, x, y, radius * 2.4);
    halo.addColorStop(0, rgba(state.skin.peg[0], state.skin.peg[1], state.skin.peg[2], 0.55 * alpha));
    halo.addColorStop(1, rgba(state.skin.peg[0], state.skin.peg[1], state.skin.peg[2], 0));
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(x, y, radius * 2.4, 0, TAU);
    ctx.fill();
    ctx.fillStyle = rgba(state.skin.peg[0], state.skin.peg[1], state.skin.peg[2], (state.light ? 0.92 : 1) * alpha);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, TAU);
    ctx.fill();
    ctx.strokeStyle = rgba(1, 1, 1, (0.4 + 0.45 * flash) * alpha);
    ctx.lineWidth = 2.2 * state.scale;
    ctx.stroke();
  } else {
    const length = 78 * state.scale;
    const thick = 14 * state.scale;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(hit.heading + Math.PI / 2);
    ctx.fillStyle = rgba(state.skin.bar[0], state.skin.bar[1], state.skin.bar[2], (0.88 + 0.12 * flash) * alpha);
    roundRect(ctx, -length / 2, -thick / 2, length, thick, thick / 2);
    ctx.fill();
    ctx.restore();
  }
  if (flash > 0.05) {
    ctx.strokeStyle = rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], 0.7 * flash);
    ctx.lineWidth = 2 * state.scale;
    ctx.beginPath();
    ctx.arc(hit.x, hit.y, state.ballR + (1 - flash) * 28 * state.scale, 0, TAU);
    ctx.stroke();
  }
  void pulse;
}

function paintTrail(ctx: CanvasRenderingContext2D, state: MapState, time: number): void {
  for (let i = 4; i >= 1; i--) {
    const ghost = position(state.hits, time - i * 0.045);
    const alpha = 0.18 * (1 - i / 5);
    ctx.fillStyle = rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], alpha);
    ctx.beginPath();
    ctx.arc(ghost.x, ghost.y, state.ballR * (1 - i * 0.12), 0, TAU);
    ctx.fill();
  }
}

function paintRings(ctx: CanvasRenderingContext2D, state: MapState, time: number): void {
  for (let i = state.hits.length - 1; i >= 0; i--) {
    const age = time - state.hits[i].time;
    if (age < 0) continue;
    if (age > 0.42) break;
    const fade = 1 - age / 0.42;
    ctx.strokeStyle = rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], 0.75 * fade);
    ctx.lineWidth = (2.4 * fade + 0.6) * state.scale;
    ctx.beginPath();
    ctx.arc(state.hits[i].x, state.hits[i].y, state.ballR + age * 160 * state.scale, 0, TAU);
    ctx.stroke();
  }
}

function paintBall(ctx: CanvasRenderingContext2D, state: MapState, x: number, y: number, pulse: number): void {
  const radius = state.ballR * (1 + 0.2 * pulse);
  const halo = ctx.createRadialGradient(x, y, radius * 0.4, x, y, radius * 3.2);
  halo.addColorStop(0, rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], 0.55));
  halo.addColorStop(1, rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], 0));
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(x, y, radius * 3.2, 0, TAU);
  ctx.fill();
  ctx.fillStyle = rgba(state.skin.ball[0], state.skin.ball[1], state.skin.ball[2], 1);
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, TAU);
  ctx.fill();
  ctx.fillStyle = rgba(1, 1, 1, state.light ? 0.55 : 0.8);
  ctx.beginPath();
  ctx.arc(x - radius * 0.32, y - radius * 0.34, radius * 0.16, 0, TAU);
  ctx.fill();
  ctx.strokeStyle = rgba(1, 1, 1, 0.35 + 0.45 * pulse);
  ctx.lineWidth = 2 * state.scale;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, TAU);
  ctx.stroke();
}
