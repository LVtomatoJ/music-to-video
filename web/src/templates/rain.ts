import type { MusicData, Section } from "../music";
import { clamp, damp, interp, lowerBound, sampleCurve, smoothstep } from "../math";
import { defineTemplate, type ParamSpec, type TemplateContext } from "../template";

const TAU = Math.PI * 2;
const FALL = 0.3;
const MAX_LIVE = 22;

type DropKind = "note" | "beat" | "echo" | "mist";

interface Drop {
  time: number;
  u: number;
  d: number;
  size: number;
  strong: boolean;
  life: number;
  rings: number;
  speed: number;
  gap: number;
  seed: number;
  kind: DropKind;
}

interface Curve {
  times: Float32Array;
  values: Float32Array;
}

export interface RainState {
  width: number;
  height: number;
  unit: number;
  horizon: number;
  duration: number;
  drops: Drop[];
  beats: number[];
  strong: number[];
  energy: Curve;
  wind: Curve;
  streaks: { sx: number; sy: number; depth: number; phase: number }[];
  shoreU: number[];
  shoreH: number[];
  cameraDt: number;
  camX: Float32Array;
  camY: Float32Array;
  camZ: Float32Array;
  maxLife: number;
}

const rainParams = {
  seed: { type: "number", label: "种子", min: 1, max: 999, step: 1, default: 42, scope: "plan" },
  mist: { type: "number", label: "细雨", min: 0, max: 1.6, step: 0.05, default: 1, scope: "plan" },
  ripple: { type: "number", label: "涟漪", min: 0.7, max: 1.5, step: 0.05, default: 1, scope: "plan" },
  gust: {
    type: "enum",
    label: "风",
    default: "natural",
    scope: "plan",
    options: [
      { value: "still", label: "无风" },
      { value: "natural", label: "自然" },
      { value: "gale", label: "一阵一阵" },
    ],
  },
  moon: { type: "number", label: "月亮", min: 0.42, max: 0.84, step: 0.01, default: 0.66, scope: "frame" },
  shine: { type: "number", label: "月光", min: 0.4, max: 1.3, step: 0.05, default: 1, scope: "frame" },
  tint: { type: "color", label: "水色", default: "#b7cfe2", scope: "frame" },
  reflection: { type: "boolean", label: "月光倒影", default: true, scope: "frame" },
} satisfies Record<string, ParamSpec>;

type RainParams = {
  seed: number;
  mist: number;
  ripple: number;
  gust: string;
  moon: number;
  shine: number;
  tint: string;
  reflection: boolean;
};

export const rainTemplate = defineTemplate<RainState, typeof rainParams>({
  meta: {
    id: "rain",
    name: "雨夜水面",
    author: "music-to-video",
    version: "1.0.0",
    description: "月光下的水面。音头落成雨滴，涟漪顺着拍子荡开。",
  },
  tier: "2d",
  params: rainParams,
  setup(ctx, music, params) {
    const view = as2d(ctx);
    return buildRain(view.width, view.height, music, params as RainParams, view.rng);
  },
  render(ctx, t, state, music, params) {
    const view = as2d(ctx);
    paintRain(view.ctx, state, music, params as RainParams, t, view);
  },
});

function as2d(ctx: TemplateContext) {
  if (ctx.tier !== "2d") throw new Error("雨夜水面用 2D 这一层画。");
  return ctx;
}

function buildRain(
  width: number,
  height: number,
  music: MusicData,
  params: RainParams,
  rng: () => number,
): RainState {
  const unit = width / 1920;
  const beats = music.beats.slice();
  const strong = (music.extensions?.strongBeats?.length ? music.extensions.strongBeats : music.downbeats).slice();
  const strongSet = new Set(strong.map((beat) => beat.toFixed(4)));
  const intervals: number[] = [];
  for (let i = 1; i < beats.length; i++) intervals.push(beats[i] - beats[i - 1]);
  intervals.sort((a, b) => a - b);
  const period = clamp(intervals.length ? intervals[intervals.length >> 1] : 0.6, 0.3, 1.2);
  const energy = {
    times: Float32Array.from(music.loudness.times),
    values: Float32Array.from(music.loudness.values),
  };
  const events: { time: number; value: number; strong: boolean; kind: DropKind }[] = [];
  for (const onset of music.onsets) events.push({ time: onset.time, value: onset.strength, strong: false, kind: "note" });
  for (const beat of beats) {
    const isStrong = strongSet.has(beat.toFixed(4));
    const near = events.findIndex((item) => Math.abs(item.time - beat) < 0.07);
    if (near >= 0) {
      const prev = events[near];
      events[near] = { time: beat, value: Math.max(prev.value, 0.5), strong: isStrong, kind: "beat" };
    } else {
      const level = interp(energy.times, energy.values, beat);
      events.push({ time: beat, value: 0.35 + 0.4 * level, strong: isStrong, kind: "beat" });
    }
  }
  events.sort((a, b) => a.time - b.time);
  const merged: typeof events = [];
  for (const item of events) {
    const prev = merged[merged.length - 1];
    if (prev && item.time - prev.time < 0.045) {
      prev.value = Math.max(prev.value, item.value);
      prev.strong = prev.strong || item.strong;
      continue;
    }
    merged.push({ ...item });
  }

  const drops: Drop[] = [];
  const speedBase = 96 * unit * Math.sqrt(0.62 / period) * params.ripple;
  const gap = clamp(period * 0.22, 0.11, 0.24);
  for (const event of merged) {
    let size = 0.55 + 0.95 * event.value + (event.strong ? 0.45 : 0);
    size = Math.min(1.95, size);
    const spot = place(drops, event.time, rng);
    const rings = 3 + (size > 1.15 ? 1 : 0) + (event.strong ? 1 : 0);
    const life = 2.6 + 1.6 * Math.min(1, size / 1.6);
    drops.push({
      time: event.time,
      u: spot.u,
      d: spot.d,
      size,
      strong: event.strong,
      life,
      rings,
      speed: speedBase * (0.8 + 0.35 * size),
      gap,
      seed: rng(),
      kind: event.kind,
    });
    if (event.strong && size > 1.3) {
      const back = 0.16 + 0.08 * rng();
      const ang = rng() * TAU;
      drops.push({
        time: event.time + back,
        u: clamp(spot.u + Math.cos(ang) * 0.035, 0.04, 0.96),
        d: clamp(spot.d + Math.sin(ang) * 0.05, 0.08, 0.98),
        size: 0.42,
        strong: false,
        life: 1.3,
        rings: 2,
        speed: speedBase * 0.8,
        gap,
        seed: rng(),
        kind: "echo",
      });
    }
  }
  drops.sort((a, b) => a.time - b.time);
  if (params.mist > 0.02) {
    let when = 0.15;
    while (when < music.duration - 0.2) {
      const level = interp(energy.times, energy.values, when);
      const rate = (1.6 + 5 * level) * params.mist;
      when += -Math.log(Math.max(1e-4, rng())) / Math.max(0.2, rate);
      if (when >= music.duration - 0.2) break;
      const spot = place(drops, when, rng);
      insertDrop(drops, {
        time: when,
        u: spot.u,
        d: spot.d,
        size: 0.26 + 0.14 * rng(),
        strong: false,
        life: 1.5 + 0.6 * rng(),
        rings: 2,
        speed: speedBase * 0.72,
        gap: gap * 0.8,
        seed: rng(),
        kind: "mist",
      });
    }
  }
  for (let index = 0; index < drops.length; index++) {
    const drop = drops[index];
    let later = 0;
    const end = Math.min(drops.length, index + 120);
    for (let j = index + 1; j < end; j++) {
      const other = drops[j];
      if (other.time < drop.time + drop.life && other.kind !== "mist") later++;
    }
    if (later > MAX_LIVE * 0.6) drop.life *= Math.max(0.55, (MAX_LIVE * 0.6) / later);
  }

  const gust = gustAmounts(params.gust);
  const wind = buildWind(music.duration, strong, music.sections, rng, gust.amp, gust.base);
  const streaks = [];
  for (let i = 0; i < 260; i++) {
    streaks.push({ sx: rng(), sy: rng(), depth: 0.6 + 0.8 * rng(), phase: rng() });
  }
  const shoreU: number[] = [];
  const shoreH: number[] = [];
  const phase = rng() * 10;
  for (let x = 0; x <= 1.02; x += 0.006) {
    shoreU.push(x);
    shoreH.push(
      0.5 +
        0.28 * Math.sin(x * 7.1 + phase) +
        0.18 * Math.sin(x * 17.3 + phase * 1.7) +
        0.08 * Math.sin(x * 41),
    );
  }
  const camera = bakeCamera(music.duration, unit, beats, strong, energy, wind);
  let maxLife = 1;
  for (const drop of drops) maxLife = Math.max(maxLife, drop.life);
  return {
    width,
    height,
    unit,
    horizon: height * 0.27,
    duration: music.duration,
    drops,
    beats,
    strong,
    energy,
    wind,
    streaks,
    shoreU,
    shoreH,
    cameraDt: camera.dt,
    camX: camera.camX,
    camY: camera.camY,
    camZ: camera.camZ,
    maxLife,
  };
}

function gustAmounts(gust: string): { amp: number; base: number } {
  if (gust === "still") return { amp: 0, base: 0 };
  if (gust === "gale") return { amp: 0.4, base: 0.16 };
  return { amp: 0.22, base: 0.1 };
}

function place(drops: Drop[], when: number, rng: () => number): { u: number; d: number } {
  const start = Math.max(0, drops.length - 40);
  const recent: Drop[] = [];
  for (let i = start; i < drops.length; i++) {
    if (when - drops[i].time < 2.6) recent.push(drops[i]);
  }
  let bestU = 0.5;
  let bestD = 0.6;
  let bestScore = -1e9;
  for (let k = 0; k < 28; k++) {
    const d = 0.08 + 0.74 * rng() ** 0.8;
    const margin = 0.05 + 0.07 * d;
    const u = margin + (1 - 2 * margin) * rng();
    let score = 0;
    for (const drop of recent) {
      const weight = Math.exp(-(when - drop.time) * 1.1);
      const du = (u - drop.u) * 1.9;
      const dd = d - drop.d;
      const dist = Math.hypot(du, dd);
      score -= weight / (dist * dist + 0.004);
    }
    score -= 0.6 * Math.abs(u - 0.55) + 0.35 * Math.abs(d - 0.55);
    score += 0.08 * rng();
    if (score > bestScore) {
      bestScore = score;
      bestU = u;
      bestD = d;
    }
  }
  return { u: bestU, d: bestD };
}

function insertDrop(drops: Drop[], drop: Drop): void {
  let lo = 0;
  let hi = drops.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (drops[mid].time <= drop.time) lo = mid + 1;
    else hi = mid;
  }
  drops.splice(lo, 0, drop);
}

function buildWind(
  duration: number,
  strong: number[],
  sections: Section[],
  rng: () => number,
  gustAmp: number,
  baseAmp: number,
): Curve {
  const safe = Math.max(duration, 1);
  const n = Math.floor(safe * 20) + 2;
  const times = new Float32Array(n);
  const values = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = (safe * i) / (n - 1);
    times[i] = t;
    values[i] = baseAmp * (1 + 0.5 * Math.sin(t * 0.21));
  }
  const gusts: number[] = [];
  let last = -99;
  strong.forEach((hit, index) => {
    if (hit - last >= 5 && index % 3 === 0) {
      gusts.push(hit);
      last = hit;
    }
  });
  let prev: Section | null = null;
  for (const section of sections) {
    if (prev && section.energy > prev.energy * 1.25 && section.start - last >= 2.5) {
      gusts.push(section.start);
      last = section.start;
    }
    prev = section;
  }
  for (const hit of gusts) {
    const sign = rng() > 0.35 ? 1 : -1;
    for (let i = 0; i < n; i++) {
      const age = times[i] - hit;
      if (age < 0) continue;
      const shape = (1 - Math.exp(-age * 5)) * Math.exp(-age * 0.7);
      values[i] += sign * gustAmp * shape;
    }
  }
  return { times, values };
}

function bakeCamera(
  duration: number,
  unit: number,
  beats: number[],
  strong: number[],
  energy: Curve,
  wind: Curve,
): { dt: number; camX: Float32Array; camY: Float32Array; camZ: Float32Array } {
  const dt = 1 / 60;
  const n = Math.ceil(Math.max(duration, 1) / dt) + 2;
  const camX = new Float32Array(n);
  const camY = new Float32Array(n);
  const camZ = new Float32Array(n);
  let x = 0;
  let y = 0;
  let z = 1;
  for (let i = 0; i < n; i++) {
    const time = i * dt;
    const push = beatPush(beats, time);
    const glow = beatGlow(strong, time);
    const level = interp(energy.times, energy.values, time);
    const slant = interp(wind.times, wind.values, time);
    const tx = 14 * unit * Math.sin(time * 0.07) + slant * 10 * unit;
    const ty = 6 * unit * Math.cos(time * 0.05);
    x = damp(x, tx, dt, 0.9);
    y = damp(y, ty, dt, 0.9);
    z = damp(z, 1 + 0.018 * level, dt, 1.2);
    camX[i] = x;
    camY[i] = y;
    camZ[i] = z + 0.008 * push + 0.01 * glow;
  }
  return { dt, camX, camY, camZ };
}

function beatGlow(hits: number[], time: number): number {
  return windowed(hits, time, 0.05, 1.4, (age) => (age < 0 ? smoothstep((age + 0.05) / 0.05) : Math.exp(-age * 2.6)));
}

function beatPush(hits: number[], time: number): number {
  return windowed(hits, time, 0, 0.9, (age) => Math.exp(-age * 5) * (1 - age / 0.9));
}

function windowed(hits: number[], time: number, pre: number, post: number, falloff: (age: number) => number): number {
  const start = lowerBound(hits, time - post);
  let level = 0;
  for (let i = start; i < hits.length; i++) {
    const age = time - hits[i];
    if (age < -pre) break;
    level = Math.max(level, falloff(age));
  }
  return level;
}

function project(state: RainState, u: number, d: number): { x: number; y: number; scale: number; aspect: number } {
  const depth = 0.16 + 0.84 * d;
  const y = state.horizon + (state.height - state.horizon) * d ** 1.35 * 1.04;
  const scale = 0.32 + depth;
  const x = state.width * 0.5 + (u - 0.5) * state.width * (0.78 + 0.5 * d);
  const aspect = 0.22 + 0.24 * d;
  return { x, y, scale, aspect };
}

function liveDrops(state: RainState, time: number): Drop[] {
  const drops = state.drops;
  const target = time - state.maxLife - 0.05;
  let lo = 0;
  let hi = drops.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (drops[mid].time < target) lo = mid + 1;
    else hi = mid;
  }
  const out: Drop[] = [];
  for (let i = lo; i < drops.length; i++) {
    const drop = drops[i];
    if (drop.time - FALL > time) break;
    if (time - drop.time <= drop.life) out.push(drop);
  }
  return out;
}

function paintRain(
  ctx: CanvasRenderingContext2D,
  state: RainState,
  music: MusicData,
  params: RainParams,
  t: number,
  view: { post: { glow: number; grain: number; vignette: number } },
): void {
  const time = clamp(t, 0, Math.max(music.duration, 0.001));
  const energy = interp(state.energy.times, state.energy.values, time);
  const glow = beatGlow(state.strong, time);
  const live = liveDrops(state, time);
  const byDepth = live.slice().sort((a, b) => a.d - b.d);
  const zoom = sampleCurve(state.camZ, state.cameraDt, time);
  const camX = sampleCurve(state.camX, state.cameraDt, time);
  const camY = sampleCurve(state.camY, state.cameraDt, time);
  const tint = parseHex(params.tint);
  const w = state.width;
  const h = state.height;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.filter = "none";
  ctx.fillStyle = "#02040a";
  ctx.fillRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.scale(zoom, zoom);
  ctx.translate(-w / 2 + camX, -h / 2 + camY);
  paintSky(ctx, state, params, time, glow);
  paintWater(ctx, state, params, energy, glow, tint);
  paintSwell(ctx, state, time, energy);
  if (params.reflection) paintMoonPath(ctx, state, params, byDepth, time, energy, glow);
  ctx.globalCompositeOperation = "lighter";
  for (const drop of byDepth) paintRipples(ctx, state, drop, time, glow, tint);
  ctx.globalCompositeOperation = "source-over";
  for (const drop of byDepth) paintSplash(ctx, state, drop, time);
  paintStreaks(ctx, state, time, energy);
  for (const drop of byDepth) paintFalling(ctx, state, drop, time);
  ctx.restore();
  view.post.glow = (0.8 + 0.5 * glow) * params.shine;
  view.post.grain = 1;
  view.post.vignette = 1;
}

function moonXy(state: RainState, params: RainParams): [number, number] {
  return [state.width * params.moon, state.horizon * 0.4];
}

function paintSky(
  ctx: CanvasRenderingContext2D,
  state: RainState,
  params: RainParams,
  time: number,
  glow: number,
): void {
  const w = state.width;
  const hz = state.horizon;
  const shine = params.shine;
  const sky = ctx.createLinearGradient(0, 0, 0, hz);
  sky.addColorStop(0, rgb(0.02, 0.032, 0.075));
  sky.addColorStop(0.65, rgb(0.05, 0.085, 0.15));
  sky.addColorStop(1, rgb(0.09, 0.14, 0.205));
  ctx.fillStyle = sky;
  ctx.fillRect(-80, -80, w + 160, hz + 90);
  const [mx, my] = moonXy(state, params);
  const haloR = 420 * state.unit;
  const halo = ctx.createRadialGradient(mx, my, 0, mx, my, haloR);
  halo.addColorStop(0, rgba(0.7, 0.82, 0.95, (0.3 + 0.1 * glow) * shine));
  halo.addColorStop(0.12, rgba(0.45, 0.62, 0.82, (0.16 + 0.06 * glow) * shine));
  halo.addColorStop(0.45, rgba(0.2, 0.32, 0.5, 0.06 * shine));
  halo.addColorStop(1, rgba(0.1, 0.2, 0.3, 0));
  ctx.fillStyle = halo;
  ctx.fillRect(-80, -80, w + 160, hz + 90);
  const discR = 34 * state.unit;
  const disc = ctx.createRadialGradient(mx - 6 * state.unit, my - 6 * state.unit, 2 * state.unit, mx, my, discR);
  disc.addColorStop(0, rgba(1, 0.99, 0.95, 0.98));
  disc.addColorStop(0.8, rgba(0.88, 0.93, 1, 0.92));
  disc.addColorStop(1, rgba(0.8, 0.88, 1, 0));
  ctx.fillStyle = disc;
  ctx.beginPath();
  ctx.arc(mx, my, discR, 0, TAU);
  ctx.fill();
  for (let index = 0; index < 4; index++) {
    const radius = (260 + index * 40) * state.unit;
    const cx = mx - 520 * state.unit + ((time * (9 + index * 3) + index * 380) * state.unit) % (1100 * state.unit);
    const cy = my - 30 * state.unit + index * 26 * state.unit;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(1, 0.16);
    const veil = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
    veil.addColorStop(0, rgba(0.3, 0.38, 0.5, 0.16));
    veil.addColorStop(1, rgba(0.3, 0.38, 0.5, 0));
    ctx.fillStyle = veil;
    ctx.beginPath();
    ctx.arc(0, 0, radius, 0, TAU);
    ctx.fill();
    ctx.restore();
  }
  ctx.beginPath();
  ctx.moveTo(-40, hz + 2);
  for (let i = 0; i < state.shoreU.length; i++) {
    ctx.lineTo(-40 + state.shoreU[i] * (w + 80), hz - 6 * state.unit - state.shoreH[i] * 34 * state.unit);
  }
  ctx.lineTo(w + 40, hz + 2);
  ctx.closePath();
  ctx.fillStyle = rgb(0.018, 0.03, 0.052);
  ctx.fill();
}

function paintWater(
  ctx: CanvasRenderingContext2D,
  state: RainState,
  params: RainParams,
  energy: number,
  glow: number,
  tint: [number, number, number],
): void {
  const w = state.width;
  const h = state.height;
  const hz = state.horizon;
  const lift = 0.026 * glow + 0.012 * energy;
  const water = ctx.createLinearGradient(0, hz, 0, h);
  const c0 = mix(0.08 + lift, 0.135 + lift, 0.18 + lift, tint, 0.18);
  const c1 = mix(0.036 + lift, 0.088 + lift, 0.12 + lift, tint, 0.22);
  const c2 = mix(0.016 + lift * 0.5, 0.05 + lift, 0.075 + lift, tint, 0.22);
  const c3 = mix(0.006, 0.022, 0.036, tint, 0.16);
  water.addColorStop(0, rgb(...c0));
  water.addColorStop(0.18, rgb(...c1));
  water.addColorStop(0.6, rgb(...c2));
  water.addColorStop(1, rgb(...c3));
  ctx.fillStyle = water;
  ctx.fillRect(-80, hz, w + 160, h - hz + 80);
  ctx.beginPath();
  ctx.moveTo(-40, hz);
  for (let i = 0; i < state.shoreU.length; i++) {
    ctx.lineTo(-40 + state.shoreU[i] * (w + 80), hz + 4 * state.unit + state.shoreH[i] * 20 * state.unit);
  }
  ctx.lineTo(w + 40, hz);
  ctx.closePath();
  ctx.fillStyle = rgba(0.01, 0.02, 0.036, 0.55);
  ctx.fill();
  const [mx] = moonXy(state, params);
  ctx.save();
  ctx.translate(mx, hz + (h - hz) * 0.42);
  ctx.scale(1, 0.55);
  const radius = 760 * state.unit;
  const sheen = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
  const alpha = (0.12 + 0.17 * glow + 0.06 * energy) * params.shine;
  const s0 = mix(0.4, 0.58, 0.7, tint, 0.35);
  const s1 = mix(0.2, 0.36, 0.48, tint, 0.35);
  sheen.addColorStop(0, rgba(s0[0], s0[1], s0[2], alpha));
  sheen.addColorStop(0.5, rgba(s1[0], s1[1], s1[2], alpha * 0.4));
  sheen.addColorStop(1, rgba(0.1, 0.2, 0.3, 0));
  ctx.fillStyle = sheen;
  ctx.beginPath();
  ctx.arc(0, 0, radius, 0, TAU);
  ctx.fill();
  ctx.restore();
}

function paintSwell(ctx: CanvasRenderingContext2D, state: RainState, time: number, energy: number): void {
  const w = state.width;
  const h = state.height;
  const hz = state.horizon;
  const unit = state.unit;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineCap = "round";
  const lines = 34;
  for (let i = 0; i < lines; i++) {
    const phase = ((i + time * 0.05) / lines) % 1;
    const d = phase ** 1.6;
    const y = hz + (h - hz) * d * 1.04 + 2;
    let seed = Math.sin(i * 91.7) * 1000;
    seed -= Math.floor(seed);
    const alpha = 0.035 * Math.sin(phase * Math.PI) * (0.7 + 0.5 * energy);
    ctx.lineWidth = (0.8 + 2.4 * d) * unit;
    const segs = 3 + Math.floor(seed * 3);
    for (let k = 0; k < segs; k++) {
      const cx = -40 + (w + 80) * fract(seed * 3.7 + k / segs + 0.02 * Math.sin(time * 0.3 + i));
      const half = (80 + 260 * d) * unit * (0.5 + 0.5 * ((seed * 13.1 + k * 0.37) % 1));
      const gradient = ctx.createLinearGradient(cx - half, 0, cx + half, 0);
      gradient.addColorStop(0, rgba(0.4, 0.6, 0.7, 0));
      gradient.addColorStop(0.5, rgba(0.4, 0.6, 0.7, alpha));
      gradient.addColorStop(1, rgba(0.4, 0.6, 0.7, 0));
      ctx.strokeStyle = gradient;
      const wob = 3 * d * unit * Math.sin(time * 0.8 + k + i);
      ctx.beginPath();
      ctx.moveTo(cx - half, y + wob);
      ctx.bezierCurveTo(cx - half * 0.3, y - 2 * d * unit + wob, cx + half * 0.3, y + 2 * d * unit - wob, cx + half, y - wob);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function displacement(state: RainState, live: Drop[], x: number, y: number, time: number): number {
  let shove = 0;
  for (const drop of live) {
    const age = time - drop.time;
    if (age <= 0) continue;
    const point = project(state, drop.u, drop.d);
    const dx = (x - point.x) / point.scale;
    const dy = (y - point.y) / (point.scale * point.aspect);
    const dist = Math.hypot(dx, dy);
    const front = drop.speed * age;
    if (dist > front + 30 * state.unit || dist < 1e-3) continue;
    const fade = Math.max(0, 1 - age / drop.life);
    const wave = Math.sin(((front - dist) * 0.16) / state.unit) * Math.exp(-Math.max(0, front - dist) / (70 * state.unit));
    shove += wave * drop.size * 9 * state.unit * fade * (dx / dist) * point.scale;
  }
  return clamp(shove, -26 * state.unit, 26 * state.unit);
}

function paintMoonPath(
  ctx: CanvasRenderingContext2D,
  state: RainState,
  params: RainParams,
  live: Drop[],
  time: number,
  energy: number,
  glow: number,
): void {
  const [mx] = moonXy(state, params);
  const hz = state.horizon;
  const h = state.height;
  const unit = state.unit;
  let y = hz + 3;
  let row = 0;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineCap = "round";
  while (y < h + 60 * unit) {
    const d = clamp((y - hz) / (h - hz), 0, 1);
    const spread = (26 + 250 * d) * unit;
    const step = (2.4 + 9 * d) * unit;
    for (let k = 0; k < 3; k++) {
      const seed = row * 7.31 + k * 3.17;
      let off = Math.sin(seed * 12.9898) * 43758.5453;
      off = (off - Math.floor(off)) * 2 - 1;
      let cx = mx + off * spread * 0.8 + Math.sin(time * 0.9 + seed) * 6 * unit * (0.4 + d);
      cx += displacement(state, live, cx, y, time);
      const length = (10 + 70 * d) * unit * (0.5 + 0.5 * Math.abs(Math.sin(time * 1.3 + seed * 1.7)));
      const fall = 1 - Math.min(1, Math.abs(cx - mx) / (spread + 1));
      const alpha = (0.05 + 0.2 * (1 - d) ** 0.6) * fall * (0.8 + 0.65 * glow + 0.25 * energy) * params.shine;
      if (alpha < 0.008) continue;
      ctx.strokeStyle = rgba(0.62, 0.72, 0.82, alpha);
      ctx.lineWidth = (0.9 + 2.4 * d) * unit;
      ctx.beginPath();
      ctx.moveTo(cx - length / 2, y);
      ctx.lineTo(cx + length / 2, y);
      ctx.stroke();
    }
    y += step;
    row++;
  }
  ctx.restore();
}

function ringAlpha(drop: Drop, age: number, k: number): number {
  const local = age - k * drop.gap;
  if (local <= 0) return 0;
  const life = drop.life - k * drop.gap * 0.6;
  if (local >= life) return 0;
  return smoothstep(local / 0.08) * (1 - local / life) ** 1.6 * (1 - 0.16 * k);
}

function paintRipples(
  ctx: CanvasRenderingContext2D,
  state: RainState,
  drop: Drop,
  time: number,
  glow: number,
  tint: [number, number, number],
): void {
  const age = time - drop.time;
  if (age <= 0) return;
  const point = project(state, drop.u, drop.d);
  let bright = (0.58 + 0.4 * Math.min(1, drop.size / 1.5)) * (1 + 0.3 * glow);
  if (drop.kind === "mist") bright *= 0.55;
  const unit = state.unit;
  for (let k = 0; k < drop.rings; k++) {
    const alpha = ringAlpha(drop, age, k) * bright;
    if (alpha < 0.01) continue;
    const local = age - k * drop.gap;
    let radius = drop.speed * local * (1 - 0.1 * k) + 2 * unit;
    radius *= 0.55 + 0.45 * drop.size;
    const width = ((1.1 + 1.6 * drop.size) * (1 + local * 0.6)) / (1 + 0.4 * k) * unit;
    if (radius < 0.5) continue;
    ctx.save();
    ctx.translate(point.x, point.y);
    ctx.scale(point.scale, point.scale * point.aspect);
    const far = mix(0.78, 0.88, 0.98, tint, 0.28);
    const mid = mix(0.42, 0.66, 0.78, tint, 0.4);
    const near = mix(0.3, 0.55, 0.66, tint, 0.4);
    const gradient = ctx.createLinearGradient(0, -radius, 0, radius);
    gradient.addColorStop(0, rgba(far[0], far[1], far[2], alpha));
    gradient.addColorStop(0.5, rgba(mid[0], mid[1], mid[2], alpha * 0.55));
    gradient.addColorStop(1, rgba(near[0], near[1], near[2], alpha * 0.32));
    ctx.strokeStyle = gradient;
    ctx.lineWidth = Math.max(0.4, width);
    ctx.beginPath();
    ctx.arc(0, 0, radius, 0, TAU);
    ctx.stroke();
    if (radius > width * 2.5) {
      const soft = ctx.createLinearGradient(0, -radius, 0, radius);
      const softC = mix(0.4, 0.58, 0.72, tint, 0.35);
      soft.addColorStop(0, rgba(softC[0], softC[1], softC[2], alpha * 0.2));
      soft.addColorStop(1, rgba(0.2, 0.4, 0.52, alpha * 0.08));
      ctx.strokeStyle = soft;
      ctx.lineWidth = width * 3.4;
      ctx.beginPath();
      ctx.arc(0, 0, Math.max(0.5, radius - width * 2), 0, TAU);
      ctx.stroke();
    }
    ctx.restore();
  }
}

function paintSplash(ctx: CanvasRenderingContext2D, state: RainState, drop: Drop, time: number): void {
  const age = time - drop.time;
  if (age < 0 || age > 0.7) return;
  const point = project(state, drop.u, drop.d);
  const s = point.scale * (0.55 + 0.45 * drop.size);
  const unit = state.unit;
  const flash = Math.exp(-age * 9) * (drop.kind === "mist" ? 0.35 : 1);
  if (flash > 0.02) {
    ctx.save();
    ctx.translate(point.x, point.y);
    ctx.scale(1, 0.5 + point.aspect);
    const radius = 40 * unit * s * (1 + 0.7 * (drop.strong ? 1 : 0));
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, radius);
    gradient.addColorStop(0, rgba(0.95, 0.98, 1, 0.95 * flash));
    gradient.addColorStop(0.18, rgba(0.7, 0.88, 1, 0.55 * flash));
    gradient.addColorStop(0.5, rgba(0.45, 0.68, 0.85, 0.18 * flash));
    gradient.addColorStop(1, rgba(0.3, 0.5, 0.7, 0));
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(0, 0, Math.max(0.5, radius), 0, TAU);
    ctx.fill();
    ctx.restore();
  }
  if (drop.kind !== "mist" && age < 0.32) {
    const u = age / 0.32;
    ctx.save();
    ctx.translate(point.x, point.y);
    ctx.scale(s, s * point.aspect);
    ctx.strokeStyle = rgba(0.85, 0.95, 1, 0.75 * (1 - u) ** 1.5);
    ctx.lineWidth = 2.6 * (1 - u) + 0.6;
    ctx.beginPath();
    ctx.arc(0, 0, Math.max(0.5, (6 + 70 * (1 - (1 - u) ** 2.2)) * unit), 0, TAU);
    ctx.stroke();
    ctx.restore();
  }
  const count = drop.kind === "echo" || drop.kind === "mist" ? 0 : 3 + Math.floor(drop.size * 2.2) + (drop.strong ? 3 : 0);
  const gravity = 1500 * unit;
  for (let i = 0; i < count; i++) {
    let jitter = Math.sin((i + 1) * 12.9898 + drop.seed * 78.233) * 43758.5453;
    jitter -= Math.floor(jitter);
    const ang = drop.seed * TAU + i * (TAU / Math.max(1, count)) + 0.9 * (jitter - 0.5);
    const speed = (60 + 90 * jitter) * unit * s;
    const up = (150 + 160 * ((jitter * 7.13 + drop.seed) % 1)) * unit * s * (drop.strong ? 1.2 : 1);
    const lift = up * age - 0.5 * gravity * s * age * age;
    if (lift < 0 && age > 0.05) continue;
    const px = point.x + Math.cos(ang) * speed * age;
    const py = point.y + Math.sin(ang) * speed * age * point.aspect - lift;
    const alpha = (1 - age / 0.7) * (0.45 + 0.4 * jitter);
    ctx.fillStyle = rgba(0.86, 0.94, 1, alpha);
    ctx.beginPath();
    ctx.arc(px, py, Math.max(0.4, (0.7 + 0.8 * jitter + 0.4 * drop.size) * point.scale * unit), 0, TAU);
    ctx.fill();
  }
  if (drop.strong && drop.size > 1.2 && age > 0.05 && age < 0.42) {
    const u = (age - 0.05) / 0.37;
    const jet = Math.sin(u * Math.PI) * 34 * unit * s;
    const jw = 2.4 * unit * s;
    const gradient = ctx.createLinearGradient(point.x, point.y - jet, point.x, point.y);
    gradient.addColorStop(0, rgba(0.95, 0.98, 1, 0.9 * (1 - u * 0.5)));
    gradient.addColorStop(1, rgba(0.55, 0.75, 0.9, 0));
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(point.x - jw, point.y);
    ctx.bezierCurveTo(
      point.x - jw * 0.6,
      point.y - jet * 0.6,
      point.x - jw * 0.4,
      point.y - jet,
      point.x,
      point.y - jet - jw,
    );
    ctx.bezierCurveTo(
      point.x + jw * 0.4,
      point.y - jet,
      point.x + jw * 0.6,
      point.y - jet * 0.6,
      point.x + jw,
      point.y,
    );
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = rgba(0.95, 0.98, 1, 0.8 * (1 - u));
    ctx.beginPath();
    ctx.arc(point.x, point.y - jet - jw * 2.2, Math.max(0.4, jw * 0.9), 0, TAU);
    ctx.fill();
  }
}

function paintFalling(ctx: CanvasRenderingContext2D, state: RainState, drop: Drop, time: number): void {
  const age = time - drop.time;
  if (age >= 0 || age < -FALL || drop.kind === "mist") return;
  const point = project(state, drop.u, drop.d);
  const u = (age + FALL) / FALL;
  const height = 520 * state.unit * point.scale;
  const slant = interp(state.wind.times, state.wind.values, drop.time);
  const lift = height * (1 - u) ** 1.25;
  const hx = point.x + slant * lift;
  const hy = point.y - lift;
  const length = (46 + 30 * drop.size) * state.unit * point.scale * (0.6 + 0.4 * u);
  const tx = hx + slant * length;
  const ty = hy - length;
  const alpha = smoothstep(u / 0.35) * (0.55 + 0.3 * Math.min(1, drop.size));
  const gradient = ctx.createLinearGradient(tx, ty, hx, hy);
  gradient.addColorStop(0, rgba(0.75, 0.86, 1, 0));
  gradient.addColorStop(1, rgba(0.92, 0.97, 1, alpha));
  ctx.strokeStyle = gradient;
  ctx.lineCap = "round";
  ctx.lineWidth = (1.3 + 0.9 * drop.size) * point.scale * state.unit;
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(hx, hy);
  ctx.stroke();
  const ry = point.y + lift * point.aspect * 0.9;
  const rl = length * point.aspect * 0.9;
  const reflection = ctx.createLinearGradient(hx, ry + rl, hx, ry);
  reflection.addColorStop(0, rgba(0.5, 0.7, 0.85, 0));
  reflection.addColorStop(1, rgba(0.6, 0.8, 0.95, alpha * 0.35));
  ctx.strokeStyle = reflection;
  ctx.lineWidth = (1 + 0.6 * drop.size) * point.scale * state.unit;
  ctx.beginPath();
  ctx.moveTo(hx, ry + rl);
  ctx.lineTo(hx, ry);
  ctx.stroke();
}

function paintStreaks(ctx: CanvasRenderingContext2D, state: RainState, time: number, energy: number): void {
  const slant = interp(state.wind.times, state.wind.values, time);
  const count = Math.floor(60 + 200 * energy);
  const unit = state.unit;
  const span = state.height + 260 * unit;
  ctx.lineCap = "round";
  for (let index = 0; index < count && index < state.streaks.length; index++) {
    const streak = state.streaks[index];
    const speed = (900 + 700 * streak.depth) * unit;
    const yy = (streak.sy * span + time * speed) % span - 130 * unit;
    const xx = streak.sx * (state.width + 400 * unit) - 200 * unit + slant * (yy - state.height * 0.5) + streak.phase * 3 * unit;
    const length = (22 + 30 * streak.depth) * unit * (0.8 + 0.5 * energy);
    const alpha = (0.03 + 0.065 * streak.depth) * (0.75 + 0.7 * energy);
    ctx.strokeStyle = rgba(0.7, 0.8, 0.92, alpha);
    ctx.lineWidth = (0.6 + 0.6 * streak.depth) * unit;
    ctx.beginPath();
    ctx.moveTo(xx, yy);
    ctx.lineTo(xx - slant * length, yy - length);
    ctx.stroke();
  }
}

function fract(value: number): number {
  return value - Math.floor(value);
}

function rgb(r: number, g: number, b: number): string {
  return `rgb(${Math.round(clamp(r, 0, 1) * 255)}, ${Math.round(clamp(g, 0, 1) * 255)}, ${Math.round(clamp(b, 0, 1) * 255)})`;
}

function rgba(r: number, g: number, b: number, a: number): string {
  return `rgba(${Math.round(clamp(r, 0, 1) * 255)}, ${Math.round(clamp(g, 0, 1) * 255)}, ${Math.round(clamp(b, 0, 1) * 255)}, ${clamp(a, 0, 1)})`;
}

function mix(r: number, g: number, b: number, tint: [number, number, number], amount: number): [number, number, number] {
  return [r + (tint[0] - r) * amount, g + (tint[1] - g) * amount, b + (tint[2] - b) * amount];
}

function parseHex(hex: string): [number, number, number] {
  const raw = hex.replace("#", "").trim();
  const full = raw.length === 3 ? raw.split("").map((c) => c + c).join("") : raw;
  const value = Number.parseInt(full, 16);
  if (!Number.isFinite(value) || full.length !== 6) return [0.72, 0.81, 0.89];
  return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
}
