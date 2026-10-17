import type { MusicData, Section } from "../music";
import { clamp, damp, interp, roundRect, sampleCurve, smoothstep, timesPulse } from "../math";
import { defineTemplate, type ParamSpec } from "../template";
import { BAKE_DT, as2d, frameCount, rgb, rgba } from "./shared";

interface Building {
  born: number;
  grow: number;
  x: number;
  width: number;
  target: number;
  energy: number;
  floors: number;
  cols: number;
  tower: boolean;
  roof: number;
  tint: number;
  sign: number;
}

interface WindowLight {
  time: number;
  index: number;
  col: number;
  row: number;
}

interface Block {
  x: number;
  width: number;
  height: number;
  roof: number;
}

interface CityState {
  width: number;
  height: number;
  unit: number;
  buildings: Building[];
  lights: WindowLight[][];
  bass: number[];
  beats: number[];
  far: Block[];
  mid: Block[];
  stars: { x: number; y: number; r: number; warmth: number; phase: number }[];
  sections: Section[];
  camX: Float32Array;
  camZ: Float32Array;
}

const NIGHT = [
  [0.09, 0.11, 0.2],
  [0.14, 0.1, 0.22],
  [0.07, 0.09, 0.14],
  [0.16, 0.1, 0.16],
  [0.08, 0.13, 0.17],
] as const;

const ACCENT: [number, number, number] = [1, 0.78, 0.42];
const ACCENT2: [number, number, number] = [0.49, 0.855, 0.965];

const cityParams = {
  seed: { type: "number", label: "种子", min: 1, max: 999, step: 1, default: 42, scope: "plan" },
} satisfies Record<string, ParamSpec>;

export const cityTemplate = defineTemplate<CityState, typeof cityParams>({
  meta: {
    id: "city",
    name: "夜城",
    author: "music-to-video",
    version: "1.0.0",
    description: "楼在拍点上长成。镜头提前跟上下一座，窗灯顺着拍子往上走。",
  },
  tier: "2d",
  params: cityParams,
  setup(ctx, music) {
    const view = as2d(ctx);
    return buildCity(view.width, view.height, music, view.rng);
  },
  render(ctx, t, state, music) {
    const view = as2d(ctx);
    paintCity(view.ctx, state, t, view);
    void music;
  },
});

function buildCity(width: number, height: number, music: MusicData, rng: () => number): CityState {
  const unit = width / 1920;
  const beats = music.beats.length ? music.beats.slice() : [0];
  const count = pickCount(beats.length);
  const indexes = pickIndexes(beats.length, count);
  const specs: {
    beat: number;
    block: number;
    target: number;
    energy: number;
    floors: number;
    cols: number;
    tower: boolean;
    roof: number;
    tint: number;
    sign: number;
  }[] = [];
  for (const index of indexes) {
    const beat = beats[index];
    const energy = sectionEnergy(music.sections, beat);
    const block = (34 + rng() * 92) * unit;
    let target = (110 + 300 * energy) * (0.7 + 0.62 * rng()) * unit;
    if (energy >= 0.55) target *= 1.08;
    target = clamp(target, 96 * unit, height * 0.58);
    const floors = clamp(Math.round(target / (32 * unit)), 3, 16);
    const cols = clamp(Math.round(block / (18 * unit)), 2, 6);
    const roof = Math.floor(rng() * 5);
    const tint = Math.floor(rng() * 5);
    const sign = rng() > 0.55 ? Math.floor(rng() * 3) : 0;
    specs.push({
      beat,
      block,
      target,
      energy,
      floors,
      cols,
      tower: energy >= 0.62 || roof === 3,
      roof,
      tint,
      sign,
    });
  }
  const gaps = specs.map(() => (14 + rng() * 34) * unit);
  const total = specs.reduce((sum, item) => sum + item.block, 0) + gaps.slice(0, -1).reduce((sum, gap) => sum + gap, 0);
  let cursor = -total / 2;
  const pattern = [0.58, 1.25, 0.72, 1.4, 0.64, 1.08, 0.8, 1.48, 0.7, 1.16];
  const buildings: Building[] = specs.map((spec, index) => {
    const target = clamp(spec.target * pattern[index % pattern.length], 110 * unit, height * 0.68);
    const building: Building = {
      born: spec.beat,
      grow: 0.34,
      x: cursor,
      width: spec.block,
      target,
      energy: spec.energy,
      floors: clamp(Math.round(target / (30 * unit)), 3, 16),
      cols: clamp(Math.round(spec.block / (18 * unit)), 2, 6),
      tower: spec.tower,
      roof: spec.roof,
      tint: spec.tint,
      sign: spec.sign,
    };
    cursor += spec.block + gaps[index];
    return building;
  });

  const birth = new Set(buildings.map((building) => building.born.toFixed(4)));
  const others = beats.filter((beat) => !birth.has(beat.toFixed(4)));
  const cells: [number, number, number][] = [];
  buildings.forEach((building, index) => {
    for (let row = 0; row < building.floors; row++) {
      for (let col = 0; col < building.cols; col++) cells.push([index, col, row]);
    }
  });
  const order = cells.slice();
  for (let i = order.length - 1; i > 0; i--) {
    const swap = Math.floor(rng() * (i + 1));
    [order[i], order[swap]] = [order[swap], order[i]];
  }
  const bornLit = new Set<string>();
  const lights: WindowLight[][] = buildings.map(() => []);
  buildings.forEach((building, index) => {
    const own = order.filter((cell) => cell[0] === index);
    for (const cell of own.slice(0, Math.max(1, Math.floor(own.length / 3)))) {
      bornLit.add(cell.join(":"));
      lights[index].push({ time: building.born, index, col: cell[1], row: cell[2] });
    }
  });
  const pool = order.filter((cell) => !bornLit.has(cell.join(":")));
  others.forEach((beat, step) => {
    const cell = pool.length ? pool.pop()! : cells.length ? cells[step % cells.length] : null;
    if (!cell) return;
    lights[cell[0]].push({ time: beat, index: cell[0], col: cell[1], row: cell[2] });
  });

  const lows = beats.map((beat) => interp(music.bands.times, music.bands.low, beat));
  let bass: number[] = [];
  const peak = Math.max(0, ...lows);
  if (peak > 0.05) {
    const ordered = lows.slice().sort((a, b) => a - b);
    const cut = Math.max(0.42, ordered[Math.floor(0.62 * (ordered.length - 1))]);
    bass = beats.filter((_, index) => lows[index] >= cut);
  }
  if (!bass.length) bass = (music.extensions?.strongBeats?.length ? music.extensions.strongBeats : beats.filter((_, i) => i % 4 === 0)).slice();

  const far = skyline(rng, unit, -2400, 2400, 48, 110, 28, 100, 0.55, 0.35, 3);
  const mid = skyline(rng, unit, -2200, 2200, 36, 80, 70, 150, 0.72, 0.45, 4);
  const stars = Array.from({ length: 70 }, () => ({
    x: rng() * width,
    y: rng() * height * 0.62,
    r: (0.6 + rng() * 1.3) * unit,
    warmth: rng(),
    phase: rng() * Math.PI * 2,
  }));
  const camera = bakeCamera(width, buildings, beats, music.sections, music.duration);
  return {
    width,
    height,
    unit,
    buildings,
    lights,
    bass,
    beats,
    far,
    mid,
    stars,
    sections: music.sections,
    camX: camera.x,
    camZ: camera.z,
  };
}

function pickCount(beatCount: number): number {
  if (beatCount >= 24) return Math.min(18, Math.max(12, Math.floor(beatCount / 3)));
  return Math.min(beatCount, beatCount >= 16 ? 18 : Math.max(1, beatCount));
}

function pickIndexes(beatCount: number, count: number): number[] {
  if (count <= 1) return [0];
  const last = Math.max(0, Math.round((beatCount - 1) * 0.72));
  const indexes: number[] = [];
  for (let step = 0; step < count; step++) {
    let index = Math.min(beatCount - 1, Math.round((step * last) / (count - 1)));
    if (indexes.length && index <= indexes[indexes.length - 1]) index = Math.min(beatCount - 1, indexes[indexes.length - 1] + 1);
    indexes.push(index);
  }
  return indexes;
}

function sectionEnergy(sections: Section[], time: number): number {
  for (const section of sections) {
    if (time < section.end) return section.energy;
  }
  return sections[sections.length - 1]?.energy ?? 0.4;
}

function skyline(
  rng: () => number,
  unit: number,
  from: number,
  to: number,
  w0: number,
  w1: number,
  h0: number,
  h1: number,
  gap0: number,
  gap1: number,
  roofs: number,
): Block[] {
  const blocks: Block[] = [];
  let x = from * unit;
  const end = to * unit;
  while (x < end) {
    const width = (w0 + rng() * w1) * unit;
    const height = (h0 + rng() * h1) * unit;
    blocks.push({ x, width, height, roof: Math.floor(rng() * roofs) });
    x += width * (gap0 + rng() * gap1);
  }
  return blocks;
}

function coverIndex(buildings: Building[], time: number): number {
  if (!buildings.length) return -1;
  const ready = time + 2.8;
  let last = 0;
  for (let index = 0; index < buildings.length; index++) {
    const building = buildings[index];
    if (building.born - building.grow <= ready) last = index;
    else break;
  }
  return Math.min(buildings.length - 1, last + 1);
}

function cameraTarget(width: number, buildings: Building[], time: number): [number, number] {
  if (!buildings.length) return [0, 1];
  const end = coverIndex(buildings, time);
  let left = buildings[0].x;
  let right = buildings[end].x + buildings[end].width;
  let span = Math.max(1, right - left);
  const pad = Math.max(48 * (width / 1920), span * 0.05);
  left -= pad;
  right += pad;
  span = right - left;
  const zoom = Math.min(1.5, (width * 0.76) / span);
  return [(left + right) * 0.5, Math.max(0.62, zoom)];
}

function bakeCamera(width: number, buildings: Building[], beats: number[], sections: Section[], duration: number) {
  const n = frameCount(duration);
  const x = new Float32Array(n);
  const z = new Float32Array(n);
  let camX = cameraTarget(width, buildings, 0)[0];
  let camZ = cameraTarget(width, buildings, 0)[1];
  for (let i = 0; i < n; i++) {
    const time = i * BAKE_DT;
    if (i > 0) {
      const [tx, tz] = cameraTarget(width, buildings, time);
      camX = damp(camX, tx, BAKE_DT, 2.6);
      camZ = damp(camZ, tz, BAKE_DT, tz < camZ ? 3.6 : 1.6);
    }
    const energy = sectionEnergy(sections, time);
    const push = 0.016 * timesPulse(beats, time, 0.04, 0.16) * (0.4 + 0.6 * energy);
    x[i] = camX;
    z[i] = camZ * (1 + push);
  }
  return { x, z };
}

function buildingHeight(building: Building, time: number): number {
  if (time <= building.born - building.grow) return 0;
  const u = (time - (building.born - building.grow)) / building.grow;
  if (u >= 1) {
    const age = time - building.born;
    if (age > 0 && age < 0.36) {
      const bump = Math.sin((Math.PI * age) / 0.36);
      const lift = 0.032 * (0.6 + 0.4 * building.energy);
      return building.target * (1 + lift * bump);
    }
    return building.target;
  }
  return building.target * smoothstep(u);
}

function energyArea(sections: Section[], time: number): number {
  let area = 0;
  for (const section of sections) {
    const start = Math.max(0, section.start);
    const end = Math.min(time, section.end);
    if (end > start) area += (end - start) * section.energy;
  }
  return area;
}

function paintCity(
  ctx: CanvasRenderingContext2D,
  state: CityState,
  time: number,
  view: { post: { glow: number; grain: number; vignette: number } },
): void {
  const { width, height } = state;
  const camX = sampleCurve(state.camX, BAKE_DT, time);
  const zoom = sampleCurve(state.camZ, BAKE_DT, time);
  const ground = height * 0.78;
  const pulse = timesPulse(state.beats, time, 0.04, 0.16);
  const glow = timesPulse(state.bass, time, 0.05, 0.42);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  const sky = ctx.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, rgb(0.03, 0.04, 0.11));
  sky.addColorStop(0.4, rgb(0.06, 0.045, 0.13));
  sky.addColorStop(0.7, rgb(0.1, 0.06, 0.14));
  sky.addColorStop(1, rgb(0.03, 0.02, 0.04));
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, width, height);
  for (const star of state.stars) {
    const twinkle = 0.72 + 0.28 * Math.sin(time * 1.7 + star.phase);
    const tone = 0.75 + 0.2 * star.warmth;
    ctx.fillStyle = rgba(tone, tone * (0.92 + 0.05 * (1 - star.warmth)), 0.85 + 0.1 * (1 - star.warmth), (0.35 + (0.4 * star.r) / (2 * state.unit)) * twinkle);
    ctx.beginPath();
    ctx.arc(star.x - camX * 0.04, star.y, star.r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = rgba(0.96, 0.95, 0.9, 0.92);
  ctx.beginPath();
  ctx.arc(width * 0.78, height * 0.16, 26 * state.unit, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = rgb(0.03, 0.025, 0.07);
  ctx.beginPath();
  ctx.arc(width * 0.78 + 11 * state.unit, height * 0.16 - 6 * state.unit, 22 * state.unit, 0, Math.PI * 2);
  ctx.fill();
  paintLayer(ctx, state, state.far, ground - 36 * state.unit, camX, zoom, 0.32, 0.45, [0.035, 0.04, 0.07]);
  paintLayer(ctx, state, state.mid, ground - 16 * state.unit, camX, zoom, 0.62, 0.72, [0.055, 0.06, 0.1]);
  ctx.save();
  ctx.translate(width / 2, ground);
  const farZoom = 1 + (zoom - 1) * 0.35;
  ctx.scale(farZoom, farZoom);
  const haze = ctx.createLinearGradient(0, -80 * state.unit, 0, 30 * state.unit);
  haze.addColorStop(0, rgba(0.45, 0.28, 0.55, 0));
  haze.addColorStop(0.55, rgba(0.85, 0.55, 0.32, 0.1));
  haze.addColorStop(1, rgba(0.4, 0.25, 0.45, 0));
  ctx.fillStyle = haze;
  ctx.fillRect(-width * 2, -80 * state.unit, width * 4, 110 * state.unit);
  ctx.restore();
  ctx.save();
  ctx.translate(width / 2, ground);
  ctx.scale(zoom, zoom);
  ctx.translate(-camX, 0);
  for (let index = 0; index < state.buildings.length; index++) {
    paintBuilding(ctx, state, state.buildings[index], index, time, glow);
  }
  paintStreet(ctx, state, time, glow, pulse);
  ctx.restore();
  view.post.glow = 0.72 + 0.45 * Math.max(pulse, glow * 0.65);
  view.post.grain = 0.85;
  view.post.vignette = 0.95;
}

function paintLayer(
  ctx: CanvasRenderingContext2D,
  state: CityState,
  blocks: Block[],
  ground: number,
  camX: number,
  zoom: number,
  parallax: number,
  zoomMix: number,
  color: [number, number, number],
): void {
  ctx.save();
  ctx.translate(state.width / 2, ground);
  const layerZoom = 1 + (zoom - 1) * zoomMix;
  ctx.scale(layerZoom, layerZoom);
  ctx.translate(-camX * parallax, 0);
  ctx.fillStyle = rgb(...color);
  for (const block of blocks) {
    ctx.fillRect(block.x, -block.height, block.width, block.height);
    if (block.roof === 1 && block.height > 50 * state.unit) {
      ctx.fillRect(block.x + block.width * 0.22, -block.height - block.height * 0.18, block.width * 0.56, block.height * 0.18);
    } else if (block.roof === 2) {
      ctx.beginPath();
      ctx.moveTo(block.x, -block.height);
      ctx.lineTo(block.x + block.width * 0.5, -block.height - 16 * state.unit);
      ctx.lineTo(block.x + block.width, -block.height);
      ctx.fill();
    }
    if (block.height > 90 * state.unit && block.width > 40 * state.unit) {
      ctx.fillStyle = rgba(1, 0.78, 0.42, 0.18);
      ctx.fillRect(block.x + block.width * 0.35, -block.height * 0.55, Math.max(4 * state.unit, block.width * 0.08), 5 * state.unit);
      ctx.fillStyle = rgb(...color);
    }
  }
  ctx.restore();
}

function paintBuilding(
  ctx: CanvasRenderingContext2D,
  state: CityState,
  building: Building,
  index: number,
  time: number,
  glow: number,
): void {
  const risen = buildingHeight(building, time);
  if (risen < 1.5) return;
  const base = NIGHT[building.tint % NIGHT.length];
  const lift = 0.04 * building.energy;
  const body: [number, number, number] = [
    Math.min(1, base[0] + lift),
    Math.min(1, base[1] + lift * 0.4),
    Math.min(1, base[2] + lift),
  ];
  roundRect(ctx, building.x, -risen, building.width, risen, 3 * state.unit);
  const fill = ctx.createLinearGradient(building.x, -risen, building.x + building.width, 0);
  fill.addColorStop(0, rgb(Math.min(1, body[0] + 0.05), Math.min(1, body[1] + 0.04), Math.min(1, body[2] + 0.07)));
  fill.addColorStop(1, rgb(body[0] * 0.62, body[1] * 0.62, body[2] * 0.72));
  ctx.fillStyle = fill;
  ctx.fill();
  const shade = Math.min(10 * state.unit, building.width * 0.18);
  ctx.fillStyle = rgba(0, 0, 0, 0.32);
  ctx.fillRect(building.x + building.width - shade, -risen, shade, risen);
  const crown = risen * 0.16;
  ctx.fillStyle = rgb(body[0] * 0.85, body[1] * 0.85, body[2] * 0.95);
  if (building.roof === 1) {
    const inset = building.width * 0.2;
    ctx.fillRect(building.x + inset, -risen - crown, building.width - 2 * inset, crown);
  } else if (building.roof === 2) {
    ctx.beginPath();
    ctx.moveTo(building.x - 2 * state.unit, -risen);
    ctx.lineTo(building.x + building.width * 0.5, -risen - crown * 1.4);
    ctx.lineTo(building.x + building.width + 2 * state.unit, -risen);
    ctx.fill();
  } else if (building.roof === 3 || building.tower) {
    ctx.beginPath();
    ctx.moveTo(building.x + building.width * 0.28, -risen);
    ctx.lineTo(building.x + building.width * 0.5, -risen - crown * 2.1);
    ctx.lineTo(building.x + building.width * 0.72, -risen);
    ctx.fill();
  } else if (building.roof === 4) {
    ctx.fillRect(building.x + building.width * 0.12, -risen - crown * 0.45, building.width * 0.76, crown * 0.45);
    ctx.fillRect(building.x + building.width * 0.28, -risen - crown, building.width * 0.44, crown * 0.55);
  }
  ctx.strokeStyle = rgba(0.82, 0.88, 0.98, 0.28);
  ctx.lineWidth = 1.2 * state.unit;
  ctx.beginPath();
  ctx.moveTo(building.x + 2 * state.unit, -risen);
  ctx.lineTo(building.x + building.width - 2 * state.unit, -risen);
  ctx.stroke();
  if (building.sign && risen > building.target * 0.72) {
    const signW = building.width * (building.sign === 1 ? 0.42 : 0.55);
    const signX = building.x + (building.width - signW) * 0.5;
    const signY = -risen * (building.roof === 0 ? 0.62 : 0.48);
    const breathe = 0.78 + 0.22 * glow;
    const color = building.sign === 1 ? [1, 0.55, 0.28, 0.85 * breathe] : [0.45, 0.85, 1, 0.8 * breathe];
    const grad = ctx.createLinearGradient(signX, signY, signX + signW, signY);
    grad.addColorStop(0, rgba(color[0], color[1], color[2], 0));
    grad.addColorStop(0.15, rgba(color[0], color[1], color[2], color[3]));
    grad.addColorStop(0.85, rgba(color[0], color[1], color[2], color[3]));
    grad.addColorStop(1, rgba(color[0], color[1], color[2], 0));
    ctx.fillStyle = grad;
    ctx.fillRect(signX, signY, signW, 7 * state.unit);
  }
  paintWindows(ctx, state, building, index, risen, time, glow);
}

function paintWindows(
  ctx: CanvasRenderingContext2D,
  state: CityState,
  building: Building,
  index: number,
  risen: number,
  time: number,
  bass: number,
): void {
  const margin = building.width * 0.16;
  const usable = Math.max(1, building.width - 2 * margin);
  const winW = Math.min(10 * state.unit, (usable / building.cols) * 0.55);
  const winH = Math.min(14 * state.unit, (risen / Math.max(building.floors, 1)) * 0.42);
  const lit = new Map<string, number>();
  for (const light of state.lights[index]) {
    if (time + 1e-6 < light.time) continue;
    const age = time - light.time;
    const flash = age < 0.36 ? smoothstep(1 - age / 0.36) : 0;
    const key = `${light.col}:${light.row}`;
    lit.set(key, Math.max(lit.get(key) ?? 0, 0.72 + 0.28 * flash));
  }
  for (let row = 0; row < building.floors; row++) {
    const lag = (building.floors - 1 - row) * 0.03;
    const ripple = timesPulse(state.beats, time - lag, 0.045, 0.14);
    for (let col = 0; col < building.cols; col++) {
      const x = building.x + margin + ((col + 0.5) * usable) / building.cols - winW / 2;
      const span = building.target * 0.78;
      const y = -building.target * 0.9 + ((row + 0.5) * span) / building.floors;
      if (y < -risen + 6 * state.unit) continue;
      const level = lit.get(`${col}:${row}`) ?? 0;
      if (level <= 0) {
        const sheen = Math.max(0.07 * ripple, 0.05 * bass);
        ctx.fillStyle = sheen <= 0.01 ? rgba(0.02, 0.03, 0.05, 0.65) : rgba(ACCENT[0], ACCENT[1], ACCENT[2], 0.1 + 0.55 * sheen);
      } else {
        ctx.fillStyle = rgba(ACCENT[0], ACCENT[1], ACCENT[2], Math.min(1, level + 0.22 * ripple + 0.16 * bass));
      }
      roundRect(ctx, x, y, winW, winH, 1.6 * state.unit);
      ctx.fill();
    }
  }
}

function paintStreet(ctx: CanvasRenderingContext2D, state: CityState, time: number, glow: number, pulse: number): void {
  if (!state.buildings.length) return;
  const left = state.buildings[0].x - 120 * state.unit;
  const right = state.buildings[state.buildings.length - 1].x + state.buildings[state.buildings.length - 1].width + 120 * state.unit;
  const span = Math.max(240 * state.unit, right - left);
  const roadH = 168 * state.unit;
  ctx.fillStyle = rgb(0.012, 0.014, 0.022);
  ctx.fillRect(left - 280 * state.unit, 0, span + 560 * state.unit, roadH);
  const wash = ctx.createLinearGradient(0, 0, 0, 36 * state.unit);
  wash.addColorStop(0, rgba(1, 0.86, 0.58, 0.08 + 0.55 * glow + 0.7 * pulse));
  wash.addColorStop(1, rgba(ACCENT2[0], ACCENT2[1], ACCENT2[2], 0));
  ctx.fillStyle = wash;
  ctx.fillRect(left - 280 * state.unit, 0, span + 560 * state.unit, 36 * state.unit);
  paintReflections(ctx, state, time);
  if (pulse > 0.04) {
    ctx.fillStyle = rgba(1, 0.9, 0.7, 0.22 * pulse);
    ctx.fillRect(left - 280 * state.unit, 2 * state.unit, span + 560 * state.unit, 5 * state.unit);
  }
  ctx.fillStyle = rgba(0.78, 0.82, 0.88, 0.22);
  for (let dash = left; dash < right; dash += 64 * state.unit) ctx.fillRect(dash, 78 * state.unit, 28 * state.unit, 2.4 * state.unit);
  paintTram(ctx, state, left, span, time, energyArea(state.sections, time) * 42 * state.unit, Math.max(glow, pulse));
  const lanes: [number, number, number][] = [
    [44, 8, 1],
    [112, 12, -1],
  ];
  for (let index = 0; index < 6; index++) {
    const [lane, thick, direction] = lanes[index % 2];
    const speed = (170 + (index % 4) * 55 + (direction < 0 ? 40 : 0)) * state.unit;
    const length = (190 + 90 * Math.max(pulse, 0.45 * glow) + (index % 3) * 24) * state.unit;
    const travel =
      (time * speed + energyArea(state.sections, time) * (64 + (index % 3) * 16) * state.unit + (index * span) / 6) %
      (span + length + 80 * state.unit);
    paintHeadlight(ctx, state, left, right, lane * state.unit, thick * state.unit, direction, length, travel, pulse);
  }
}

function paintReflections(ctx: CanvasRenderingContext2D, state: CityState, time: number): void {
  for (const building of state.buildings) {
    const risen = buildingHeight(building, time);
    if (risen < 12 * state.unit) continue;
    const smear = Math.min(78 * state.unit, risen * 0.28);
    for (let col = 0; col < building.cols; col++) {
      const x = building.x + (building.width * (col + 0.5)) / building.cols - 1.4 * state.unit;
      const grad = ctx.createLinearGradient(0, 8 * state.unit, 0, 8 * state.unit + smear);
      grad.addColorStop(0, rgba(1, 0.78, 0.45, 0.22));
      grad.addColorStop(1, rgba(1, 0.7, 0.4, 0));
      ctx.fillStyle = grad;
      ctx.fillRect(x, 12 * state.unit, 2.8 * state.unit, smear);
    }
  }
}

function paintTram(
  ctx: CanvasRenderingContext2D,
  state: CityState,
  left: number,
  span: number,
  time: number,
  boost: number,
  flare: number,
): void {
  const tramW = 340 * state.unit;
  const tramH = 32 * state.unit;
  const tramX = left + ((time * 78 * state.unit + boost) % (span + tramW + 160 * state.unit)) - tramW;
  const y = 14 * state.unit;
  ctx.fillStyle = rgba(0.04, 0.045, 0.06, 0.98);
  roundRect(ctx, tramX, y, tramW, tramH, 5 * state.unit);
  ctx.fill();
  ctx.strokeStyle = rgba(0.55, 0.62, 0.7, 0.35);
  ctx.lineWidth = 1.4 * state.unit;
  ctx.beginPath();
  ctx.moveTo(tramX + tramW * 0.5, y);
  ctx.lineTo(tramX + tramW * 0.5, y - 16 * state.unit);
  ctx.moveTo(tramX + tramW * 0.38, y - 16 * state.unit);
  ctx.lineTo(tramX + tramW * 0.62, y - 16 * state.unit);
  ctx.stroke();
  const window = ctx.createLinearGradient(tramX, 0, tramX + tramW, 0);
  window.addColorStop(0, rgba(1, 0.78, 0.42, 0));
  window.addColorStop(0.08, rgba(1, 0.82, 0.48, 0.28 + 0.45 * flare));
  window.addColorStop(0.92, rgba(1, 0.82, 0.48, 0.28 + 0.45 * flare));
  window.addColorStop(1, rgba(1, 0.78, 0.42, 0));
  ctx.fillStyle = window;
  ctx.fillRect(tramX + 18 * state.unit, y + 8 * state.unit, tramW - 36 * state.unit, 8 * state.unit);
  const beam = (180 + 140 * flare) * state.unit;
  const head = ctx.createLinearGradient(tramX + tramW - 8 * state.unit, 0, tramX + tramW + beam, 0);
  head.addColorStop(0, rgba(0.8, 0.94, 1, 0.9));
  head.addColorStop(1, rgba(0.8, 0.94, 1, 0));
  ctx.fillStyle = head;
  ctx.fillRect(tramX + tramW - 8 * state.unit, y + 14 * state.unit, beam, 8 * state.unit);
}

function paintHeadlight(
  ctx: CanvasRenderingContext2D,
  state: CityState,
  left: number,
  right: number,
  lane: number,
  thick: number,
  direction: number,
  length: number,
  travel: number,
  pulse: number,
): void {
  const alpha = 0.82 + 0.18 * pulse;
  void state;
  if (direction > 0) {
    const nose = left + travel;
    const streak = ctx.createLinearGradient(nose - length, lane, nose, lane);
    streak.addColorStop(0, rgba(0.95, 0.18, 0.12, 0));
    streak.addColorStop(0.18, rgba(1, 0.28, 0.16, 0.75 * alpha));
    streak.addColorStop(0.42, rgba(1, 0.9, 0.7, 0.9 * alpha));
    streak.addColorStop(1, rgba(1, 0.98, 0.92, alpha));
    ctx.fillStyle = streak;
    ctx.fillRect(nose - length, lane, length, thick);
  } else {
    const nose = right - travel;
    const streak = ctx.createLinearGradient(nose, lane, nose + length, lane);
    streak.addColorStop(0, rgba(1, 0.98, 0.92, alpha));
    streak.addColorStop(0.58, rgba(1, 0.9, 0.7, 0.9 * alpha));
    streak.addColorStop(0.82, rgba(1, 0.28, 0.16, 0.75 * alpha));
    streak.addColorStop(1, rgba(0.95, 0.18, 0.12, 0));
    ctx.fillStyle = streak;
    ctx.fillRect(nose, lane, length, thick);
  }
}
