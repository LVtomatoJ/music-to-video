import type { MusicData } from "../music";
import { interp } from "../math";

const BALL_R = 28;
const SPREAD = 0.58;
const CLEARANCE = 72;
const BODY_GAP = 96;
const MIN_BODY = 48;
const SOLID_EPS = 1.5;

export interface Segment {
  t0: number;
  t1: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  bow: number;
}

export interface Obstacle {
  kind: string;
  hitTime: number;
  x: number;
  y: number;
  nx: number;
  ny: number;
  tx: number;
  ty: number;
  strong: boolean;
  energy: number;
  section: string;
  size: number;
  pegR: number;
  variant: string;
}

type Pt = [number, number];
type Poly = [Pt, Pt, number];
type Phrase = "cluster" | "stretch" | "corridor" | "curve";
type Mood = "calm" | "mid" | "intense";

interface PhraseSection {
  start: number;
  end: number;
  energy: number;
  label: Mood;
}

const SECTION_SIZE: Record<Mood, number> = { calm: 0.74, mid: 1, intense: 1.36 };
const VARIANT_SCALE: Record<string, number> = {
  short: 0.58,
  long: 1.7,
  corner: 1.2,
  double: 1.15,
  arch: 1.35,
  single: 0.85,
  pair: 1,
  chord: 1.05,
  pattern: 1.1,
  crack: 1.25,
  block: 1,
  stack: 1.2,
};
const QUIET = new Set(["short", "single", "crack"]);
const RICH = new Set(["long", "double", "corner", "stack", "chord", "pattern", "arch", "pair"]);
const SIMPLER: Record<string, string[]> = {
  double: ["long", "short"],
  corner: ["long", "short"],
  long: ["short"],
  arch: ["long", "short"],
  stack: ["block", "crack"],
  block: ["crack"],
  pattern: ["chord", "single"],
  chord: ["single"],
  pair: ["single"],
};
const POOLS: Record<string, string[]> = {
  neon: ["wall"],
  pastel: ["wall", "platform"],
  arcade: ["peg", "pad", "spinner"],
};
const VARIANTS: Record<string, string[]> = {
  wall: ["short", "long", "corner", "double"],
  platform: ["short", "long", "arch"],
  peg: ["single", "pair"],
  pad: ["short", "long"],
  spinner: ["short", "long"],
};

export function solveMap(music: MusicData, seed: number, skinName: string): { obstacles: Obstacle[]; segments: Segment[]; ballR: number } {
  const rng = { v: seed & 0xffffffff || 1 };
  const phrases = buildPhrases(music);
  const beats = music.beats.filter((beat) => beat >= -1e-4);
  const strong = music.extensions?.strongBeats ?? [];
  const flags = beats.map((beat) => strong.some((hit) => Math.abs(hit - beat) <= 1e-3));
  const deduped: [number, boolean][] = [];
  for (let index = 0; index < beats.length; index++) {
    const beat = beats[index];
    if (deduped.length && Math.abs(beat - deduped[deduped.length - 1][0]) < 1e-3) continue;
    deduped.push([beat, flags[index]]);
  }
  const obstacles: Obstacle[] = [];
  const segments: Segment[] = [];
  if (!deduped.length) {
    segments.push({ t0: 0, t1: Math.max(music.duration, 0.1), x0: 0, y0: 0, x1: 680 * music.duration, y1: 0, bow: 0 });
    return { obstacles, segments, ballR: BALL_R };
  }
  const pool = POOLS[skinName] ?? POOLS.neon;
  let lane = (xorshift32(rng) * 2 - 1) * Math.PI;
  let side = xorshift32(rng) < 0.5 ? 1 : -1;
  let heading = lane + side * SPREAD;
  let turnSide = side;
  let runLeft = 0;
  let lastAngle = 0;
  const flownTurns: number[] = [];
  let stepOut = false;
  let pos: Pt = [0, 0];
  let tPrev = 0;
  if (deduped[0][0] > 1e-3) {
    const [ivx, ivy] = unit(Math.cos(heading), Math.sin(heading));
    const dist = 680 * deduped[0][0];
    pos = [-ivx * dist, -ivy * dist];
  }
  const counts: Record<string, number> = {};
  for (const name of pool) counts[name] = 0;
  let lastKind = "";
  let lastPhrase = "";
  let strongOrdinal = 0;
  const polys: Poly[] = [];
  deduped.forEach(([beat, isStrong], index) => {
    try {
    const section = phraseAt(phrases, beat);
    const [phrase, speed, spread, extend] = phraseMotion(section.label, index, isStrong);
    if (extend && phrase !== lastPhrase) stepOut = true;
    lastPhrase = phrase;
    const dt = Math.max(1e-3, beat - tPrev);
    let length = speed * dt * (0.55 + spread * 0.05);
    if (stepOut) {
      length *= 1.06;
      stepOut = false;
    }
    if (runLeft <= 0 || sameSideRun(flownTurns, turnSide) >= 3) {
      turnSide = -turnSide;
      const roll = xorshift32(rng);
      runLeft = roll < 0.62 ? 1 : roll < 0.9 ? 2 : 3;
    }
    runLeft -= 1;
    length = Math.max(MIN_SEGMENT, Math.min(520, length));
    const drive = bounceDrive(music, beat, isStrong, tPrev, section.energy, rng);
    const angle = separateAngle(bounceAngle(drive, rng), lastAngle, (72 * Math.PI) / 180, rng);
    const prevSeg = segments[segments.length - 1];
    const base = prevSeg ? Math.atan2(prevSeg.y1 - prevSeg.y0, prevSeg.x1 - prevSeg.x0) : heading;
    const flight = chooseFlight(base, turnSide, angle, length, pos, segments, flownTurns);
    const bend = flight.bend;
    if (Math.sign(bend) && Math.sign(bend) !== turnSide) {
      turnSide = Math.sign(bend);
      runLeft = xorshift32(rng) < 0.75 ? 0 : 1;
    }
    lastAngle = Math.abs(bend);
    const flown = base + bend;
    let ivx = flight.ivx;
    let ivy = flight.ivy;
    let hit = flight.hit;
    const nextLane = flown;
    const nextSide = turnSide;
    const out = flown + bend * 0.35;
    let [ovx, ovy] = unit(Math.cos(out), Math.sin(out));
    const planned = out;
    [hit, ivx, ivy, ovx, ovy] = limitTurn(prevSeg, pos, hit, ivx, ivy, ovx, ovy);
    flownTurns.push(headingDelta(prevSeg, pos, hit));
    const kind = chooseKind(pool, isStrong, counts, lastKind, rng);
    counts[kind] += 1;
    lastKind = kind;
    let flourish = false;
    if (isStrong) {
      strongOrdinal += 1;
      flourish = strongOrdinal % 4 === 0;
    }
    const variant = chooseVariant(kind, section.label, flourish, rng);
    const fit = Math.min(1, length / 280);
    let size = bases(kind) * (1 + 0.22 * section.energy + (isStrong ? 0.28 : 0)) * Math.max(0.55, fit);
    size = Math.min(420, size * (VARIANT_SCALE[variant] ?? 1) * SECTION_SIZE[section.label]);
    size = capSize(hit, size, obstacles);
    const pegR = kind === "peg" ? size : 0;
    const bow = 0;
    if (obstacles.length && segments.length) {
      const arrival = tangent(segments[segments.length - 1], true);
      const departure = tangentOf(pos, hit, bow, false);
      writeNormal(obstacles[obstacles.length - 1], arrival, departure);
    }
    const [endTx, endTy] = tangentOf(pos, hit, bow, true);
    const normal = normalFrom(endTx, endTy, ovx, ovy) ?? unit(ivx - ovx, ivy - ovy);
    const [nx, ny] = normal;
    obstacles.push({
      kind,
      hitTime: beat,
      x: hit[0],
      y: hit[1],
      nx,
      ny,
      tx: -ny,
      ty: nx,
      strong: isStrong,
      energy: section.energy,
      section: section.label,
      size: kind === "peg" ? pegR : size,
      pegR,
      variant,
    });
    segments.push({ t0: tPrev, t1: beat, x0: pos[0], y0: pos[1], x1: hit[0], y1: hit[1], bow });
    polys.push([pos, hit, bow]);
    pos = hit;
    tPrev = beat;
    lane = nextLane;
    side = nextSide;
    const actual = Math.atan2(ovy, ovx);
    heading = aimsInto(hit, actual, obstacles) && !aimsInto(hit, planned, obstacles) ? planned : actual;
    if (!Number.isFinite(heading)) heading = lane;
    void flourish;
    } catch {
      pushStraight(obstacles, segments, pos, tPrev, beat, heading, isStrong);
      const last = obstacles[obstacles.length - 1];
      pos = [last.x, last.y];
      tPrev = beat;
    }
  });
  if (music.duration > tPrev + 1e-3) {
    try {
      const [ivx, ivy] = unit(Math.cos(heading), Math.sin(heading));
      const dist = 680 * (music.duration - tPrev);
      const guess: Pt = [pos[0] + ivx * dist, pos[1] + ivy * dist];
      const routed = avoidOverlap(pos, guess, [ivx, ivy], [ivx, ivy], polys, dist, obstacles, segments, tPrev, music.duration, false);
      const limited = limitTurn(segments[segments.length - 1], pos, routed[0], routed[1], routed[2], routed[3], routed[4]);
      const end = limited[0];
      segments.push({
        t0: tPrev,
        t1: music.duration,
        x0: Number.isFinite(pos[0]) ? pos[0] : 0,
        y0: Number.isFinite(pos[1]) ? pos[1] : 0,
        x1: Number.isFinite(end[0]) ? end[0] : (Number.isFinite(pos[0]) ? pos[0] : 0) + dist,
        y1: Number.isFinite(end[1]) ? end[1] : Number.isFinite(pos[1]) ? pos[1] : 0,
        bow: 0,
      });
    } catch {
      pushStraight(obstacles, segments, pos, tPrev, music.duration, heading, false);
    }
  }
  try {
    alignNormals(obstacles, segments);
    seat(obstacles, segments);
    spread(obstacles, segments);
  } catch {
    // A failed tidy-up keeps the course that was already flown.
  }
  sanitizeCourse(obstacles, segments, music.duration);
  return { obstacles, segments, ballR: BALL_R };
}

function pushStraight(
  obstacles: Obstacle[],
  segments: Segment[],
  pos: Pt,
  tPrev: number,
  beat: number,
  heading: number,
  strong: boolean,
): void {
  const [ux, uy] = unit(Math.cos(heading), Math.sin(heading));
  const originX = Number.isFinite(pos[0]) ? pos[0] : 0;
  const originY = Number.isFinite(pos[1]) ? pos[1] : 0;
  const aimed: Pt = [originX + ux * 220, originY + uy * 220];
  const limited = limitTurn(segments[segments.length - 1], [originX, originY], aimed, ux, uy, ux, uy);
  const hit = limited[0];
  obstacles.push({
    kind: "wall",
    hitTime: Number.isFinite(beat) ? beat : tPrev,
    x: hit[0],
    y: hit[1],
      nx: -limited[2],
      ny: limited[1],
      tx: limited[1],
      ty: limited[2],
    strong,
    energy: 0.4,
    section: "mid",
    size: 168,
    pegR: 0,
    variant: "long",
  });
  segments.push({
    t0: Number.isFinite(tPrev) ? tPrev : 0,
    t1: Math.max(Number.isFinite(beat) ? beat : tPrev + 0.2, (Number.isFinite(tPrev) ? tPrev : 0) + 1e-3),
    x0: originX,
    y0: originY,
    x1: hit[0],
    y1: hit[1],
    bow: 0,
  });
}

function num(value: number, fallback = 0): number {
  return Number.isFinite(value) ? value : fallback;
}

function sanitizeCourse(obstacles: Obstacle[], segments: Segment[], duration: number): void {
  for (const segment of segments) {
    segment.t0 = num(segment.t0);
    segment.t1 = num(segment.t1, segment.t0 + 0.1);
    if (segment.t1 <= segment.t0) segment.t1 = segment.t0 + 0.1;
    segment.x0 = num(segment.x0);
    segment.y0 = num(segment.y0);
    segment.x1 = num(segment.x1, segment.x0 + 200);
    segment.y1 = num(segment.y1, segment.y0);
    segment.bow = 0;
  }
  for (const obstacle of obstacles) {
    obstacle.x = num(obstacle.x);
    obstacle.y = num(obstacle.y);
    obstacle.hitTime = num(obstacle.hitTime);
    obstacle.size = num(obstacle.size, 168);
    if (!(obstacle.size > 8)) obstacle.size = 168;
    const [nx, ny] = unit(num(obstacle.nx), num(obstacle.ny, -1));
    obstacle.nx = nx;
    obstacle.ny = ny;
    obstacle.tx = -ny;
    obstacle.ty = nx;
    obstacle.pegR = num(obstacle.pegR);
    if (!obstacle.kind) obstacle.kind = "wall";
    if (!obstacle.variant) obstacle.variant = "long";
  }
  if (!segments.length) {
    segments.push({ t0: 0, t1: Math.max(duration, 0.5), x0: 0, y0: 0, x1: 640, y1: 0, bow: 0 });
  }
}

export function pointOn(segments: Segment[], time: number): Pt {
  if (!segments.length) return [0, 0];
  if (time <= segments[0].t0) return [segments[0].x0, segments[0].y0];
  const last = segments[segments.length - 1];
  if (time >= last.t1) return [last.x1, last.y1];
  let lo = 0;
  let hi = segments.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (segments[mid].t1 < time) lo = mid + 1;
    else hi = mid;
  }
  const seg = segments[lo];
  const span = seg.t1 - seg.t0;
  const u = span <= 1e-9 ? 0 : Math.max(0, Math.min(1, (time - seg.t0) / span));
  return bulge([seg.x0, seg.y0], [seg.x1, seg.y1], seg.bow, u);
}

function xorshift32(state: { v: number }): number {
  let x = state.v >>> 0;
  x = (x ^ ((x << 13) >>> 0)) >>> 0;
  x = (x ^ (x >>> 17)) >>> 0;
  x = (x ^ ((x << 5) >>> 0)) >>> 0;
  state.v = x;
  return x / 4294967296;
}

function loudAt(music: MusicData, time: number): number {
  const value = interp(music.loudness.times, music.loudness.values, time);
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0.4));
}

function nearestRow(times: number[], time: number): number {
  if (!times.length) return 0;
  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < time) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - time) < Math.abs(times[lo] - time)) return lo - 1;
  return lo;
}

function onsetAt(music: MusicData, time: number): number {
  let best = 0;
  for (const onset of music.onsets) {
    const gap = Math.abs(onset.time - time);
    if (gap > 0.12) continue;
    best = Math.max(best, onset.strength * (1 - gap / 0.12));
  }
  return Math.max(0, Math.min(1, best));
}

function accentAt(music: MusicData, time: number, strong: boolean): number {
  const marks = music.downbeats.length ? music.downbeats : music.beats.filter((_, index) => index % 4 === 0);
  const downbeat = marks.some((hit) => Math.abs(hit - time) <= 0.045);
  return Math.min(1, (strong ? 0.62 : 0) + (downbeat ? 0.48 : 0));
}

function pitchMove(music: MusicData, time: number, prev: number): number {
  const chroma = music.extensions?.chroma;
  if (chroma && chroma.times.length > 1 && chroma.values.length > 1) {
    const before = chroma.values[nearestRow(chroma.times, prev)] ?? [];
    const after = chroma.values[nearestRow(chroma.times, time)] ?? [];
    let delta = 0;
    for (let index = 0; index < 12; index++) delta += Math.abs((after[index] ?? 0) - (before[index] ?? 0));
    return Math.min(1, delta / 1.35);
  }
  const tones = music.extensions?.tones;
  if (!tones?.length) return 0;
  const midiNear = (at: number) => {
    let best = tones[0];
    let gap = Infinity;
    for (const tone of tones) {
      const distance = Math.abs(tone.time - at);
      if (distance < gap) {
        gap = distance;
        best = tone;
      }
    }
    return gap < 0.45 ? best.midi : null;
  };
  const from = midiNear(prev);
  const to = midiNear(time);
  if (from == null || to == null) return 0;
  return Math.min(1, Math.abs(to - from) / 7);
}

/** 0 is a quiet drift, 1 is a hard accent. */
function bounceDrive(
  music: MusicData,
  beat: number,
  strong: boolean,
  prev: number,
  energy: number,
  rng: { v: number },
): number {
  const loud = loudAt(music, beat);
  const onset = music.onsets.length ? onsetAt(music, beat) : loud;
  const accent = accentAt(music, beat, strong);
  const pitch = pitchMove(music, beat, prev);
  const jitter = (xorshift32(rng) - 0.5) * 0.16;
  const drive = 0.2 * loud + 0.28 * onset + 0.28 * accent + 0.14 * pitch + 0.16 * Math.max(0, Math.min(1, energy)) + jitter;
  return Math.max(0, Math.min(1, drive));
}

function bounceAngle(drive: number, rng: { v: number }): number {
  const jitter = (xorshift32(rng) - 0.5) * ((22 * Math.PI) / 180);
  const angle = (16 * Math.PI) / 180 + ((50 * Math.PI) / 180) * drive + jitter;
  return Math.max((14 * Math.PI) / 180, Math.min((70 * Math.PI) / 180, angle));
}

function recentTurnSum(turns: number[], extra: number): number {
  const start = Math.max(0, turns.length - (TURN_WINDOW - 1));
  let sum = extra;
  for (let index = start; index < turns.length; index++) sum += turns[index];
  return sum;
}

function sameSideRun(turns: number[], side: number): number {
  if (!side) return 0;
  let count = 0;
  for (let index = turns.length - 1; index >= 0; index--) {
    if (Math.sign(turns[index]) !== side) break;
    count += 1;
  }
  return count;
}

function headingDelta(prev: Segment | undefined, from: Pt, to: Pt): number {
  if (!prev) return 0;
  const ux = prev.x1 - prev.x0;
  const uy = prev.y1 - prev.y0;
  const vx = to[0] - from[0];
  const vy = to[1] - from[1];
  if (Math.hypot(ux, uy) < 1e-6 || Math.hypot(vx, vy) < 1e-6) return 0;
  return Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy);
}

/** Keeps the signed heading change over the last eight bounces inside ±200°. */
function fitWindow(turns: number[], bend: number): number {
  const base = recentTurnSum(turns, 0);
  if (Math.abs(base + bend) <= OPEN_TURN + 1e-6) return bend;
  if (Math.abs(base - bend) <= OPEN_TURN + 1e-6) return -bend;
  const lo = -OPEN_TURN - base;
  const hi = OPEN_TURN - base;
  return Math.max(lo, Math.min(hi, bend));
}

function trailCrosses(from: Pt, to: Pt, segments: Segment[]): boolean {
  const start = Math.max(0, segments.length - 16);
  for (let index = start; index < segments.length - 1; index++) {
    const segment = segments[index];
    if (segmentsCross(from, to, [segment.x0, segment.y0], [segment.x1, segment.y1])) return true;
  }
  return false;
}

function segmentsCross(a: Pt, b: Pt, c: Pt, d: Pt): boolean {
  const side = (p: Pt, q: Pt, r: Pt) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  const ab1 = side(a, b, c);
  const ab2 = side(a, b, d);
  const cd1 = side(c, d, a);
  const cd2 = side(c, d, b);
  const eps = 2;
  if (Math.abs(ab1) <= eps || Math.abs(ab2) <= eps || Math.abs(cd1) <= eps || Math.abs(cd2) <= eps) return false;
  return ab1 * ab2 < 0 && cd1 * cd2 < 0;
}

function chooseFlight(
  base: number,
  side: number,
  angle: number,
  length: number,
  pos: Pt,
  segments: Segment[],
  turns: number[],
): { bend: number; hit: Pt; ivx: number; ivy: number } {
  const magnitudes = [
    angle,
    Math.min((70 * Math.PI) / 180, angle + (18 * Math.PI) / 180),
    Math.max(MIN_TURN, angle - (18 * Math.PI) / 180),
    (28 * Math.PI) / 180,
    (52 * Math.PI) / 180,
  ];
  let fallback: { bend: number; hit: Pt; ivx: number; ivy: number } | null = null;
  for (const trySide of [side, -side]) {
    for (const magnitude of magnitudes) {
      let bend = fitWindow(turns, trySide * magnitude);
      if (sameSideRun(turns, Math.sign(bend)) >= 3) bend = fitWindow(turns, -bend);
      bend = Math.max(-MAX_TURN, Math.min(MAX_TURN, bend));
      const heading = base + bend;
      const [ivx, ivy] = unit(Math.cos(heading), Math.sin(heading));
      const hit: Pt = [pos[0] + ivx * length, pos[1] + ivy * length];
      const candidate = { bend, hit, ivx, ivy };
      if (!fallback) fallback = candidate;
      if (!trailCrosses(pos, hit, segments)) return candidate;
    }
  }
  const straight = fitWindow(turns, side * MIN_TURN);
  const heading = base + straight;
  const [ivx, ivy] = unit(Math.cos(heading), Math.sin(heading));
  return (
    fallback ?? {
      bend: straight,
      hit: [pos[0] + ivx * length, pos[1] + ivy * length],
      ivx,
      ivy,
    }
  );
}

function separateAngle(angle: number, previous: number, limit: number, rng: { v: number }): number {
  const gap = (18 * Math.PI) / 180;
  const clamped = Math.max(MIN_TURN, Math.min(limit, angle));
  if (!(previous > 0) || Math.abs(clamped - previous) >= gap) return clamped;
  const kick = gap + xorshift32(rng) * ((20 * Math.PI) / 180);
  const first = xorshift32(rng) < 0.5 ? previous + kick : previous - kick;
  const candidate = Math.max(MIN_TURN, Math.min(limit, first));
  if (Math.abs(candidate - previous) >= gap) return candidate;
  const other = Math.max(MIN_TURN, Math.min(limit, previous * 2 - first));
  if (Math.abs(other - previous) >= gap - 1e-6) return other;
  const step = Math.min(gap * 0.4, (limit - MIN_TURN) * 0.45);
  const nudged = clamped >= (limit + MIN_TURN) / 2 ? clamped - step : clamped + step;
  return Math.max(MIN_TURN, Math.min(limit, nudged));
}

function unit(x: number, y: number): Pt {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return [1, 0];
  const mag = Math.hypot(x, y);
  if (!(mag > 1e-8)) return [1, 0];
  return [x / mag, y / mag];
}

function buildPhrases(music: MusicData): PhraseSection[] {
  const sections: PhraseSection[] = [];
  const energies: number[] = [];
  for (let start = 0; start < music.duration - 1e-6; start += 2) {
    const end = Math.min(music.duration, start + 2);
    let sum = 0;
    let count = 0;
    for (let time = start; time < end; time += 0.05) {
      sum += interp(music.loudness.times, music.loudness.values, time);
      count += 1;
    }
    const energy = count ? sum / count : 0;
    sections.push({ start, end, energy, label: "mid" });
    energies.push(energy);
  }
  const sorted = energies.slice().sort((a, b) => a - b);
  const at = (ratio: number) => sorted[Math.min(sorted.length - 1, Math.floor(ratio * (sorted.length - 1)))] ?? 0.4;
  const low = energies.length >= 3 ? at(0.34) : 0.35;
  const high = energies.length >= 3 ? at(0.67) : 0.62;
  for (const section of sections) {
    section.label = section.energy <= low ? "calm" : section.energy >= high ? "intense" : "mid";
  }
  return sections.length ? sections : [{ start: 0, end: Math.max(music.duration, 0.5), energy: 0.4, label: "mid" }];
}

function phraseAt(sections: PhraseSection[], time: number): PhraseSection {
  for (const section of sections) if (time < section.end) return section;
  return sections[sections.length - 1];
}

function phraseMotion(label: Mood, index: number, strong: boolean): [Phrase, number, number, boolean] {
  if (label === "intense" && index % 10 !== 0) return ["cluster", 560, strong ? 0.48 : 0.4, false];
  if (label === "calm") {
    if (Math.floor(index / 4) % 2 === 0) return ["stretch", 860, 0.22, true];
    return ["corridor", 680, 0.36, false];
  }
  if (index % 3 !== 2) return ["curve", 640, 0.5, false];
  return ["corridor", 720, 0.32, false];
}

function bases(kind: string): number {
  if (kind === "platform") return 230;
  if (kind === "peg") return 30;
  if (kind === "pad") return 120;
  if (kind === "spinner") return 128;
  return 150;
}

function chooseKind(pool: string[], strong: boolean, counts: Record<string, number>, last: string, rng: { v: number }): string {
  let options = pool.slice();
  if (strong) {
    const heavy = options.filter((name) => name === "wall" || name === "pad" || name === "spinner");
    if (heavy.length && xorshift32(rng) < 0.7) options = heavy;
  }
  const least = options.slice().sort((a, b) => counts[a] - counts[b] || Number(a === last) - Number(b === last))[0];
  if (counts[least] === 0 || xorshift32(rng) < 0.6) return least;
  const rest = options.filter((name) => name !== last);
  const bag = rest.length ? rest : options;
  return bag[Math.floor(xorshift32(rng) * bag.length) % bag.length];
}

function chooseVariant(kind: string, label: Mood, flourish: boolean, rng: { v: number }): string {
  const options = VARIANTS[kind] ?? ["plain"];
  if (flourish) {
    const rich = options.filter((name) => RICH.has(name));
    if (rich.length) return rich[Math.floor(xorshift32(rng) * rich.length) % rich.length];
  }
  if (label === "calm") {
    const quiet = options.filter((name) => QUIET.has(name));
    if (quiet.length && xorshift32(rng) < 0.72) return quiet[Math.floor(xorshift32(rng) * quiet.length) % quiet.length];
  }
  if (label === "intense") {
    const rich = options.filter((name) => RICH.has(name));
    if (rich.length && xorshift32(rng) < 0.7) return rich[Math.floor(xorshift32(rng) * rich.length) % rich.length];
  }
  return options[Math.floor(xorshift32(rng) * options.length) % options.length];
}

function cloudCenter(obstacles: Obstacle[]): Pt {
  const count = obstacles.length;
  return [obstacles.reduce((sum, obs) => sum + obs.x, 0) / count, obstacles.reduce((sum, obs) => sum + obs.y, 0) / count];
}

function insideCloud(pos: Pt, obstacles: Obstacle[], margin: number): boolean {
  if (obstacles.length < 6) return false;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const obs of obstacles) {
    minX = Math.min(minX, obs.x);
    maxX = Math.max(maxX, obs.x);
    minY = Math.min(minY, obs.y);
    maxY = Math.max(maxY, obs.y);
  }
  return pos[0] >= minX - margin && pos[0] <= maxX + margin && pos[1] >= minY - margin && pos[1] <= maxY + margin;
}

/** A bounce may be sharp, but it still has to keep going forward. */
const MAX_TURN = (75 * Math.PI) / 180;
const MIN_TURN = (12 * Math.PI) / 180;
const TURN_WINDOW = 8;
const OPEN_TURN = (200 * Math.PI) / 180;
/** Long enough that a wall of about 156 world units reads as its own object. */
const MIN_SEGMENT = 156 * 2.5;

function limitTurn(
  prev: Segment | undefined,
  pos: Pt,
  hit: Pt,
  ivx: number,
  ivy: number,
  ovx: number,
  ovy: number,
): [Pt, number, number, number, number] {
  if (!prev) return [hit, ivx, ivy, ovx, ovy];
  const [ux, uy] = unit(prev.x1 - prev.x0, prev.y1 - prev.y0);
  const dx = hit[0] - pos[0];
  const dy = hit[1] - pos[1];
  const len = Math.hypot(dx, dy);
  if (!(len > 1)) return [hit, ivx, ivy, ovx, ovy];
  const vx = dx / len;
  const vy = dy / len;
  const dot = Math.max(-1, Math.min(1, ux * vx + uy * vy));
  const cross = ux * vy - uy * vx;
  const angle = Math.atan2(cross, dot);
  if (Math.abs(angle) <= MAX_TURN) return [hit, ivx, ivy, ovx, ovy];
  const capped = Math.sign(angle) * MAX_TURN;
  const cosine = Math.cos(capped);
  const sine = Math.sin(capped);
  const nx = ux * cosine - uy * sine;
  const ny = ux * sine + uy * cosine;
  const delta = capped - angle;
  const oc = Math.cos(delta);
  const os = Math.sin(delta);
  return [[pos[0] + nx * len, pos[1] + ny * len], nx, ny, ovx * oc - ovy * os, ovx * os + ovy * oc];
}

function aimsInto(origin: Pt, angle: number, obstacles: Obstacle[]): boolean {
  if (obstacles.length < 6 || !insideCloud(origin, obstacles, 80)) return false;
  const [cx, cy] = cloudCenter(obstacles);
  const vx = cx - origin[0];
  const vy = cy - origin[1];
  const dist = Math.hypot(vx, vy);
  if (dist < 1) return true;
  return (Math.cos(angle) * vx + Math.sin(angle) * vy) / dist > 0.3;
}

function avoidOverlap(
  pos: Pt,
  hit: Pt,
  incoming: Pt,
  outgoing: Pt,
  polys: Poly[],
  length: number,
  obstacles: Obstacle[],
  segments: Segment[],
  t0: number,
  t1: number,
  allowBend: boolean,
): [Pt, number, number, number, number] {
  const ix = incoming[0];
  const iy = incoming[1];
  const ox = outgoing[0];
  const oy = outgoing[1];
  const flown = flownPoints(segments);
  const angles = [0];
  for (const degrees of [12, 22, 34, 46]) angles.push(degrees, -degrees);
  const search = (originIn: Pt, originOut: Pt, chords: number[], yaw: number[]) => {
    let best: [number, Pt, number, number, number, number] | null = null;
    for (const chord of chords) {
      if (chord < 24) continue;
      for (const degrees of yaw) {
        const rad = (degrees * Math.PI) / 180;
        const cosine = Math.cos(rad);
        const sine = Math.sin(rad);
        const ux = originIn[0] * cosine - originIn[1] * sine;
        const uy = originIn[0] * sine + originIn[1] * cosine;
        const vx = originOut[0] * cosine - originOut[1] * sine;
        const vy = originOut[0] * sine + originOut[1] * cosine;
        const candidate: Pt = [pos[0] + ux * chord, pos[1] + uy * chord];
        if (blocked(pos, candidate, polys)) continue;
        const pen = legSolidPen(pos, candidate, 0, t0, t1, obstacles, segments, flown);
        if (pen <= 1e-6) return { found: [candidate, ux, uy, vx, vy] as [Pt, number, number, number, number], best };
        if (!best || pen < best[0]) best = [pen, candidate, ux, uy, vx, vy];
      }
    }
    return { found: null, best };
  };
  const probe = () => {
    let result = search([ix, iy], [ox, oy], [length, length * 0.78, length * 0.58, length * 0.4, length * 0.28], angles);
    if (result.found) return result;
    for (const extra of [24, -24, 40, -40]) {
      const rad = (extra * Math.PI) / 180;
      const cosine = Math.cos(rad);
      const sine = Math.sin(rad);
      const ex = ox * cosine - oy * sine;
      const ey = ox * sine + oy * cosine;
      const alt = search([ix, iy], [ex, ey], [length, length * 0.62, length * 0.4], [0, 16, -16, 32, -32, 46, -46]);
      if (alt.found) return { found: alt.found, best: result.best };
      if (alt.best && (!result.best || alt.best[0] < result.best[0])) result = { found: null, best: alt.best };
    }
    if (allowBend && Math.hypot(ix - ox, iy - oy) > 1e-3) {
      const [nx, ny] = unit(ix - ox, iy - oy);
      const tx = -iy;
      const ty = ix;
      for (const bend of [0.35, -0.35, 0.6, -0.6]) {
        const [ux, uy] = unit(ix + tx * bend, iy + ty * bend);
        const candidate: Pt = [pos[0] + ux * length, pos[1] + uy * length];
        if (blocked(pos, candidate, polys)) continue;
        const [vx, vy] = reflect(ux, uy, nx, ny);
        const pen = legSolidPen(pos, candidate, 0, t0, t1, obstacles, segments, flown);
        if (pen <= 1e-6) return { found: [candidate, ux, uy, vx, vy] as [Pt, number, number, number, number], best: result.best };
        if (!result.best || pen < result.best[0]) result = { found: null, best: [pen, candidate, ux, uy, vx, vy] };
      }
    }
    return result;
  };
  let { found, best } = probe();
  if (found) return found;
  for (let pass = 0; pass < 3; pass++) {
    if (!best || !obstacles.length) break;
    if (!shrinkBlockers(pos, best[1], t0, t1, obstacles, segments, flown)) break;
    const retry = probe();
    if (retry.found) return retry.found;
    if (retry.best && (!best || retry.best[0] < best[0])) best = retry.best;
    else break;
  }
  if (best) return [best[1], best[2], best[3], best[4], best[5]];
  for (const origin of [[ix, iy], [-ix, -iy]] as Pt[]) {
    for (const chord of [length, length * 0.55, Math.max(24, length * 0.3)]) {
      for (const degrees of angles) {
        const rad = (degrees * Math.PI) / 180;
        const cosine = Math.cos(rad);
        const sine = Math.sin(rad);
        const ux = origin[0] * cosine - origin[1] * sine;
        const uy = origin[0] * sine + origin[1] * cosine;
        const candidate: Pt = [pos[0] + ux * chord, pos[1] + uy * chord];
        if (blocked(pos, candidate, polys)) continue;
        const vx = ox * cosine - oy * sine;
        const vy = ox * sine + oy * cosine;
        return [candidate, ux, uy, vx, vy];
      }
    }
  }
  return [hit, ix, iy, ox, oy];
}

function reflect(ix: number, iy: number, nx: number, ny: number): Pt {
  const dot = ix * nx + iy * ny;
  return unit(ix - 2 * dot * nx, iy - 2 * dot * ny);
}

function bulge(a: Pt, b: Pt, bow: number, u: number): Pt {
  const x = a[0] + (b[0] - a[0]) * u;
  const y = a[1] + (b[1] - a[1]) * u;
  const offset = 4 * u * (1 - u) * bow;
  if (Math.abs(offset) < 1e-8) return [x, y];
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const length = Math.hypot(dx, dy) || 1;
  return [x + (-dy / length) * offset, y + (dx / length) * offset];
}

function velocityOn(seg: Segment, u: number): Pt {
  const span = Math.max(1e-6, seg.t1 - seg.t0);
  const dx = seg.x1 - seg.x0;
  const dy = seg.y1 - seg.y0;
  const lateral = 4 * (1 - 2 * u) * seg.bow;
  if (Math.abs(lateral) < 1e-8) return [dx / span, dy / span];
  const length = Math.hypot(dx, dy) || 1;
  return [(dx + (-dy / length) * lateral) / span, (dy + (dx / length) * lateral) / span];
}

function tangent(seg: Segment, atEnd: boolean): Pt {
  return unit(...velocityOn(seg, atEnd ? 1 : 0));
}

function tangentOf(a: Pt, b: Pt, bow: number, atEnd: boolean): Pt {
  return tangent({ t0: 0, t1: 1, x0: a[0], y0: a[1], x1: b[0], y1: b[1], bow }, atEnd);
}

function normalFrom(ix: number, iy: number, ox: number, oy: number): Pt | null {
  const dx = ix - ox;
  const dy = iy - oy;
  if (Math.hypot(dx, dy) < 1e-4) return null;
  return unit(dx, dy);
}

function writeNormal(obs: Obstacle, arrival: Pt, departure: Pt): void {
  const normal = normalFrom(arrival[0], arrival[1], departure[0], departure[1]);
  if (!normal) return;
  obs.nx = normal[0];
  obs.ny = normal[1];
  obs.tx = -normal[1];
  obs.ty = normal[0];
}

function alignNormals(obstacles: Obstacle[], segments: Segment[]): void {
  obstacles.forEach((obs, index) => {
    if (index + 1 >= segments.length) return;
    const incoming = tangent(segments[index], true);
    const outgoing = tangent(segments[index + 1], false);
    writeNormal(obs, incoming, outgoing);
  });
}

function polyPieces(a: Pt, b: Pt, bow: number): [Pt, Pt][] {
  if (Math.abs(bow) < 1e-3) return [[a, b]];
  const points = [0, 0.25, 0.5, 0.75, 1].map((u) => bulge(a, b, bow, u));
  const pieces: [Pt, Pt][] = [];
  for (let index = 0; index < points.length - 1; index++) pieces.push([points[index], points[index + 1]]);
  return pieces;
}

function segmentDistance(a: Pt, b: Pt, c: Pt, d: Pt): number {
  const d1x = b[0] - a[0];
  const d1y = b[1] - a[1];
  const d2x = d[0] - c[0];
  const d2y = d[1] - c[1];
  const rx = a[0] - c[0];
  const ry = a[1] - c[1];
  const aa = d1x * d1x + d1y * d1y;
  const ee = d2x * d2x + d2y * d2y;
  const ff = d2x * rx + d2y * ry;
  const bb = d1x * d2x + d1y * d2y;
  const cc = d1x * rx + d1y * ry;
  const denom = aa * ee - bb * bb;
  let s = denom > 1e-8 && aa > 1e-8 ? Math.min(1, Math.max(0, (bb * ff - cc * ee) / denom)) : 0;
  let t = ee > 1e-8 ? (bb * s + ff) / ee : 0;
  if (t < 0) {
    t = 0;
    s = aa <= 1e-8 ? 0 : Math.min(1, Math.max(0, -cc / aa));
  } else if (t > 1) {
    t = 1;
    s = aa <= 1e-8 ? 0 : Math.min(1, Math.max(0, (bb - cc) / aa));
  }
  const cx1 = a[0] + d1x * s;
  const cy1 = a[1] + d1y * s;
  const cx2 = c[0] + d2x * t;
  const cy2 = c[1] + d2y * t;
  return Math.hypot(cx1 - cx2, cy1 - cy2);
}

function intersects(a: Pt, b: Pt, c: Pt, d: Pt): boolean {
  const cross = (p: Pt, q: Pt, r: Pt) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  const straddles = (p: Pt, q: Pt, r: Pt, s: Pt) => cross(p, q, r) * cross(p, q, s) < 0;
  return straddles(a, b, c, d) && straddles(c, d, a, b);
}

function foldsBack(a: Pt, b: Pt, p: Pt, q: Pt): boolean {
  if (intersects(a, b, p, q) && Math.hypot(b[0] - a[0], b[1] - a[1]) > 24) return true;
  for (let step = 2; step < 7; step++) {
    const u = step / 6;
    const point: Pt = [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
    if (Math.hypot(point[0] - a[0], point[1] - a[1]) < 80) continue;
    if (segmentDistance(point, point, p, q) < 18) return true;
  }
  return false;
}

function blocked(a: Pt, b: Pt, polys: Poly[]): boolean {
  for (let index = 0; index < polys.length; index++) {
    const [c, d, bow] = polys[index];
    const recent = index === polys.length - 1;
    for (const [p, q] of polyPieces(c, d, bow)) {
      if (recent) {
        if (foldsBack(a, b, p, q) || segmentDistance(b, b, p, q) < CLEARANCE) return true;
        continue;
      }
      if (intersects(a, b, p, q) || segmentDistance(a, b, p, q) < CLEARANCE) return true;
    }
  }
  return false;
}

function iterPoints(a: Pt, b: Pt, bow: number, t0: number, t1: number, step = 8): [number, number, number][] {
  const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const steps = Math.max(1, Math.ceil(length / step));
  const span = t1 - t0;
  const points: [number, number, number][] = [];
  for (let index = 0; index <= steps; index++) {
    const u = index / steps;
    const [x, y] = bulge(a, b, bow, u);
    points.push([x, y, t0 + span * u]);
  }
  return points;
}

function flownPoints(segments: Segment[]): [number, number, number][] {
  const points: [number, number, number][] = [];
  for (const seg of segments) points.push(...iterPoints([seg.x0, seg.y0], [seg.x1, seg.y1], seg.bow, seg.t0, seg.t1));
  return points;
}

function circlePen(px: number, py: number, cx: number, cy: number, cr: number, radius: number): number {
  return radius + cr - Math.hypot(px - cx, py - cy);
}

function rectPen(lx: number, ly: number, x: number, y: number, w: number, h: number, radius: number): number {
  if (w < 0) {
    x += w;
    w = -w;
  }
  if (h < 0) {
    y += h;
    h = -h;
  }
  const nearestX = Math.min(Math.max(lx, x), x + w);
  const nearestY = Math.min(Math.max(ly, y), y + h);
  if (lx < x || lx > x + w || ly < y || ly > y + h) return radius - Math.hypot(lx - nearestX, ly - nearestY);
  return radius + Math.min(lx - x, x + w - lx, ly - y, y + h - ly);
}

function solidRects(obs: Obstacle): [number, number, number, number][] {
  const length = obs.size;
  if (obs.kind === "wall") {
    const thick = obs.variant === "short" ? 14 : 18;
    const rects: [number, number, number, number][] = [[0, -length / 2, thick, length]];
    if (obs.variant === "double") rects.push([thick + 16, -length * 0.42, thick * 0.75, length * 0.84]);
    if (obs.variant === "corner") rects.push([0, length / 2 - thick, Math.max(54, length * 0.42), thick]);
    return rects;
  }
  if (obs.kind === "platform") {
    const thick = obs.variant === "short" ? 16 : 22;
    return [
      [0, -length / 2, thick, length],
      [thick * 0.35, -length * 0.32 - 5, thick * 1.3, 10],
      [thick * 0.35, length * 0.32 - 5, thick * 1.3, 10],
    ];
  }
  if (obs.kind === "pad") {
    const thick = 28 + (obs.strong ? 10 : 0);
    return [[0, -length / 2, thick, length]];
  }
  return [[0, -length / 2, 16, length]];
}

function penetration(obs: Obstacle, px: number, py: number, radius: number, normal: Pt | null): number {
  const nx = normal ? normal[0] : obs.nx;
  const ny = normal ? normal[1] : obs.ny;
  const tx = -ny;
  const ty = nx;
  const reach = obs.kind === "peg" ? BALL_R + Math.max(obs.pegR, 1) * 3.5 + 8 : obs.kind === "spinner" ? obs.size * 2 + 48 : obs.size * 0.9 + 80;
  if (Math.hypot(px - obs.x, py - obs.y) > reach + radius) return -1;
  if (obs.kind === "peg") {
    const cx = obs.x + nx * (BALL_R + obs.pegR);
    const cy = obs.y + ny * (BALL_R + obs.pegR);
    let worst = circlePen(px, py, cx, cy, obs.pegR, radius);
    if (obs.variant === "pair") {
      worst = Math.max(worst, circlePen(px, py, cx + tx * obs.pegR * 2.4, cy + ty * obs.pegR * 2.4, obs.pegR * 0.78, radius));
    }
    return worst;
  }
  if (obs.kind === "spinner") {
    return circlePen(px, py, obs.x, obs.y, Math.abs(obs.size) * 0.5, radius);
  }
  const ox = obs.x + nx * BALL_R;
  const oy = obs.y + ny * BALL_R;
  const dx = px - ox;
  const dy = py - oy;
  const lx = dx * nx + dy * ny;
  const ly = dx * tx + dy * ty;
  let worst = -1e9;
  for (const [x, y, w, h] of solidRects(obs)) worst = Math.max(worst, rectPen(lx, ly, x, y, w, h, radius));
  return worst;
}

function countPen(obs: Obstacle, px: number, py: number, time: number, normal: Pt | null): number {
  let pen = penetration(obs, px, py, BALL_R, normal);
  if (pen <= 0) return 0;
  if (Math.abs(time - obs.hitTime) <= 0.5 / 30 + 1e-4) pen -= SOLID_EPS;
  return pen > 0 ? pen : 0;
}

function legSolidPen(
  a: Pt,
  b: Pt,
  bow: number,
  t0: number,
  t1: number,
  obstacles: Obstacle[],
  segments: Segment[],
  flown: [number, number, number][],
): number {
  if (!obstacles.length) return 0;
  const overrides = new Map<number, Pt>();
  if (segments.length) {
    const normal = normalFrom(...tangent(segments[segments.length - 1], true), ...tangentOf(a, b, bow, false));
    if (!normal) return 1e6;
    overrides.set(obstacles.length - 1, normal);
  }
  let worst = 0;
  for (const [px, py, time] of iterPoints(a, b, bow, t0, t1)) {
    obstacles.forEach((obs, index) => {
      worst = Math.max(worst, countPen(obs, px, py, time, overrides.get(index) ?? null));
    });
  }
  if (flown.length && overrides.size) {
    const normal = overrides.get(obstacles.length - 1)!;
    const obs = obstacles[obstacles.length - 1];
    for (const [px, py, time] of flown) worst = Math.max(worst, countPen(obs, px, py, time, normal));
  }
  return worst;
}

function shrinkBlockers(
  pos: Pt,
  candidate: Pt,
  t0: number,
  t1: number,
  obstacles: Obstacle[],
  segments: Segment[],
  flown: [number, number, number][],
): boolean {
  const points = iterPoints(pos, candidate, 0, t0, t1);
  const normal = segments.length ? normalFrom(...tangent(segments[segments.length - 1], true), ...tangentOf(pos, candidate, 0, false)) : null;
  let changed = false;
  obstacles.forEach((obs, index) => {
    const use = index === obstacles.length - 1 ? normal : null;
    const pen = () => {
      let worst = 0;
      for (const [px, py, time] of points) worst = Math.max(worst, countPen(obs, px, py, time, use));
      if (use) for (const [px, py, time] of flown) worst = Math.max(worst, countPen(obs, px, py, time, use));
      return worst;
    };
    if (relax(obs, pen)) changed = true;
  });
  return changed;
}

function relax(obs: Obstacle, pen: () => number): boolean {
  if (pen() <= 1e-6) return false;
  const originalVariant = obs.variant;
  const originalSize = Math.max(obs.size, 1);
  const originalPeg = obs.pegR;
  const variants = [originalVariant];
  for (const name of SIMPLER[originalVariant] ?? []) if (!variants.includes(name)) variants.push(name);
  const sizes: number[] = [];
  let size = originalSize;
  while (size > 18) {
    size = Math.max(18, size * 0.72);
    sizes.push(size);
  }
  for (const variant of variants) {
    const trySizes = variant === originalVariant ? sizes : [originalSize, ...sizes];
    for (const next of trySizes) {
      obs.variant = variant;
      obs.size = next;
      if (obs.kind === "peg") obs.pegR = Math.max(10, originalPeg * next / originalSize);
      if (pen() <= 1e-6) return true;
    }
  }
  return true;
}

function capSize(hit: Pt, size: number, obstacles: Obstacle[]): number {
  if (!obstacles.length) return size;
  let nearest = obstacles[0];
  let best = Infinity;
  for (const obs of obstacles) {
    const dist = Math.hypot(hit[0] - obs.x, hit[1] - obs.y);
    if (dist < best) {
      best = dist;
      nearest = obs;
    }
  }
  const room = best - nearest.size * 0.5 - BODY_GAP;
  return Math.min(size, Math.max(MIN_BODY, room * 2));
}

function bodyBoxes(obs: Obstacle): [number, number, number, number][] {
  const nx = obs.nx;
  const ny = obs.ny;
  const tx = -ny;
  const ty = nx;
  if (obs.kind === "peg") {
    const radius = Math.max(obs.pegR, 8);
    const cx = obs.x + nx * (BALL_R + obs.pegR);
    const cy = obs.y + ny * (BALL_R + obs.pegR);
    return [[cx - radius, cy - radius, cx + radius, cy + radius]];
  }
  const ox = obs.x + nx * BALL_R;
  const oy = obs.y + ny * BALL_R;
  return solidRects(obs).map(([x, y, w, h]) => {
    const corners: Pt[] = [
      [x, y],
      [x + w, y],
      [x + w, y + h],
      [x, y + h],
    ];
    const xs = corners.map(([lx, ly]) => ox + nx * lx + tx * ly);
    const ys = corners.map(([lx, ly]) => oy + ny * lx + ty * ly);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  });
}

function boxSep(a: number[], b: number[]): number {
  const dx = a[2] < b[0] ? b[0] - a[2] : b[2] < a[0] ? a[0] - b[2] : -Math.min(a[2] - b[0], b[2] - a[0]);
  const dy = a[3] < b[1] ? b[1] - a[3] : b[3] < a[1] ? a[1] - b[3] : -Math.min(a[3] - b[1], b[3] - a[1]);
  if (dx >= 0 && dy >= 0) return Math.hypot(dx, dy);
  if (dx >= 0) return dx;
  if (dy >= 0) return dy;
  return Math.max(dx, dy);
}

function pairSep(left: Obstacle, right: Obstacle): number {
  let best = Infinity;
  for (const a of bodyBoxes(left)) for (const b of bodyBoxes(right)) best = Math.min(best, boxSep(a, b));
  return best;
}

function flightSamples(segments: Segment[]): [number, number, number][] {
  const points: [number, number, number][] = [];
  for (const seg of segments) points.push(...iterPoints([seg.x0, seg.y0], [seg.x1, seg.y1], seg.bow, seg.t0, seg.t1, 12));
  return points;
}

function penOn(obs: Obstacle, samples: [number, number, number][]): number {
  let worst = 0;
  for (const [px, py, time] of samples) worst = Math.max(worst, countPen(obs, px, py, time, null));
  return worst;
}

function seat(obstacles: Obstacle[], segments: Segment[]): void {
  if (!obstacles.length || !segments.length) return;
  const samples = flightSamples(segments);
  for (const obs of obstacles) {
    if (penOn(obs, samples) <= 1e-6) continue;
    const originalSize = obs.size;
    const originalPeg = obs.pegR;
    const originalVariant = obs.variant;
    const variants = [originalVariant, ...(SIMPLER[originalVariant] ?? [])];
    const sizes = [originalSize];
    let size = originalSize;
    while (size > 18) {
      size = Math.max(18, size * 0.72);
      sizes.push(size);
    }
    let bestPen = penOn(obs, samples);
    let best = { variant: originalVariant, size: originalSize, pegR: originalPeg };
    let cleared = false;
    for (const variant of variants) {
      for (const next of sizes) {
        if (variant === originalVariant && Math.abs(next - originalSize) < 1e-6) continue;
        obs.variant = variant;
        obs.size = next;
        if (obs.kind === "peg") obs.pegR = Math.max(10, originalPeg * (next / Math.max(originalSize, 1)));
        const pen = penOn(obs, samples);
        if (pen <= 1e-6) {
          cleared = true;
          break;
        }
        if (pen < bestPen) {
          bestPen = pen;
          best = { variant, size: next, pegR: obs.pegR };
        }
      }
      if (cleared) break;
    }
    if (!cleared) {
      obs.variant = best.variant;
      obs.size = best.size;
      obs.pegR = best.pegR;
    }
  }
}

function spread(obstacles: Obstacle[], segments: Segment[]): void {
  if (obstacles.length < 2 || !segments.length) return;
  const samples = flightSamples(segments);
  for (let index = 0; index < obstacles.length; index++) {
    const newer = obstacles[index];
    for (const older of obstacles.slice(0, index)) {
      for (let pass = 0; pass < 8; pass++) {
        if (pairSep(older, newer) >= BODY_GAP) break;
        const before = newer.size;
        if (newer.size > MIN_BODY + 0.5) {
          const penBefore = penOn(newer, samples);
          newer.size = Math.max(MIN_BODY, newer.size * 0.82);
          if (newer.kind === "peg") newer.pegR = Math.max(10, newer.pegR * newer.size / before);
          if (penOn(newer, samples) <= penBefore + 1e-4) continue;
          newer.size = before;
        }
        if (older.size > MIN_BODY + 0.5) {
          const penBefore = penOn(older, samples);
          const olderBefore = older.size;
          older.size = Math.max(MIN_BODY, older.size * 0.82);
          if (penOn(older, samples) <= penBefore + 1e-4) continue;
          older.size = olderBefore;
        }
        break;
      }
    }
  }
}
