import type { MusicData, Section, ToneEvent } from "../music";
import { chromaAt, sectionAt, soundAt } from "../music";
import { clamp, damp, roundRect, sampleCurve, smoothstep, timesPulse } from "../math";
import { defineTemplate, type ParamSpec } from "../template";
import { BAKE_DT, TAU, as2d, frameCount, rgb, rgba } from "./shared";

const NOTE_COLORS: [number, number, number][] = [
  [0.95, 0.45, 0.48],
  [0.96, 0.62, 0.38],
  [0.96, 0.82, 0.42],
  [0.62, 0.86, 0.46],
  [0.38, 0.82, 0.72],
  [0.38, 0.72, 0.95],
  [0.45, 0.55, 0.95],
  [0.62, 0.48, 0.95],
  [0.82, 0.45, 0.9],
  [0.95, 0.45, 0.72],
  [0.9, 0.55, 0.48],
  [0.78, 0.8, 0.86],
];
const WHITES = new Set([0, 2, 4, 5, 7, 9, 11]);

interface KeyNote {
  time: number;
  duration: number;
  midi: number;
  display: number;
  source: string;
  span: number;
}

interface PianoState {
  width: number;
  height: number;
  unit: number;
  notes: KeyNote[];
  lo: number;
  hi: number;
  keyW: number;
  keyLeft: number;
  keyTop: number;
  whiteH: number;
  lead: number;
  beats: number[];
  strong: number[];
  sections: Section[];
  salt: number;
  energy: Float32Array;
  zoom: Float32Array;
  dip: Float32Array;
}

const pianoParams = {
  seed: { type: "number", label: "种子", min: 1, max: 999, step: 1, default: 42, scope: "plan" },
  harmony: { type: "boolean", label: "和声", default: true, scope: "plan" },
} satisfies Record<string, ParamSpec>;

export const pianoTemplate = defineTemplate<PianoState, typeof pianoParams>({
  meta: {
    id: "piano",
    name: "瀑布琴",
    author: "music-to-video",
    version: "1.0.0",
    description: "音符落到琴键上。旋律来自频谱音高，和声来自色度，同一琴键不会叠在一起。",
  },
  tier: "2d",
  params: pianoParams,
  setup(ctx, music, params) {
    const view = as2d(ctx);
    return buildPiano(view.width, view.height, music, params.harmony, params.seed);
  },
  render(ctx, t, state) {
    const view = as2d(ctx);
    paintPiano(view.ctx, state, t, view);
  },
});

function buildPiano(width: number, height: number, music: MusicData, harmony: boolean, seed: number): PianoState {
  const unit = height / 1080;
  const tones = music.extensions?.tones?.length ? music.extensions.tones : fallbackTones(music);
  const midis = tones.map((tone) => tone.midi).sort((a, b) => a - b);
  const center = midis[midis.length >> 1] ?? 60;
  const [lo, hi] = fourOctaveSpan(center);
  const notes: KeyNote[] = [];
  for (const tone of tones) {
    if (tone.time < 0 || tone.time > music.duration + 1e-3) continue;
    notes.push({
      time: tone.time,
      duration: tone.duration,
      midi: tone.midi,
      display: clampMidi(tone.midi, lo, hi),
      source: tone.source,
      span: tone.duration,
    });
  }
  if (harmony) addArrangement(notes, music, lo, hi);
  openSameKeyGaps(notes);
  notes.sort((a, b) => a.time - b.time || a.display - b.display);
  const whiteCount = Math.max(1, countWhites(lo, hi));
  const board = width * 0.875;
  const keyW = board / whiteCount;
  const whiteH = Math.min(height * 0.24, keyW * 4.35);
  const beats = music.beats.slice();
  const strong = (music.extensions?.strongBeats?.length ? music.extensions.strongBeats : music.downbeats).slice();
  const baked = bakePiano(music.duration, music.sections, beats, strong);
  return {
    width,
    height,
    unit,
    notes,
    lo,
    hi,
    keyW,
    keyLeft: (width - board) / 2,
    keyTop: height - whiteH - height * 0.11,
    whiteH,
    lead: 1.68,
    beats,
    strong,
    sections: music.sections,
    salt: seed,
    energy: baked.energy,
    zoom: baked.zoom,
    dip: baked.dip,
  };
}

function fallbackTones(music: MusicData): ToneEvent[] {
  const beats = music.beats.length ? music.beats : [];
  return beats.filter((time) => soundAt(music, time)).map((time) => ({
    time,
    duration: 0.42,
    midi: 60 + Math.round(chromaPeak(music, time)),
    source: "chroma" as const,
  }));
}

function chromaPeak(music: MusicData, time: number): number {
  const column = chromaAt(music, time);
  let best = 0;
  let pc = 0;
  for (let i = 0; i < 12; i++) {
    if (column[i] > best) {
      best = column[i];
      pc = i;
    }
  }
  return pc;
}

function fourOctaveSpan(center: number): [number, number] {
  let best: [number, number, number] | null = null;
  for (let lo = 36; lo <= 60; lo += 12) {
    const hi = lo + 47;
    if (center < lo || center > hi) continue;
    const score = Math.abs((center - lo) / 47 - 0.62);
    if (!best || score < best[0]) best = [score, lo, hi];
  }
  return best ? [best[1], best[2]] : [48, 95];
}

function clampMidi(midi: number, lo: number, hi: number): number {
  let shown = midi;
  while (shown < lo) shown += 12;
  while (shown > hi) shown -= 12;
  return shown;
}

function countWhites(lo: number, hi: number): number {
  let count = 0;
  for (let tone = lo; tone <= hi; tone++) if (WHITES.has(tone % 12)) count++;
  return count;
}

function whitesBefore(lo: number, midi: number): number {
  let count = 0;
  for (let tone = lo; tone < midi; tone++) if (WHITES.has(tone % 12)) count++;
  return count;
}

function keysFor(lo: number, hi: number, pitchClass: number): number[] {
  const first = lo + ((pitchClass - (lo % 12) + 12) % 12);
  const keys: number[] = [];
  for (let midi = first; midi <= hi; midi += 12) keys.push(midi);
  return keys;
}

function slotFree(taken: Map<number, [number, number][]>, midi: number, time: number, duration: number): boolean {
  const start = time;
  const end = time + duration + 0.05;
  for (const [other, length] of taken.get(midi) ?? []) {
    if (start < other + length + 0.05 && other < end) return false;
  }
  return true;
}

function hold(taken: Map<number, [number, number][]>, midi: number, time: number, duration: number): void {
  const list = taken.get(midi) ?? [];
  list.push([time, duration]);
  taken.set(midi, list);
}

function chordClasses(column: number[], count: number): number[] {
  if (!column.length || Math.max(...column) <= 1e-4) return [];
  const order = [...column.keys()].sort((a, b) => column[b] - column[a]);
  return order.slice(0, count);
}

function voice(
  lo: number,
  hi: number,
  pitchClass: number,
  time: number,
  duration: number,
  taken: Map<number, [number, number][]>,
  prefer: "bass" | "high" | "mid",
): number | null {
  const keys = keysFor(lo, hi, pitchClass);
  if (!keys.length) return null;
  let ordered = keys;
  if (prefer === "bass") ordered = keys.slice(0, 2);
  else if (prefer === "high") ordered = (keys.slice(1).length ? keys.slice(1) : keys).slice().reverse();
  else ordered = keys.slice(1, -1).length ? keys.slice(1, -1) : keys;
  for (const midi of ordered) if (slotFree(taken, midi, time, duration)) return midi;
  for (const midi of keys) if (slotFree(taken, midi, time, duration)) return midi;
  return null;
}

function addArrangement(notes: KeyNote[], music: MusicData, lo: number, hi: number): void {
  const beats = music.beats;
  if (!beats.length) return;
  const taken = new Map<number, [number, number][]>();
  for (const note of notes) hold(taken, note.display, note.time, note.duration);
  const strong = new Set((music.extensions?.strongBeats ?? []).map((hit) => hit.toFixed(5)));
  const bassHits = new Set(bassTimes(music).map((hit) => hit.toFixed(5)));
  for (const beat of beats) {
    if (beat > music.duration + 1e-3) return;
    if (!soundAt(music, beat)) continue;
    const energy = sectionAt(music, beat).energy;
    const count = energy < 0.62 ? 3 : 4;
    const column = chromaAt(music, beat);
    const classes = chordClasses(column, count);
    if (!classes.length) continue;
    const duration = clamp(0.12 + 0.2 * energy, 0.12, 0.34);
    classes.forEach((pitchClass, chordIndex) => {
      const prefer = chordIndex === count - 1 && energy >= 0.62 ? "high" : "mid";
      const midi = voice(lo, hi, pitchClass, beat, duration, taken, prefer);
      if (midi === null) return;
      hold(taken, midi, beat, duration);
      notes.push({ time: beat, duration, midi, display: midi, source: "chroma", span: duration });
    });
    if (bassHits.has(beat.toFixed(5)) && classes.length) {
      const bassDuration = clamp(0.18 + 0.2 * energy, 0.16, 0.42);
      const midi = voice(lo, hi, classes[0], beat, bassDuration, taken, "bass");
      if (midi !== null) {
        hold(taken, midi, beat, bassDuration);
        notes.push({ time: beat, duration: bassDuration, midi, display: midi, source: "bass", span: bassDuration });
      }
    }
    if (strong.has(beat.toFixed(5))) {
      const steps = energy < 0.45 ? 2 : energy < 0.72 ? 3 : 4;
      for (let step = 0; step < steps; step++) {
        const when = beat + 0.085 * (step + 1);
        if (when > music.duration) break;
        const ripple = clamp(0.1 + 0.08 * energy, 0.12, 0.22);
        const midi = voice(lo, hi, classes[step % classes.length], when, ripple, taken, "high");
        if (midi === null) continue;
        hold(taken, midi, when, ripple);
        notes.push({ time: when, duration: ripple, midi, display: midi, source: "arp", span: ripple });
      }
    }
  }
}

function bassTimes(music: MusicData): number[] {
  const beats = music.beats;
  if (!beats.length) return [];
  const lows = beats.map((beat) => {
    const times = music.bands.times;
    const values = music.bands.low;
    if (!times.length) return 0;
    let index = 0;
    let best = Infinity;
    for (let i = 0; i < times.length; i++) {
      const delta = Math.abs(times[i] - beat);
      if (delta < best) {
        best = delta;
        index = i;
      }
    }
    return values[index] ?? 0;
  });
  if (Math.max(...lows) > 0.05) {
    const ordered = lows.slice().sort((a, b) => a - b);
    const cut = Math.max(0.42, ordered[Math.floor(0.62 * (ordered.length - 1))]);
    const found = beats.filter((_, index) => lows[index] >= cut);
    if (found.length) return found;
  }
  return music.extensions?.strongBeats?.length ? music.extensions.strongBeats.slice() : beats.filter((_, index) => index % 4 === 0);
}

function openSameKeyGaps(notes: KeyNote[]): void {
  const groups = new Map<number, KeyNote[]>();
  for (const note of notes) {
    const group = groups.get(note.display) ?? [];
    group.push(note);
    groups.set(note.display, group);
  }
  for (const group of groups.values()) {
    group.sort((a, b) => a.time - b.time || b.duration - a.duration);
    for (let index = 0; index < group.length - 1; index++) {
      const note = group[index];
      const interval = group[index + 1].time - note.time;
      if (interval <= 0.03) note.span = 0;
      else note.span = Math.min(note.duration, Math.max(0, interval - Math.min(0.05, interval * 0.36)));
    }
    group[group.length - 1].span = group[group.length - 1].duration;
  }
}

function bakePiano(duration: number, sections: Section[], beats: number[], strong: number[]) {
  const n = frameCount(duration);
  const energy = new Float32Array(n);
  const zoom = new Float32Array(n);
  const dip = new Float32Array(n);
  let level = sectionEnergy(sections, 0);
  let cam = 1 + 0.04 * level;
  for (let i = 0; i < n; i++) {
    const time = i * BAKE_DT;
    const raw = sectionEnergy(sections, time);
    if (i > 0) {
      level = damp(level, raw, BAKE_DT, 1.15);
      cam = damp(cam, 1 + 0.04 * level, BAKE_DT, 1.25);
    }
    const beat = timesPulse(beats, time, 0.04, 0.15);
    energy[i] = level;
    zoom[i] = cam * (1 + 0.012 * beat * (0.35 + 0.65 * level));
    dip[i] = (3 + 4.2 * level) * timesPulse(strong, time, 0.045, 0.18);
  }
  return { energy, zoom, dip };
}

function sectionEnergy(sections: Section[], time: number): number {
  for (const section of sections) if (time < section.end) return section.energy;
  return sections[sections.length - 1]?.energy ?? 0.4;
}

function keyCenter(state: PianoState, midi: number): [number, number] {
  const shown = clampMidi(midi, state.lo, state.hi);
  const x = WHITES.has(shown % 12)
    ? state.keyLeft + whitesBefore(state.lo, shown) * state.keyW + state.keyW * 0.5
    : state.keyLeft + whitesBefore(state.lo, shown) * state.keyW;
  return [x, state.keyTop];
}

function fallOffset(state: PianoState, remain: number): number {
  const travel = state.keyTop * 0.92;
  if (remain <= 0) return 0;
  const speed = travel / state.lead;
  if (remain >= state.lead) return travel + (remain - state.lead) * speed * 0.72;
  const t = remain / state.lead;
  return travel * (0.72 * t + 0.28 * smoothstep(t));
}

function paintPiano(
  ctx: CanvasRenderingContext2D,
  state: PianoState,
  time: number,
  view: { post: { glow: number; grain: number; vignette: number } },
): void {
  const energy = sampleCurve(state.energy, BAKE_DT, time);
  const zoom = sampleCurve(state.zoom, BAKE_DT, time);
  const dip = sampleCurve(state.dip, BAKE_DT, time) * state.unit;
  const beat = timesPulse(state.beats, time, 0.04, 0.15);
  const camX = (1.6 + 6 * energy) * Math.sin(time * 0.62) * state.unit;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  const sky = ctx.createLinearGradient(0, 0, 0, state.height);
  sky.addColorStop(0, rgb(0.02, 0.024, 0.043));
  sky.addColorStop(1, rgb(0.008, 0.01, 0.02));
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, state.width, state.height);
  ctx.save();
  ctx.translate(state.width / 2, state.keyTop);
  ctx.translate(camX, dip);
  ctx.scale(zoom, zoom);
  ctx.translate(-state.width / 2, -state.keyTop);
  paintAir(ctx, state, time, energy, beat);
  paintBeams(ctx, state, time);
  paintNotes(ctx, state, time, energy, beat);
  paintKeyboard(ctx, state, time);
  paintFloor(ctx, state, time);
  paintSparks(ctx, state, time, energy);
  ctx.restore();
  view.post.glow = 0.78 + 0.35 * beat;
  view.post.grain = 0.7;
  view.post.vignette = 1.05;
}

function liveColors(state: PianoState, time: number): [number, number, number][] {
  const colors: [number, number, number][] = [];
  for (const note of state.notes) {
    const remain = note.time - time;
    const age = time - note.time;
    if ((remain >= -0.05 && remain <= state.lead) || (age >= 0 && age <= note.duration + 0.2)) {
      colors.push(NOTE_COLORS[note.midi % 12]);
      if (colors.length >= 8) break;
    }
  }
  return colors.length ? colors : [[0.478, 0.78, 0.98]];
}

function paintAir(ctx: CanvasRenderingContext2D, state: PianoState, time: number, energy: number, beat: number): void {
  const colors = liveColors(state, time);
  const red = colors.reduce((sum, color) => sum + color[0], 0) / colors.length;
  const green = colors.reduce((sum, color) => sum + color[1], 0) / colors.length;
  const blue = colors.reduce((sum, color) => sum + color[2], 0) / colors.length;
  const wash = 0.11 + 0.16 * energy + 0.09 * beat;
  const glow = ctx.createRadialGradient(state.width * 0.5, state.keyTop * 0.72, 40, state.width * 0.5, state.keyTop * 0.35, state.width * 0.72);
  glow.addColorStop(0, rgba(red, green, blue, wash));
  glow.addColorStop(0.55, rgba(red * 0.7, green * 0.75, blue, wash * 0.45));
  glow.addColorStop(1, rgba(red, green, blue, 0));
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, state.width, state.keyTop + 4);
  const span = state.keyW * countWhites(state.lo, state.hi);
  const count = 28 + Math.floor(22 * energy);
  for (let index = 0; index < count; index++) {
    const drift = (index * 83 + state.salt * 0.37 + time * (14 + (index % 5) * 3)) % span;
    const x = state.keyLeft + drift;
    const travel = state.keyTop * 0.9;
    const y = travel - ((time * (18 + (index % 6) * 4) + index * 37) % travel);
    const twinkle = 0.45 + 0.55 * (0.5 + 0.5 * Math.sin(time * 1.6 + index * 2.399));
    const alpha = (0.06 + 0.1 * energy + 0.05 * beat) * twinkle;
    const tone = colors[index % colors.length];
    ctx.fillStyle = rgba(tone[0], tone[1], tone[2], alpha);
    ctx.beginPath();
    ctx.arc(x, y, (1.15 + (index % 3) * 0.35) * state.unit, 0, TAU);
    ctx.fill();
  }
}

function hitLevel(note: KeyNote, time: number): number {
  const age = time - note.time;
  if (age < 0 || age > note.duration + 0.28) return 0;
  if (age <= note.duration) return 1;
  return smoothstep(1 - (age - note.duration) / 0.28);
}

function paintBeams(ctx: CanvasRenderingContext2D, state: PianoState, time: number): void {
  const best = new Map<number, [number, [number, number, number]]>();
  for (const note of state.notes) {
    let level = hitLevel(note, time);
    const remain = note.time - time;
    if (remain > 0 && remain < state.lead) level = Math.max(level, 0.28 * (1 - remain / state.lead));
    if (level <= 0.04) continue;
    const shown = clampMidi(note.display, state.lo, state.hi);
    const color = NOTE_COLORS[note.midi % 12];
    const previous = best.get(shown);
    if (!previous || level > previous[0]) best.set(shown, [level, color]);
  }
  for (const [midi, [level, color]] of best) {
    const [x, keyY] = keyCenter(state, midi);
    const black = !WHITES.has(midi % 12);
    const beamW = state.keyW * (black ? 0.32 : 0.46);
    const beam = ctx.createLinearGradient(0, 24, 0, keyY);
    beam.addColorStop(0, rgba(color[0], color[1], color[2], 0));
    beam.addColorStop(0.55, rgba(color[0], color[1], color[2], 0.1 * level));
    beam.addColorStop(1, rgba(color[0], color[1], color[2], 0.38 * level));
    ctx.fillStyle = beam;
    ctx.fillRect(x - beamW / 2, 24, beamW, keyY - 24);
  }
}

function paintNotes(ctx: CanvasRenderingContext2D, state: PianoState, time: number, energy: number, beat: number): void {
  const visible: KeyNote[] = [];
  for (const note of state.notes) {
    if (note.span <= 0.02) continue;
    if (time < note.time - state.lead - 0.15 || time > note.time + note.span + 0.05) continue;
    const top = state.keyTop - fallOffset(state, note.time + note.span - time);
    const bottom = state.keyTop - fallOffset(state, note.time - time);
    if (bottom - top < 1.2 || bottom < -24 || top > state.keyTop + 6) continue;
    visible.push(note);
  }
  visible.sort((a, b) => {
    const ay = state.keyTop - fallOffset(state, a.time - time);
    const by = state.keyTop - fallOffset(state, b.time - time);
    return ay - by || a.time - b.time;
  });
  for (const note of visible) {
    const top = state.keyTop - fallOffset(state, note.time + note.span - time);
    const bottom = state.keyTop - fallOffset(state, note.time - time);
    paintPill(ctx, state, note, top, bottom, time, energy, beat);
  }
}

function noteRgb(note: KeyNote, energy: number): [number, number, number] {
  const base = NOTE_COLORS[note.midi % 12];
  const lift = 0.05 + 0.1 * energy;
  return base.map((channel) => Math.min(1, channel + (1 - channel) * lift)) as [number, number, number];
}

function paintPill(
  ctx: CanvasRenderingContext2D,
  state: PianoState,
  note: KeyNote,
  top: number,
  bottom: number,
  time: number,
  energy: number,
  beat: number,
): void {
  const color = noteRgb(note, energy);
  const [x] = keyCenter(state, note.display);
  const black = !WHITES.has(clampMidi(note.display, state.lo, state.hi) % 12);
  const noteW = state.keyW * (black ? 0.34 : 0.48);
  const height = bottom - top;
  const remain = note.time - time;
  let hot = remain > 0 ? 0.8 + 0.2 * smoothstep(1 - Math.min(1, remain / 0.32)) : 1 + 0.28 * pulse(time - note.time, 0.03, 0.11);
  hot *= 1 + 0.14 * beat * (0.45 + 0.55 * energy);
  const rectX = x - noteW / 2;
  const radius = noteW * 0.5;
  roundRect(ctx, rectX - 1.6, top - 1.2, noteW + 3.2, height + 2.4, radius);
  ctx.fillStyle = rgba(color[0], color[1], color[2], 0.14 * Math.min(hot, 1.35));
  ctx.fill();
  roundRect(ctx, rectX, top, noteW, height, radius);
  const fill = ctx.createLinearGradient(0, top, 0, bottom);
  fill.addColorStop(0, rgba(color[0], color[1], color[2], 0.28 * hot));
  fill.addColorStop(0.45, rgba(color[0], color[1], color[2], 0.9 * Math.min(hot, 1)));
  fill.addColorStop(1, rgba(Math.min(1, color[0] + 0.18), Math.min(1, color[1] + 0.16), Math.min(1, color[2] + 0.12), Math.min(1, hot)));
  ctx.fillStyle = fill;
  ctx.fill();
  if (height > noteW * 0.9) {
    const shineH = Math.min(height * 0.42, noteW * 1.3);
    roundRect(ctx, rectX + 2.2, bottom - shineH, noteW - 4.4, shineH - 1.5, Math.max(1, radius - 2.2));
    ctx.fillStyle = rgba(1, 0.98, 0.96, 0.22 * Math.min(hot, 1.1));
    ctx.fill();
  }
}

function pulse(age: number, attack: number, release: number): number {
  if (age < 0 || age >= attack + release) return 0;
  if (age < attack) return smoothstep(age / attack);
  return smoothstep(1 - (age - attack) / release);
}

function pressLevel(state: PianoState, time: number): Map<number, number> {
  const pressed = new Map<number, number>();
  for (const note of state.notes) {
    if (time < note.time || time > note.time + note.duration + 0.22) continue;
    const age = time - note.time;
    const level = age <= note.duration ? 1 : smoothstep(1 - (age - note.duration) / 0.22);
    const shown = clampMidi(note.display, state.lo, state.hi);
    pressed.set(shown, Math.max(pressed.get(shown) ?? 0, level));
  }
  return pressed;
}

function paintKeyboard(ctx: CanvasRenderingContext2D, state: PianoState, time: number): void {
  const pressed = pressLevel(state, time);
  for (let tone = state.lo; tone <= state.hi; tone++) {
    if (!WHITES.has(tone % 12)) continue;
    const x = state.keyLeft + whitesBefore(state.lo, tone) * state.keyW;
    const level = pressed.get(tone) ?? 0;
    const sink = 46 * state.unit * level;
    const gap = Math.max(2.2, state.keyW * 0.07);
    ctx.fillStyle = rgb(0.01, 0.011, 0.016);
    roundRect(ctx, x + gap * 0.5, state.keyTop, state.keyW - gap, state.whiteH, 4);
    ctx.fill();
    roundRect(ctx, x + gap * 0.5 + 0.6, state.keyTop + sink, state.keyW - gap - 1.2, state.whiteH - sink, 4);
    const fill = ctx.createLinearGradient(0, state.keyTop + sink, 0, state.keyTop + state.whiteH);
    if (level > 0) {
      const color = NOTE_COLORS[tone % 12];
      fill.addColorStop(0, rgb(0.35 + 0.6 * color[0], 0.32 + 0.55 * color[1], 0.34 + 0.5 * color[2]));
      fill.addColorStop(0.12, rgb(0.55, 0.57, 0.62));
      fill.addColorStop(1, rgb(0.72, 0.74, 0.78));
    } else {
      fill.addColorStop(0, rgb(0.99, 0.99, 1));
      fill.addColorStop(0.08, rgb(0.93, 0.94, 0.96));
      fill.addColorStop(1, rgb(0.74, 0.76, 0.8));
    }
    ctx.fillStyle = fill;
    ctx.fill();
    if (level > 0.05) {
      ctx.fillStyle = rgba(1, 0.98, 0.94, 0.55 + 0.4 * level);
      ctx.fillRect(x + gap, state.keyTop + sink, state.keyW - gap * 2, 3.5);
    }
  }
  const blackH = state.whiteH * 0.64;
  const blackW = state.keyW * 0.58;
  for (let tone = state.lo; tone <= state.hi; tone++) {
    if (WHITES.has(tone % 12)) continue;
    const x = state.keyLeft + whitesBefore(state.lo, tone) * state.keyW - blackW / 2;
    const level = pressed.get(tone) ?? 0;
    const sink = 24 * state.unit * level;
    ctx.fillStyle = rgb(0.008, 0.008, 0.012);
    roundRect(ctx, x, state.keyTop, blackW, blackH, 3);
    ctx.fill();
    roundRect(ctx, x + 0.8, state.keyTop + sink, blackW - 1.6, blackH - sink, 3);
    if (level > 0) {
      const color = NOTE_COLORS[tone % 12];
      ctx.fillStyle = rgb(0.22 + 0.55 * color[0], 0.16 + 0.42 * color[1], 0.18 + 0.4 * color[2]);
    } else {
      const shade = ctx.createLinearGradient(0, state.keyTop, 0, state.keyTop + blackH);
      shade.addColorStop(0, rgb(0.16, 0.17, 0.2));
      shade.addColorStop(0.2, rgb(0.05, 0.052, 0.07));
      shade.addColorStop(1, rgb(0.02, 0.02, 0.03));
      ctx.fillStyle = shade;
    }
    ctx.fill();
    ctx.fillStyle = rgba(1, 1, 1, level <= 0 ? 0.22 : 0.7);
    ctx.fillRect(x + 2.5, state.keyTop + sink + 2, blackW - 5, 2.2);
  }
}

function paintFloor(ctx: CanvasRenderingContext2D, state: PianoState, time: number): void {
  const pressed = pressLevel(state, time);
  const top = state.keyTop + state.whiteH;
  const whites = countWhites(state.lo, state.hi);
  const board = state.keyW * whites;
  const reflectH = Math.min(state.height - top - 6, state.whiteH * 0.62);
  if (reflectH < 8) return;
  ctx.fillStyle = rgba(0, 0, 0, 0.45);
  ctx.fillRect(state.keyLeft - 6, top, board + 12, 7);
  for (let tone = state.lo; tone <= state.hi; tone++) {
    if (!WHITES.has(tone % 12)) continue;
    const x = state.keyLeft + whitesBefore(state.lo, tone) * state.keyW;
    const level = pressed.get(tone) ?? 0;
    const color = level > 0 ? NOTE_COLORS[tone % 12] : ([0.75, 0.77, 0.82] as [number, number, number]);
    const alpha = 0.16 + 0.28 * level;
    const grad = ctx.createLinearGradient(0, top, 0, top + reflectH);
    grad.addColorStop(0, rgba(color[0], color[1], color[2], alpha));
    grad.addColorStop(1, rgba(color[0], color[1], color[2], 0));
    ctx.fillStyle = grad;
    ctx.fillRect(x + 1.4, top + 6, state.keyW - 2.8, reflectH - 6);
  }
}

function paintSparks(ctx: CanvasRenderingContext2D, state: PianoState, time: number, energy: number): void {
  ctx.lineCap = "round";
  for (const note of state.notes) {
    const age = time - note.time;
    if (age < 0 || age > 0.46) continue;
    const color = noteRgb(note, energy);
    const [x, keyY] = keyCenter(state, note.display);
    const fade = smoothstep(1 - age / 0.46);
    const half = state.keyW * 0.55;
    const splash = ctx.createLinearGradient(x - half, keyY, x + half, keyY);
    splash.addColorStop(0, rgba(color[0], color[1], color[2], 0));
    splash.addColorStop(0.5, rgba(1, 0.98, 0.95, 0.9 * fade));
    splash.addColorStop(1, rgba(color[0], color[1], color[2], 0));
    ctx.fillStyle = splash;
    ctx.fillRect(x - half, keyY - 7, half * 2, 9);
    const ring = smoothstep(Math.min(1, age / 0.32));
    ctx.strokeStyle = rgba(color[0], color[1], color[2], fade * 0.9);
    ctx.lineWidth = 3.2 * fade + 1.4;
    ctx.beginPath();
    ctx.arc(x, keyY + 2, (22 + ring * 70) * state.unit, Math.PI * 1.08, Math.PI * 1.92);
    ctx.stroke();
    if (note.source === "chroma" || note.source === "bass" || note.source === "arp") continue;
    for (let spark = 0; spark < 7; spark++) {
      const angle = Math.PI * 1.05 + spark * ((Math.PI * 0.9) / 6);
      const dist = (46 + age * (280 + spark * 16)) * state.unit;
      const x2 = x + Math.cos(angle) * dist;
      const y2 = keyY + Math.sin(angle) * dist;
      const x1 = x + Math.cos(angle) * (dist - 18 * state.unit);
      const y1 = keyY + Math.sin(angle) * (dist - 18 * state.unit);
      ctx.strokeStyle = rgba(color[0], color[1], color[2], fade * 0.85);
      ctx.lineWidth = 3.4 * fade + 0.6;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }
}
