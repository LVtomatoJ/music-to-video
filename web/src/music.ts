import { ellisBeats, estimateBpm, strongBeats } from "./beats";
import { fftMagnitude } from "./fft";
import { clamp, interp, percentile } from "./math";

export const MUSIC_DATA_VERSION = 1 as const;

export type SectionLabel = "intro" | "verse" | "chorus" | "bridge" | "outro";

export interface Onset {
  time: number;
  strength: number;
}

export interface Section {
  start: number;
  end: number;
  energy: number;
  label: SectionLabel;
}

export interface MusicData {
  version: typeof MUSIC_DATA_VERSION;
  duration: number;
  sampleRate: number;
  /** Beats per minute, from the Ellis tracker. */
  tempo: number;
  /** Beats per bar. v1 always uses 4. */
  meter: number;
  beats: number[];
  /** First beat of each bar. The spans between them are the bars. */
  downbeats: number[];
  onsets: Onset[];
  /** Loudness envelope, smoothed over about 1.2 s and normalized to 0..1. */
  loudness: { times: number[]; values: number[] };
  /** Band energy over the same hop grid, lightly smoothed and normalized per band. */
  bands: { times: number[]; low: number[]; mid: number[]; high: number[] };
  sections: Section[];
  /**
   * Optional fields. Templates must ignore keys they do not know.
   * `strongBeats` is the accented subset of `beats` (high-frequency flux).
   */
  extensions?: {
    analyzer?: string;
    strongBeats?: number[];
  };
}

const ANALYSIS_RATE = 22050;
const FFT_SIZE = 1024;
const HOP = 512;

export interface AnalyzedAudio {
  music: MusicData;
  audio: AudioBuffer;
}

export async function decodeAndAnalyze(
  bytes: ArrayBuffer,
  onProgress?: (ratio: number) => void,
): Promise<AnalyzedAudio> {
  onProgress?.(0.02);
  const decodeCtx = new AudioContext();
  const audio = await decodeCtx.decodeAudioData(bytes.slice(0));
  await decodeCtx.close();
  onProgress?.(0.08);
  const samples = await renderMono(audio, ANALYSIS_RATE);
  onProgress?.(0.16);
  const music = await analyzeSamples(samples, ANALYSIS_RATE, audio.duration, onProgress);
  onProgress?.(1);
  return { music, audio };
}

async function renderMono(audio: AudioBuffer, sampleRate: number): Promise<Float32Array> {
  const length = Math.max(1, Math.ceil(audio.duration * sampleRate));
  const offline = new OfflineAudioContext(1, length, sampleRate);
  const source = offline.createBufferSource();
  source.buffer = audio;
  source.connect(offline.destination);
  source.start(0);
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0).slice();
}

async function analyzeSamples(
  input: Float32Array,
  sampleRate: number,
  duration: number,
  onProgress?: (ratio: number) => void,
): Promise<MusicData> {
  let samples = input;
  if (samples.length < FFT_SIZE) {
    const padded = new Float32Array(FFT_SIZE);
    padded.set(samples);
    samples = padded;
  }
  const frameRate = sampleRate / HOP;
  const nFrames = Math.max(1, Math.floor((samples.length - FFT_SIZE) / HOP) + 1);
  const window = new Float32Array(FFT_SIZE);
  for (let i = 0; i < FFT_SIZE; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)));
  }
  const frame = new Float32Array(FFT_SIZE);
  const re = new Float64Array(FFT_SIZE);
  const im = new Float64Array(FFT_SIZE);
  const mag = new Float32Array(FFT_SIZE >> 1);
  const prev = new Float32Array(FFT_SIZE >> 1);
  const flux = new Float32Array(nFrames);
  const hfFlux = new Float32Array(nFrames);
  const rms = new Float32Array(nFrames);
  const low = new Float32Array(nFrames);
  const mid = new Float32Array(nFrames);
  const high = new Float32Array(nFrames);
  const binHz = sampleRate / FFT_SIZE;
  const hfStart = Math.ceil(2000 / binHz);
  const lowBin = Math.max(1, Math.floor(20 / binHz));
  const midBin = Math.ceil(250 / binHz);
  const highBin = Math.ceil(2000 / binHz);
  const airBin = Math.min(mag.length, Math.ceil(8000 / binHz));

  for (let i = 0; i < nFrames; i++) {
    const offset = i * HOP;
    let energy = 0;
    const hopEnd = Math.min(HOP, samples.length - offset);
    for (let n = 0; n < hopEnd; n++) {
      const s = samples[offset + n];
      energy += s * s;
    }
    rms[i] = Math.sqrt(energy / Math.max(1, hopEnd));
    for (let n = 0; n < FFT_SIZE; n++) frame[n] = (samples[offset + n] || 0) * window[n];
    fftMagnitude(frame, re, im, mag);
    let full = 0;
    let hf = 0;
    let lowSum = 0;
    let midSum = 0;
    let highSum = 0;
    for (let k = 1; k < mag.length; k++) {
      const power = Math.log1p(mag[k]);
      const delta = power - prev[k];
      if (delta > 0) {
        full += delta;
        if (k >= hfStart) hf += delta;
      }
      prev[k] = power;
      if (k >= lowBin && k < midBin) lowSum += power;
      else if (k >= midBin && k < highBin) midSum += power;
      else if (k >= highBin && k < airBin) highSum += power;
    }
    flux[i] = full;
    hfFlux[i] = hf;
    low[i] = lowSum;
    mid[i] = midSum;
    high[i] = highSum;
    if (i % 180 === 0) {
      onProgress?.(0.16 + 0.7 * (i / nFrames));
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  const times = Array.from({ length: nFrames }, (_, i) => i / frameRate);
  const loudness = normalize(smoothSeries(rms, frameRate, 1.2));
  const bands = {
    times,
    low: asArray(normalize(smoothSeries(low, frameRate, 0.35))),
    mid: asArray(normalize(smoothSeries(mid, frameRate, 0.35))),
    high: asArray(normalize(smoothSeries(high, frameRate, 0.35))),
  };
  const novelty = halfWave(flux, smoothSeries(flux, frameRate, 0.35));
  let tempo = estimateBpm(novelty, frameRate);
  if (!Number.isFinite(tempo) || tempo < 70 || tempo > 180) tempo = 120;
  let beats = ellisBeats(novelty, frameRate, tempo).filter((t) => t > 0.05 && t < duration - 0.05);
  if (beats.length < 4 && duration > 2) {
    beats = [];
    const period = 60 / tempo;
    for (let t = period * 0.5; t < duration - 0.05; t += period) beats.push(t);
  }
  const accents = strongBeats(beats, hfFlux, frameRate);
  const downbeats = chooseDownbeats(beats, times, bands.low);
  const onsets = pickOnsets(novelty, frameRate, duration);
  const sections = buildSections(times, loudness, duration);
  return {
    version: MUSIC_DATA_VERSION,
    duration,
    sampleRate,
    tempo,
    meter: 4,
    beats,
    downbeats,
    onsets,
    loudness: { times, values: asArray(loudness) },
    bands,
    sections,
    extensions: {
      analyzer: "webaudio-fft-ellis-v1",
      strongBeats: accents,
    },
  };
}

function asArray(values: Float32Array): number[] {
  return Array.from(values);
}

function smoothSeries(values: Float32Array, frameRate: number, windowSec: number): Float32Array {
  const radius = Math.max(1, Math.round((windowSec * frameRate) / 2));
  const prefix = new Float64Array(values.length + 1);
  for (let i = 0; i < values.length; i++) prefix[i + 1] = prefix[i] + values[i];
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const a = Math.max(0, i - radius);
    const b = Math.min(values.length - 1, i + radius);
    out[i] = (prefix[b + 1] - prefix[a]) / (b - a + 1);
  }
  return out;
}

function normalize(values: Float32Array): Float32Array {
  const scale = percentile(values, 97) + 1e-9;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = clamp(values[i] / scale, 0, 1);
  return out;
}

function halfWave(values: Float32Array, slow: Float32Array): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = Math.max(0, values[i] - slow[i]);
  return out;
}

function pickOnsets(flux: Float32Array, frameRate: number, duration: number): Onset[] {
  const peak = percentile(flux, 99) + 1e-9;
  const out: Onset[] = [];
  let lastTime = -1;
  let lastStrength = 0;
  for (let i = 2; i < flux.length - 2; i++) {
    if (flux[i] < flux[i - 1] || flux[i] <= flux[i + 1]) continue;
    const w0 = Math.max(0, i - 8);
    const w1 = Math.min(flux.length, i + 9);
    let local = 0;
    for (let k = w0; k < w1; k++) local += flux[k];
    local /= w1 - w0;
    if (flux[i] < local * 1.35 + peak * 0.02) continue;
    const time = i / frameRate;
    if (time < 0.03 || time > duration) continue;
    const strength = clamp(flux[i] / peak, 0, 1);
    if (strength < 0.08) continue;
    if (time - lastTime < 0.05) {
      if (strength > lastStrength && out.length) {
        out[out.length - 1] = { time, strength };
        lastTime = time;
        lastStrength = strength;
      }
      continue;
    }
    out.push({ time, strength });
    lastTime = time;
    lastStrength = strength;
  }
  return out;
}

function chooseDownbeats(beats: number[], times: number[], low: number[]): number[] {
  if (beats.length < 4) return beats.slice();
  let bestPhase = 0;
  let bestScore = -Infinity;
  for (let phase = 0; phase < 4; phase++) {
    let sum = 0;
    let count = 0;
    for (let i = phase; i < beats.length; i += 4) {
      sum += interp(times, low, beats[i]);
      count++;
    }
    const score = sum / count;
    if (score > bestScore) {
      bestScore = score;
      bestPhase = phase;
    }
  }
  const out: number[] = [];
  for (let i = bestPhase; i < beats.length; i += 4) out.push(beats[i]);
  return out;
}

function buildSections(times: number[], loudness: Float32Array, duration: number): Section[] {
  if (duration <= 0.5 || times.length === 0) {
    return [{ start: 0, end: Math.max(duration, 0.5), energy: 0.4, label: "verse" }];
  }
  const step = 0.5;
  const samples: { t: number; e: number; kind: "low" | "mid" | "high" }[] = [];
  const values: number[] = [];
  for (let t = 0; t < duration; t += step) {
    values.push(interp(times, loudness, t));
  }
  const lowCut = percentile(values, 34);
  const highCut = percentile(values, 67);
  const flat = highCut - lowCut < 0.08;
  values.forEach((e, index) => {
    const t = index * step;
    let kind: "low" | "mid" | "high";
    if (flat) kind = t > duration * 0.28 && t < duration * 0.78 ? "high" : "mid";
    else if (e >= highCut) kind = "high";
    else if (e <= lowCut) kind = "low";
    else kind = "mid";
    samples.push({ t, e, kind });
  });
  type Run = { start: number; end: number; kind: "low" | "mid" | "high"; sum: number; count: number };
  const runs: Run[] = [];
  for (const sample of samples) {
    const end = Math.min(duration, sample.t + step);
    const prev = runs[runs.length - 1];
    if (prev && prev.kind === sample.kind) {
      prev.end = end;
      prev.sum += sample.e;
      prev.count += 1;
    } else {
      runs.push({ start: sample.t, end, kind: sample.kind, sum: sample.e, count: 1 });
    }
  }
  for (let i = 0; i < runs.length; i++) {
    const run = runs[i];
    if (run.end - run.start >= 4 || runs.length === 1) continue;
    const host = i > 0 ? runs[i - 1] : runs[i + 1];
    if (!host) continue;
    host.start = Math.min(host.start, run.start);
    host.end = Math.max(host.end, run.end);
    host.sum += run.sum;
    host.count += run.count;
    runs.splice(i, 1);
    i--;
  }
  const highIndexes = new Set(runs.map((run, index) => (run.kind === "high" ? index : -1)).filter((i) => i >= 0));
  const sections: Section[] = runs.map((run, index) => {
    const prevHigh = highIndexes.has(index - 1);
    const nextHigh = highIndexes.has(index + 1);
    let label: SectionLabel;
    if (index === 0 && run.kind !== "high" && run.end - run.start <= 28) label = "intro";
    else if (index === runs.length - 1 && run.kind !== "high" && duration - run.start <= 28) label = "outro";
    else if (run.kind === "high") label = "chorus";
    else if (prevHigh && nextHigh) label = "bridge";
    else label = "verse";
    return { start: run.start, end: run.end, energy: run.sum / Math.max(1, run.count), label };
  });
  if (sections.length) {
    sections[0].start = 0;
    sections[sections.length - 1].end = duration;
  }
  return sections.length ? sections : [{ start: 0, end: duration, energy: 0.4, label: "verse" }];
}

/** A short made-up groove so the pond is already moving before a file is chosen. */
export function demoMusic(): MusicData {
  const duration = 24;
  const tempo = 100;
  const period = 60 / tempo;
  const beats: number[] = [];
  for (let t = 0.35; t < duration - 0.1; t += period) beats.push(Number(t.toFixed(4)));
  const downbeats = beats.filter((_, index) => index % 4 === 0);
  const onsets: Onset[] = [];
  for (let i = 0; i < beats.length; i++) {
    const time = beats[i];
    const chorus = time >= 8 && time < 18;
    onsets.push({ time, strength: i % 4 === 0 ? 0.92 : chorus ? 0.62 : 0.4 });
    if (chorus) onsets.push({ time: time + period * 0.5, strength: 0.48 });
  }
  const times: number[] = [];
  const values: number[] = [];
  const low: number[] = [];
  const mid: number[] = [];
  const high: number[] = [];
  for (let t = 0; t <= duration; t += 0.05) {
    times.push(t);
    const env = t < 4 ? 0.28 : t < 8 ? 0.42 : t < 18 ? 0.86 : 0.34;
    const wobble = 0.06 * Math.sin(t * 1.7);
    values.push(clamp(env + wobble, 0, 1));
    low.push(clamp(env * 0.9, 0, 1));
    mid.push(clamp(env * 0.75 + (t >= 8 && t < 18 ? 0.1 : 0), 0, 1));
    high.push(clamp((t >= 8 && t < 18 ? 0.7 : 0.25) + wobble, 0, 1));
  }
  return {
    version: MUSIC_DATA_VERSION,
    duration,
    sampleRate: ANALYSIS_RATE,
    tempo,
    meter: 4,
    beats,
    downbeats,
    onsets,
    loudness: { times, values },
    bands: { times, low, mid, high },
    sections: [
      { start: 0, end: 4, energy: 0.28, label: "intro" },
      { start: 4, end: 8, energy: 0.42, label: "verse" },
      { start: 8, end: 18, energy: 0.86, label: "chorus" },
      { start: 18, end: 24, energy: 0.34, label: "outro" },
    ],
    extensions: {
      analyzer: "demo",
      strongBeats: downbeats,
    },
  };
}
