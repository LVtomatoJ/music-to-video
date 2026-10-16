/** Ellis spectral-flux beat tracker, ported from tools/flux_tracker.py. */

export function pyRound(x: number): number {
  const negative = x < 0;
  const ax = Math.abs(x);
  const base = Math.floor(ax);
  const frac = ax - base;
  let rounded: number;
  if (Math.abs(frac - 0.5) < 1e-8) {
    rounded = base % 2 === 0 ? base : base + 1;
  } else if (frac > 0.5) {
    rounded = base + 1;
  } else {
    rounded = base;
  }
  return negative ? -rounded : rounded;
}

export function estimateBpm(flux: ArrayLike<number>, frameRate: number): number {
  let mean = 0;
  for (let i = 0; i < flux.length; i++) mean += flux[i];
  mean /= flux.length;
  const centered = new Float64Array(flux.length);
  for (let i = 0; i < flux.length; i++) centered[i] = flux[i] - mean;
  const ac = autocorr(centered);
  const norm = ac[0] + 1e-12;
  let bestScore = -Infinity;
  let bestBpm = 120;
  for (let bpm = 80; bpm <= 170; bpm++) {
    const lag = pyRound((frameRate * 60) / bpm);
    if (lag <= 0 || lag >= ac.length) continue;
    const lo = Math.max(1, lag - 1);
    const hi = Math.min(ac.length, lag + 2);
    let peak = -Infinity;
    for (let i = lo; i < hi; i++) peak = Math.max(peak, ac[i] / norm);
    const prior = Math.exp(-((bpm - 120) ** 2) / (2 * 30 ** 2));
    const score = peak * prior;
    if (score > bestScore) {
      bestScore = score;
      bestBpm = (60 * frameRate) / pyRound((frameRate * 60) / bpm);
    }
  }
  return bestBpm;
}

function autocorr(x: Float64Array): Float64Array {
  const n = x.length;
  const ac = new Float64Array(n);
  for (let lag = 0; lag < n; lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) sum += x[i] * x[i + lag];
    ac[lag] = sum;
  }
  return ac;
}

export function ellisBeats(onset: ArrayLike<number>, frameRate: number, bpm: number): number[] {
  const tightness = 100;
  const period = Math.max(1, (60 / bpm) * frameRate);
  const n = onset.length;
  let prStart = pyRound(-2 * period);
  let prEnd = pyRound(-period / 2);
  if (prEnd >= -1) prEnd = -1;
  if (prStart > prEnd) prStart = prEnd;
  const prange: number[] = [];
  for (let lag = prStart; lag <= prEnd; lag++) prange.push(lag);
  const txwt = prange.map((lag) => -tightness * Math.log(-lag / period) ** 2);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += onset[i];
  mean /= n;
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const d = onset[i] - mean;
    varSum += d * d;
  }
  const std = Math.sqrt(varSum / n);
  const local = new Float64Array(n);
  for (let i = 0; i < n; i++) local[i] = std > 1e-9 ? onset[i] / std : onset[i];
  const backlink = new Int32Array(n);
  backlink.fill(-1);
  const cumscore = new Float64Array(n);
  cumscore.fill(-Infinity);
  const startI = Math.max(0, -prange[0]);
  for (let i = 0; i < Math.min(startI, n); i++) cumscore[i] = local[i];
  for (let i = startI; i < n; i++) {
    let best = -Infinity;
    let chosen = -1;
    let any = false;
    for (let k = 0; k < prange.length; k++) {
      const timer = i + prange[k];
      if (timer >= 0 && timer < i) {
        const score = txwt[k] + cumscore[timer];
        if (!any || score > best) {
          best = score;
          chosen = timer;
          any = true;
        }
      }
    }
    if (!any) {
      cumscore[i] = local[i];
      continue;
    }
    cumscore[i] = local[i] + best;
    backlink[i] = chosen;
  }
  const endLo = Math.max(0, n - Math.max(0, pyRound(period)));
  let end = endLo;
  let best = -Infinity;
  for (let i = endLo; i < n; i++) {
    if (i === endLo || cumscore[i] > best) {
      best = cumscore[i];
      end = i;
    }
  }
  const frames: number[] = [];
  const seen = new Uint8Array(n);
  let cursor = end;
  while (cursor >= 0 && !seen[cursor]) {
    seen[cursor] = 1;
    frames.push(cursor);
    cursor = backlink[cursor];
  }
  frames.reverse();
  return frames.map((frame) => frame / frameRate);
}

export function strongBeats(beatTimes: number[], hfFlux: ArrayLike<number>, frameRate: number): number[] {
  if (!beatTimes.length || !hfFlux.length) return [];
  const vals = beatTimes.map((t) => {
    let idx = pyRound(t * frameRate);
    if (idx < 0) idx = 0;
    if (idx > hfFlux.length - 1) idx = hfFlux.length - 1;
    return hfFlux[idx];
  });
  const threshold = median(vals) * 1.35;
  const out: number[] = [];
  for (let i = 0; i < beatTimes.length; i++) {
    if (vals[i] >= threshold && vals[i] > 0) out.push(beatTimes[i]);
  }
  return out;
}

function median(values: number[]): number {
  const copy = values.slice().sort((a, b) => a - b);
  const n = copy.length;
  if (n % 2 === 1) return copy[(n / 2) | 0];
  return 0.5 * (copy[n / 2 - 1] + copy[n / 2]);
}

export function trackFlux(flux: ArrayLike<number>, hfFlux: ArrayLike<number>, frameRate: number) {
  const tempo = estimateBpm(flux, frameRate);
  const beats = ellisBeats(flux, frameRate, tempo);
  const strong = strongBeats(beats, hfFlux, frameRate);
  return { tempo, beats, strong_beats: strong };
}
