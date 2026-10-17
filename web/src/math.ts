export function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

export function smoothstep(value: number): number {
  const u = clamp(value, 0, 1);
  return u * u * (3 - 2 * u);
}

export function damp(current: number, target: number, dt: number, rate: number): number {
  return target + (current - target) * Math.exp(-rate * dt);
}

export function interp(times: ArrayLike<number>, values: ArrayLike<number>, time: number): number {
  const n = times.length;
  if (n === 0) return 0;
  if (time <= times[0]) return values[0];
  if (time >= times[n - 1]) return values[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= time) lo = mid;
    else hi = mid;
  }
  const span = times[hi] - times[lo] || 1;
  return values[lo] + ((values[hi] - values[lo]) * (time - times[lo])) / span;
}

export function sampleCurve(values: ArrayLike<number>, dt: number, time: number): number {
  if (values.length === 0) return 0;
  const x = time / dt;
  const i = Math.floor(x);
  if (i <= 0) return values[0];
  if (i >= values.length - 1) return values[values.length - 1];
  const f = x - i;
  return values[i] * (1 - f) + values[i + 1] * f;
}

export function percentile(values: ArrayLike<number>, p: number): number {
  const n = values.length;
  if (n === 0) return 0;
  const copy = new Float64Array(n);
  for (let i = 0; i < n; i++) copy[i] = values[i];
  copy.sort();
  const index = clamp(p / 100, 0, 1) * (n - 1);
  const lo = Math.floor(index);
  const hi = Math.min(n - 1, lo + 1);
  return copy[lo] + (copy[hi] - copy[lo]) * (index - lo);
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashString(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function mixSeed(a: number, b: number): number {
  let x = (a >>> 0) ^ Math.imul(b >>> 0, 0x9e3779b1);
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  x = (x ^ (x >>> 16)) >>> 0;
  return x || 1;
}

export function lowerBound(times: ArrayLike<number>, target: number): number {
  let lo = 0;
  let hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
