/** In-place radix-2 FFT. `re` and `im` are reused across frames by the analyzer. */

export function fftMagnitude(
  frame: Float32Array,
  re: Float64Array,
  im: Float64Array,
  mag: Float32Array,
): void {
  const n = frame.length;
  for (let i = 0; i < n; i++) {
    re[i] = frame[i];
    im[i] = 0;
  }
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const theta = (-2 * Math.PI) / size;
    const wpr = Math.cos(theta);
    const wpi = Math.sin(theta);
    for (let i = 0; i < n; i += size) {
      let wr = 1;
      let wi = 0;
      for (let j = 0; j < half; j++) {
        const k = i + j;
        const m = k + half;
        const tr = wr * re[m] - wi * im[m];
        const ti = wr * im[m] + wi * re[m];
        re[m] = re[k] - tr;
        im[m] = im[k] - ti;
        re[k] += tr;
        im[k] += ti;
        const next = wr * wpr - wi * wpi;
        wi = wr * wpi + wi * wpr;
        wr = next;
      }
    }
  }
  const bins = n >> 1;
  for (let i = 0; i < bins; i++) mag[i] = Math.hypot(re[i], im[i]);
}
