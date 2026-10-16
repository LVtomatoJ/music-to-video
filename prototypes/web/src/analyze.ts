/**
 * Web Audio analysis: decodeAudioData + OfflineAudioContext AnalyserNode.
 * AnalyserNode only exposes the current spectrum, so rendering suspends once
 * per hop (512 samples) and reads getFloatFrequencyData. That series feeds
 * the same Ellis tracker as the Rust prototype.
 */

import { trackFlux } from "./tracker";

const HOP = 512;
const FFT = 2048;

export interface Analysis {
  tempo: number;
  beats: number[];
  strong_beats: number[];
  duration: number;
  sampleRate: number;
}

export async function analyze(arrayBuffer: ArrayBuffer): Promise<Analysis> {
  const decodeCtx = new AudioContext();
  const audio = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
  await decodeCtx.close();
  const { flux, hfFlux, frameRate } = await fluxViaAnalyser(audio);
  const tracked = trackFlux(flux, hfFlux, frameRate);
  return {
    tempo: tracked.tempo,
    beats: tracked.beats,
    strong_beats: tracked.strong_beats,
    duration: audio.duration,
    sampleRate: audio.sampleRate,
  };
}

async function fluxViaAnalyser(audio: AudioBuffer): Promise<{
  flux: Float64Array;
  hfFlux: Float64Array;
  frameRate: number;
}> {
  const sr = audio.sampleRate;
  const nFrames = 1 + Math.floor((audio.length - FFT) / HOP);
  const offline = new OfflineAudioContext(1, audio.length, sr);
  const source = offline.createBufferSource();
  source.buffer = audio;
  const analyser = offline.createAnalyser();
  analyser.fftSize = FFT;
  analyser.smoothingTimeConstant = 0;
  analyser.minDecibels = -140;
  analyser.maxDecibels = 0;
  source.connect(analyser);
  analyser.connect(offline.destination);
  source.start(0);

  const flux = new Float64Array(nFrames);
  const hfFlux = new Float64Array(nFrames);
  const prev = new Float64Array(analyser.frequencyBinCount);
  const binHz = sr / FFT;
  const hfStart = Math.ceil(2000 / binHz);
  let captured = 0;

  for (let i = 0; i < nFrames; i++) {
    const endSample = i * HOP + FFT;
    const time = endSample / sr;
    if (time >= audio.duration) break;
    const frameIndex = i;
    offline.suspend(time).then(() => {
      const db = new Float32Array(analyser.frequencyBinCount);
      analyser.getFloatFrequencyData(db);
      let full = 0;
      let high = 0;
      for (let k = 0; k < db.length; k++) {
        const mag = db[k] <= -140 ? 0 : 10 ** (db[k] / 20);
        const delta = mag - prev[k];
        if (delta > 0) {
          full += delta;
          if (k >= hfStart) high += delta;
        }
        prev[k] = mag;
      }
      flux[frameIndex] = full;
      hfFlux[frameIndex] = high;
      captured += 1;
      offline.resume();
    });
  }

  await offline.startRendering();
  if (captured < nFrames * 0.9) {
    throw new Error(`AnalyserNode captured ${captured}/${nFrames} frames`);
  }
  return { flux, hfFlux, frameRate: sr / HOP };
}
