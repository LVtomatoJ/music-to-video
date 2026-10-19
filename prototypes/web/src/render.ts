/** Canvas2D frames encoded with WebCodecs and muxed by mp4-muxer. */

import { ArrayBufferTarget, Muxer } from "mp4-muxer";
import { cssColor, hslToRgb, makeScene, stepScene, type Scene } from "./scene";
import type { Analysis } from "./analyze";

const FPS = 30;

export async function renderVideo(analysis: Analysis, width: number, height: number): Promise<Uint8Array> {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) throw new Error("canvas 2d unavailable");

  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    video: { codec: "avc", width, height, frameRate: FPS },
    fastStart: "in-memory",
    firstTimestampBehavior: "offset",
  });
  let encoderError: unknown = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (error) => {
      encoderError = error;
    },
  });
  encoder.configure({
    codec: "avc1.640028",
    width,
    height,
    bitrate: 8_000_000,
    framerate: FPS,
    avc: { format: "avc" },
    latencyMode: "quality",
    hardwareAcceleration: "prefer-software",
  });

  const scene = makeScene(width, height);
  const strong = strongFlags(analysis.beats, analysis.strong_beats);
  const nframes = Math.floor(analysis.duration * FPS);
  const dt = 1 / FPS;
  let beatIndex = 0;
  let flash = 0;
  for (let frame = 0; frame < nframes; frame++) {
    const t0 = frame * dt;
    const t1 = t0 + dt;
    let beat = false;
    let isStrong = false;
    while (beatIndex < analysis.beats.length && analysis.beats[beatIndex] < t1) {
      if (analysis.beats[beatIndex] >= t0) {
        beat = true;
        isStrong = isStrong || strong[beatIndex];
      }
      beatIndex += 1;
    }
    const pulse = stepScene(scene, beat, isStrong, dt);
    flash = Math.max(flash * 0.86, pulse);
    draw(ctx, scene, flash, width, height);
    const videoFrame = new VideoFrame(canvas, {
      timestamp: Math.round((frame * 1_000_000) / FPS),
      duration: Math.round(1_000_000 / FPS),
    });
    encoder.encode(videoFrame, { keyFrame: frame % 60 === 0 });
    videoFrame.close();
    if (encoderError) throw encoderError;
    if (frame % 20 === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }
  await encoder.flush();
  if (encoderError) throw encoderError;
  muxer.finalize();
  return new Uint8Array(target.buffer);
}

function strongFlags(beats: number[], strongBeats: number[]): boolean[] {
  const flags: boolean[] = [];
  let si = 0;
  for (const time of beats) {
    while (si < strongBeats.length && strongBeats[si] < time - 1e-6) si += 1;
    const isStrong = si < strongBeats.length && Math.abs(strongBeats[si] - time) <= 1e-6;
    flags.push(isStrong);
    if (isStrong) si += 1;
  }
  return flags;
}

function draw(ctx: CanvasRenderingContext2D, scene: Scene, flash: number, width: number, height: number) {
  const lift = flash * 36;
  ctx.fillStyle = `rgb(${14 + lift},${16 + lift},${24 + lift})`;
  ctx.fillRect(0, 0, width, height);
  for (const particle of scene.particles) {
    const k = Math.max(0, particle.life / particle.maxLife);
    const [r, g, b] = hslToRgb(particle.hue, 0.8, 0.35 + 0.35 * k);
    ctx.beginPath();
    ctx.fillStyle = cssColor(r, g, b);
    ctx.arc(particle.x, particle.y, particle.r * (0.35 + 0.65 * k), 0, Math.PI * 2);
    ctx.fill();
  }
  for (const ball of scene.balls) {
    const [r, g, b] = hslToRgb(ball.hue, 0.75, 0.55);
    ctx.beginPath();
    ctx.fillStyle = cssColor(r, g, b);
    ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = ball.flash > 0.4 ? "#fff" : cssColor(...hslToRgb(ball.hue, 0.8, 0.75));
    ctx.stroke();
  }
}
