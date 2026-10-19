import type { PostSettings } from "./template";
import { clamp, mulberry32 } from "./math";

const quarter = document.createElement("canvas");
const blurred = document.createElement("canvas");
const grain = document.createElement("canvas");

/**
 * The same film pass the Python renderer applies after a scene:
 * thresholded bloom, a vignette, and seeded grain. Grain is a soft-light
 * overlay so the main canvas can stay on the GPU.
 */
export function applyPost(ctx: CanvasRenderingContext2D, post: PostSettings, time: number): void {
  const canvas = ctx.canvas;
  const width = canvas.width;
  const height = canvas.height;
  bloom(ctx, width, height, post.glow);
  vignette(ctx, width, height, post.vignette);
  filmGrain(ctx, width, height, post.grain, time);
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
  ctx.filter = "none";
}

function bloom(ctx: CanvasRenderingContext2D, width: number, height: number, amount: number): void {
  const bw = Math.max(2, Math.floor(width / 4));
  const bh = Math.max(2, Math.floor(height / 4));
  quarter.width = bw;
  quarter.height = bh;
  const small = quarter.getContext("2d", { willReadFrequently: true });
  if (!small) return;
  small.setTransform(bw / width, 0, 0, bh / height, 0, 0);
  small.drawImage(ctx.canvas, 0, 0);
  small.setTransform(1, 0, 0, 1, 0, 0);
  const image = small.getImageData(0, 0, bw, bh);
  const data = image.data;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = Math.max(0, data[i] - 24);
    data[i + 1] = Math.max(0, data[i + 1] - 24);
    data[i + 2] = Math.max(0, data[i + 2] - 24);
    data[i + 3] = 255;
  }
  small.putImageData(image, 0, 0);
  blurred.width = bw;
  blurred.height = bh;
  const soft = blurred.getContext("2d");
  if (!soft) return;
  soft.clearRect(0, 0, bw, bh);
  soft.filter = "blur(5px)";
  soft.drawImage(quarter, 0, 0);
  soft.filter = "none";
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = "lighter";
  ctx.globalAlpha = 0.3 * clamp(amount, 0, 1.6);
  ctx.drawImage(blurred, 0, 0, width, height);
  ctx.restore();
}

function vignette(ctx: CanvasRenderingContext2D, width: number, height: number, amount: number): void {
  const cx = width / 2;
  const cy = height / 2;
  const inner = Math.min(width, height) * 0.28;
  const outer = Math.hypot(width, height) * 0.58;
  const edge = 0.38 * clamp(amount, 0, 1.5);
  const gradient = ctx.createRadialGradient(cx, cy, inner, cx, cy, outer);
  gradient.addColorStop(0, "rgba(158, 198, 230, 0.05)");
  gradient.addColorStop(0.42, "rgba(0, 0, 0, 0)");
  gradient.addColorStop(1, `rgba(5, 3, 7, ${edge})`);
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  ctx.restore();
}

function filmGrain(ctx: CanvasRenderingContext2D, width: number, height: number, amount: number, time: number): void {
  const size = 160;
  grain.width = size;
  grain.height = size;
  const g = grain.getContext("2d", { willReadFrequently: true });
  if (!g) return;
  const image = g.createImageData(size, size);
  const data = image.data;
  const seed = (Math.floor(time * 30) * 1103515245 + 12345) >>> 0;
  const rng = mulberry32(seed || 1);
  for (let i = 0; i < data.length; i += 4) {
    const n = 108 + Math.floor(rng() * 40);
    data[i] = n;
    data[i + 1] = n;
    data[i + 2] = n;
    data[i + 3] = 255;
  }
  g.putImageData(image, 0, 0);
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = "soft-light";
  ctx.globalAlpha = 0.28 * clamp(amount, 0, 1.5);
  ctx.drawImage(grain, 0, 0, width, height);
  ctx.restore();
}
