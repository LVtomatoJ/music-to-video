import type { TemplateContext } from "../template";

export const TAU = Math.PI * 2;
export const BAKE_DT = 1 / 60;

export function as2d(ctx: TemplateContext) {
  if (ctx.tier !== "2d") throw new Error("这个模板用 2D 这一层画。");
  return ctx;
}

export function rgb(r: number, g: number, b: number): string {
  return `rgb(${(r * 255).toFixed(1)},${(g * 255).toFixed(1)},${(b * 255).toFixed(1)})`;
}

export function rgba(r: number, g: number, b: number, a: number): string {
  return `rgba(${(r * 255).toFixed(1)},${(g * 255).toFixed(1)},${(b * 255).toFixed(1)},${a})`;
}

export function frameCount(duration: number): number {
  return Math.floor(Math.max(duration, 0.5) / BAKE_DT) + 3;
}
