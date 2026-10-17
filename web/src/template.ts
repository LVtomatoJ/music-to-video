import type { MusicData } from "./music";

export interface TemplateMeta {
  id: string;
  name: string;
  author: string;
  version: string;
  description: string;
}

export interface PostSettings {
  /** Highlight bloom. 1 matches the Python film pass; beats may push it a little higher. */
  glow: number;
  /** Seeded grain amount. 1 is the normal film grain. */
  grain: number;
  /** Edge darkening. 1 is the rain palette's vignette. */
  vignette: number;
}

export interface Canvas2DContext {
  tier: "2d";
  width: number;
  height: number;
  ctx: CanvasRenderingContext2D;
  /**
   * Seeded uniform random in [0, 1). During setup the stream runs forward once.
   * During render the platform reseeds it from (template, seed, frame), so a
   * frame never depends on which frame was drawn before it.
   */
  rng: () => number;
  post: PostSettings;
}

export interface WebGPUContext {
  tier: "webgpu";
  width: number;
  height: number;
  /** GPUDevice from the browser. Typed loosely so the spec compiles without a WebGPU lib. */
  device: object;
  /** GPUCanvasContext. */
  context: object;
  /** GPUTextureFormat, for example "bgra8unorm". */
  format: string;
  rng: () => number;
  post: PostSettings;
}

export type TemplateContext = Canvas2DContext | WebGPUContext;

export type ParamSpec =
  | {
      type: "number";
      label: string;
      min: number;
      max: number;
      step?: number;
      default: number;
      /** "frame" knobs skip setup(); the plan stays put and only the picture changes. */
      scope?: "plan" | "frame";
    }
  | {
      type: "color";
      label: string;
      default: string;
      scope?: "plan" | "frame";
    }
  | {
      type: "enum";
      label: string;
      options: { value: string; label: string }[];
      default: string;
      scope?: "plan" | "frame";
    }
  | {
      type: "boolean";
      label: string;
      default: boolean;
      scope?: "plan" | "frame";
    };

export type ParamValues = Record<string, number | string | boolean>;

export type ParamsOf<Spec extends Record<string, ParamSpec>> = {
  [K in keyof Spec]: Spec[K] extends { type: "number" }
    ? number
    : Spec[K] extends { type: "boolean" }
      ? boolean
      : string;
};

export interface TemplateDefinition<State, Spec extends Record<string, ParamSpec>> {
  meta: TemplateMeta;
  tier: "2d" | "webgpu";
  params: Spec;
  setup(ctx: TemplateContext, music: MusicData, params: ParamsOf<Spec>): State;
  render(ctx: TemplateContext, t: number, state: State, music: MusicData, params: ParamsOf<Spec>): void;
}

export function defineTemplate<State, Spec extends Record<string, ParamSpec>>(
  definition: TemplateDefinition<State, Spec>,
): TemplateDefinition<State, Spec> {
  return definition;
}

export function defaultParams<Spec extends Record<string, ParamSpec>>(spec: Spec): ParamsOf<Spec> {
  const out: Record<string, number | string | boolean> = {};
  for (const key of Object.keys(spec)) out[key] = spec[key].default;
  return out as ParamsOf<Spec>;
}

export function defaultPost(): PostSettings {
  return { glow: 1, grain: 1, vignette: 1 };
}

/** WebGPU templates fall back to the 2D context when the browser has no GPU adapter. */
export function resolveTier(requested: "2d" | "webgpu"): "2d" | "webgpu" {
  if (requested === "2d") return "2d";
  if (typeof navigator !== "undefined" && "gpu" in navigator) return "webgpu";
  return "2d";
}
