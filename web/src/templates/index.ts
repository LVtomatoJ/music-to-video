import type { MusicData } from "../music";
import type { ParamSpec, ParamValues, ParamsOf, TemplateContext, TemplateDefinition, TemplateMeta } from "../template";
import { rainTemplate } from "./rain";

export interface RunnableTemplate {
  meta: TemplateMeta;
  tier: "2d" | "webgpu";
  params: Record<string, ParamSpec>;
  setup(ctx: TemplateContext, music: MusicData, params: ParamValues): unknown;
  render(ctx: TemplateContext, t: number, state: unknown, music: MusicData, params: ParamValues): void;
}

function adapt<State, Spec extends Record<string, ParamSpec>>(
  template: TemplateDefinition<State, Spec>,
): RunnableTemplate {
  return {
    meta: template.meta,
    tier: template.tier,
    params: template.params,
    setup: (ctx, music, params) => template.setup(ctx, music, params as ParamsOf<Spec>),
    render: (ctx, t, state, music, params) =>
      template.render(ctx, t, state as State, music, params as ParamsOf<Spec>),
  };
}

export const templates: RunnableTemplate[] = [adapt(rainTemplate)];
