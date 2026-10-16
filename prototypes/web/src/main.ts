import { analyze } from "./analyze";
import { renderVideo } from "./render";

interface RunResult {
  ok: boolean;
  error?: string;
  tempo?: number;
  beats?: number[];
  strong_beats?: number[];
  duration?: number;
  analysis_ms?: number;
  render_ms?: number;
}

declare global {
  interface Window {
    runPipeline: (wavUrl: string, analyzeOnly: boolean, width: number, height: number) => Promise<RunResult>;
  }
}

window.runPipeline = async (wavUrl, analyzeOnly, width, height) => {
  try {
    const response = await fetch(wavUrl);
    if (!response.ok) throw new Error(`fetch ${wavUrl} failed`);
    const bytes = await response.arrayBuffer();
    const t0 = performance.now();
    const analysis = await analyze(bytes);
    const t1 = performance.now();
    if (!analyzeOnly) {
      const video = await renderVideo(analysis, width, height);
      const upload = await fetch("/upload", {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body: video,
      });
      if (!upload.ok) throw new Error("upload failed");
    }
    const t2 = performance.now();
    return {
      ok: true,
      tempo: analysis.tempo,
      beats: analysis.beats,
      strong_beats: analysis.strong_beats,
      duration: analysis.duration,
      analysis_ms: t1 - t0,
      render_ms: analyzeOnly ? 0 : t2 - t1,
    };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.stack || error.message : String(error) };
  }
};
