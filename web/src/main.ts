import { exportMp4 } from "./export";
import { hashString, mixSeed, mulberry32, clamp } from "./math";
import { demoMusic, decodeAndAnalyze, type MusicData, type SectionLabel } from "./music";
import { applyPost } from "./post";
import { defaultParams, defaultPost, type Canvas2DContext, type ParamValues } from "./template";
import { templates, type RunnableTemplate } from "./templates";

const VIEW_W = 1600;
const VIEW_H = 900;
const EXPORT_W = 1280;
const EXPORT_H = 720;

const view = document.querySelector<HTMLCanvasElement>("#view")!;
const playButton = document.querySelector<HTMLButtonElement>("#play")!;
const seek = document.querySelector<HTMLInputElement>("#seek")!;
const clock = document.querySelector<HTMLElement>("#clock")!;
const fileInput = document.querySelector<HTMLInputElement>("#file")!;
const drop = document.querySelector<HTMLElement>("#drop")!;
const status = document.querySelector<HTMLElement>("#status")!;
const templateRow = document.querySelector<HTMLElement>("#templates")!;
const paramsForm = document.querySelector<HTMLFormElement>("#params")!;
const facts = document.querySelector<HTMLElement>("#facts")!;
const exportButton = document.querySelector<HTMLButtonElement>("#export")!;
const jsonButton = document.querySelector<HTMLButtonElement>("#json")!;
const audio = document.querySelector<HTMLAudioElement>("#song")!;
const stage = document.querySelector<HTMLElement>("#stage")!;

view.width = VIEW_W;
view.height = VIEW_H;
const context = view.getContext("2d", { alpha: false });
if (!context) throw new Error("画布不可用");
const viewCtx: CanvasRenderingContext2D = context;

const labels: Record<SectionLabel, string> = {
  intro: "引子",
  verse: "主歌",
  chorus: "副歌",
  bridge: "桥段",
  outro: "尾声",
};

let current: RunnableTemplate = templates[0];
let params: ParamValues = defaultParams(current.params);
let music: MusicData = demoMusic();
let plan: unknown = null;
let audioBuffer: AudioBuffer | null = null;
let usingDemo = true;
let playing = true;
let ready = false;
let exporting = false;
let objectUrl: string | null = null;
let demoAnchor = performance.now();
let demoOrigin = 0;
let raf = 0;
let songName = "示例节奏";

function planSeed(): number {
  const numeric = typeof params.seed === "number" ? params.seed : 42;
  return mixSeed(hashString(current.meta.id), numeric);
}

function makeContext(target: CanvasRenderingContext2D, width: number, height: number, time: number | null): Canvas2DContext {
  const seed = time === null ? planSeed() : mixSeed(planSeed(), Math.floor(time * 30));
  return {
    tier: "2d",
    width,
    height,
    ctx: target,
    rng: mulberry32(seed),
    post: defaultPost(),
  };
}

function rebuild(): void {
  plan = current.setup(makeContext(viewCtx, view.width, view.height, null), music, params);
}

function renderAt(time: number): void {
  const frame = makeContext(viewCtx, view.width, view.height, time);
  current.render(frame, time, plan, music, params);
  applyPost(viewCtx, frame.post, time);
  if (!seek.matches(":active")) {
    seek.max = String(music.duration);
    seek.value = String(time);
  }
  clock.textContent = `${format(time)} / ${format(music.duration)}`;
}

function currentTime(): number {
  if (!usingDemo) return clamp(audio.currentTime || 0, 0, music.duration || 0);
  if (!playing) return demoOrigin;
  return (demoOrigin + (performance.now() - demoAnchor) / 1000) % Math.max(music.duration, 0.001);
}

function seekTo(time: number): void {
  const next = clamp(time, 0, Math.max(music.duration, 0.001));
  if (usingDemo) {
    demoOrigin = next;
    demoAnchor = performance.now();
  } else {
    audio.currentTime = next;
  }
  renderAt(next);
}

function tick(): void {
  renderAt(currentTime());
  raf = playing ? requestAnimationFrame(tick) : 0;
}

function startLoop(): void {
  if (!raf) raf = requestAnimationFrame(tick);
}

function setPlaying(next: boolean): void {
  playing = next;
  playButton.textContent = playing ? "❚❚" : "▶";
  playButton.setAttribute("aria-label", playing ? "暂停" : "播放");
  if (playing) startLoop();
}

async function toggle(): Promise<void> {
  if (usingDemo) {
    if (playing) {
      demoOrigin = currentTime();
      setPlaying(false);
    } else {
      demoAnchor = performance.now();
      setPlaying(true);
    }
    return;
  }
  if (audio.paused) {
    await audio.play();
    setPlaying(true);
  } else {
    audio.pause();
    setPlaying(false);
  }
}

async function loadFile(file: File): Promise<void> {
  status.textContent = "正在听这首歌…";
  exportButton.disabled = true;
  try {
    const analyzed = await decodeAndAnalyze(await file.arrayBuffer(), (ratio) => {
      status.textContent = `正在分析 ${Math.round(ratio * 100)}%`;
    });
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    audio.src = objectUrl;
    music = analyzed.music;
    audioBuffer = analyzed.audio;
    usingDemo = false;
    songName = file.name;
    audio.pause();
    setPlaying(false);
    rebuild();
    seekTo(0);
    renderFacts();
    renderTemplates();
    status.textContent = `${file.name} · ${Math.round(music.tempo)} BPM`;
    ready = true;
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "这首歌没有听成";
  } finally {
    exportButton.disabled = false;
  }
}

function renderTemplates(): void {
  templateRow.replaceChildren();
  for (const template of templates) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = template.meta.name;
    button.title = template.meta.description;
    button.classList.toggle("on", template === current);
    button.addEventListener("click", () => {
      if (template === current) return;
      current = template;
      params = defaultParams(template.params);
      renderParams();
      renderTemplates();
      rebuild();
      renderAt(currentTime());
    });
    templateRow.appendChild(button);
  }
}

function renderParams(): void {
  paramsForm.replaceChildren();
  for (const key of Object.keys(current.params)) {
    const spec = current.params[key];
    const row = document.createElement("label");
    row.className = "knob";
    const name = document.createElement("span");
    name.textContent = spec.label;
    row.appendChild(name);
    if (spec.type === "number") {
      const input = document.createElement("input");
      input.type = "range";
      input.min = String(spec.min);
      input.max = String(spec.max);
      input.step = String(spec.step ?? 0.01);
      input.value = String(params[key]);
      const output = document.createElement("output");
      output.textContent = formatNumber(Number(params[key]));
      input.addEventListener("input", () => {
        const value = Number(input.value);
        params[key] = value;
        output.textContent = formatNumber(value);
        applyParam(key);
      });
      row.append(input, output);
    } else if (spec.type === "color") {
      const input = document.createElement("input");
      input.type = "color";
      input.value = String(params[key]);
      input.addEventListener("input", () => {
        params[key] = input.value;
        applyParam(key);
      });
      row.appendChild(input);
    } else if (spec.type === "enum") {
      const select = document.createElement("select");
      for (const option of spec.options) {
        const node = document.createElement("option");
        node.value = option.value;
        node.textContent = option.label;
        select.appendChild(node);
      }
      select.value = String(params[key]);
      select.addEventListener("change", () => {
        params[key] = select.value;
        applyParam(key);
      });
      row.appendChild(select);
    } else {
      const input = document.createElement("input");
      input.type = "checkbox";
      input.className = "check";
      input.checked = Boolean(params[key]);
      input.addEventListener("change", () => {
        params[key] = input.checked;
        applyParam(key);
      });
      row.appendChild(input);
    }
    paramsForm.appendChild(row);
  }
}

function applyParam(key: string): void {
  if (current.params[key]?.scope !== "frame") rebuild();
  renderAt(currentTime());
}

function renderFacts(): void {
  const sectionText = music.sections
    .map((section) => `${labels[section.label]} ${format(section.start)}`)
    .join("  ");
  const rows: [string, string][] = [
    ["速度", `${Math.round(music.tempo)} BPM`],
    ["拍", String(music.beats.length)],
    ["小节", String(music.downbeats.length)],
    ["音头", String(music.onsets.length)],
    ["段落", sectionText || "—"],
  ];
  facts.replaceChildren();
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    facts.append(dt, dd);
  }
}

function format(time: number): string {
  const whole = Math.max(0, Math.floor(time));
  const minutes = Math.floor(whole / 60);
  const seconds = whole % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

async function runExport(seconds?: number): Promise<{ bytes: number; head: string } | void> {
  if (exporting) return;
  exporting = true;
  exportButton.disabled = true;
  const previous = exportButton.textContent;
  const duration = seconds ?? (usingDemo ? Math.min(12, music.duration) : music.duration);
  try {
    const scratch = document.createElement("canvas");
    scratch.width = EXPORT_W;
    scratch.height = EXPORT_H;
    const scratchCtx = scratch.getContext("2d", { alpha: false });
    if (!scratchCtx) throw new Error("导出画布不可用");
    const exportPlan = current.setup(makeContext(scratchCtx, EXPORT_W, EXPORT_H, null), music, params);
    const audioForExport = !usingDemo && audioBuffer ? await sliceAudio(audioBuffer, duration) : null;
    const blob = await exportMp4({
      width: EXPORT_W,
      height: EXPORT_H,
      fps: 30,
      duration,
      audio: audioForExport,
      draw: (drawCtx, time) => {
        const frame = makeContext(drawCtx, EXPORT_W, EXPORT_H, time);
        current.render(frame, time, exportPlan, music, params);
        applyPost(drawCtx, frame.post, time);
      },
      onProgress: (ratio) => {
        exportButton.textContent = `编码中 ${Math.round(ratio * 100)}%`;
      },
    });
    if (seconds !== undefined) {
      const bytes = new Uint8Array(await blob.arrayBuffer());
      const head = Array.from(bytes.subarray(0, 12))
        .map((value) => value.toString(16).padStart(2, "0"))
        .join("");
      return { bytes: bytes.byteLength, head };
    }
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${songName.replace(/\.[^.]+$/, "")}-rain.mp4`;
    link.click();
    URL.revokeObjectURL(url);
    status.textContent = usingDemo ? "已导出示例画面，没有声音。" : "已导出，画面和声音都在这个文件里。";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "导出失败";
    if (seconds !== undefined) throw error;
  } finally {
    exporting = false;
    exportButton.disabled = false;
    exportButton.textContent = previous;
  }
}

async function sliceAudio(source: AudioBuffer, seconds: number): Promise<AudioBuffer> {
  const frames = Math.max(1, Math.min(source.length, Math.floor(seconds * source.sampleRate)));
  const offline = new OfflineAudioContext(source.numberOfChannels, frames, source.sampleRate);
  const copy = offline.createBuffer(source.numberOfChannels, frames, source.sampleRate);
  for (let channel = 0; channel < source.numberOfChannels; channel++) {
    copy.copyToChannel(source.getChannelData(channel).subarray(0, frames), channel);
  }
  return copy;
}

playButton.addEventListener("click", () => void toggle());
seek.addEventListener("input", () => seekTo(Number(seek.value)));
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) void loadFile(file);
});
exportButton.addEventListener("click", () => void runExport());
jsonButton.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(music)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${songName.replace(/\.[^.]+$/, "")}.music.json`;
  link.click();
  URL.revokeObjectURL(url);
});
audio.addEventListener("ended", () => setPlaying(false));
window.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  event.preventDefault();
  void toggle();
});

for (const target of [drop, stage]) {
  target.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  target.addEventListener("dragleave", () => drop.classList.remove("over"));
  target.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    const file = event.dataTransfer?.files?.[0];
    if (file) void loadFile(file);
  });
}

window.__m2v = {
  get ready() { return ready; },
  get demo() { return usingDemo; },
  get music() { return music; },
  get time() { return currentTime(); },
  seek: seekTo,
  play() {
    if (usingDemo) {
      demoAnchor = performance.now();
      setPlaying(true);
      return;
    }
    void audio.play().then(() => setPlaying(true));
  },
  pause() {
    if (!usingDemo) audio.pause();
    if (usingDemo) demoOrigin = currentTime();
    setPlaying(false);
  },
  async exportSeconds(seconds: number) {
    const result = await runExport(seconds);
    if (!result) throw new Error("导出没有返回文件");
    return result;
  },
};

declare global {
  interface Window {
    __m2v: {
      readonly ready: boolean;
      readonly demo: boolean;
      readonly music: MusicData;
      readonly time: number;
      seek: (time: number) => void;
      play: () => void;
      pause: () => void;
      exportSeconds: (seconds: number) => Promise<{ bytes: number; head: string }>;
    };
  }
}

rebuild();
renderTemplates();
renderParams();
renderFacts();
setPlaying(true);
ready = true;
status.textContent = "示例节奏。上传后会换成这首歌自己的拍子。";
