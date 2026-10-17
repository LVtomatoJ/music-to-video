import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer-core";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const song = process.argv[2];
const outDir = process.argv[3] || "/opt/cursor/artifacts/web-scenes";
if (!song) {
  console.error("usage: node scripts/capture.mjs <song> [outDir]");
  process.exit(1);
}
fs.mkdirSync(outDir, { recursive: true });

const preview = spawn("npx", ["vite", "preview", "--host", "127.0.0.1", "--port", "4173", "--strictPort"], {
  cwd: root,
  stdio: ["ignore", "pipe", "pipe"],
  detached: true,
});
preview.stdout.on("data", (chunk) => process.stdout.write(chunk));
preview.stderr.on("data", (chunk) => process.stderr.write(chunk));

let code = 0;
try {
  await waitForServer("http://127.0.0.1:4173/");
  const browser = await puppeteer.launch({
    executablePath: process.env.CHROME_PATH || "/usr/bin/google-chrome",
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--autoplay-policy=no-user-gesture-required"],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(180000);
  page.on("pageerror", (error) => console.error("pageerror", error));
  page.on("console", (message) => {
    if (message.type() === "error") console.error("console", message.text());
  });
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
  await page.goto("http://127.0.0.1:4173/", { waitUntil: "networkidle0" });
  await page.waitForFunction(() => window.__m2v && window.__m2v.ready);
  const input = await page.$("#file");
  await input.uploadFile(song);
  await page.waitForFunction(() => window.__m2v.ready && window.__m2v.demo === false, { timeout: 180000 });
  const music = await page.evaluate(() => window.__m2v.music);
  const ids = await page.evaluate(() => window.__m2v.templates);
  fs.writeFileSync(path.join(outDir, "music.json"), JSON.stringify(summary(music), null, 2));
  console.log("tempo", music.tempo, "beats", music.beats.length, "onsets", music.onsets.length, "tones", music.extensions?.tones?.length);
  console.log("templates", ids);
  console.log("sections", music.sections);

  const times = pickTimes(music);
  console.log("moments", times);
  const stable = await page.evaluate(async (time) => {
    window.__m2v.pause();
    window.__m2v.seek(time);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const first = document.querySelector("#view").toDataURL("image/png");
    window.__m2v.seek(time + 1.7);
    window.__m2v.seek(time);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const second = document.querySelector("#view").toDataURL("image/png");
    return first === second;
  }, times.quiet);
  console.log("deterministic", stable);
  if (!stable) code = 1;

  for (const id of ids) {
    for (const [name, time] of Object.entries(times)) {
      await page.evaluate(async (templateId, at) => {
        window.__m2v.pause();
        if (!window.__m2v.setTemplate(templateId)) throw new Error(`missing template ${templateId}`);
        window.__m2v.seek(at);
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      }, id, time);
      const dataUrl = await page.evaluate(() => document.querySelector("#view").toDataURL("image/png"));
      const file = path.join(outDir, `${id}-${name}.png`);
      fs.writeFileSync(file, Buffer.from(dataUrl.split(",")[1], "base64"));
      console.log("wrote", file);
    }
  }
  await page.screenshot({ path: path.join(outDir, "page.png") });
  await browser.close();
  console.log(JSON.stringify({ stable, frames: ids.length * 3 }));
} catch (error) {
  console.error(error);
  code = 1;
} finally {
  if (preview.pid) {
    try {
      process.kill(-preview.pid, "SIGTERM");
    } catch {
      preview.kill("SIGTERM");
    }
  }
}
process.exit(code);

function summary(music) {
  const tones = music.extensions?.tones ?? [];
  return {
    version: music.version,
    duration: music.duration,
    tempo: music.tempo,
    meter: music.meter,
    beats: music.beats.length,
    downbeats: music.downbeats.length,
    onsets: music.onsets.length,
    sections: music.sections,
    analyzer: music.extensions?.analyzer,
    strongBeats: music.extensions?.strongBeats?.length ?? 0,
    tones: tones.length,
    pitchTones: tones.filter((tone) => tone.source === "pitch").length,
    chromaTones: tones.filter((tone) => tone.source === "chroma").length,
    chromaFrames: music.extensions?.chroma?.times?.length ?? 0,
  };
}

function pickTimes(music) {
  const quiet = music.sections.filter((section) => section.label === "intro" || section.label === "verse" || section.label === "outro");
  quiet.sort((a, b) => a.energy - b.energy);
  const chorus = music.sections.filter((section) => section.label === "chorus");
  chorus.sort((a, b) => b.energy - a.energy);
  const quietSection = quiet[0] || music.sections[0];
  const chorusSection = chorus[0] || music.sections[music.sections.length - 1];
  return {
    quiet: clamp((quietSection.start + quietSection.end) / 2, 0.4, music.duration - 0.4),
    dense: clamp(densest(music.onsets), 0.4, music.duration - 0.4),
    chorus: clamp((chorusSection.start + chorusSection.end) / 2, 0.4, music.duration - 0.4),
  };
}

function densest(onsets) {
  if (!onsets.length) return 8;
  let bestTime = onsets[0].time;
  let best = 0;
  for (let i = 0; i < onsets.length; i++) {
    const start = onsets[i].time;
    let count = 0;
    for (let j = i; j < onsets.length && onsets[j].time < start + 2; j++) count++;
    if (count > best) {
      best = count;
      bestTime = start + 0.35;
    }
  }
  return bestTime;
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

async function waitForServer(url) {
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
  }
  throw new Error(`server did not start: ${url}`);
}
