import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer-core";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const song = process.argv[2];
const outDir = process.argv[3] || "/opt/cursor/artifacts/rain-web";
if (!song) {
  console.error("usage: node scripts/capture.mjs <song> [outDir]");
  process.exit(1);
}
fs.mkdirSync(outDir, { recursive: true });

const preview = spawn("npx", ["vite", "preview", "--host", "127.0.0.1", "--port", "4173", "--strictPort"], {
  cwd: root,
  stdio: ["ignore", "pipe", "pipe"],
});
preview.stdout.on("data", (chunk) => process.stdout.write(chunk));
preview.stderr.on("data", (chunk) => process.stderr.write(chunk));

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
  fs.writeFileSync(path.join(outDir, "music.json"), JSON.stringify(summary(music), null, 2));
  console.log("tempo", music.tempo, "beats", music.beats.length, "onsets", music.onsets.length);
  console.log("sections", music.sections);

  const times = pickTimes(music);
  console.log("frames", times);
  const stable = await page.evaluate(async (time) => {
    window.__m2v.seek(time);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const first = document.querySelector("#view").toDataURL("image/png");
    window.__m2v.seek(time);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const second = document.querySelector("#view").toDataURL("image/png");
    return first === second;
  }, times[0].time);
  console.log("deterministic", stable);

  for (const frame of times) {
    await page.evaluate(async (time) => {
      window.__m2v.pause();
      window.__m2v.seek(time);
      await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    }, frame.time);
    const dataUrl = await page.evaluate(() => document.querySelector("#view").toDataURL("image/png"));
    const file = path.join(outDir, `${frame.name}.png`);
    fs.writeFileSync(file, Buffer.from(dataUrl.split(",")[1], "base64"));
    console.log("wrote", file);
  }
  await page.screenshot({ path: path.join(outDir, "page.png") });

  let recorded = false;
  if (typeof page.screencast === "function") {
    const clip = path.join(outDir, "preview.webm");
    const recorder = await page.screencast({ path: clip, fps: 12 });
    await page.evaluate((time) => {
      window.__m2v.seek(time);
      window.__m2v.play();
    }, times.find((frame) => frame.name.startsWith("chorus"))?.time ?? times[0].time);
    await new Promise((resolve) => setTimeout(resolve, 5000));
    await recorder.stop();
    recorded = true;
    console.log("wrote", clip);
  } else {
    console.log("screencast API missing");
  }

  try {
    const exported = await page.evaluate(() => window.__m2v.exportSeconds(1));
    fs.writeFileSync(path.join(outDir, "export-1s.json"), JSON.stringify(exported, null, 2));
    console.log("export", exported);
  } catch (error) {
    fs.writeFileSync(path.join(outDir, "export-error.txt"), String(error));
    console.error("export failed", error);
  }

  await browser.close();
  if (!stable) {
    console.error("render was not deterministic");
    process.exitCode = 1;
  }
  console.log(JSON.stringify({ stable, recorded, frames: times.length }));
} finally {
  preview.kill("SIGTERM");
}

function summary(music) {
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
  };
}

function pickTimes(music) {
  const quiet = [...music.sections].filter((section) => section.label === "intro" || section.label === "verse" || section.label === "outro");
  quiet.sort((a, b) => a.energy - b.energy);
  const chorus = [...music.sections].filter((section) => section.label === "chorus");
  chorus.sort((a, b) => b.energy - a.energy);
  const quietSection = quiet[0] || music.sections[0];
  const chorusSection = chorus[0] || music.sections[music.sections.length - 1];
  const downbeat = nearest(music.downbeats, (quietSection.start + quietSection.end) / 2) ?? music.beats[Math.floor(music.beats.length / 3)] ?? 4;
  const chorusBeat = nearest(music.downbeats, (chorusSection.start + chorusSection.end) / 2) ?? chorusSection.start + 1;
  const dense = densest(music.onsets);
  return [
    { name: "01-quiet", time: clamp((quietSection.start + quietSection.end) / 2, 0.2, music.duration - 0.2) },
    { name: "02-verse", time: clamp(quietSection.start + Math.min(2, (quietSection.end - quietSection.start) / 3), 0.2, music.duration - 0.2) },
    { name: "03-falling", time: clamp(downbeat - 0.12, 0.05, music.duration - 0.2) },
    { name: "04-splash", time: clamp(downbeat + 0.07, 0.05, music.duration - 0.2) },
    { name: "05-chorus", time: clamp((chorusSection.start + chorusSection.end) / 2, 0.2, music.duration - 0.2) },
    { name: "06-chorus-later", time: clamp((chorusSection.start + chorusSection.end) / 2 + 0.48, 0.2, music.duration - 0.2) },
    { name: "07-dense", time: clamp(dense, 0.2, music.duration - 0.2) },
    { name: "08-chorus-hit", time: clamp(chorusBeat + 0.08, 0.05, music.duration - 0.2) },
  ];
}

function nearest(times, target) {
  let best = null;
  let score = Infinity;
  for (const time of times) {
    const delta = Math.abs(time - target);
    if (delta < score) {
      score = delta;
      best = time;
    }
  }
  return best;
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
      bestTime = start + 0.4;
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
