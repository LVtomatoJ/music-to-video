import fs from "node:fs";
import type { MusicData } from "../src/music";
import { sampleBallScreens } from "../src/templates/map";

const width = 1600;
const height = 900;

function song(duration: number, beats: number[], seedLabel: string): { name: string; music: MusicData } {
  const times: number[] = [];
  const values: number[] = [];
  for (let time = 0; time <= duration; time += 0.1) {
    times.push(time);
    values.push(0.2 + 0.6 * (0.5 + 0.5 * Math.sin(time * 0.37)));
  }
  return {
    name: seedLabel,
    music: {
      version: 1,
      duration,
      sampleRate: 44100,
      tempo: 120,
      meter: 4,
      beats,
      downbeats: beats.filter((_, index) => index % 4 === 0),
      onsets: [],
      loudness: { times, values },
      bands: { times, low: values, mid: values, high: values },
      sections: [],
      extensions: { strongBeats: beats.filter((_, index) => index % 4 === 0) },
    },
  };
}

function regular(bpm: number, count: number, name: string) {
  const step = 60 / bpm;
  const beats: number[] = [];
  for (let index = 0; index < count; index++) beats.push(0.2 + index * step);
  const duration = beats[beats.length - 1] + 1.5;
  return song(duration, beats, name);
}

function bursts() {
  const beats: number[] = [];
  let time = 0.3;
  for (let group = 0; group < 18; group++) {
    for (let index = 0; index < 6; index++) beats.push(time + index * 0.18);
    time += 2.6;
  }
  return song(time + 1, beats, "bursts");
}

const cases = [
  regular(117, 245, "frankie-117"),
  regular(96, 289, "grid-96"),
  regular(200, 80, "fast-200"),
  bursts(),
  song(40, [], "no-beats"),
  song(30, [1.2], "one-beat"),
  song(50, Array.from({ length: 40 }, (_, index) => index * 0.4 + (index % 5) * 0.17), "jitter"),
];

if (fs.existsSync("/tmp/beat-thee-music.json")) {
  const raw = JSON.parse(fs.readFileSync("/tmp/beat-thee-music.json", "utf8"));
  cases.push({
    name: "beat-thee",
    music: {
      version: 1,
      duration: raw.duration,
      sampleRate: 44100,
      tempo: 96,
      meter: 4,
      beats: raw.beats,
      downbeats: raw.beats.filter((_: number, index: number) => index % 4 === 0),
      onsets: [],
      loudness: raw.loudness,
      bands: { times: raw.loudness.times, low: raw.loudness.values, mid: raw.loudness.values, high: raw.loudness.values },
      sections: [],
      extensions: { strongBeats: raw.strongBeats },
    },
  });
}

const seeds = [1, 42, 900];
let failures = 0;
let samples = 0;
for (const item of cases) {
  const useSeeds = item.music.beats.length > 8 ? seeds : [42];
  for (const seed of useSeeds) {
    const times: number[] = [0, 0.4];
    const duration = item.music.duration;
    for (let time = 0; time <= duration; time += Math.max(4, duration / 12)) times.push(Math.min(time, duration));
    times.push(Math.max(0, duration - 0.05));
    const frames = sampleBallScreens(item.music, seed, "neon", width, height, times);
    samples += frames.length;
    const missed = frames.filter((frame) => !frame.onScreen);
    if (missed.length) {
      failures += missed.length;
      console.log("MISS", item.name, "seed", seed, missed.slice(0, 4));
    } else {
      console.log("ok", item.name, "seed", seed, "frames", frames.length, "ball", frames[Math.floor(frames.length / 2)]);
    }
  }
}
console.log(JSON.stringify({ samples, failures }));
if (failures) process.exit(1);
