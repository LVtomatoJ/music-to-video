/**
 * Headless Chrome driver: serves the page, runs Web Audio analysis + WebCodecs
 * encode, then ffmpeg-muxes the original wav as AAC.
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import puppeteer from "puppeteer-core";

declare const __dirname: string;

interface Args {
  input: string;
  output: string;
  beatsOut: string;
  analyzeOnly: boolean;
  width: number;
  height: number;
}

function parseArgs(argv: string[]): Args {
  const args: Args = {
    input: "",
    output: "",
    beatsOut: "",
    analyzeOnly: false,
    width: 1920,
    height: 1080,
  };
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (token === "--input") args.input = argv[++i];
    else if (token === "--output") args.output = argv[++i];
    else if (token === "--beats-out") args.beatsOut = argv[++i];
    else if (token === "--width") args.width = Number(argv[++i]);
    else if (token === "--height") args.height = Number(argv[++i]);
    else if (token === "--analyze-only") args.analyzeOnly = true;
    else throw new Error(`unknown argument ${token}`);
  }
  if (!args.input || !args.beatsOut || (!args.analyzeOnly && !args.output)) {
    throw new Error("usage: drive --input track.wav --output out.mp4 --beats-out beats.json [--analyze-only]");
  }
  return args;
}

function ffmpeg(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn("ffmpeg", args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg exited ${code}: ${stderr}`));
    });
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const here = __dirname;
  const mainJs = fs.readFileSync(path.join(here, "main.js"));
  const html = `<!doctype html><meta charset="utf-8"><script src="/main.js"></script>`;
  let upload: Buffer | null = null;
  const server = http.createServer((req, res) => {
    if (!req.url) {
      res.writeHead(400);
      res.end();
      return;
    }
    if (req.url.startsWith("/track.wav")) {
      res.writeHead(200, { "content-type": "audio/wav" });
      fs.createReadStream(args.input).pipe(res);
      return;
    }
    if (req.url.startsWith("/main.js")) {
      res.writeHead(200, { "content-type": "text/javascript" });
      res.end(mainJs);
      return;
    }
    if (req.url.startsWith("/upload") && req.method === "POST") {
      const chunks: Buffer[] = [];
      req.on("data", (chunk) => chunks.push(chunk as Buffer));
      req.on("end", () => {
        upload = Buffer.concat(chunks);
        res.writeHead(200);
        res.end("ok");
      });
      return;
    }
    res.writeHead(200, { "content-type": "text/html" });
    res.end(html);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("no port");
  const origin = `http://127.0.0.1:${address.port}`;

  const browser = await puppeteer.launch({
    executablePath: "/usr/bin/google-chrome-stable",
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--autoplay-policy=no-user-gesture-required"],
  });
  try {
    const page = await browser.newPage();
    page.on("console", (msg) => console.error("page:", msg.text()));
    page.on("pageerror", (err) => console.error("pageerror:", err));
    await page.goto(origin, { waitUntil: "load" });
    const result = await page.evaluate(
      async (analyzeOnly, width, height) => window.runPipeline("/track.wav", analyzeOnly, width, height),
      args.analyzeOnly,
      args.width,
      args.height,
    );
    if (!result.ok) throw new Error(result.error || "pipeline failed");
    let renderS = (result.render_ms || 0) / 1000;
    if (!args.analyzeOnly) {
      if (!upload) throw new Error("browser did not upload a video");
      const silent = args.output.replace(/\.mp4$/i, "") + ".silent.mp4";
      fs.mkdirSync(path.dirname(path.resolve(args.output)), { recursive: true });
      fs.writeFileSync(silent, upload);
      const muxStart = performance.now();
      await ffmpeg([
        "-y",
        "-loglevel",
        "error",
        "-i",
        silent,
        "-i",
        args.input,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        args.output,
      ]);
      renderS += (performance.now() - muxStart) / 1000;
      fs.unlinkSync(silent);
    }
    const analysisS = (result.analysis_ms || 0) / 1000;
    const payload = {
      stack: "web-webaudio-webcodecs",
      tempo: result.tempo,
      beats: result.beats,
      strong_beats: result.strong_beats,
      duration: result.duration,
      timing: {
        analysis_s: analysisS,
        render_encode_s: args.analyzeOnly ? 0 : renderS,
        total_s: analysisS + (args.analyzeOnly ? 0 : renderS),
      },
    };
    fs.mkdirSync(path.dirname(path.resolve(args.beatsOut)), { recursive: true });
    fs.writeFileSync(args.beatsOut, JSON.stringify(payload, null, 2) + "\n");
    console.log(JSON.stringify(payload.timing));
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
