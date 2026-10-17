import { ArrayBufferTarget, Muxer } from "mp4-muxer";

export interface ExportRequest {
  width: number;
  height: number;
  fps: number;
  duration: number;
  audio: AudioBuffer | null;
  draw: (ctx: CanvasRenderingContext2D, time: number) => void;
  onProgress?: (ratio: number) => void;
}

export async function exportMp4(request: ExportRequest): Promise<Blob> {
  if (typeof VideoEncoder === "undefined") {
    throw new Error("这个浏览器没有 WebCodecs，还不能在本地导出。");
  }
  const width = request.width - (request.width % 2);
  const height = request.height - (request.height % 2);
  const fps = request.fps;
  const frames = Math.max(1, Math.round(request.duration * fps));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) throw new Error("导出用的画布没有准备好。");

  const videoCodec = await pickVideoCodec(width, height, fps);
  const audioBuffer = request.audio ? await renderStereo(request.audio) : null;
  const audioOk = audioBuffer ? await audioSupported(audioBuffer) : false;
  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    video: { codec: "avc", width, height, frameRate: fps },
    audio: audioOk && audioBuffer ? { codec: "aac", numberOfChannels: 2, sampleRate: audioBuffer.sampleRate } : undefined,
    fastStart: "in-memory",
    firstTimestampBehavior: "strict",
  });

  let encoderError: Error | null = null;
  const videoEncoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (error) => {
      encoderError = error instanceof Error ? error : new Error(String(error));
    },
  });
  videoEncoder.configure({
    codec: videoCodec,
    width,
    height,
    bitrate: 5_000_000,
    framerate: fps,
    avc: { format: "avc" },
    latencyMode: "quality",
  });

  let audioEncoder: AudioEncoder | null = null;
  if (audioOk && audioBuffer) {
    audioEncoder = new AudioEncoder({
      output: (chunk, meta) => muxer.addAudioChunk(chunk, meta),
      error: (error) => {
        encoderError = error instanceof Error ? error : new Error(String(error));
      },
    });
    audioEncoder.configure({
      codec: "mp4a.40.2",
      numberOfChannels: 2,
      sampleRate: audioBuffer.sampleRate,
      bitrate: 160_000,
    });
  }

  const frameDuration = 1_000_000 / fps;
  for (let index = 0; index < frames; index++) {
    if (encoderError) throw encoderError;
    const time = index / fps;
    request.draw(ctx, time);
    const frame = new VideoFrame(canvas, { timestamp: Math.round(index * frameDuration), duration: Math.round(frameDuration) });
    videoEncoder.encode(frame, { keyFrame: index % (fps * 2) === 0 });
    frame.close();
    while (videoEncoder.encodeQueueSize > 6) {
      await new Promise((resolve) => setTimeout(resolve, 8));
    }
    if (index % 3 === 0) {
      request.onProgress?.(index / frames);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  await videoEncoder.flush();
  videoEncoder.close();

  if (audioEncoder && audioBuffer) {
    const channels = audioBuffer.numberOfChannels;
    const block = 1024;
    const total = audioBuffer.length;
    for (let offset = 0; offset < total; offset += block) {
      const count = Math.min(block, total - offset);
      const planar = new Float32Array(channels * count);
      for (let c = 0; c < channels; c++) {
        planar.set(audioBuffer.getChannelData(c).subarray(offset, offset + count), c * count);
      }
      const data = new AudioData({
        format: "f32-planar",
        sampleRate: audioBuffer.sampleRate,
        numberOfFrames: count,
        numberOfChannels: channels,
        timestamp: Math.round((offset / audioBuffer.sampleRate) * 1_000_000),
        data: planar,
      });
      audioEncoder.encode(data);
      data.close();
      if (audioEncoder.encodeQueueSize > 8) {
        await new Promise((resolve) => setTimeout(resolve, 4));
      }
    }
    await audioEncoder.flush();
    audioEncoder.close();
  }
  if (encoderError) throw encoderError;
  muxer.finalize();
  request.onProgress?.(1);
  return new Blob([target.buffer], { type: "video/mp4" });
}

async function pickVideoCodec(width: number, height: number, fps: number): Promise<string> {
  const candidates = ["avc1.42001f", "avc1.4d001f", "avc1.640028"];
  for (const codec of candidates) {
    const support = await VideoEncoder.isConfigSupported({
      codec,
      width,
      height,
      bitrate: 5_000_000,
      framerate: fps,
    });
    if (support.supported) return codec;
  }
  throw new Error("这个浏览器不能把画面编成 H.264。");
}

async function audioSupported(audio: AudioBuffer): Promise<boolean> {
  if (typeof AudioEncoder === "undefined") return false;
  const support = await AudioEncoder.isConfigSupported({
    codec: "mp4a.40.2",
    sampleRate: audio.sampleRate,
    numberOfChannels: audio.numberOfChannels,
    bitrate: 160_000,
  });
  return Boolean(support.supported);
}

async function renderStereo(audio: AudioBuffer): Promise<AudioBuffer> {
  const sampleRate = 48000;
  const length = Math.max(1, Math.ceil(audio.duration * sampleRate));
  const offline = new OfflineAudioContext(2, length, sampleRate);
  const source = offline.createBufferSource();
  source.buffer = audio;
  source.connect(offline.destination);
  source.start(0);
  return offline.startRendering();
}
