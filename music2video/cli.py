"""CLI: one song in, one MP4 out."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

from music2video.analysis import analyze
from music2video.encode import FfmpegEncoder
from music2video.render import CairoRenderer
from music2video.scene import Scene

ASPECTS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music2video",
        description="Render a beat-reactive ball animation and mux the original audio into an MP4.",
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="wav, mp3, or anything ffmpeg can read")
    parser.add_argument("-o", "--output", type=Path, required=True, help="destination mp4")
    parser.add_argument("--aspect", choices=sorted(ASPECTS), default="16:9")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=None, help="trim the song before analysis")
    parser.add_argument("--dump-beats", type=Path, default=None, help="write detected beats as JSON")
    return parser


def strong_flags(beats: list[float], strong: list[float]) -> list[bool]:
    flags: list[bool] = []
    index = 0
    for time in beats:
        while index < len(strong) and strong[index] < time - 1e-6:
            index += 1
        is_strong = index < len(strong) and abs(strong[index] - time) <= 1e-6
        flags.append(is_strong)
        if is_strong:
            index += 1
    return flags


def render_video(args: argparse.Namespace) -> dict:
    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")
    if args.fps < 1:
        raise SystemExit("--fps must be positive")
    width, height = ASPECTS[args.aspect]
    with tempfile.TemporaryDirectory(prefix="music2video-") as tmp:
        analysis = analyze(args.input, Path(tmp), max_seconds=args.max_seconds)
        scene = Scene.create(width, height, seed=args.seed)
        renderer = CairoRenderer(width, height)
        encoder = FfmpegEncoder(args.output, width, height, args.fps, analysis.wav_path, crf=args.crf)
        flags = strong_flags(analysis.beats, analysis.strong_beats)
        nframes = int(math.floor(analysis.duration * args.fps + 1e-6))
        dt = 1.0 / args.fps
        beat_index = 0
        try:
            for frame in range(nframes):
                t0 = frame * dt
                t1 = t0 + dt
                beat = False
                strong = False
                while beat_index < len(analysis.beats) and analysis.beats[beat_index] < t1:
                    if analysis.beats[beat_index] >= t0:
                        beat = True
                        strong = strong or flags[beat_index]
                    beat_index += 1
                scene.update(dt, beat, strong)
                encoder.write_frame(renderer.draw(scene))
                if frame % args.fps == 0:
                    print(f"\rframe {frame}/{nframes}", file=sys.stderr, end="")
        finally:
            encoder.close()
        payload = {
            "tempo": analysis.tempo,
            "beats": analysis.beats,
            "strong_beats": analysis.strong_beats,
            "duration": analysis.duration,
            "width": width,
            "height": height,
            "fps": args.fps,
            "output": str(args.output),
        }
    print(file=sys.stderr)
    if args.dump_beats is not None:
        args.dump_beats.parent.mkdir(parents=True, exist_ok=True)
        args.dump_beats.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"tempo {analysis.tempo:.2f} BPM, {len(analysis.beats)} beats, "
        f"{len(analysis.strong_beats)} strong, wrote {args.output}"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_video(args)
    return 0
