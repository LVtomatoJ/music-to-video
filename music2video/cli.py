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
from music2video.scene import build_scene
from music2video.themes import THEMES, pick_theme

ASPECTS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music2video",
        description="Render a beat-synced picture and mux the original audio into an MP4.",
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="wav, mp3, or anything ffmpeg can read")
    parser.add_argument("-o", "--output", type=Path, required=True, help="destination mp4")
    parser.add_argument("--aspect", choices=sorted(ASPECTS), default="16:9")
    parser.add_argument(
        "--scene",
        choices=("map", "balls", "city", "piano", "stars"),
        default="map",
        help="map: one ball on a beat map; balls: the older multi-ball field; city, piano, stars: other beat-synced forms",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=None, help="trim the song before analysis")
    parser.add_argument(
        "--beat-mode",
        choices=("default", "soft"),
        default="default",
        help="soft prefers a half-time pulse when it hits onsets at least as cleanly",
    )
    parser.add_argument(
        "--theme",
        choices=tuple(THEMES) + ("auto",),
        default="neon",
        help="one visual skin for the whole video; auto picks from section energy",
    )
    parser.add_argument("--title", default="", help="optional corner title; hidden when empty")
    parser.add_argument("--artist", default="", help="optional corner credit under the title; hidden when empty")
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
        analysis = analyze(args.input, Path(tmp), max_seconds=args.max_seconds, beat_mode=args.beat_mode)
        theme = pick_theme(analysis) if args.theme == "auto" else args.theme
        scene = build_scene(args.scene, width, height, analysis, args.seed, theme)
        if hasattr(scene, "caption_title"):
            scene.caption_title = args.title
            scene.caption_artist = args.artist
        renderer = CairoRenderer(width, height, fps=args.fps)
        encoder = FfmpegEncoder(args.output, width, height, args.fps, analysis.wav_path, crf=args.crf)
        flags = strong_flags(analysis.beats, analysis.strong_beats)
        nframes = int(math.floor(analysis.duration * args.fps + 1e-6))
        dt = 1.0 / args.fps
        beat_index = 0
        try:
            for frame in range(nframes):
                t0 = frame * dt
                t1 = t0 + dt
                events: list[tuple[float, bool]] = []
                while beat_index < len(analysis.beats) and analysis.beats[beat_index] < t1:
                    if analysis.beats[beat_index] >= t0:
                        events.append((analysis.beats[beat_index], flags[beat_index]))
                    beat_index += 1
                # The frame that contains a beat is drawn at that beat, so the hit is the picture.
                sample = events[0][0] if events else t0
                scene.update(dt, sample, events)
                encoder.write_frame(renderer.draw(scene))
                if frame % args.fps == 0:
                    print(f"\rframe {frame}/{nframes}", file=sys.stderr, end="")
        finally:
            encoder.close()
        timing = getattr(scene, "timing", None)
        payload = {
            "tempo": analysis.tempo,
            "beats": analysis.beats,
            "strong_beats": analysis.strong_beats,
            "duration": analysis.duration,
            "width": width,
            "height": height,
            "fps": args.fps,
            "scene": args.scene,
            "theme": getattr(getattr(scene, "skin", None), "name", args.theme),
            "beat_mode": args.beat_mode,
            "output": str(args.output),
        }
        if timing:
            payload["hit_timing"] = timing
            kinds = []
            for obs in getattr(scene, "obstacles", []):
                kinds.append({"time": obs.hit_time, "kind": obs.kind, "strong": obs.strong, "section": obs.section})
            payload["obstacles"] = kinds
    print(file=sys.stderr)
    if args.dump_beats is not None:
        args.dump_beats.parent.mkdir(parents=True, exist_ok=True)
        args.dump_beats.write_text(json.dumps(payload, indent=2) + "\n")
    timing = payload.get("hit_timing")
    extra = ""
    if timing:
        extra = (
            f", hit timing max {timing['max_timing_error_s'] * 1000:.3f} ms"
            f", position {timing['max_position_error_px']:.4f} px"
        )
    report = getattr(scene, "sync_report", None)
    if report:
        detail = ", ".join(f"{key} {value}" for key, value in report.items())
        extra = f"{extra}, {detail}"
    print(
        f"tempo {analysis.tempo:.2f} BPM, {len(analysis.beats)} beats, "
        f"{len(analysis.strong_beats)} strong, scene {args.scene}, theme {payload['theme']}{extra}, wrote {args.output}"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_video(args)
    return 0
