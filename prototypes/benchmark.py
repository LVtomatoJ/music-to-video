#!/usr/bin/env python3
"""Re-run the three prototypes and write docs/benchmark-results.json."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "prototypes" / "out"
DOCS = ROOT / "docs" / "benchmark-results.json"
PYTHON = sys.executable
RUST = ROOT / "prototypes" / "rust" / "target" / "release" / "beatballs-rust"
WEB = ROOT / "prototypes" / "web" / "dist" / "drive.js"
TRACK = ROOT / "assets" / "test-track.wav"
TRUTH = ROOT / "assets" / "ground-truth.json"
REAL_MP3 = ROOT / "Alex MakeMusic - I Am Good - Instrumental Version.mp3"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=cwd)


def nonempty_loc(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                total += 1
    return total


def ffprobe(path: Path) -> dict:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,bit_rate",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    info = json.loads(raw)
    video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in info["streams"] if stream["codec_type"] == "audio")
    return {
        "duration": float(info["format"]["duration"]),
        "size_bytes": int(info["format"]["size"]),
        "bit_rate": int(info["format"]["bit_rate"]),
        "video_codec": video["codec_name"],
        "width": video["width"],
        "height": video["height"],
        "fps": video["avg_frame_rate"],
        "nb_frames": int(video.get("nb_frames") or 0),
        "audio_codec": audio["codec_name"],
        "sample_rate": int(audio["sample_rate"]),
        "channels": int(audio["channels"]),
    }


def interval_stats(beats: list[float]) -> dict:
    if len(beats) < 3:
        return {}
    gaps = [b - a for a, b in zip(beats, beats[1:])]
    ordered = sorted(gaps)
    med = ordered[len(ordered) // 2]
    mean = sum(gaps) / len(gaps)
    var = sum((gap - mean) ** 2 for gap in gaps) / len(gaps)
    regular = sum(1 for gap in gaps if abs(gap - med) <= 0.10 * med) / len(gaps)
    return {
        "median_interval_s": med,
        "implied_bpm": 60.0 / med,
        "interval_cv": (var**0.5) / mean,
        "regular_fraction": regular,
    }


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from tools.score_beats import score_file

    OUT.mkdir(parents=True, exist_ok=True)
    run([PYTHON, str(ROOT / "tools" / "make_test_track.py"), "--wav", str(TRACK), "--truth", str(TRUTH)])
    stacks = {
        "python": [
            PYTHON,
            str(ROOT / "prototypes" / "python" / "demo.py"),
            "--input",
            str(TRACK),
            "--output",
            str(OUT / "python.mp4"),
            "--beats-out",
            str(OUT / "python-beats.json"),
        ],
        "rust": [
            str(RUST),
            "--input",
            str(TRACK),
            "--output",
            str(OUT / "rust.mp4"),
            "--beats-out",
            str(OUT / "rust-beats.json"),
        ],
        "web": [
            "node",
            str(WEB),
            "--input",
            str(TRACK),
            "--output",
            str(OUT / "web.mp4"),
            "--beats-out",
            str(OUT / "web-beats.json"),
        ],
    }
    for cmd in stacks.values():
        run(cmd)

    real_wav = OUT / "real30.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-t",
            "30",
            "-i",
            str(REAL_MP3),
            "-ac",
            "1",
            "-ar",
            "44100",
            str(real_wav),
        ]
    )
    real_cmds = {
        "python": [
            PYTHON,
            str(ROOT / "prototypes" / "python" / "demo.py"),
            "--input",
            str(real_wav),
            "--output",
            str(OUT / "unused.mp4"),
            "--beats-out",
            str(OUT / "real-python.json"),
            "--analyze-only",
        ],
        "rust": [
            str(RUST),
            "--input",
            str(real_wav),
            "--output",
            str(OUT / "unused.mp4"),
            "--beats-out",
            str(OUT / "real-rust.json"),
            "--analyze-only",
        ],
        "web": [
            "node",
            str(WEB),
            "--input",
            str(real_wav),
            "--output",
            str(OUT / "unused.mp4"),
            "--beats-out",
            str(OUT / "real-web.json"),
            "--analyze-only",
        ],
    }
    for cmd in real_cmds.values():
        run(cmd)

    loc = {
        "python": nonempty_loc(list((ROOT / "prototypes" / "python").glob("*.py"))),
        "rust": nonempty_loc(list((ROOT / "prototypes" / "rust" / "src").glob("*.rs"))),
        "web": nonempty_loc(list((ROOT / "prototypes" / "web" / "src").glob("*.ts"))),
    }
    results = {
        "machine": {
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
            "python": platform.python_version(),
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "node": subprocess.check_output(["node", "--version"], text=True).strip(),
            "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        },
        "clip": {
            "duration_s": 30,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "encoder_note": "Python and Rust pipe raw frames to ffmpeg libx264 CRF 20 veryfast. Web uses WebCodecs AVC at 8 Mbps, then ffmpeg copies video and encodes AAC.",
        },
        "code_nonempty_lines": loc,
        "synthetic": {},
        "real_excerpt": {
            "source": REAL_MP3.name,
            "seconds": 30,
            "note": "No beat annotations. Compared by tempo regularity only.",
        },
    }
    for name, filename in (("python", "python"), ("rust", "rust"), ("web", "web")):
        estimate = OUT / f"{filename}-beats.json"
        payload = json.loads(estimate.read_text())
        scored = score_file(TRUTH, estimate)
        probed = ffprobe(OUT / f"{filename}.mp4")
        results["synthetic"][name] = {
            "timing": payload["timing"],
            "tempo": payload["tempo"],
            "score": scored,
            "output": probed,
        }
        real = json.loads((OUT / f"real-{filename}.json").read_text())
        results["real_excerpt"][name] = {
            "analysis_s": real["timing"]["analysis_s"],
            "tempo": real["tempo"],
            "beats": len(real["beats"]),
            "strong_beats": len(real["strong_beats"]),
            **interval_stats(real["beats"]),
        }
    DOCS.parent.mkdir(parents=True, exist_ok=True)
    DOCS.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {DOCS}")


if __name__ == "__main__":
    if not RUST.exists() or not WEB.exists():
        sys.exit(f"missing built prototypes: rust={RUST.exists()} web={WEB.exists()}")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is required")
    main()
