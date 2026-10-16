"""Pipe raw frames to ffmpeg and mux the source audio."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path


class FfmpegEncoder:
    def __init__(self, output: Path, width: int, height: int, fps: int, audio: Path, crf: int = 20) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgra",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._stderr: list[bytes] = []
        self._thread = threading.Thread(target=self._drain)
        self._thread.start()

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        self._stderr.append(self.proc.stderr.read())
        self.proc.stderr.close()

    def write_frame(self, frame: memoryview | bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(frame)

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self._thread.join()
        code = self.proc.wait()
        if code != 0:
            message = b"".join(self._stderr).decode(errors="replace")
            raise RuntimeError(f"ffmpeg exited {code}: {message}")
