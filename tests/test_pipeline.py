"""Beat scores on the synthetic track, plus a short render in both aspects."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from music2video.analysis import analyze  # noqa: E402
from music2video.cli import main  # noqa: E402
from tools.make_test_track import synthesize, write_wav  # noqa: E402
from tools.score_beats import match_counts  # noqa: E402


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wav = Path(cls.tmp.name) / "track.wav"
        y, truth = synthesize()
        write_wav(cls.wav, y)
        cls.truth = truth

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_beat_f1(self) -> None:
        analysis = analyze(self.wav, Path(self.tmp.name))
        beats = match_counts(self.truth["beats"], analysis.beats)
        strong = match_counts(self.truth["strong_beats"], analysis.strong_beats)
        self.assertGreaterEqual(beats["f1"], 0.95)
        self.assertGreaterEqual(strong["f1"], 0.80)
        self.assertAlmostEqual(analysis.tempo, 126.0, delta=1.0)

    def test_both_aspects_have_audio(self) -> None:
        for aspect, width, height in (("16:9", 1920, 1080), ("9:16", 1080, 1920)):
            output = Path(self.tmp.name) / f"{aspect.replace(':', 'x')}.mp4"
            code = main(
                [
                    "-i",
                    str(self.wav),
                    "-o",
                    str(output),
                    "--aspect",
                    aspect,
                    "--max-seconds",
                    "1.5",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_streams",
                        "-of",
                        "json",
                        str(output),
                    ]
                )
            )
            video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
            audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
            self.assertEqual(video["width"], width)
            self.assertEqual(video["height"], height)
            self.assertEqual(audio["codec_name"], "aac")
            self.assertGreater(float(audio["duration"]), 1.0)


if __name__ == "__main__":
    unittest.main()
