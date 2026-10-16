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

import math  # noqa: E402

from music2video.analysis import Analysis, Section, analyze  # noqa: E402
from music2video.cli import main  # noqa: E402
from music2video.mapscene import SPEED, MapScene, _reflect, _unit  # noqa: E402
from music2video.themes import THEMES, pick_theme  # noqa: E402
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

    def test_soft_mode_does_not_halve_a_clear_kick(self) -> None:
        analysis = analyze(self.wav, Path(self.tmp.name), beat_mode="soft")
        self.assertAlmostEqual(analysis.tempo, 126.0, delta=2.0)
        self.assertGreaterEqual(len(analysis.beats), 50)

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

    def test_map_hits_are_beat_accurate(self) -> None:
        analysis = analyze(self.wav, Path(self.tmp.name))
        scene = MapScene.create(1920, 1080, analysis, seed=42, theme="neon")
        timing = scene.timing
        self.assertEqual(timing["hits"], len(analysis.beats))
        self.assertLess(timing["max_position_error_px"], 1e-6)
        self.assertEqual(timing["max_timing_error_s"], 0.0)
        self.assertGreaterEqual(timing["min_clearance_px"], 72.0)
        self.assertEqual(set(timing["kinds"]), {"wall"})

    def test_segments_stay_straight_and_bounces_reflect(self) -> None:
        analysis = analyze(self.wav, Path(self.tmp.name))
        for seed in (1, 42, 99):
            scene = MapScene.create(1920, 1080, analysis, seed=seed, theme="neon")
            self.assertGreaterEqual(scene.timing["min_clearance_px"], 72.0)
            self.assertEqual(scene.timing["max_timing_error_s"], 0.0)
            self.assertEqual(len(scene.segments), len(scene.obstacles) + 1)
            for index, obs in enumerate(scene.obstacles):
                arrived = scene.segments[index]
                left = scene.segments[index + 1]
                ix, iy = _unit(arrived.x1 - arrived.x0, arrived.y1 - arrived.y0)
                ox, oy = _unit(left.x1 - left.x0, left.y1 - left.y0)
                rx, ry = _reflect(ix, iy, obs.nx, obs.ny)
                self.assertGreater(ix * obs.nx + iy * obs.ny, 0.0, f"seed {seed} hit {index} does not approach")
                self.assertLess(ox * obs.nx + oy * obs.ny, 0.0, f"seed {seed} hit {index} does not leave")
                self.assertGreater(rx * ox + ry * oy, 0.999, f"seed {seed} hit {index} reflection")
                span = arrived.t1 - arrived.t0
                vx, vy = arrived.x1 - arrived.x0, arrived.y1 - arrived.y0
                length = math.hypot(vx, vy)
                angles = []
                for step in range(1, 6):
                    t = arrived.t0 + span * step / 6.0
                    px, py = scene.position_at(t)
                    cross = abs((px - arrived.x0) * vy - (py - arrived.y0) * vx)
                    self.assertLess(cross / length, 1e-6)
                    self.assertGreater((px - arrived.x0) * ix + (py - arrived.y0) * iy, 0.0)
                    vel = scene.velocity_at(t)
                    angles.append(math.atan2(vel[1], vel[0]))
                    self.assertGreater(vel[0] * ix + vel[1] * iy, SPEED * 0.99)
                self.assertLess(max(angles) - min(angles), 1e-9)

    def test_themes_keep_their_own_objects_and_feel(self) -> None:
        analysis = analyze(self.wav, Path(self.tmp.name))
        projectiles = set()
        feels = set()
        for name, theme in THEMES.items():
            scene = MapScene.create(1920, 1080, analysis, seed=42, theme=name)
            kinds = set(scene.timing["kinds"])
            self.assertTrue(kinds)
            self.assertTrue(kinds <= set(theme.objects))
            self.assertEqual(scene.skin.name, name)
            self.assertEqual(scene.skin.projectile, theme.projectile)
            self.assertEqual(scene.timing["max_timing_error_s"], 0.0)
            self.assertLess(scene.timing["max_position_error_px"], 1e-6)
            projectiles.add(theme.projectile)
            feels.add((theme.feel.mass, theme.feel.stiffness, theme.feel.damping, theme.feel.trail, theme.projectile))
            beat = scene.obstacles[0].hit_time
            scene.update(1 / 30, beat, [(beat, scene.obstacles[0].strong)])
            self.assertGreater(scene.squash, 0.05)
            samples = [scene.squash]
            for step in range(24):
                scene.update(1 / 30, beat + (step + 1) / 30, [])
                samples.append(scene.squash)
            self.assertTrue(all(math.isfinite(value) for value in samples))
            self.assertLess(max(abs(value) for value in samples), 1.36)
            jumps = [abs(samples[i + 1] - samples[i]) for i in range(len(samples) - 1)]
            self.assertLess(max(jumps), 0.15, name)
        self.assertEqual(len(projectiles), len(THEMES))
        self.assertEqual(len(feels), len(THEMES))

    def test_auto_theme_follows_energy(self) -> None:
        calm = Analysis(
            60.0,
            [1.0, 2.0, 3.0],
            [],
            4.0,
            44100,
            Path("unused.wav"),
            sections=(Section(0.0, 4.0, 0.12, "calm"),),
        )
        hot = Analysis(
            126.0,
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            4.0,
            44100,
            Path("unused.wav"),
            sections=(Section(0.0, 4.0, 0.55, "intense"),),
        )
        self.assertEqual(pick_theme(calm), "pastel")
        self.assertEqual(pick_theme(hot), "shatter")

    def test_balls_scene_still_renders(self) -> None:
        output = Path(self.tmp.name) / "balls.mp4"
        code = main(
            [
                "-i",
                str(self.wav),
                "-o",
                str(output),
                "--scene",
                "balls",
                "--max-seconds",
                "1.2",
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(output.is_file())
        probe = json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)]
            )
        )
        self.assertTrue(any(stream["codec_type"] == "audio" for stream in probe["streams"]))


if __name__ == "__main__":
    unittest.main()
