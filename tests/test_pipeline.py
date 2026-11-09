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
from music2video.mapscene import MapScene, _reflect, _segment_tangent, max_flyer_penetration  # noqa: E402
from music2video.render import CairoRenderer  # noqa: E402
from music2video.scene import build_scene  # noqa: E402
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

    def test_flight_follows_the_bounce(self) -> None:
        analysis = analyze(self.wav, Path(self.tmp.name))
        for seed in (1, 42, 99):
            scene = MapScene.create(1920, 1080, analysis, seed=seed, theme="neon")
            self.assertGreaterEqual(scene.timing["min_clearance_px"], 72.0)
            self.assertEqual(scene.timing["max_timing_error_s"], 0.0)
            self.assertEqual(len(scene.segments), len(scene.obstacles) + 1)
            for index, obs in enumerate(scene.obstacles):
                arrived = scene.segments[index]
                left = scene.segments[index + 1]
                ix, iy = _segment_tangent(arrived, True)
                ox, oy = _segment_tangent(left, False)
                rx, ry = _reflect(ix, iy, obs.nx, obs.ny)
                self.assertGreater(ix * obs.nx + iy * obs.ny, 0.0, f"seed {seed} hit {index} does not approach")
                self.assertLess(ox * obs.nx + oy * obs.ny, 0.0, f"seed {seed} hit {index} does not leave")
                self.assertGreater(rx * ox + ry * oy, 0.999, f"seed {seed} hit {index} reflection")
                span = arrived.t1 - arrived.t0
                dx, dy = arrived.x1 - arrived.x0, arrived.y1 - arrived.y0
                length = math.hypot(dx, dy)
                angles = []
                for step in range(1, 7):
                    t = arrived.t0 + span * step / 7.0
                    px, py = scene.position_at(t)
                    distance = abs((px - arrived.x0) * dy - (py - arrived.y0) * dx) / length
                    self.assertLess(distance, abs(arrived.bow) + 1.0)
                    self.assertGreater((px - arrived.x0) * dx + (py - arrived.y0) * dy, 0.0)
                    vel = scene.velocity_at(t)
                    self.assertGreater(vel[0] * dx + vel[1] * dy, 0.0)
                    angles.append(math.atan2(vel[1], vel[0]))
                unwrapped = [angles[0]]
                for angle in angles[1:]:
                    turn = (angle - unwrapped[-1] + math.pi) % (2 * math.pi) - math.pi
                    self.assertLess(abs(turn), 0.4)
                    unwrapped.append(unwrapped[-1] + turn)
                span_angle = max(unwrapped) - min(unwrapped)
                if abs(arrived.bow) < 1e-3:
                    self.assertLess(span_angle, 1e-6)
                else:
                    self.assertLess(span_angle, 0.95)

    def test_within_theme_variety(self) -> None:
        analysis = analyze(self.wav, Path(self.tmp.name))
        expected = {
            "neon": {"short", "long", "corner", "double"},
            "pastel": {"short", "long", "arch"},
            "piano": {"single", "chord", "pattern"},
            "shatter": {"crack", "block", "stack"},
        }
        for name, wanted in expected.items():
            scene = MapScene.create(1920, 1080, analysis, seed=42, theme=name)
            self.assertTrue(set(scene.timing["kinds"]) <= set(THEMES[name].objects))
            variants = {obs.variant for obs in scene.obstacles}
            self.assertGreaterEqual(len(variants & wanted), 2, (name, variants))
            self.assertTrue(any(obs.flourish for obs in scene.obstacles), name)
            speeds = []
            for seg in scene.segments[:-1]:
                dt = max(1e-6, seg.t1 - seg.t0)
                speeds.append(math.hypot(seg.x1 - seg.x0, seg.y1 - seg.y0) / dt)
            self.assertGreater(max(speeds) / min(speeds), 1.35, name)
            self.assertEqual(scene.timing["max_timing_error_s"], 0.0)
            self.assertGreaterEqual(scene.timing["min_clearance_px"], 72.0)
        neon = MapScene.create(1920, 1080, analysis, seed=42, theme="neon")
        self.assertTrue(any(abs(seg.bow) > 1.0 for seg in neon.segments))
        calm = [obs.size for obs in neon.obstacles if obs.section == "calm"]
        hot = [obs.size for obs in neon.obstacles if obs.section == "intense"]
        if calm and hot:
            self.assertGreater(sum(hot) / len(hot), sum(calm) / len(calm))

    def test_flyer_does_not_cross_placed_obstacles(self) -> None:
        analysis = analyze(self.wav, Path(self.tmp.name))
        for name in ("neon", "pastel", "arcade", "piano", "shatter"):
            for seed in (7, 42):
                scene = MapScene.create(1920, 1080, analysis, seed=seed, theme=name)
                pen = max_flyer_penetration(scene)
                self.assertLessEqual(pen, 0.0, (name, seed, round(pen, 2)))
                self.assertEqual(scene.timing["max_timing_error_s"], 0.0, name)
                self.assertGreaterEqual(scene.timing["min_clearance_px"], 72.0, (name, seed))

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
            self.assertLess(max(jumps), 0.22, name)
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

    def test_forms_land_on_the_music(self) -> None:
        analysis = analyze(self.wav, Path(self.tmp.name))
        city = build_scene("city", 1920, 1080, analysis, 42, "neon")
        self.assertEqual(city.form, "city")
        self.assertGreaterEqual(len(city.buildings), 8)
        births = {building.born for building in city.buildings}
        answered = births | {light.time for light in city.lights}
        for beat in analysis.beats:
            self.assertTrue(any(abs(beat - when) < 1e-6 for when in answered), beat)
        for building in city.buildings:
            self.assertAlmostEqual(city.building_height(building, building.born), building.target, places=4)
            self.assertGreater(city.building_height(building, building.born + 0.16), building.target)
            self.assertAlmostEqual(city.building_height(building, building.born + 0.4), building.target, places=3)
            self.assertEqual(city.building_height(building, building.born - building.grow - 0.05), 0.0)
        calm = [building.target for building in city.buildings if building.energy <= 0.35]
        hot = [building.target for building in city.buildings if building.energy >= 0.55]
        if len(calm) >= 2 and len(hot) >= 2:
            self.assertGreater(sum(hot) / len(hot), sum(calm) / len(calm))
        self.assertEqual(city.timing["max_timing_error_s"], 0.0)
        self.assertGreater(len(city.bass_times), 0)
        framed = _growth_starts_in_frame(city, analysis)
        self.assertEqual(framed, len(city.buildings))
        self.assertGreater(framed, 0)

        piano = build_scene("piano", 1920, 1080, analysis, 42, "neon")
        self.assertGreaterEqual(len(piano.notes), 8)
        for note in piano.notes:
            x, y = piano.note_head(note, note.time)
            kx, ky = piano.key_center(note.display)
            self.assertAlmostEqual(x, kx, places=4)
            self.assertAlmostEqual(y, ky, places=4)
            self.assertGreaterEqual(note.duration, 0.12)
        overlaps, worst_gap, checked = _same_key_gaps(piano)
        self.assertEqual(overlaps, 0)
        self.assertGreater(checked, 0)
        self.assertGreaterEqual(worst_gap, 3.0)
        whites = sum(1 for tone in range(piano.lo, piano.hi + 1) if tone % 12 in {0, 2, 4, 5, 7, 9, 11})
        board = piano.key_w * whites
        self.assertGreaterEqual(board / piano.width, 0.84)
        self.assertLessEqual(board / piano.width, 0.91)
        self.assertGreaterEqual(whites, 21)
        self.assertGreater(piano.sync_report["chroma"], 0)
        self.assertGreaterEqual(_visible_note_average(piano), 3.0)
        self.assertEqual(piano.timing["max_timing_error_s"], 0.0)
        self.assertLess(piano.timing["max_position_error_px"], 1e-4)

        for aspect_width, aspect_height in ((1920, 1080), (1080, 1920)):
            stars = build_scene("stars", aspect_width, aspect_height, analysis, 42, "neon")
            self.assertGreaterEqual(len(stars.stars), 8)
            self.assertGreater(len(stars.links), 0)
            for star in stars.stars:
                self.assertAlmostEqual(stars.brightness(star, star.time), 1.0, places=4)
                self.assertEqual(stars.brightness(star, star.time - 1.0), 0.0)
                self.assertGreaterEqual(star.x, 100.0)
                self.assertGreaterEqual(star.y, 100.0)
                self.assertLessEqual(star.x, stars.width - 100.0)
                self.assertLessEqual(star.y, stars.height - 100.0)
            for left_index, left in enumerate(stars.stars):
                for right in stars.stars[left_index + 1 :]:
                    self.assertGreaterEqual(math.hypot(left.x - right.x, left.y - right.y), 60.0)
            for index, (a, b) in enumerate(stars.links):
                for c, d in stars.links[index + 1 :]:
                    if len({a, b, c, d}) < 4:
                        continue
                    self.assertFalse(_segments_cross(stars.stars[a], stars.stars[b], stars.stars[c], stars.stars[d]))
            self.assertGreaterEqual(_constellation_gap(stars), 24.0)
            self.assertTrue(_stars_answer_a_beat(stars))
            self.assertEqual(stars.timing["max_timing_error_s"], 0.0)
            self.assertEqual(stars.slow_mix(1.0), 0.0)
            hit = float(analysis.beats[4])
            self.assertEqual(stars.beat_level(hit), 1.0)
            self.assertEqual(stars.beat_level(hit + 0.22), 0.0)
            self.assertAlmostEqual(stars.beat_level(hit + 0.11), 0.5, places=4)
            self.assertAlmostEqual(stars.glint_u(hit + 0.16), 0.5, places=4)
        stars = build_scene("stars", 1920, 1080, analysis, 42, "neon")

        renderer = CairoRenderer(640, 360, fps=30)
        for scene in (city, piano, stars):
            scene.update(1 / 30, min(1.0, analysis.duration * 0.5), [])
            frame = renderer.draw(scene)
            self.assertEqual(len(frame), 640 * 360 * 4)

    def test_rain_drops_land_on_the_music(self) -> None:
        analysis = analyze(self.wav, Path(self.tmp.name))
        rain = build_scene("rain", 1920, 1080, analysis, 42, "neon")
        self.assertEqual(rain.form, "rain")
        main_drops = [drop for drop in rain.drops if drop.kind in ("note", "beat")]
        self.assertGreaterEqual(len(main_drops), len(analysis.beats))
        landed = [drop.time for drop in main_drops]
        # Every beat has a drop that splashes exactly on it.
        for beat in analysis.beats:
            self.assertTrue(any(abs(beat - when) < 1e-9 for when in landed), beat)
        strong = {round(beat, 4) for beat in analysis.strong_beats}
        for drop in main_drops:
            if round(drop.time, 4) in strong:
                self.assertTrue(drop.strong)
        # Drops landing close in time are well apart on screen, and never pile up.
        self.assertGreaterEqual(rain.sync_report["min_landing_gap_px"], 120.0)
        positions = {(round(drop.u, 3), round(drop.d, 3)) for drop in main_drops}
        self.assertEqual(len(positions), len(main_drops))
        self.assertLessEqual(rain.sync_report["peak_live"], 60)
        self.assertEqual(rain.timing["max_timing_error_s"], 0.0)
        renderer = CairoRenderer(640, 360, fps=30)
        rain.update(1 / 30, min(1.0, analysis.duration * 0.5), [])
        self.assertEqual(len(renderer.draw(rain)), 640 * 360 * 4)

    def test_slow_stars_keep_moving_between_beats(self) -> None:
        analysis = _ballad_analysis()
        stars = build_scene("stars", 1920, 1080, analysis, 42, "neon")
        self.assertGreaterEqual(len(stars.stars), 4)
        self.assertGreater(stars.slow_mix(2.0), 0.9)
        self.assertLess(stars.pulse_gap(2.0), 1.2)
        hit = float(analysis.beats[3])
        gap = stars.pulse_gap(hit)
        self.assertEqual(stars.beat_level(hit), 1.0)
        # The rise is already partway up a frame before the hit, and the release
        # is still lit well after a fast song would have gone dark.
        self.assertGreater(stars.beat_level(hit - 0.06), 0.55)
        self.assertGreater(stars.beat_level(hit + 0.22), 0.35)
        self.assertLess(stars.beat_level(hit + gap * 0.78), 0.2)
        window = stars._event_window(hit, 0.32, 0.88, 1.25)
        self.assertGreater(window, gap * 0.7)
        self.assertLess(window, gap)
        self.assertAlmostEqual(stars.brightness(stars.stars[0], stars.stars[0].time), 1.0, places=4)
        live = next(star for star in stars.stars if star.phrase == stars.active_phrase(2.0))
        on_beat = stars.brightness(live, hit + 0.04)
        mid = stars.brightness(live, hit + gap * 0.72)
        self.assertGreater(on_beat, mid + 0.08)

        fps = 30
        dt = 1.0 / fps
        beats = [float(beat) for beat in analysis.beats]
        steps: list[float] = []
        previous = None
        for frame in range(int(3.2 * fps)):
            t0 = 1.0 + frame * dt
            t1 = t0 + dt
            sample = t0
            for beat in beats:
                if beat >= t1:
                    break
                if beat >= t0:
                    sample = beat
                    break
            stars.update(dt, sample, [])
            if stars.active_phrase() != live.phrase:
                previous = None
                continue
            for star in stars.stars:
                if stars.life(star.phrase) < 0.2:
                    continue
                px, py = stars.placed(star)
                self.assertGreater(px, 36.0, star.phrase)
                self.assertLess(px, stars.width - 36.0)
                self.assertGreater(py, 36.0)
                self.assertLess(py, stars.height - 36.0)
            point = stars.placed(live)
            if previous is not None:
                steps.append(math.hypot(point[0] - previous[0], point[1] - previous[1]))
            previous = point
        self.assertGreater(len(steps), 60)
        self.assertGreater(sum(steps), 12.0)
        self.assertLess(max(steps), 2.0)
        stalls = sum(1 for step in steps if step < 0.05)
        self.assertLess(stalls / len(steps), 0.12)

    def test_glint_stays_on_stars_that_are_already_lit(self) -> None:
        from music2video.forms import _glint_point

        analysis = _tempo_analysis(68, 14)
        stars = build_scene("stars", 1920, 1080, analysis, 42, "neon")
        samples = 0
        partial = 0
        time = 0.2
        while time < analysis.duration - 0.05:
            stars.time = time
            phrase = stars.active_phrase(time)
            edges = stars.lit_edges(time)
            progress = stars.glint_u(time)
            whole = [
                (stars.stars[a], stars.stars[b])
                for a, b in stars.links
                if stars.stars[a].phrase == phrase
            ]
            if progress is not None:
                samples += 1
                age = _beat_age(analysis.beats, time)
                beat_at = time if age is None else time - age
                for left, right in edges:
                    self.assertGreater(stars.brightness(left, time), 0.02)
                    self.assertGreater(stars.brightness(right, time), 0.02)
                    self.assertLessEqual(left.time, beat_at + 1e-3)
                    self.assertLessEqual(right.time, beat_at + 1e-3)
                if edges:
                    for sample in (progress, max(0.0, progress - 0.10)):
                        x, y = _glint_point(stars, edges, sample)
                        self.assertTrue(
                            any(_near_segment(stars, left, right, x, y, time) for left, right in edges),
                            (time, sample),
                        )
                if whole and len(edges) < len(whole):
                    partial += 1
            time += 1.0 / 30.0
        self.assertGreater(samples, 30)
        self.assertGreater(partial, 8)

    def test_ripple_centers_on_visible_stars(self) -> None:
        analysis = _tempo_analysis(68, 14)
        stars = build_scene("stars", 1920, 1080, analysis, 42, "neon")
        found = 0
        time = 0.05
        while time < analysis.duration - 0.05:
            phrase = stars.active_phrase(time)
            members = [star for star in stars.stars if star.phrase == phrase]
            visible = stars.ripple_members(time)
            age = _beat_age(analysis.beats, time)
            window = stars._event_window(time, 0.42, 0.78, 1.10)
            if (
                age is not None
                and age <= window
                and len(visible) >= 1
                and len(members) > len(visible)
            ):
                vx = sum(stars.placed(star, time)[0] for star in visible) / len(visible)
                vy = sum(stars.placed(star, time)[1] for star in visible) / len(visible)
                ax = sum(stars.placed(star, time)[0] for star in members) / len(members)
                ay = sum(stars.placed(star, time)[1] for star in members) / len(members)
                self.assertGreater(math.hypot(vx - ax, vy - ay), 18.0, time)
                # Every star inside the ring's focus is already drawn.
                for star in visible:
                    self.assertGreater(stars.brightness(star, time), 0.02)
                found += 1
            time += 0.05
        self.assertGreater(found, 4)

    def test_midtempo_easing_moves_through_the_bar(self) -> None:
        analysis = _tempo_analysis(90, 16)
        stars = build_scene("stars", 1920, 1080, analysis, 42, "neon")
        gap = stars.pulse_gap(4.0)
        self.assertGreater(gap, 0.64)
        self.assertLess(gap, 0.70)
        self.assertGreater(stars.slow_mix(4.0), 0.9)
        # A bar around 0.66–0.77s (about 78–90 BPM) is in the slow drift.
        for bpm in (78, 85, 90):
            sky = build_scene("stars", 1920, 1080, _tempo_analysis(bpm, 10), 42, "neon")
            self.assertGreater(sky.slow_mix(3.0), 0.9, bpm)
        # 110–123 BPM keeps the short hit. The second half of the bar does not drift.
        for bpm in (110, 116, 123):
            sky = build_scene("stars", 1920, 1080, _tempo_analysis(bpm, 10), 42, "neon")
            self.assertEqual(sky.slow_mix(3.0), 0.0, bpm)
            hit = float(sky.analysis.beats[4])
            self.assertEqual(sky.beat_level(hit + 0.22), 0.0, bpm)
            self.assertAlmostEqual(sky._event_window(hit, 0.32, 0.88, 1.25), 0.32, places=4)

        steps, second = _drift_steps(stars)
        self.assertGreater(len(steps), 60)
        # The attack still moves a couple of pixels in one frame. A hitch would be far larger.
        self.assertLess(max(steps), 3.5)
        self.assertGreater(len(second), 30)
        self.assertGreater(sum(second) / len(second), 0.25)
        stalls = sum(1 for step in second if step < 0.05)
        self.assertLess(stalls / len(second), 0.12)

        fast = build_scene("stars", 1920, 1080, _tempo_analysis(120, 12), 42, "neon")
        _steps, fast_second = _drift_steps(fast, seconds=4.0)
        self.assertGreater(len(fast_second), 15)
        self.assertLess(sum(fast_second) / len(fast_second), 0.12)

    def test_a_rest_keeps_a_constellation_up(self) -> None:
        analysis = _tempo_analysis(90, 28, rest=(8.0, 18.5))
        for width, height in ((1920, 1080), (1080, 1920)):
            stars = build_scene("stars", width, height, analysis, 42, "neon")
            self.assertGreaterEqual(len({star.phrase for star in stars.stars}), 2)
            empty = 0
            single = 0
            frames = 0
            time = 9.0
            while time < 19.0:
                shown = [star for star in stars.stars if stars.brightness(star, time) > 0.02]
                if not shown:
                    empty += 1
                if len(shown) < 2:
                    single += 1
                frames += 1
                time += 0.1
            self.assertGreater(frames, 50)
            self.assertEqual(empty, 0, (width, height))
            self.assertEqual(single, 0, (width, height))

    def test_portrait_sky_uses_the_lower_half(self) -> None:
        analysis = _tempo_analysis(90, 36)
        for seed in (1, 7, 42):
            stars = build_scene("stars", 1080, 1920, analysis, seed, "neon")
            groups: dict[int, list] = {}
            for star in stars.stars:
                groups.setdefault(star.phrase, []).append(star)
            self.assertGreaterEqual(len(groups), 3, seed)
            self.assertGreaterEqual(_constellation_gap(stars), 24.0, seed)
            cents = []
            earliest = None
            for members in groups.values():
                cy = sum(star.y for star in members) / len(members)
                cents.append(cy)
                start = min(star.time for star in members)
                if earliest is None or start < earliest[0]:
                    earliest = (start, cy)
            lower = sum(1 for y in cents if y >= stars.height * 0.5)
            self.assertEqual(lower, len(cents), (seed, [round(y) for y in cents]))
            piled = sum(1 for y in cents if y < stars.height * 0.22)
            self.assertEqual(piled, 0, (seed, [round(y) for y in cents]))
            self.assertGreater(max(cents), stars.height * 0.75, seed)
            self.assertIsNotNone(earliest)
            self.assertGreater(earliest[1], stars.height * 0.50, seed)

    def test_a_lone_star_grows_a_small_figure(self) -> None:
        from music2video.forms import _company_for_strays
        from music2video.notes import Tone

        analysis = _tempo_analysis(90, 16)
        stray = [Tone(0.10, 0.2, 60, "pitch")]
        later = [Tone(12.0, 0.2, 62, "pitch"), Tone(12.67, 0.2, 64, "pitch")]
        filled = _company_for_strays([stray, later], analysis)
        self.assertGreaterEqual(len(filled[0]), 3)
        self.assertLess(filled[0][-1].time, 3.0)
        self.assertEqual([tone.time for tone in filled[1]], [12.0, 12.67])
        # A one-star phrase whose neighbor arrives quickly stays one star.
        close = _company_for_strays(
            [[Tone(1.0, 0.2, 60, "pitch")], [Tone(3.0, 0.2, 62, "pitch"), Tone(3.6, 0.2, 64, "pitch")]],
            analysis,
        )
        self.assertEqual(len(close[0]), 1)
        whole = [Tone(index * 0.67, 0.2, 60, "pitch") for index in range(6)]
        self.assertEqual(len(_company_for_strays([whole], analysis)[0]), 6)


def _growth_starts_in_frame(city, analysis, fps: int = 30) -> int:
    """Step like the CLI. Count towers whose body is inside a 10% margin at growth start.

    The next tower, the one about to grow, has to clear the same margin on that frame.
    """
    dt = 1.0 / fps
    nframes = int(math.floor(analysis.duration * fps + 1e-6))
    beats = [float(beat) for beat in analysis.beats]
    beat_index = 0
    seen: set[int] = set()
    for frame in range(nframes):
        t0 = frame * dt
        t1 = t0 + dt
        events: list[tuple[float, bool]] = []
        while beat_index < len(beats) and beats[beat_index] < t1 - 1e-9:
            if beats[beat_index] >= t0 - 1e-9:
                events.append((beats[beat_index], False))
            beat_index += 1
        sample = events[0][0] if events else t0
        city.update(dt, sample, events)
        for index, building in enumerate(city.buildings):
            if index in seen or city.time + 1e-9 < building.born - building.grow:
                continue
            _assert_box_margin(city, building, 0.0)
            if index + 1 < len(city.buildings):
                _assert_box_margin(city, city.buildings[index + 1], 0.0)
            seen.add(index)
    return len(seen)


def _assert_box_margin(city, building, risen: float) -> None:
    left, top, right, bottom = city.screen_box(building, risen)
    margin = 0.10
    if left < city.width * margin or right > city.width * (1.0 - margin):
        raise AssertionError(
            f"building x={building.x:.1f} horizontal margin {left:.1f}..{right:.1f} "
            f"on {city.width:.0f} at t={city.time:.3f}"
        )
    if top < city.height * margin or bottom > city.height * (1.0 - margin):
        raise AssertionError(
            f"building x={building.x:.1f} vertical margin {top:.1f}..{bottom:.1f} "
            f"on {city.height:.0f} at t={city.time:.3f}"
        )


def _visible_note_average(scene, step: float = 0.25) -> float:
    total = 0
    frames = 0
    time = 0.4
    end = max(0.4, scene.duration - 0.4)
    while time <= end + 1e-6:
        scene.time = time
        visible = 0
        for note in scene.notes:
            if note.span <= 0.02:
                continue
            top, bottom = scene.note_bounds(note)
            if bottom - top >= 1.2 and bottom >= -24 and top <= scene.key_top + 6:
                visible += 1
        total += visible
        frames += 1
        time += step
    return total / max(frames, 1)


def _same_key_gaps(scene, step: float = 1.0 / 30.0) -> tuple[int, float, int]:
    """Return overlap count, the smallest positive gap in pixels, and pairs checked."""
    groups: dict[int, list] = {}
    for note in scene.notes:
        groups.setdefault(note.display, []).append(note)
    overlaps = 0
    worst = 1e9
    checked = 0
    time = 0.0
    while time <= scene.duration + 1e-6:
        scene.time = time
        for group in groups.values():
            boxes = []
            for note in group:
                if note.span <= 0.02:
                    continue
                top, bottom = scene.note_bounds(note)
                if bottom - top < 1.2:
                    continue
                boxes.append((top, bottom))
            for index, (top, bottom) in enumerate(boxes):
                for other_top, other_bottom in boxes[index + 1 :]:
                    if bottom <= other_top:
                        gap = other_top - bottom
                    elif other_bottom <= top:
                        gap = top - other_bottom
                    else:
                        gap = -min(bottom, other_bottom) + max(top, other_top)
                    checked += 1
                    if gap < 3.0:
                        overlaps += 1
                    else:
                        worst = min(worst, gap)
        time += step
    if worst > 1e8:
        worst = 0.0
    return overlaps, worst, checked


def _ballad_analysis() -> Analysis:
    """A flat 64 BPM grid with no audio file, so the sky is built from the beats."""
    return _tempo_analysis(64.0, 12.0, energy=0.28)


def _tempo_analysis(
    bpm: float,
    duration: float = 16.0,
    rest: tuple[float, float] | None = None,
    energy: float = 0.4,
) -> Analysis:
    """A flat grid with no audio file. `rest` drops beats, which drops the melody too."""
    period = 60.0 / bpm
    beats: list[float] = []
    time = 0.0
    while time < duration - 1e-9:
        if rest is None or not (rest[0] <= time < rest[1]):
            beats.append(time)
        time += period
    sections = tuple(
        Section(index * 2.0, min(duration, (index + 1) * 2.0), energy, "verse")
        for index in range(int(duration / 2.0) + 1)
    )
    return Analysis(
        tempo=bpm,
        beats=beats,
        strong_beats=beats[1::2],
        duration=duration,
        sample_rate=44100,
        wav_path=Path(f"/tmp/music2video-no-such-{bpm:.0f}.wav"),
        sections=sections,
        rms_times=(0.0, duration),
        rms=(energy, energy),
        bands=((0.08,) * 8, (0.08,) * 8),
    )


def _beat_age(beats, now: float) -> float | None:
    last = None
    for hit in beats:
        if hit > now:
            break
        last = hit
    if last is None:
        return None
    return now - last


def _near_segment(scene, left, right, x: float, y: float, time: float, tol: float = 1.5) -> bool:
    ax, ay = scene.placed(left, time)
    bx, by = scene.placed(right, time)
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 < 1e-6:
        return math.hypot(x - ax, y - ay) <= tol
    along = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length2))
    return math.hypot(x - (ax + along * dx), y - (ay + along * dy)) <= tol


def _drift_steps(scene, seconds: float = 6.0, fps: int = 30) -> tuple[list[float], list[float]]:
    """Frame steps of the live star, and the subset that falls in the second half of a bar."""
    dt = 1.0 / fps
    beats = [float(beat) for beat in scene.analysis.beats]
    steps: list[float] = []
    second: list[float] = []
    previous = None
    previous_phrase = None
    for frame in range(int(seconds * fps)):
        t0 = 1.0 + frame * dt
        t1 = t0 + dt
        sample = t0
        for beat in beats:
            if beat >= t1:
                break
            if beat >= t0:
                sample = beat
                break
        scene.update(dt, sample, [])
        phrase = scene.active_phrase()
        if phrase < 0 or scene.life(phrase) < 0.2:
            previous = None
            continue
        gap = scene.pulse_gap()
        age = _beat_age(beats, scene.time)
        live = next(star for star in scene.stars if star.phrase == phrase)
        point = scene.placed(live)
        if previous is not None and previous_phrase == phrase:
            step = math.hypot(point[0] - previous[0], point[1] - previous[1])
            steps.append(step)
            if age is not None and 0.45 * gap <= age <= 0.92 * gap:
                second.append(step)
        previous = point
        previous_phrase = phrase
    return steps, second


def _constellation_gap(scene) -> float:
    """Smallest gap between the bounding circles of two constellations. Empty sky returns a large gap."""
    groups: dict[int, list] = {}
    for star in scene.stars:
        groups.setdefault(star.phrase, []).append(star)
    circles = []
    for group in groups.values():
        cx = sum(star.x for star in group) / len(group)
        cy = sum(star.y for star in group) / len(group)
        radius = max(math.hypot(star.x - cx, star.y - cy) for star in group)
        circles.append((cx, cy, radius))
    if len(circles) < 2:
        return 1e6
    gap = 1e6
    for index, (ax, ay, ar) in enumerate(circles):
        for bx, by, br in circles[index + 1 :]:
            gap = min(gap, math.hypot(ax - bx, ay - by) - ar - br)
    return gap


def _stars_answer_a_beat(scene) -> bool:
    """A star in the live phrase is brighter on a beat than halfway to the next one."""
    beats = [float(beat) for beat in scene.analysis.beats]
    for star in scene.stars:
        later = [beat for beat in beats if beat > star.time + 0.45]
        if len(later) < 2:
            continue
        for hit, nxt in zip(later, later[1:]):
            if nxt - hit < 0.28 or nxt - hit > 1.4:
                continue
            if scene.active_phrase(hit + 0.04) != star.phrase:
                continue
            if scene.life(star.phrase, hit + 0.04) < 0.9:
                continue
            mid = hit + (nxt - hit) * 0.55
            if scene.brightness(star, hit + 0.04) > scene.brightness(star, mid) + 0.08:
                return True
    return False


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r) -> float:
        return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)

    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


if __name__ == "__main__":
    unittest.main()
