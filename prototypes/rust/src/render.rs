//! tiny-skia frames piped to ffmpeg (H.264 + AAC from the source wav).

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread;

use tiny_skia::{Color, FillRule, Paint, PathBuilder, Pixmap, Stroke, Transform};

use crate::scene::{hsl_to_rgb, Scene};

const FPS: f64 = 30.0;

pub fn render_mp4(
    audio: &str,
    output: &str,
    beats: &[f64],
    strong_flags: &[bool],
    duration: f64,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let nframes = (duration * FPS).floor() as usize;
    let mut scene = Scene::new(width, height);
    let mut pixmap = Pixmap::new(width, height).ok_or("pixmap allocation failed")?;
    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{width}x{height}"),
            "-r",
            "30",
            "-i",
            "pipe:0",
            "-i",
            audio,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            output,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("ffmpeg: {err}"))?;
    let mut stdin = child.stdin.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let stderr_thread = thread::spawn(move || {
        let mut buf = Vec::new();
        stderr.read_to_end(&mut buf).ok();
        buf
    });

    let mut beat_index = 0usize;
    let mut flash = 0.0;
    let dt = 1.0 / FPS;
    for frame in 0..nframes {
        let t0 = frame as f64 * dt;
        let t1 = t0 + dt;
        let mut beat = false;
        let mut strong = false;
        while beat_index < beats.len() && beats[beat_index] < t1 {
            if beats[beat_index] >= t0 {
                beat = true;
                strong = strong || strong_flags[beat_index];
            }
            beat_index += 1;
        }
        let pulse = scene.step(beat, strong, dt);
        flash = f64::max(flash * 0.86, pulse);
        draw(&mut pixmap, &scene, flash);
        stdin.write_all(pixmap.data()).map_err(|err| format!("pipe: {err}"))?;
    }
    drop(stdin);
    let stderr_buf = stderr_thread.join().unwrap_or_default();
    let status = child.wait().map_err(|err| err.to_string())?;
    if !status.success() {
        return Err(format!(
            "ffmpeg exited {status}: {}",
            String::from_utf8_lossy(&stderr_buf)
        ));
    }
    Ok(())
}

fn draw(pixmap: &mut Pixmap, scene: &Scene, flash: f64) {
    let lift = flash * 36.0;
    pixmap.fill(rgb((14.0 + lift) / 255.0, (16.0 + lift) / 255.0, (24.0 + lift) / 255.0));
    for particle in &scene.particles {
        let k = (particle.life / particle.max_life).max(0.0);
        let (r, g, b) = hsl_to_rgb(particle.hue, 0.80, 0.35 + 0.35 * k);
        fill_circle(
            pixmap,
            particle.x as f32,
            particle.y as f32,
            (particle.r * (0.35 + 0.65 * k)) as f32,
            rgb(r, g, b),
        );
    }
    for ball in &scene.balls {
        let (r, g, b) = hsl_to_rgb(ball.hue, 0.75, 0.55);
        fill_circle(pixmap, ball.x as f32, ball.y as f32, ball.r as f32, rgb(r, g, b));
        let stroke = if ball.flash > 0.4 {
            rgb(1.0, 1.0, 1.0)
        } else {
            let (sr, sg, sb) = hsl_to_rgb(ball.hue, 0.80, 0.75);
            rgb(sr, sg, sb)
        };
        stroke_circle(pixmap, ball.x as f32, ball.y as f32, ball.r as f32, stroke);
    }
}

fn rgb(r: f64, g: f64, b: f64) -> Color {
    Color::from_rgba(r.clamp(0.0, 1.0) as f32, g.clamp(0.0, 1.0) as f32, b.clamp(0.0, 1.0) as f32, 1.0)
        .unwrap()
}

fn circle_path(x: f32, y: f32, r: f32) -> Option<tiny_skia::Path> {
    if r <= 0.2 {
        return None;
    }
    let mut builder = PathBuilder::new();
    builder.push_circle(x, y, r);
    builder.finish()
}

fn fill_circle(pixmap: &mut Pixmap, x: f32, y: f32, r: f32, color: Color) {
    let Some(path) = circle_path(x, y, r) else {
        return;
    };
    let mut paint = Paint::default();
    paint.anti_alias = true;
    paint.set_color(color);
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
}

fn stroke_circle(pixmap: &mut Pixmap, x: f32, y: f32, r: f32, color: Color) {
    let Some(path) = circle_path(x, y, r) else {
        return;
    };
    let mut paint = Paint::default();
    paint.anti_alias = true;
    paint.set_color(color);
    let mut stroke = Stroke::default();
    stroke.width = 3.0;
    pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
}
