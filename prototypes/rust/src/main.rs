mod analyze;
mod render;
mod scene;

use std::env;
use std::fs;
use std::time::Instant;

use analyze::analyze;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut input = String::new();
    let mut output = String::new();
    let mut beats_out = String::new();
    let mut analyze_only = false;
    let mut width: u32 = 1920;
    let mut height: u32 = 1080;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => {
                input = need(&args, index)?;
                index += 2;
            }
            "--output" => {
                output = need(&args, index)?;
                index += 2;
            }
            "--beats-out" => {
                beats_out = need(&args, index)?;
                index += 2;
            }
            "--width" => {
                width = need(&args, index)?.parse().map_err(|err| format!("width: {err}"))?;
                index += 2;
            }
            "--height" => {
                height = need(&args, index)?.parse().map_err(|err| format!("height: {err}"))?;
                index += 2;
            }
            "--analyze-only" => {
                analyze_only = true;
                index += 1;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if input.is_empty() || beats_out.is_empty() || (!analyze_only && output.is_empty()) {
        return Err("usage: beatballs-rust --input track.wav --output out.mp4 --beats-out beats.json [--analyze-only]".into());
    }
    if let Some(parent) = std::path::Path::new(&beats_out).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
    }
    let t0 = Instant::now();
    let analysis = analyze(&input)?;
    let t1 = Instant::now();
    if !analyze_only {
        if let Some(parent) = std::path::Path::new(&output).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|err| err.to_string())?;
            }
        }
        let flags = strong_flags(&analysis.beats, &analysis.strong_beats);
        render::render_mp4(
            &input,
            &output,
            &analysis.beats,
            &flags,
            analysis.duration,
            width,
            height,
        )?;
    }
    let t2 = Instant::now();
    let analysis_s = t1.duration_since(t0).as_secs_f64();
    let render_s = if analyze_only { 0.0 } else { t2.duration_since(t1).as_secs_f64() };
    let total_s = t2.duration_since(t0).as_secs_f64();
    let body = format!(
        "{{\n  \"stack\": \"rust-rustfft-tiny-skia\",\n  \"tempo\": {:.8},\n  \"beats\": [{}],\n  \"strong_beats\": [{}],\n  \"duration\": {:.8},\n  \"timing\": {{\"analysis_s\": {:.6}, \"render_encode_s\": {:.6}, \"total_s\": {:.6}}}\n}}\n",
        analysis.tempo,
        join_floats(&analysis.beats),
        join_floats(&analysis.strong_beats),
        analysis.duration,
        analysis_s,
        render_s,
        total_s
    );
    fs::write(&beats_out, body).map_err(|err| err.to_string())?;
    println!(
        "{{\"analysis_s\": {analysis_s:.6}, \"render_encode_s\": {render_s:.6}, \"total_s\": {total_s:.6}}}"
    );
    Ok(())
}

fn need(args: &[String], index: usize) -> Result<String, String> {
    args.get(index + 1)
        .cloned()
        .ok_or_else(|| format!("missing value for {}", args[index]))
}

fn join_floats(values: &[f64]) -> String {
    values.iter().map(|value| format!("{value:.8}")).collect::<Vec<_>>().join(", ")
}

fn strong_flags(beats: &[f64], strong: &[f64]) -> Vec<bool> {
    let mut flags = Vec::with_capacity(beats.len());
    let mut si = 0usize;
    for time in beats {
        while si < strong.len() && strong[si] < time - 1e-6 {
            si += 1;
        }
        let is_strong = si < strong.len() && (strong[si] - time).abs() <= 1e-6;
        flags.push(is_strong);
        if is_strong {
            si += 1;
        }
    }
    flags
}
