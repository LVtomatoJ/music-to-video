//! Spectral-flux + Ellis beat tracker, ported from tools/flux_tracker.py.
//! Python 3 round-half-even is matched so tempo lags land on the same frames.

use std::f64::consts::PI;

use hound::{SampleFormat, WavReader};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

const N_FFT: usize = 2048;
const HOP: usize = 512;
const HF_HZ: f64 = 2000.0;

#[derive(Clone, Debug)]
pub struct Analysis {
    pub tempo: f64,
    pub beats: Vec<f64>,
    pub strong_beats: Vec<f64>,
    pub duration: f64,
}

pub fn read_wav_mono(path: &str) -> Result<(Vec<f64>, u32), String> {
    let mut reader = WavReader::open(path).map_err(|err| format!("wav: {err}"))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err("wav has zero channels".into());
    }
    let interleaved: Vec<f64> = match spec.sample_format {
        SampleFormat::Int => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f64 / 32768.0).map_err(|err| err.to_string()))
            .collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| sample.map(|value| value as f64).map_err(|err| err.to_string()))
            .collect::<Result<Vec<_>, _>>()?,
    };
    let frames = interleaved.len() / channels;
    let mut mono = vec![0.0; frames];
    for (index, sample) in interleaved.iter().enumerate() {
        mono[index / channels] += sample / channels as f64;
    }
    Ok((mono, spec.sample_rate))
}

fn py_round(x: f64) -> i32 {
    let negative = x < 0.0;
    let ax = x.abs();
    let base = ax.floor();
    let frac = ax - base;
    let rounded = if (frac - 0.5).abs() < 1e-8 {
        if (base as i64) % 2 == 0 {
            base
        } else {
            base + 1.0
        }
    } else if frac > 0.5 {
        base + 1.0
    } else {
        base
    };
    let signed = if negative { -rounded } else { rounded };
    signed as i32
}

fn hanning(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos())
        .collect()
}

fn flux(y: &[f64], sr: u32) -> (Vec<f64>, Vec<f64>, f64) {
    let window = hanning(N_FFT);
    if y.len() < N_FFT {
        return (Vec::new(), Vec::new(), sr as f64 / HOP as f64);
    }
    let n_frames = 1 + (y.len() - N_FFT) / HOP;
    let spec_bins = N_FFT / 2 + 1;
    let mut hf_index = Vec::new();
    for k in 0..spec_bins {
        let freq = k as f64 * sr as f64 / N_FFT as f64;
        if freq >= HF_HZ {
            hf_index.push(k);
        }
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buffer = vec![Complex::new(0.0, 0.0); N_FFT];
    let mut prev = vec![0.0; spec_bins];
    let mut flux = vec![0.0; n_frames];
    let mut hf_flux = vec![0.0; n_frames];
    for frame in 0..n_frames {
        let start = frame * HOP;
        for k in 0..N_FFT {
            buffer[k] = Complex::new(y[start + k] * window[k], 0.0);
        }
        fft.process(&mut buffer);
        let mut full = 0.0;
        let mut high = 0.0;
        let mut next_prev = vec![0.0; spec_bins];
        for k in 0..spec_bins {
            let mag = buffer[k].norm();
            let delta = mag - prev[k];
            if delta > 0.0 {
                full += delta;
            }
            next_prev[k] = mag;
        }
        for &k in &hf_index {
            let delta = next_prev[k] - prev[k];
            if delta > 0.0 {
                high += delta;
            }
        }
        prev = next_prev;
        flux[frame] = full;
        hf_flux[frame] = high;
    }
    (flux, hf_flux, sr as f64 / HOP as f64)
}

fn autocorr(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut ac = vec![0.0; n];
    for lag in 0..n {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += x[i] * x[i + lag];
        }
        ac[lag] = sum;
    }
    ac
}

fn estimate_bpm(flux: &[f64], frame_rate: f64) -> f64 {
    let mean = flux.iter().sum::<f64>() / flux.len() as f64;
    let centered: Vec<f64> = flux.iter().map(|v| v - mean).collect();
    let ac = autocorr(&centered);
    let norm = ac[0] + 1e-12;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_bpm = 120.0;
    for bpm in 80..171 {
        let lag = py_round(frame_rate * 60.0 / bpm as f64);
        if lag <= 0 || lag as usize >= ac.len() {
            continue;
        }
        let lo = 1.max(lag - 1) as usize;
        let hi = ac.len().min(lag as usize + 2);
        let mut peak = f64::MIN;
        for value in &ac[lo..hi] {
            peak = peak.max(*value / norm);
        }
        let prior = (-((bpm as f64 - 120.0).powi(2)) / (2.0 * 30.0_f64.powi(2))).exp();
        let score = peak * prior;
        if score > best_score {
            best_score = score;
            let lag_f = py_round(frame_rate * 60.0 / bpm as f64) as f64;
            best_bpm = 60.0 * frame_rate / lag_f;
        }
    }
    best_bpm
}

fn ellis_beats(onset: &[f64], frame_rate: f64, bpm: f64) -> Vec<f64> {
    let tightness = 100.0;
    let period = ((60.0 / bpm) * frame_rate).max(1.0);
    let n = onset.len();
    let mut pr_start = py_round(-2.0 * period);
    let mut pr_end = py_round(-period / 2.0);
    if pr_end >= -1 {
        pr_end = -1;
    }
    if pr_start > pr_end {
        pr_start = pr_end;
    }
    let prange: Vec<i32> = (pr_start..=pr_end).collect();
    let txwt: Vec<f64> = prange
        .iter()
        .map(|lag| -tightness * ((-(*lag) as f64) / period).ln().powi(2))
        .collect();
    let mean = onset.iter().sum::<f64>() / n as f64;
    let var = onset.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt();
    let local: Vec<f64> = if std > 1e-9 {
        onset.iter().map(|v| v / std).collect()
    } else {
        onset.to_vec()
    };
    let mut backlink = vec![-1i32; n];
    let mut cumscore = vec![f64::NEG_INFINITY; n];
    let start_i = 0.max(-prange[0]) as usize;
    for i in 0..start_i.min(n) {
        cumscore[i] = local[i];
    }
    for i in start_i..n {
        let mut best = f64::NEG_INFINITY;
        let mut chosen = -1i32;
        let mut any = false;
        for (idx, lag) in prange.iter().enumerate() {
            let timer = i as i32 + *lag;
            if timer >= 0 && (timer as usize) < i {
                let score = txwt[idx] + cumscore[timer as usize];
                if !any || score > best {
                    best = score;
                    chosen = timer;
                    any = true;
                }
            }
        }
        if !any {
            cumscore[i] = local[i];
            continue;
        }
        cumscore[i] = local[i] + best;
        backlink[i] = chosen;
    }
    let end_lo = n.saturating_sub(py_round(period).max(0) as usize);
    let mut end = end_lo;
    let mut best = f64::NEG_INFINITY;
    for (offset, score) in cumscore[end_lo..].iter().enumerate() {
        if offset == 0 || *score > best {
            best = *score;
            end = end_lo + offset;
        }
    }
    let mut frames = Vec::new();
    let mut seen = vec![false; n];
    let mut cursor = end as i32;
    while cursor >= 0 && !seen[cursor as usize] {
        seen[cursor as usize] = true;
        frames.push(cursor);
        cursor = backlink[cursor as usize];
    }
    frames.reverse();
    frames.into_iter().map(|frame| frame as f64 / frame_rate).collect()
}

fn median(values: &[f64]) -> f64 {
    let mut copy = values.to_vec();
    copy.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = copy.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        copy[n / 2]
    } else {
        0.5 * (copy[n / 2 - 1] + copy[n / 2])
    }
}

fn strong_beats(beat_times: &[f64], hf_flux: &[f64], frame_rate: f64) -> Vec<f64> {
    if beat_times.is_empty() || hf_flux.is_empty() {
        return Vec::new();
    }
    let vals: Vec<f64> = beat_times
        .iter()
        .map(|t| {
            let mut idx = py_round(t * frame_rate);
            if idx < 0 {
                idx = 0;
            }
            let max_idx = hf_flux.len() as i32 - 1;
            if idx > max_idx {
                idx = max_idx;
            }
            hf_flux[idx as usize]
        })
        .collect();
    let threshold = median(&vals) * 1.35;
    beat_times
        .iter()
        .zip(vals.iter())
        .filter_map(|(t, v)| if *v >= threshold && *v > 0.0 { Some(*t) } else { None })
        .collect()
}

pub fn analyze(path: &str) -> Result<Analysis, String> {
    let (y, sr) = read_wav_mono(path)?;
    let duration = y.len() as f64 / sr as f64;
    let (flux, hf_flux, frame_rate) = flux(&y, sr);
    if flux.is_empty() {
        return Err("audio shorter than one FFT window".into());
    }
    let tempo = estimate_bpm(&flux, frame_rate);
    let beats = ellis_beats(&flux, frame_rate, tempo);
    let strong = strong_beats(&beats, &hf_flux, frame_rate);
    Ok(Analysis {
        tempo,
        beats,
        strong_beats: strong,
        duration,
    })
}
