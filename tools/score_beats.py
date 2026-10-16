#!/usr/bin/env python3
"""Score estimated beats against ground truth with a 70ms match window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def match_counts(reference: list[float], estimate: list[float], tol: float = 0.070) -> dict:
    used = [False] * len(estimate)
    tp = 0
    for ref in reference:
        best_i = None
        best_d = tol
        for i, est in enumerate(estimate):
            if used[i]:
                continue
            delta = abs(est - ref)
            if delta <= best_d:
                best_d = delta
                best_i = i
        if best_i is not None:
            used[best_i] = True
            tp += 1
    fp = len(estimate) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reference": len(reference),
        "estimate": len(estimate),
    }


def interval_stats(beats: list[float]) -> dict:
    if len(beats) < 3:
        return {"median_interval": None, "cv": None, "regular_fraction": None}
    gaps = [b - a for a, b in zip(beats, beats[1:])]
    ordered = sorted(gaps)
    med = ordered[len(ordered) // 2]
    mean = sum(gaps) / len(gaps)
    var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    std = var**0.5
    regular = sum(1 for g in gaps if abs(g - med) <= 0.10 * med) / len(gaps)
    return {
        "median_interval": med,
        "implied_bpm": (60.0 / med) if med else None,
        "cv": (std / mean) if mean else None,
        "regular_fraction": regular,
    }


def score_file(truth_path: Path, estimate_path: Path, tol: float = 0.070) -> dict:
    truth = json.loads(truth_path.read_text())
    est = json.loads(estimate_path.read_text())
    beats = est.get("beats", [])
    strong = est.get("strong_beats", [])
    return {
        "tempo_estimate": est.get("tempo"),
        "tempo_target": truth.get("bpm"),
        "tempo_error_bpm": abs(float(est.get("tempo", 0)) - float(truth["bpm"])),
        "beats": match_counts(truth["beats"], beats, tol),
        "strong_beats": match_counts(truth["strong_beats"], strong, tol),
        "intervals": interval_stats(beats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=0.070)
    args = parser.parse_args()
    print(json.dumps(score_file(args.truth, args.estimate, args.tol), indent=2))


if __name__ == "__main__":
    main()
