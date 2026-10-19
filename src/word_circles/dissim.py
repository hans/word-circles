"""Dissim. baseline (Malan et al. 2024, s-malan/prom-wordseg), reimplemented.

Features are standardised over the whole split. The cosine distance between
adjacent frames is z-scored per utterance, smoothed with a left-padded box
filter, and its peaks are boundaries. A final boundary at T is appended
unless a peak already lies within one frame of it. Primary settings: HuBERT
L9, window 4, prominence 0.75.
"""

import argparse
from pathlib import Path

import numpy as np
from concept_analysis.features import NpzFeatureReader
from scipy.signal import find_peaks
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler

from word_circles.evaluation import write_predictions

FRAME_S = 0.02


def normalize_split(features):
    """One StandardScaler fit on every frame of the split, applied to each utterance."""
    scaler = StandardScaler()
    scaler.partial_fit(np.concatenate(features))
    return [scaler.transform(x) for x in features]


def distance_curve(x):
    """Cosine distance between adjacent frames, z-scored within the utterance."""
    d = np.array([distance.cosine(x[i], x[i + 1]) for i in range(x.shape[0] - 1)])
    return StandardScaler().fit_transform(d.reshape(-1, 1)).reshape(-1)


def smoothed_curve(d, window):
    padded = np.pad(d, ((window // 2) + 1, 0), mode="edge")
    return np.convolve(padded, np.ones(window) / window, "valid")


def trough_peaks(x, prominence, window):
    peaks, _ = find_peaks(smoothed_curve(distance_curve(x), window), prominence=prominence)
    return peaks


def with_final_boundary(peaks, n_frames):
    if len(peaks) == 0:
        return np.array([n_frames])
    if peaks[-1] not in (n_frames, n_frames - 1):
        return np.append(peaks, n_frames)
    return peaks


def boundary_frames(x, prominence, window):
    if x.shape[0] == 1:
        return np.array([1])
    return with_final_boundary(trough_peaks(x, prominence, window), x.shape[0])


def predict(features, stems, prominence, window):
    return {stem: [float(f) * FRAME_S for f in boundary_frames(x, prominence, window)]
            for stem, x in zip(stems, features)}


def load_split(cache_dir, utts, layer):
    stems = Path(utts).read_text().split()
    return stems, normalize_split(NpzFeatureReader(cache_dir).read_layer(stems, layer))


def retune(caches, utts, refs, evaluator_dir, out_csv, layer, windows, prominences):
    """Band-miss diagnostic: best dev R-value over a window x prominence grid, then its test row.

    Reported as a side row only; the primary row keeps Malan24's settings.
    """
    import csv
    import tempfile

    from word_circles.evaluation import score_boundaries

    rows = []
    with tempfile.TemporaryDirectory() as work:
        stems, features = load_split(caches["dev"], utts["dev"], layer)
        for window in windows:
            for prominence in prominences:
                result = score_boundaries(predict(features, stems, prominence, window), refs["dev"], evaluator_dir,
                                          Path(work) / "pred")
                rows.append({"split": "dev", "window": window, "prominence": prominence, **result.scores})
        best = max(rows, key=lambda r: r["R"])
        stems, features = load_split(caches["test"], utts["test"], layer)
        result = score_boundaries(predict(features, stems, best["prominence"], best["window"]), refs["test"],
                                  evaluator_dir, Path(work) / "pred")
        rows.append({"split": "test", "window": best["window"], "prominence": best["prominence"], **result.scores})
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("predict")
    pre.add_argument("cache_dir", type=Path)
    pre.add_argument("utts", type=Path)
    pre.add_argument("out", type=Path)
    pre.add_argument("--layer", type=int, default=9)
    pre.add_argument("--window", type=int, default=4)
    pre.add_argument("--prominence", type=float, default=0.75)
    ret = sub.add_parser("retune")
    for name in ("dev_cache", "test_cache", "dev_utts", "test_utts", "dev_refs", "test_refs", "evaluator_dir",
                 "out_csv"):
        ret.add_argument(name, type=Path)
    ret.add_argument("--layer", type=int, default=9)
    args = parser.parse_args()
    if args.command == "predict":
        stems, features = load_split(args.cache_dir, args.utts, args.layer)
        write_predictions(predict(features, stems, args.prominence, args.window), args.out)
    else:
        retune({"dev": args.dev_cache, "test": args.test_cache}, {"dev": args.dev_utts, "test": args.test_utts},
               {"dev": args.dev_refs, "test": args.test_refs}, args.evaluator_dir, args.out_csv, args.layer,
               windows=range(2, 9), prominences=np.round(np.linspace(0.25, 1.5, 6), 3))


if __name__ == "__main__":
    main()
