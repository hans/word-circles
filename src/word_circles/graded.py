"""Per-frame RoSeg boundary confidence in abstractography's graded-score parquet layout.

The score is `detect_boundaries`'s trough signal before peak picking:
minus the smoothed angle to the utterance's first frame in the 2-D temporal
subspace, so higher means more boundary-like. Prominence and theta_max only
gate peaks and do not enter it. Its per-utterance `max - ` offset is dropped
so that scores stay comparable across utterances.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from concept_analysis.trajectory import get_sequence_relative_angles, smoothing

from word_circles.roseg import _no_threshold, grid, load_subspace, project_utterances, score_configs, select_config, write_rows

COLUMNS = {"stem": "object", "frame": "int64", "graded_score": "float64", "layer": "int64"}


def graded_curve(projected, window):
    """(T,) boundary confidence for one utterance's (T, 2) projected trajectory."""
    angles = get_sequence_relative_angles(projected, relative_to=0, angle_mode="cosine_arccos")
    return -smoothing(angles, window=window)


def select_window(rows):
    """The replication's selected config, or the best R ignoring empty utterances when none is eligible."""
    try:
        return select_config(rows), False
    except ValueError:
        thresholded = sorted((r for r in rows if not _no_threshold(r)), key=lambda r: r["param_id"])
        return max(thresholded, key=lambda r: r["R"]), True


def tune_window(subspace_dir, cache_dir, utts, ref_dir, evaluator_dir, grid_csv, window_json, configs=None):
    """Score the RoSeg grid on dev and record the selected config's smoothing window."""
    subspace, layer = load_subspace(subspace_dir)
    stems = Path(utts).read_text().split()
    projected = project_utterances(subspace, cache_dir, stems, layer)
    rows = score_configs(projected, grid() if configs is None else configs, ref_dir, evaluator_dir)
    write_rows(rows, grid_csv)
    selected, fallback = select_window(rows)
    Path(window_json).write_text(json.dumps({"layer": layer, "smooth_window_size": selected["smooth_window_size"],
                                             "fallback_ignoring_empties": fallback, "selected": selected},
                                            indent=1) + "\n")


def graded_frame(stems, subspace, cache_dir, layer, window):
    projected = project_utterances(subspace, cache_dir, stems, layer)
    frames = []
    for stem in stems:
        scores = graded_curve(projected[stem], window)
        frames.append(pd.DataFrame({"stem": stem, "frame": np.arange(len(scores)), "graded_score": scores,
                                    "layer": layer}))
    return pd.concat(frames, ignore_index=True)


def export(subspace_dirs, window_jsons, cache_dir, utts, out):
    """One long parquet (stem, frame, graded_score, layer) over every layer's subspace."""
    stems = Path(utts).read_text().split()
    frames = []
    for subspace_dir, window_json in zip(subspace_dirs, window_jsons):
        subspace, layer = load_subspace(subspace_dir)
        window = json.loads(Path(window_json).read_text())
        if window["layer"] != layer:
            raise ValueError(f"{window_json} is for layer {window['layer']}, {subspace_dir} for layer {layer}")
        frames.append(graded_frame(stems, subspace, cache_dir, layer, window["smooth_window_size"]))
    df = pd.concat(frames, ignore_index=True).astype(COLUMNS)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tun = sub.add_parser("tune-window")
    for name in ("subspace_dir", "cache_dir", "utts", "ref_dir", "evaluator_dir", "grid_csv", "window_json"):
        tun.add_argument(name, type=Path)
    tun.add_argument("--grid", type=json.loads, default=None,
                     help='reduced grid as JSON {"prominences": [...], "thetas_over_pi": [...], "windows": [...]}')
    exp = sub.add_parser("export")
    exp.add_argument("cache_dir", type=Path)
    exp.add_argument("utts", type=Path)
    exp.add_argument("out", type=Path)
    exp.add_argument("--subspaces", type=Path, nargs="+", required=True)
    exp.add_argument("--windows", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    if args.command == "tune-window":
        configs = None
        if args.grid:
            configs = grid(args.grid["prominences"], [t * math.pi for t in args.grid["thetas_over_pi"]],
                           args.grid["windows"])
        tune_window(args.subspace_dir, args.cache_dir, args.utts, args.ref_dir, args.evaluator_dir, args.grid_csv,
                    args.window_json, configs)
    else:
        if len(args.subspaces) != len(args.windows):
            raise SystemExit("--subspaces and --windows must pair up")
        export(args.subspaces, args.windows, args.cache_dir, args.utts, args.out)


if __name__ == "__main__":
    main()
