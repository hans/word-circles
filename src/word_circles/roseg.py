"""RoSeg: word boundaries from troughs of the angle in a 2-D temporal subspace.

Prediction uses the authors' `detect_boundaries` on L9 projected through a
CPCA or LEACE subspace fit on the 10 h train sample. RoSeg never emits a
final boundary. Configs come from the authors' 300-point grid, tuned for
dev-clean R-value.
"""

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from concept_analysis.annotations import get_word_alignment, load_data
from concept_analysis.features import NpzFeatureReader
from concept_analysis.projector import LEACEProjection, PCAProjection
from concept_analysis.trajectory import detect_boundaries, get_mean_trajectories, group_and_preprocess_feature

from word_circles.evaluation import score_boundaries, write_predictions
from concept_analysis.subspaces.leace_np import LeaceFitterCCA

LEACE_CHUNK_FRAMES = 50_000


def grid(prominences=None, thetas=None, windows=(1, 2, 3)):
    """The authors' tuning grid in their loop order: prominence, then theta_max, then window.

    Defaults reproduce `tune_peak_detection.py`: prominence in
    linspace(0, pi/4, 10) rounded to 3 dp, theta_max in linspace(pi/4, pi, 10).
    """
    prominences = np.linspace(0, np.pi / 4, 10) if prominences is None else prominences
    thetas = np.linspace(np.pi / 4, np.pi, 10) if thetas is None else thetas
    configs = []
    for prominence in prominences:
        for theta in thetas:
            for window in windows:
                configs.append({"param_id": len(configs), "prominence": round(float(prominence), 3),
                                "threshold_max_angle": float(theta), "smooth_window_size": int(window)})
    return configs


def _no_threshold(config):
    return math.isclose(config["threshold_max_angle"], math.pi)


def _best(rows):
    eligible = [r for r in rows if r["n_empty"] == 0]
    if not eligible:
        raise ValueError("no eligible config: every config leaves some utterance without boundaries")
    # max() keeps the first of equal maxima; rows are compared in loop (param_id) order.
    return max(sorted(eligible, key=lambda r: r["param_id"]), key=lambda r: r["R"])


def select_config(rows):
    """Best dev R-value with theta_max < pi among configs with no empty utterance; ties to loop order."""
    return _best([r for r in rows if not _no_threshold(r)])


def paired_no_threshold(rows, selected):
    """The "-abs threshold" ablation: the selected config with theta_max = pi, not re-tuned."""
    for r in rows:
        if (_no_threshold(r) and r["prominence"] == selected["prominence"]
                and r["smooth_window_size"] == selected["smooth_window_size"]):
            return r
    raise ValueError(f"no theta_max = pi partner for {selected}")


def select_no_threshold(rows):
    """Diagnostic: the ablation re-tuned over the theta_max = pi slice."""
    return _best([r for r in rows if _no_threshold(r)])


def on_grid_edge(selected, configs):
    """Whether a selection sits where a wider grid could beat it: extreme prominence or lowest theta_max."""
    prominences = [c["prominence"] for c in configs]
    thetas = [c["threshold_max_angle"] for c in configs]
    return (selected["prominence"] in (min(prominences), max(prominences))
            or math.isclose(selected["threshold_max_angle"], min(thetas)))


def widened_grid():
    """Edge diagnostic: theta_max down to pi/8 and prominence up to pi/2, same step sizes."""
    thetas = np.concatenate([np.linspace(np.pi / 8, np.pi / 4, 4)[:-1], np.linspace(np.pi / 4, np.pi, 10)])
    prominences = np.concatenate([np.linspace(0, np.pi / 4, 10), np.linspace(np.pi / 4, np.pi / 2, 10)[1:]])
    return grid(prominences, thetas)


def fit_leace_chunked(X_group, chunk_frames=LEACE_CHUNK_FRAMES):
    """The authors' LEACE temporal subspace, accumulated in chunks instead of one concatenated fit.

    `X_group` maps word duration L (frames) to an (n_tokens, L, H) array, as
    built by the authors' `group_and_preprocess_feature`. The label is the
    one-hot within-word frame index, pooled over durations. The fitter's
    Welford `update()` gives the one-shot statistics up to float rounding
    without materialising the full frame matrix.
    """
    durations = sorted(X_group)
    classes = np.arange(max(durations))
    H = X_group[durations[0]].shape[-1]
    projection = LEACEProjection(name="LEACE_temporal")
    fitter = LeaceFitterCCA(H, len(classes), method=projection._fit_method(), svd_tol=1e-7,
                            shrinkage=projection.shrinkage, dtype=X_group[durations[0]].dtype)
    eye = np.eye(len(classes), dtype=fitter.mean_x.dtype)
    for L in durations:
        frames = X_group[L].reshape(-1, H)
        labels = np.tile(np.arange(L), len(X_group[L]))
        for start in range(0, len(frames), chunk_frames):
            fitter.update(frames[start:start + chunk_frames], eye[labels[start:start + chunk_frames]])
    return projection._set_fitted_eraser(fitter.eraser, classes)


# Configs the authors shipped for HuBERT L9 (their dev selections), trained and untrained.
AUTHORS_CONFIGS = {
    "trained": {
        "leace": {"prominence": 0.175, "smooth_window_size": 3, "threshold_max_angle": math.pi / 4},
        "cpca": {"prominence": 0.349, "smooth_window_size": 2, "threshold_max_angle": math.pi / 3},
    },
    "untrained": {
        "leace": {"prominence": 0.698, "smooth_window_size": 3, "threshold_max_angle": math.pi / 4},
        "cpca": {"prominence": 0.175, "smooth_window_size": 2, "threshold_max_angle": math.pi / 4},
    },
}
ESTIMATORS = {"cpca": ("Z_CPCA", PCAProjection), "leace": ("LEACE", LEACEProjection)}
FRAME_RATE = 50.0


def fit_leace_subspace(cache_dir, ali_path, layer, out_dir, *, min_dur, max_dur, low_freq_threshold):
    """`fit_temporal_subspace.py --estimator LEACE`, with the LEACE fit done in chunks.

    Writes the authors' artifacts (temporal_subspace.pkl, mean_trajectories.npz,
    metadata.json) so their tools read the result unchanged.
    """
    store = NpzFeatureReader(cache_dir)
    alignment = get_word_alignment(ali_path, drop_silence=True, feature_frame_rate=FRAME_RATE)
    ids = alignment["utt_id"].drop_duplicates().tolist()
    X, y = load_data(id2features=dict(zip(ids, store.read_layer(ids, layer))), alignment=alignment,
                     label_types=["token_pos"], njobs=1, preserve_sequence=True)
    X_group, _ = group_and_preprocess_feature(X, y, min_dur=min_dur, max_dur=max_dur,
                                              low_freq_threshold=low_freq_threshold, label_type="token_pos",
                                              feature_frame_rate=FRAME_RATE)
    del X, y
    subspace = fit_leace_chunked(X_group)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subspace.save(out_dir / "temporal_subspace.pkl")
    templates = get_mean_trajectories(X_group)
    np.savez_compressed(out_dir / "mean_trajectories.npz", **{str(k): v for k, v in templates.items()})
    metadata = {"estimator": "LEACE", "layer_id": layer, "annotation_unit": "word_token_pos",
                "feature_frame_rate": FRAME_RATE, "feature_frame_rate_mode": "feature_frame_rate",
                "min_dur": min_dur, "max_dur": max_dur, "low_freq_threshold": low_freq_threshold,
                "drop_silence": True, "leace_fit": f"chunked update(), {LEACE_CHUNK_FRAMES} frames"}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_subspace(subspace_dir):
    metadata = json.loads((Path(subspace_dir) / "metadata.json").read_text())
    cls = {name: c for name, c in ESTIMATORS.values()}[metadata["estimator"]]
    return cls.load(Path(subspace_dir) / "temporal_subspace.pkl"), metadata["layer_id"]


def project_utterances(subspace, cache_dir, stems, layer):
    """Each utterance's whole L9 sequence (silence included) in the 2-D subspace."""
    features = NpzFeatureReader(cache_dir).read_layer(list(stems), layer)
    return {stem: subspace.transform(x, n_components=2) for stem, x in zip(stems, features)}


def predict(projected, config):
    """Boundary times (s) per utterance; no final boundary is added."""
    return {
        stem: [float(i) / FRAME_RATE for i in detect_boundaries(
            z, prominence=config["prominence"], smooth_window_size=config["smooth_window_size"],
            threshold_max_angle=config["threshold_max_angle"])]
        for stem, z in projected.items()
    }


def score_configs(projected, configs, ref_dir, evaluator_dir):
    rows = []
    with tempfile.TemporaryDirectory() as work:
        for config in configs:
            result = score_boundaries(predict(projected, config), ref_dir, evaluator_dir, Path(work) / "pred")
            rows.append({**config, **result.scores, "n_empty": len(result.empty)})
    return rows


def write_rows(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    numeric = {"param_id": int, "smooth_window_size": int, "n_empty": int}
    with open(path, newline="") as f:
        return [{k: numeric.get(k, float)(v) if k != "variant" else v for k, v in r.items()}
                for r in csv.DictReader(f)]


def tune(subspace_dir, cache_dir, utts, ref_dir, evaluator_dir, grid_csv, selection_json, *, widen=True,
         configs=None):
    """Score the grid on dev, select, and run the widened-grid diagnostic when the pick is on an edge."""
    subspace, layer = load_subspace(subspace_dir)
    stems = Path(utts).read_text().split()
    projected = project_utterances(subspace, cache_dir, stems, layer)
    configs = grid() if configs is None else configs
    rows = score_configs(projected, configs, ref_dir, evaluator_dir)
    write_rows(rows, grid_csv)

    selected = select_config(rows)
    by_r = sorted(rows, key=lambda r: (-r["R"], r["param_id"]))
    selection = {
        "selected": selected,
        "no_threshold": paired_no_threshold(rows, selected),
        "no_threshold_retuned": select_no_threshold(rows),
        "top5": [r for r in by_r if not _no_threshold(r)][:5],
        "best_ignoring_empties": by_r[0],
        "on_grid_edge": on_grid_edge(selected, configs),
    }
    if widen and selection["on_grid_edge"]:
        wide_rows = score_configs(projected, widened_grid(), ref_dir, evaluator_dir)
        write_rows(wide_rows, Path(grid_csv).with_suffix(".widened.csv"))
        selection["widened"] = select_config(wide_rows)
    Path(selection_json).write_text(json.dumps(selection, indent=1) + "\n")


def find_config(configs, target):
    for c in configs:
        if (c["prominence"] == target["prominence"] and c["smooth_window_size"] == target["smooth_window_size"]
                and math.isclose(c["threshold_max_angle"], target["threshold_max_angle"])):
            return c
    raise ValueError(f"{target} is not on the grid")


def report(subspace_dir, selection_json, estimator, caches, utts, refs, evaluator_dir, out_csv,
           authors_check=None, kaldi_refs=None):
    """Score every reported variant (selection, ablations, diagnostics) on dev and test.

    `authors_check` ("trained" or "untrained") adds the authors' shipped config for that model kind.

    With `kaldi_refs`, the selected config and its theta = pi partner are also
    scored on test against Kaldi-derived references (split `test_kaldi`).
    """
    selection = json.loads(Path(selection_json).read_text())
    variants = {k: selection[k] for k in ("selected", "no_threshold", "no_threshold_retuned", "widened")
                if k in selection}
    if authors_check:
        variants["authors"] = find_config(grid(), AUTHORS_CONFIGS[authors_check][estimator])
    subspace, layer = load_subspace(subspace_dir)
    rows = []
    for split in ("dev", "test"):
        stems = Path(utts[split]).read_text().split()
        projected = project_utterances(subspace, caches[split], stems, layer)
        with tempfile.TemporaryDirectory() as work:
            for variant, config in variants.items():
                config = {k: config[k] for k in ("param_id", "prominence", "threshold_max_angle", "smooth_window_size")}
                predictions = predict(projected, config)
                result = score_boundaries(predictions, refs[split], evaluator_dir, Path(work) / "pred")
                rows.append({"variant": variant, "split": split, **config, **result.scores,
                             "n_empty": len(result.empty)})
                if split == "test" and kaldi_refs and variant in ("selected", "no_threshold"):
                    result = score_boundaries(predictions, kaldi_refs, evaluator_dir, Path(work) / "pred")
                    rows.append({"variant": variant, "split": "test_kaldi", **config, **result.scores,
                                 "n_empty": len(result.empty)})
    write_rows(rows, out_csv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-leace")
    fit.add_argument("cache_dir", type=Path)
    fit.add_argument("ali", type=Path)
    fit.add_argument("out_dir", type=Path)
    fit.add_argument("--layer", type=int, default=9)
    fit.add_argument("--min-dur", type=float, required=True)
    fit.add_argument("--max-dur", type=float, required=True)
    fit.add_argument("--low-freq-threshold", type=int, required=True)
    tun = sub.add_parser("tune")
    for name in ("subspace_dir", "cache_dir", "utts", "ref_dir", "evaluator_dir", "grid_csv", "selection_json"):
        tun.add_argument(name, type=Path)
    tun.add_argument("--grid", type=json.loads, default=None,
                     help='reduced grid as JSON {"prominences": [...], "thetas_over_pi": [...], "windows": [...]}')
    rep = sub.add_parser("report")
    for name in ("subspace_dir", "selection_json", "dev_cache", "test_cache", "dev_utts", "test_utts",
                 "dev_refs", "test_refs", "evaluator_dir", "out_csv"):
        rep.add_argument(name, type=Path)
    rep.add_argument("--estimator", choices=ESTIMATORS, required=True)
    rep.add_argument("--authors-check", choices=AUTHORS_CONFIGS)
    rep.add_argument("--kaldi-refs", type=Path, help="Kaldi-derived test references for the diagnostic")
    pre = sub.add_parser("predict", help="write one selected variant's predictions")
    pre.add_argument("subspace_dir", type=Path)
    pre.add_argument("selection_json", type=Path)
    pre.add_argument("variant")
    pre.add_argument("cache_dir", type=Path)
    pre.add_argument("utts", type=Path)
    pre.add_argument("out", type=Path)
    args = parser.parse_args()

    if args.command == "fit-leace":
        fit_leace_subspace(args.cache_dir, args.ali, args.layer, args.out_dir, min_dur=args.min_dur,
                           max_dur=args.max_dur, low_freq_threshold=args.low_freq_threshold)
    elif args.command == "tune":
        configs = None
        if args.grid:
            configs = grid(args.grid["prominences"], [t * math.pi for t in args.grid["thetas_over_pi"]],
                           args.grid["windows"])
        tune(args.subspace_dir, args.cache_dir, args.utts, args.ref_dir, args.evaluator_dir, args.grid_csv,
             args.selection_json, configs=configs, widen=args.grid is None)
    elif args.command == "predict":
        subspace, layer = load_subspace(args.subspace_dir)
        config = json.loads(args.selection_json.read_text())[args.variant]
        stems = args.utts.read_text().split()
        write_predictions(predict(project_utterances(subspace, args.cache_dir, stems, layer), config), args.out)
    else:
        report(args.subspace_dir, args.selection_json, args.estimator,
               {"dev": args.dev_cache, "test": args.test_cache}, {"dev": args.dev_utts, "test": args.test_utts},
               {"dev": args.dev_refs, "test": args.test_refs}, args.evaluator_dir, args.out_csv,
               args.authors_check, args.kaldi_refs)


if __name__ == "__main__":
    main()
