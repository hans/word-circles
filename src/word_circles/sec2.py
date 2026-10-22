"""§II figures from the authors' metric table (Fig 2 panels, Fig 3 layer curves) and the R² unwrap diagnostic."""

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concept_analysis.projector import PCAProjection
from concept_analysis.trajectory.geometry import get_mean_trajectory_angles
from concept_analysis.trajectory.metrics import get_elapsed_angle_regression_features
from sklearn.linear_model import LinearRegression

TRAINED = "hubert-base-ls960"
RANDOM_PREFIX = "hubert-base-random-s"
PRIMARY_RANDOM = f"{RANDOM_PREFIX}0"
LOGMEL = "logmel"
FIG1A_LAYER = 9


def unwrap_diagnostic(subspaces_dir, min_frames=8):
    """Per-layer in-sample R² with and without the duration groups whose unwrapped angle steps against the rotation.

    Short groups (5-7 frames) need near-π steps per frame for a full turn, so np.unwrap can take the short way
    round. The authors' metric keeps them; this side table shows how much of the R² drop they carry.
    `r2_long` is the authors' metric on groups of at least `min_frames` frames only, where no unwrap rule is
    ambiguous: the like-for-like comparison across models. `flipped_groups` is only meaningful for trained
    models; untrained trajectories genuinely backtrack.
    """
    rows = []
    layer_dirs = [p for p in Path(subspaces_dir).iterdir() if p.is_dir() and p.name.isdigit()]
    for layer_dir in sorted(layer_dirs, key=lambda p: int(p.name)):
        subspace = PCAProjection.load(layer_dir / "temporal_subspace.pkl")
        with np.load(layer_dir / "mean_trajectories.npz") as z:
            groups = {int(k): z[k].astype(np.float64) for k in z.keys()}
        angles = get_mean_trajectory_angles(subspace, groups, "centroid")
        r2, slope = _polar_fit(angles)
        flipped = [n for n, theta in angles.items() if (np.sign(np.diff(np.unwrap(theta))) != np.sign(slope)).any()]
        kept = {n: theta for n, theta in angles.items() if n not in flipped}
        long = {n: theta for n, theta in angles.items() if n >= min_frames}
        rows.append({"layer_id": int(layer_dir.name), "r2_train": r2, "r2_without_flipped": _polar_fit(kept)[0],
                     "r2_long": _polar_fit(long)[0], "flipped_groups": flipped})
    return pd.DataFrame(rows)


def _polar_fit(angles):
    elapsed, angle = get_elapsed_angle_regression_features(angles, flatten=True)
    model = LinearRegression(fit_intercept=False).fit(angle, elapsed)
    return model.score(angle, elapsed), model.coef_[0, 0]


def best_layers(table):
    """Layer with the highest in-sample (dev-clean) R² per model: the Fig 2 panel rule."""
    rows = table.loc[table.groupby("model_name")["r2_train"].idxmax()]
    return dict(zip(rows["model_name"], rows["layer_id"].astype(int)))


def fig2(table_path, trajectories_dir, out_dir):
    """Collect each model's trajectory plots at its best-R² layer (log Mel has only layer 0), plus Fig 1a."""
    table = pd.read_csv(table_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = best_layers(table)
    for model, layer in chosen.items():
        model_dir = Path(trajectories_dir) / model
        pattern = "*" if model == LOGMEL else f"*_layer{layer:02d}*"
        plots = sorted(p for p in model_dir.rglob(pattern) if p.is_file() and not p.name.startswith("."))
        if not plots:
            raise FileNotFoundError(f"no trajectory plots matching {pattern} under {model_dir}")
        for plot in plots:
            shutil.copy(plot, out_dir / f"{model}_{plot.parent.name}_{plot.name}")
    # Fig 1a: the trained model's L9 circle, whatever its best layer.
    for plot in sorted((Path(trajectories_dir) / TRAINED).rglob(f"*_layer{FIG1A_LAYER:02d}*")):
        shutil.copy(plot, out_dir / f"fig1a_{plot.parent.name}_{plot.name}")
    pd.Series(chosen, name="layer_id").rename_axis("model_name").to_csv(out_dir / "layers.csv")


def fig3(table_path, out_path):
    """R² and completion% per layer: trained, untrained seed 0 (seed range shaded), log Mel marker."""
    table = pd.read_csv(table_path)
    table["completion"] = 100 * table["endpoint_completion"]
    random = table[table["model_name"].str.startswith(RANDOM_PREFIX)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, column, label in ((axes[0], "r2_train", "R²"), (axes[1], "completion", "completion %")):
        trained = table[table["model_name"] == TRAINED].sort_values("layer_id")
        ax.plot(trained["layer_id"], trained[column], "o-", label="HuBERT")
        primary = random[random["model_name"] == PRIMARY_RANDOM].sort_values("layer_id")
        ax.plot(primary["layer_id"], primary[column], "^:", label="Unt. HuBERT (seed 0)")
        if random["model_name"].nunique() > 1:
            spread = random.groupby("layer_id")[column].agg(["min", "max"])
            ax.fill_between(spread.index, spread["min"], spread["max"], alpha=0.2, color=ax.lines[-1].get_color(),
                            label="Unt. seed range")
        logmel = table[table["model_name"] == LOGMEL]
        if not logmel.empty:
            ax.axhline(logmel[column].iloc[0], color="gray", ls="--", lw=1, label="log Mel")
        ax.set_xlabel("layer")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f2 = sub.add_parser("fig2")
    f2.add_argument("table", type=Path)
    f2.add_argument("trajectories_dir", type=Path)
    f2.add_argument("out_dir", type=Path)
    f3 = sub.add_parser("fig3")
    f3.add_argument("table", type=Path)
    f3.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "fig2":
        fig2(args.table, args.trajectories_dir, args.out_dir)
    else:
        fig3(args.table, args.out)


if __name__ == "__main__":
    main()
