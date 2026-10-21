"""Fig 6 figure: per-syllable-count F1 / OS / Token F1 / R-value per method, word counts as bars."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PANELS = [("F1", "F1 (↑)"), ("OS", "OS (→0)"), ("token_F1", "Tok F1 (↑)"), ("R", "R-val (↑)")]
LABELS = {
    "gold_syllables": "Gold syllables",
    "dissim": "Dissim.",
    "roseg_cpca_selected": "RoSeg-CPCA",
    "roseg_cpca_no_threshold": "RoSeg-CPCA −abs",
    "roseg_leace_selected": "RoSeg-LEACE",
    "roseg_leace_no_threshold": "RoSeg-LEACE −abs",
}


def plot(score_csvs, out_path):
    table = pd.concat(pd.read_csv(p, dtype={"group": str}) for p in score_csvs)
    groups = ["all", "1", "2", "3", "4", "5", "6"]
    groups = [g for g in groups if g in set(table["group"])]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    counts = table.drop_duplicates("group").set_index("group").loc[groups, "n_words"]
    for ax, (metric, title) in zip(axes, PANELS):
        bars = ax.twinx()
        bars.bar(range(len(groups)), counts.values, color="0.85", zorder=0)
        bars.set_yscale("log")
        ax.set_zorder(bars.get_zorder() + 1)
        ax.patch.set_visible(False)
        for method, rows in table.groupby("method", sort=False):
            rows = rows.set_index("group").loc[groups]
            ax.plot(range(len(groups)), 100 * rows[metric], "o-", label=LABELS.get(method, method))
        ax.set_xticks(range(len(groups)), groups)
        ax.set_xlabel("#syllable")
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("scores", type=Path, nargs="+")
    args = parser.parse_args()
    plot(args.scores, args.out)


if __name__ == "__main__":
    main()
