"""Pre-registered acceptance checks (#7, #11) for the prod full run.

Numbers are percentages as printed in the paper. The comparison notebook
renders these tables; nothing here is asserted in pytest.
"""

import csv
import json
from pathlib import Path

import pandas as pd

METRICS = ["F1", "OS", "token_F1", "R"]
BANDS = {"F1": 2.0, "OS": 3.0, "token_F1": 2.0, "R": 2.0}
R2_L9_PAPER, R2_BAND = 0.983, 0.005
TRAINED = "hubert-base-ls960"

# Paper Table II: (split, row) -> F1, OS, Tok F1, R-val.
PAPER_TABLE_II = {
    ("dev", "gold_syllables"): (85.5, 33.9, 65.9, 71.1),
    ("dev", "dissim"): (43.9, -3.3, 17.0, 52.6),
    ("dev", "ssl_cpca"): (49.1, -12.9, 20.5, 57.6),
    ("dev", "ssl_cpca_no_threshold"): (48.4, -4.8, 19.9, 56.6),
    ("dev", "ssl_leace"): (66.0, -4.6, 42.3, 71.2),
    ("dev", "ssl_leace_no_threshold"): (58.6, 28.9, 31.0, 56.2),
    ("dev", "unt_cpca"): (12.8, -54.0, 1.1, 34.3),
    ("dev", "unt_leace"): (16.5, -32.5, 1.8, 35.0),
    ("test", "dissim"): (43.9, -2.0, 16.9, 52.4),
    ("test", "ssl_cpca"): (48.0, -11.7, 19.7, 56.8),
    ("test", "ssl_leace"): (64.3, -4.9, 39.8, 69.8),
}
# Rows judged numerically; untrained rows are qualitative only.
NUMERIC_ROWS = {k for k in PAPER_TABLE_II if not k[1].startswith("unt_")}


def _pct(scores):
    return {m: 100 * float(scores[m]) for m in METRICS}


def _read_single(path):
    with open(path, newline="") as f:
        return _pct(next(csv.DictReader(f)))


def _roseg_rows(out, est, model):
    table = pd.read_csv(Path(out) / "tables" / "roseg" / f"{est}_{model}.csv")
    return {(r.split, r.variant): {m: 100 * getattr(r, m) for m in METRICS} | {"n_empty": r.n_empty}
            for r in table.itertuples()}


def our_table_ii(out, random_models):
    """Our scores for every Table II row; untrained rows use seed 0 (the first of `random_models`)."""
    rows = {}
    rows[("dev", "gold_syllables")] = _read_single(Path(out) / "scores" / "gold_syllables_dev.csv")
    for split in ("dev", "test"):
        rows[(split, "dissim")] = _read_single(Path(out) / "scores" / f"dissim_{split}.csv")
    for est in ("cpca", "leace"):
        ssl = _roseg_rows(out, est, TRAINED)
        for split in ("dev", "test"):
            rows[(split, f"ssl_{est}")] = ssl[(split, "selected")]
            rows[(split, f"ssl_{est}_authors_config")] = ssl[(split, "authors")]
        rows[("dev", f"ssl_{est}_no_threshold")] = ssl[("dev", "no_threshold")]
        unt = _roseg_rows(out, est, random_models[0])
        rows[("dev", f"unt_{est}")] = unt[("dev", "selected")]
        rows[("test", f"unt_{est}")] = unt[("test", "selected")]
    return rows


def compare_table_ii(ours):
    """Side-by-side rows with per-metric deltas and band pass/fail for the numeric tier."""
    records = []
    for (split, row), paper in PAPER_TABLE_II.items():
        got = ours[(split, row)]
        rec = {"split": split, "row": row, "tier": "numeric" if (split, row) in NUMERIC_ROWS else "qualitative"}
        for m, p in zip(METRICS, paper):
            rec[f"{m}_paper"], rec[f"{m}_ours"] = p, round(got[m], 2)
        if row == "gold_syllables":
            rec["in_band"] = all(round(got[m], 1) == p for m, p in zip(METRICS, paper))
        elif rec["tier"] == "numeric":
            rec["in_band"] = all(abs(got[m] - p) <= BANDS[m] for m, p in zip(METRICS, paper))
        else:
            rec["in_band"] = None
        rec["n_empty"] = got.get("n_empty")
        records.append(rec)
    # Authors' shipped configs on our features: a second required in-band check (#7 drift matrix).
    for est in ("cpca", "leace"):
        for split in ("dev", "test"):
            got, paper = ours[(split, f"ssl_{est}_authors_config")], PAPER_TABLE_II[(split, f"ssl_{est}")]
            rec = {"split": split, "row": f"ssl_{est}_authors_config", "tier": "numeric"}
            for m, p in zip(METRICS, paper):
                rec[f"{m}_paper"], rec[f"{m}_ours"] = p, round(got[m], 2)
            rec["in_band"] = all(abs(got[m] - p) <= BANDS[m] for m, p in zip(METRICS, paper))
            rec["n_empty"] = got.get("n_empty")
            records.append(rec)
    return pd.DataFrame(records)


def drift_diagnosis(comparison):
    """#7's matrix: our tuned row vs the authors' config scored on our features."""
    out = []
    for est in ("cpca", "leace"):
        for split in ("dev", "test"):
            tuned = comparison.query("split == @split and row == @f'ssl_{est}'")["in_band"].item()
            authors = comparison.query("split == @split and row == @f'ssl_{est}_authors_config'")["in_band"].item()
            verdict = ("pass" if tuned and authors else "tuning drift" if authors
                       else "feature or fitting drift" if not tuned else "authors' config out of band only")
            out.append({"subspace": est, "split": split, "tuned_in_band": tuned,
                        "authors_config_in_band": authors, "diagnosis": verdict})
    return pd.DataFrame(out)


def ordering_claims(ours, untrained_by_seed, sec2):
    """Every ordering claim that must hold regardless of the bands.

    `untrained_by_seed` maps a random model name to its {(split, est): scores}.
    Thresholds for "strongly" and "little" are our operationalisation of the
    paper's prose (LEACE OS -4.6 -> +28.9, CPCA -12.9 -> -4.8).
    """
    claims = []

    def add(claim, holds, detail):
        claims.append({"claim": claim, "holds": bool(holds), "detail": detail})

    for split in ("dev", "test"):
        leace, cpca = ours[(split, "ssl_leace")], ours[(split, "ssl_cpca")]
        for m in ("F1", "token_F1", "R"):
            add(f"LEACE > CPCA on {m} ({split})", leace[m] > cpca[m], f"{leace[m]:.1f} vs {cpca[m]:.1f}")
        dissim = ours[(split, "dissim")]
        for m in ("F1", "R"):
            add(f"Dissim. < both SSL RoSeg on {m} ({split})", dissim[m] < min(leace[m], cpca[m]),
                f"{dissim[m]:.1f} vs {cpca[m]:.1f}/{leace[m]:.1f}")
        add(f"Dissim. < RoSeg-LEACE on token_F1 ({split})", dissim["token_F1"] < leace["token_F1"],
            f"{dissim['token_F1']:.1f} vs {leace['token_F1']:.1f}")

    d_leace = ours[("dev", "ssl_leace_no_threshold")]["OS"] - ours[("dev", "ssl_leace")]["OS"]
    d_cpca = ours[("dev", "ssl_cpca_no_threshold")]["OS"] - ours[("dev", "ssl_cpca")]["OS"]
    add("Removing abs threshold: LEACE OS rises strongly (> 15 pt, ends > 10)",
        d_leace > 15 and ours[("dev", "ssl_leace_no_threshold")]["OS"] > 10, f"ΔOS {d_leace:+.1f}")
    add("Removing abs threshold: CPCA OS changes little (|Δ| < half LEACE's Δ)", abs(d_cpca) < d_leace / 2,
        f"ΔOS {d_cpca:+.1f}")

    for model, rows in untrained_by_seed.items():
        for (split, est), unt in rows.items():
            ssl = ours[(split, f"ssl_{est}")]
            gaps = [ssl[m] - unt[m] for m in ("F1", "token_F1", "R")]
            add(f"SSL ≫ untrained, {est} {split} ({model})",
                min(gaps) >= 10 and abs(ssl["OS"]) < abs(unt["OS"]),
                f"min gap {min(gaps):.1f}; |OS| {abs(ssl['OS']):.1f} vs {abs(unt['OS']):.1f}")

    trained = sec2[sec2["model_name"] == TRAINED]
    best = trained["r2_train"].max()
    r2_l9 = trained.loc[trained["layer_id"] == 9, "r2_train"].item()
    add("§II: L9 at or near the best-R² layer (within 0.01)", best - r2_l9 <= 0.01,
        f"L9 {r2_l9:.4f}, best {best:.4f} (L{int(trained.loc[trained['r2_train'].idxmax(), 'layer_id'])})")
    return pd.DataFrame(claims)


def r2_l9_check(sec2):
    r2 = sec2.loc[(sec2["model_name"] == TRAINED) & (sec2["layer_id"] == 9), "r2_train"].item()
    return {"r2_l9": r2, "paper": R2_L9_PAPER, "in_band": abs(r2 - R2_L9_PAPER) <= R2_BAND}


def qualitative_sec2(sec2, random_models):
    """Untrained: R² high but below trained, over-completes; log Mel: no circle (low R²)."""
    best = sec2.loc[sec2.groupby("model_name")["r2_train"].idxmax()].set_index("model_name")
    rows = []
    trained = best.loc[TRAINED]
    rows.append({"pattern": "trained under-completes at its best layer",
                 "holds": trained["endpoint_completion"] < 1, "detail": f"{trained['endpoint_completion']:.3f}"})
    for model in random_models:
        unt = best.loc[model]
        rows.append({"pattern": f"{model}: best R² high but below trained",
                     "holds": 0.8 < unt["r2_train"] < trained["r2_train"],
                     "detail": f"{unt['r2_train']:.3f} vs {trained['r2_train']:.3f}"})
        rows.append({"pattern": f"{model}: over-completes at its best layer",
                     "holds": unt["endpoint_completion"] > 1, "detail": f"{unt['endpoint_completion']:.3f}"})
    if "logmel" in best.index:
        mel = best.loc["logmel"]
        rows.append({"pattern": "log Mel: R² far below trained (≥ 0.1 lower)",
                     "holds": trained["r2_train"] - mel["r2_train"] >= 0.1, "detail": f"{mel['r2_train']:.3f}"})
    return pd.DataFrame(rows)


def fig6_trend(fig6_dir, roseg_method="roseg_leace_selected", min_words=20):
    """Trend-only tier: from 1 to n syllables, RoSeg's OS rises less and its R falls less than gold's."""
    gold = pd.read_csv(Path(fig6_dir) / "gold_syllables_dev.csv").set_index("group")
    roseg = pd.read_csv(Path(fig6_dir) / f"{roseg_method}_dev.csv").set_index("group")
    rows = []
    for g in [g for g in ("2", "3", "4", "5", "6") if g in gold.index and gold.loc[g, "n_words"] >= min_words]:
        d_os_gold, d_os_roseg = (100 * (t.loc[g, "OS"] - t.loc["1", "OS"]) for t in (gold, roseg))
        d_r_gold, d_r_roseg = (100 * (t.loc[g, "R"] - t.loc["1", "R"]) for t in (gold, roseg))
        rows.append({"group": g, "ΔOS gold": d_os_gold, "ΔOS RoSeg": d_os_roseg, "ΔR gold": d_r_gold,
                     "ΔR RoSeg": d_r_roseg, "holds": d_os_roseg < d_os_gold and d_r_roseg > d_r_gold})
    return pd.DataFrame(rows)


def empty_counts(out):
    """Every scored predictions file's empty-utterance count (nonzero is a deviation to flag)."""
    rows = [{"file": p.name, **json.loads(p.read_text())} for p in sorted((Path(out) / "scores").glob("*.empty.json"))]
    return pd.DataFrame(rows)
