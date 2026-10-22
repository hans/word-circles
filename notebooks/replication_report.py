# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Replication report: HuBERT-base word circles
#
# Our numbers next to Naowarat, Tang & Goldwater (Table II, §II, Fig 6), judged by the
# pre-registered tiers in spec #16: numeric bands **and** ordering claims, qualitative
# patterns, and the Fig 6 trend. Verdicts count only on the prod full run; a smoke
# run renders this notebook to check plumbing.

# %% tags=["parameters"]
out_dir = "outputs"
random_models = ["hubert-base-random-s0"]
is_full_run = True

# %%
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from word_circles import acceptance

out = Path(out_dir)
pd.set_option("display.width", 200, "display.max_columns", 40)
if not is_full_run:
    print("SMOKE RUN: subset of utterances; no verdict below is an acceptance judgment.")

# %% [markdown]
# ## Documented deviations and assumptions
#
# - **Untrained HuBERT:** the authors' init and seed are unpublished. We use HF default
#   init of the full `HubertModel` from the ls960 config, seeds 0-2 (seed 0 primary), each
#   frozen to a safetensors file. These rows are judged qualitatively only.
# - **Log Mel:** the recipe is ours (80 mel, power 2, `log(x + 1e-6)`, no normalisation;
#   unlike the sister repo, a log is applied). #10 assumed `n_fft=512, center=False` already
#   gives HuBERT's frame count. It does not: it gives `(N-512)//320+1`. So the input is
#   zero-padded by 56 samples on each side, which puts window k exactly on HuBERT frame k.
# - **Fig 6 scoring** was locked after the fact (#15). None of the 4 pre-registered candidates
#   fit. The locked rule is the evaluator's formulas without its last-boundary drop, with
#   `all` = groups 1-6 pooled.
# - **Alignments:** subspaces are fit on the authors' Kaldi `.ali` files and scored against the
#   s-malan TextGrids (sd of the offset ≈ 30 ms), as the authors did. The Kaldi-reference
#   rescoring below measures the effect.
# - **Evaluator:** the authors ran an edited copy (`--alignment_type=word`). We run the
#   unmodified evaluator at `c9074a4` with `--alignment_type=words`; scoring is unchanged
#   across those versions. Its printed metrics are parsed by our wrapper, not by WTT's gather
#   script.
# - **§II R² = 0.983 is the best layer (L1), not L9.** Spec #16 attached the paper's 0.983 to L9,
#   the Fig 1a layer. Three things say otherwise: the paper's own Fig 1a polar panel shows the
#   same two short-duration groups unwrapping the wrong way as ours does, which caps L9 near
#   0.87 under the authors' metric; the paper's Fig 3 marks HuBERT's best R² at L1; and our
#   per-layer R² matches the paper's Fig 3 trace layer by layer (L1 0.983, L4-6 ≈ 0.76, L9 ≈
#   0.87). The check now compares the best layer. The ordering claim "L9 at or near the
#   best-R² layer" came from the same misreading and is retracted.
# - **Empty predictions** get a placeholder boundary at 0.02 s. Every count is tabulated below,
#   and selection only admits configs with none.
# - **Smoke run** uses 100 utterances per split, not ~300 (#5). 300 measured ~4 GB and more
#   than 10 min of extraction alone.
# - **CUDA wheels:** prod extracts on Titan V (Volta) with torch's cu126 build, because the
#   cu13x builds have no Volta kernels. Volta has no TF32, so the TF32-off flags have no effect there.

# %% [markdown]
# ## Table II

# %%
ours = acceptance.our_table_ii(out, random_models)
comparison = acceptance.compare_table_ii(ours)
comparison

# %% [markdown]
# Drift matrix: our tuned config vs the authors' shipped config, both scored on our features.

# %%
acceptance.drift_diagnosis(comparison)

# %% [markdown]
# Selected configs and their empty-utterance counts. Selection is restricted to configs with
# none, so a nonzero count can only appear on the θ = π partner or a diagnostic row.

# %%
selections = []
for model in ["hubert-base-ls960", *random_models]:
    for est in ("cpca", "leace"):
        table = pd.read_csv(out / "tables" / "roseg" / f"{est}_{model}.csv")
        selections.append(table.assign(model=model, subspace=est))
selections = pd.concat(selections)
selections[["model", "subspace", "variant", "split", "prominence", "threshold_max_angle",
            "smooth_window_size", "F1", "OS", "token_F1", "R", "n_empty"]]

# %% [markdown]
# ## Ordering claims (all required; a failure is a headline finding)

# %%
untrained = {}
for model in random_models:
    rows = {}
    for est in ("cpca", "leace"):
        table = pd.read_csv(out / "tables" / "roseg" / f"{est}_{model}.csv").query("variant == 'selected' and split in ['dev', 'test']")
        for r in table.itertuples():
            rows[(r.split, est)] = {m: 100 * getattr(r, m) for m in acceptance.METRICS}
    untrained[model] = rows
sec2 = pd.read_csv(out / "tables" / "sec2_metrics.csv")
acceptance.ordering_claims(ours, untrained)

# %% [markdown]
# ## §II
#
# HuBERT best-layer R² is in the numeric tier (paper 0.983 ± 0.005). The rest is qualitative.

# %%
acceptance.r2_best_check(sec2)

# %%
acceptance.qualitative_sec2(sec2, random_models)

# %%
sec2.pivot_table(index="layer_id", columns="model_name", values=["r2_train", "r2_test", "endpoint_completion"])

# %%
display(Image(str(out / "figures" / "fig3.png")))

# %% [markdown]
# Unwrap diagnostic (not the authors' metric): R² with the duration groups whose unwrapped
# angle steps against the rotation dropped. These are 5-7 frame groups, where a full turn
# needs near-π steps. They account for the jagged trained-HuBERT layer curve.

# %%
from word_circles.sec2 import unwrap_diagnostic

unwrap = {m: unwrap_diagnostic(out / "sec2" / "subspaces" / m) for m in [acceptance.TRAINED, *random_models]}
unwrap[acceptance.TRAINED]

# %% [markdown]
# Like-for-like R² across models: the authors' metric on duration groups of at least 8 frames
# (160 ms), where unwrapping is unambiguous. Under the full metric, untrained HuBERT looks close to
# trained at L2-12 only because trained HuBERT's short groups flip.

# %%
pd.DataFrame({m: t.set_index("layer_id")["r2_long"] for m, t in unwrap.items()}).round(3)

# %% [markdown]
# Untrained seed range (side table, no acceptance criteria except the ordering claim above).

# %%
if len(random_models) > 1:
    unt = sec2[sec2["model_name"].isin(random_models)]
    display(unt.groupby("layer_id")[["r2_train", "endpoint_completion"]].agg(["min", "max"]))
    display(pd.concat({m: pd.DataFrame(r).T for m, r in untrained.items()}))

# %% [markdown]
# ## Fig 6 (trend only)

# %%
display(Image(str(out / "figures" / "fig6.png")))
acceptance.fig6_trend(out / "scores" / "fig6")

# %% [markdown]
# ## Diagnostics
#
# Kaldi-reference rescoring on test (never used for selection), and every empty-prediction count.

# %%
kaldi = {"dissim": acceptance._read_single(out / "scores" / "kaldi" / "dissim_test.csv")}
for est in ("cpca", "leace"):
    t = pd.read_csv(out / "tables" / "roseg" / f"{est}_hubert-base-ls960.csv")
    for r in t[t["split"].isin(["test", "test_kaldi"])].itertuples():
        kaldi[f"ssl_{est}_{r.variant}_{r.split}"] = {m: 100 * getattr(r, m) for m in acceptance.METRICS}
pd.DataFrame(kaldi).T.round(2)

# %%
acceptance.empty_counts(out)

# %% [markdown]
# Widened-grid diagnostic: runs on the full grid only, for selections on a grid edge.

# %%
import json

edges = []
for model in ["hubert-base-ls960", *random_models]:
    for est in ("cpca", "leace"):
        sel = json.loads((out / "roseg" / "tune" / est / model / "selection.json").read_text())
        edges.append({"model": model, "subspace": est, "on_grid_edge": sel["on_grid_edge"],
                      "widened_run": "widened" in sel})
display(pd.DataFrame(edges))
selections[selections["variant"] == "widened"]

# %%
dissim_retune = out / "tables" / "dissim_retune.csv"
pd.read_csv(dissim_retune) if dissim_retune.exists() else "Dissim. re-tune diagnostic not run"
