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
        table = pd.read_csv(out / "tables" / "roseg" / f"{est}_{model}.csv").query("variant == 'selected'")
        for r in table.itertuples():
            rows[(r.split, est)] = {m: 100 * getattr(r, m) for m in acceptance.METRICS}
    untrained[model] = rows
sec2 = pd.read_csv(out / "tables" / "sec2_metrics.csv")
acceptance.ordering_claims(ours, untrained, sec2)

# %% [markdown]
# ## §II
#
# HuBERT L9 R² is in the numeric tier (paper 0.983 ± 0.005). The rest is qualitative.

# %%
acceptance.r2_l9_check(sec2)

# %%
acceptance.qualitative_sec2(sec2, random_models)

# %%
sec2.pivot_table(index="layer_id", columns="model_name", values=["r2_train", "r2_test", "endpoint_completion"])

# %%
display(Image(str(out / "figures" / "fig3.png")))

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

# %%
widened = selections[selections["variant"] == "widened"]
widened if len(widened) else "no selection sat on a grid edge: widened-grid diagnostic not run"

# %%
dissim_retune = out / "tables" / "dissim_retune.csv"
pd.read_csv(dissim_retune) if dissim_retune.exists() else "Dissim. re-tune diagnostic not run"
