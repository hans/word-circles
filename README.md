# word-circles

Replication of Naowarat, Tang & Goldwater, *Deep Speech Representations Track Word Timing along a Low-Dimensional Circular Trajectory* (`docs/ref.pdf`) with HuBERT-base. The spec is [#16](https://github.com/hans/word-circles/issues/16).

## Running

```sh
uv sync
# full run (prod, one GPU)
uv run snakemake -j4 --resources gpu=1
# local smoke run (CPU, ~300 utterances per split, plumbing only)
uv run snakemake -j4 --resources gpu=1 --configfile config.smoke.yaml
uv run pytest
```

Rules call `python` directly; never run several `uv run` commands at once, because they deadlock on uv's lock.

`HF_HOME`, `XDG_CACHE_HOME` (Snakemake) and `MPLCONFIGDIR` must point at writable directories. Without it, `transformers` and Snakemake fail when the default caches under `$HOME` are read-only.

## Graded RoSeg confidence on TIMIT

`outputs/graded/roseg_{cpca,leace}_frame_graded_scores.parquet` holds per-frame RoSeg boundary confidence for the 499 TIMIT sentences scored by [abstractography](https://github.com/ruca-boka/abstractography). The confidence is minus the smoothed angle to the first frame, for every layer of causal wav2vec2-base-960h. The files use the schema of abstractography's `rsa_wbc_frame_graded_scores.parquet`. Design: `docs/superpowers/plans/2026-10-22-roseg-graded-timit.md`.

These targets need the HF `timit-no_repeats` dataset at `data/timit-no_repeats`, since TIMIT is not fetched. The causal features go to memory-mapped `features.h5` caches under `w2v2.cache_dir` (default `/scratch/jgauthier/word-circles`, about 72 GB for the train sample). To follow extraction progress, run `tail -f outputs/w2v2/logs/extract_train.log`. To check our features against abstractography's own `.npz` files, set `w2v2.reference_npz_dir`.

Tests marked `data` need the fetched TextGrids (`uv run snakemake -j1 data/textgrids/dev`). The `gpu` test is a one-off CPU-vs-GPU parity check to run on prod.
