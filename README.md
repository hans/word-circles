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

Tests marked `data` need the fetched TextGrids (`uv run snakemake -j1 data/textgrids/dev`). The `gpu` test is a one-off CPU-vs-GPU parity check to run on prod.
