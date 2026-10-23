# Graded RoSeg confidence on TIMIT for abstractography

Export per-frame RoSeg boundary confidence for the 499 TIMIT sentences that
`ruca-boka/abstractography` scores. Its `graded_metrics.py` cross-correlates the
result against its RSA, dissimilarity and norm-trajectory curves. Two methods
are exported: `roseg_cpca` and `roseg_leace`.

## Output

`outputs/graded/roseg_{cpca,leace}_frame_graded_scores.parquet` uses the schema of
`rsa_wbc_frame_graded_scores.parquet`:

| column | type | meaning |
|---|---|---|
| `stem` | string | `<speaker_folder_lower>_<id_lower>` (abstractography's `stem_for`) |
| `frame` | int64 | 0 .. `len(audio)//320 - 1`, every frame, silence included |
| `graded_score` | float64 | `-smooth(angle_to_frame0, W)`; higher means more boundary-like |
| `layer` | int64 | 0..12, all layers in one file |

`graded_metrics.load_rsa_frame_graded_scores(path, layer, ...)` reads either
file unchanged.

## Decisions

1. **Features.** We port abstractography's causal extractor (`extract_model_features.py`)
   for `facebook/wav2vec2-base-960h`, pinned to revision `22aad52d`.
   - Frame f is the last position of a forward pass on `audio[:f*320+400]`,
     normalized per prefix. That gives `len//320` frames, and the last frame
     repeats when the final prefix is shorter than 400 samples past it.
   - Batching: the CNN and feature projection run once per prefix, unbatched,
     so group norm only sees that prefix. Only the Transformer is batched,
     right-padded with an attention mask.
   - A CPU check on 50 frames matched the exact per-prefix path to 1e-4
     relative at L12 (cosine 1.000000).
   - The pseudo-causal script in ideal-word-representations does not match
     (worst cosine 0.57 at L12). It normalizes the whole utterance and pads the
     input under group norm.
2. **TIMIT audio.** The audio comes from `timit-no_repeats` `input_values`.
   - Those values were normalized over the whole utterance but still sit on the
     int16 grid, so we rescale them to the int16/32768 scale abstractography read
     from WAV. Per-prefix normalization makes that equal up to the 1e-7 epsilon.
   - The 499 stems are fixed in `resources/timit_stems.txt`, taken from
     abstractography's RSA parquet.
3. **Subspace.** CPCA and LEACE are fit per layer (0..12) on causal features of the
   LibriSpeech 10 h train sample with Kaldi word alignments, using the same
   settings as the HuBERT replication (§II `min_dur`/`max_dur`/`low_freq_threshold`).
   The fitted subspace is then applied to TIMIT.
4. **W.** The smoothing window W is selected per (estimator, layer) on a seeded
   500-utterance dev-clean subset, over the authors' 300-config grid.
   - The rule is the replication's `select_config`: best R with theta_max < pi
     and no empty utterance.
   - Only W enters the curve, since prominence and theta_max just gate peaks.
   - If no config is eligible, the best R ignoring empties is used and flagged
     in `window.json`.
5. **Graded quantity.** The curve is `-concept_analysis.smoothing(get_sequence_relative_angles(z, relative_to=0), W)`.
   - This is `detect_boundaries`'s inverted signal without its per-utterance
     max offset, so values are comparable across sentences under the pooled
     z-score.
   - The zero-padded smoothing edges are kept.
6. **Parity.** The `w2v2_parity` rule compares our TIMIT cache with abstractography's
   `.npz` files. It runs on prod once `w2v2.reference_npz_dir` is set.

## Cost (to be measured on prod)

The causal pass is quadratic in utterance length. It runs over about 10 h of
train, 500 dev utterances and 499 TIMIT sentences, for all 13 layers. Expect
tens of GPU-hours and about 80 GB of cache.
