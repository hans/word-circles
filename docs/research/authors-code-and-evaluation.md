# Authors' code and evaluation: what the RoSeg / word-circle pipeline actually does

Research ticket for replicating Naowarat, Tang & Goldwater, *Deep Speech Representations Track Word Timing along a Low-Dimensional Circular Trajectory* (arXiv:2610.16273v1, `docs/ref.pdf`) with `facebook/hubert-base-ls960` on CPU.

## Sources and citation keys

| Key | Source | Commit cloned |
|---|---|---|
| **WTT** | github.com/burin-n/word_timing_trajectory (the authors' code) | `e36e7e684ba7c7b43f349420006ebfd3ef0e2a9e` |
| **EVAL** | github.com/s-malan/evaluation (the evaluator; full history cloned) | `c9074a4256ad56caa2f2acaa00af0345f040e398` (HEAD, 2026-09-22) |
| **PWS** | github.com/s-malan/prom-wordseg (the Dissim. baseline) | `4e210251bd20e2f867563529d51eeef807a73555` |
| **CS** | github.com/burin-n/concept_space (code for ref [28], CoNLL 2026, arXiv:2605.01381). Same authors, sister repo, **not** the paper's code | `b4f89a4667af6b95d1477506ad0d3bd62d5e381a` |
| **PSC** | github.com/s-malan/prom-seg-clus | `92f65679770c402440d2ae8a8ce9b114de614e04` |
| **Paper** | `docs/ref.pdf` | — |
| **Malan24** | Malan, van Niekerk & Kamper, arXiv:2409.14486v2 (= paper ref [19]) | — |
| **CoNLL26** | Naowarat, Tang & Goldwater, arXiv:2605.01381 (= paper ref [28]) | — |

Citations look like `WTT:path:Lstart-Lend`. Line numbers refer to the commits above. Every quote from Malan24 and CoNLL26 comes from my own `pypdf` text extraction, not from a WebFetch summary.

## Summary (the key answers)

- **The WTT code covers almost the whole pipeline.** It has no feature extraction and no Fig 6 / gold-syllable code. It reads precomputed `(T, D)` per-layer `.npz` caches at 50 Hz (`WTT:README.md:L7-44`).
- **LEACE label: one-hot within-word frame index.** The label is `token_pos` = 0..L−1, pooled across all word durations, which gives up to 35 classes (0..34) at 50 Hz with the 35-frame cap. It is not (cos, sin), not binned proportion, and has no duration term. That is what the paper means by "without using word duration information".
  - The LEACE code is a hand-ported **NumPy copy of EleutherAI `concept-erasure` 0.2.4**: shrinkage whitening, `svd_tol=1e-7`, `affine=True`.
  - The 2-D subspace is the top-2 left singular vectors `u` of `W Σ_xz`.
  - Coordinates are `h = (x − μ_train) @ W u[:, :2]`, so the points live in whitened space.
- **"CPCA" in the paper is `Z_CPCA` in the code.** It is *Centroid* PCA, not contrastive PCA: plain `sklearn.PCA` fit on the stacked per-(duration, timestep) mean frames. There is no foreground/background covariance and no alpha.
  - The code has a separate `CPCAProjection` class (centroids per timestep only, no duration). The paper does **not** use it.
- **§II angles.**
  - Project each duration group's mean trajectory and centre it on **that group's own 2-D centroid** (footnote 3).
  - Take `arctan2` mod 2π, then `np.unwrap`, then subtract the first angle.
  - Regress elapsed proportion `t/(L−1)` on angle with `fit_intercept=False`.
  - R² comes from sklearn `.score`, and completion% = `|1/slope| / 2π`.
  - The plotted R² is fitted and scored on dev-clean, the same data the subspace was estimated on.
- **Word filters.** Silence frames are masked out. A word's frame span is `floor(start·50) + ceil(dur·50)`. Durations of **5–35 frames inclusive** are kept, and **duration groups with fewer than 50 tokens are dropped**.
- **RoSeg.**
  - `a_t = arccos(cos∠(h_t, h_0))` relative to the estimator's mean: the LEACE training-frame mean, or the mean of the CPCA centroid rows.
  - Smoothing is a zero-padded box moving average with window ∈ {1, 2, 3} frames.
  - Troughs come from inverting the signal (`max − s`), then `scipy.signal.find_peaks(·, prominence=p)` only (no `distance`, `height` or `threshold`).
  - An "absolute threshold" then keeps a trough only if the **unsmoothed** `a_t ≤ θ_max`.
  - Tuning grid: prominence ∈ linspace(0, π/4, 10), θ_max ∈ linspace(π/4, π, 10) (π means no threshold), window ∈ {1, 2, 3}. That is 300 configs, selected by **best R-value on dev-clean**.
  - Selected for SSL HuBERT L9: LEACE p=0.175, w=3, θ=π/4; CPCA p=0.349, w=2, θ=π/3.
- **Evaluation (EVAL `boundary_eval.py`).**
  - Times are rounded to 20 ms frames, and the tolerance is ±1 frame.
  - Matching is greedy and one-to-one: each reference boundary takes the first unused prediction within tolerance.
  - Reference boundaries are the `xmax` of **every** words-tier interval, including silence intervals and the utterance end. The utterance start is never a boundary.
  - Token F1 uses segments `[0]+boundaries`, so silence intervals count as reference tokens.
  - OS = N_pred/N_ref − 1.
  - R-value = `1 − (|√((1−R)²+OS²)| + |(−OS+R−1)/√2|)/2`.
- **Data shipped.**
  - WTT ships Kaldi word `.ali` files for dev-clean, test-clean and the **exact 10 h train-clean-100 sample**: 2834 utterances, 10.00 h, all 251 speakers. Subspace fitting uses these.
  - WTT also ships flattened MFA-style TextGrids for dev-clean and test-clean with **words, syllables and phones** tiers. Evaluation uses these.
  - EVAL release v0.1.0 has the full nested TextGrid set (597 MB).
- **Dissim. (PWS).**
  - Features are standardised **corpus-wide** (one `StandardScaler` over the batch, default 25k utterances, i.e. all of dev-clean). This is not per-utterance.
  - Cosine (or Euclidean) distance between adjacent frames is computed, then **z-scored per utterance**.
  - Smoothing is a left-edge-padded box moving average, followed by `find_peaks(prominence=p)`. A final boundary at the last frame is appended.
  - No hyperparameters are shipped. Malan24 used **HuBERT L9, cosine, window 4, prominence 0.75**.

---

## 1. LEACE label encoding and subspace

**Label.** `get_temporal_subspace` groups word segments by duration in frames. For every group of length `L` it labels each frame with `np.arange(L)`, its position relative to word onset, and concatenates all groups. With estimator `LEACE` there is *no* offset per duration. `Z_LEACE` would add one, but it is not used.

```python
# WTT:concept_analysis/trajectory/estimation.py:L64-86
for k in sorted(X_group):
    X = X_group[k]; N, L, H = X.shape
    X_batches.append(X.reshape(-1, H))
    y = np.arange(L)
    if estimator in ["Z_LEACE"]:
        y += y_offset; y_offset += L      # (duration, position) pairs
    else:
        pass                              # y = position relative to segment onset
    y_batches.append(np.tile(y, N))
...
return proj_cls(name=f"{estimator}_{subspace_name}").fit(X, y)
```

`LEACEProjection.fit` one-hot encodes `y` with `sklearn.OneHotEncoder` and fits `LeaceFitterCCA(method="leace", svd_tol=1e-7, shrinkage=True)` (`WTT:concept_analysis/projector.py:L312-326`). Class `t` therefore means "the t-th frame of a word, for any word long enough to have one".

- The classes are imbalanced: class 0..4 appears in every word, while class 34 appears only in 35-frame words.
- With the 5–35 frame filter there are up to 35 classes (0..34), so Σ_xz has rank ≤ 34.

**Why there is no duration information.** The label encodes only the absolute frame offset from onset. It does not encode the proportion elapsed or the total length. The paper says "cross-covariance between whitened features and word-timing labels (without using word duration information)" (Paper §III-A). The code confirms this.

**Implementation: hand-ported, faithful to EleutherAI.** The port lives in `WTT:concept_analysis/subspaces/leace_np.py`:

> "NumPy LEACE fitter and eraser … Ported from the original repository's `spaces/leace.py` and the inherited fitter calculations in concept-erasure 0.2.4" (`L1-15`).

It contains:

- **Running moments.** These are Welford-style updates (`L197-227`).
- **Σ_xx with shrinkage.** The covariance is `(S + Sᵀ)/2 / n`, passed through `optimal_linear_shrinkage`, the RMT linear shrinkage toward `tr(S)/p · I` (`L25-75`, `L311-328`).
- **Σ_xz.** The cross-covariance is `sigma_xz_ / (n−1)` (`L330-334`).
- **Whitening.** This is ZCA (symmetric) whitening via `eigh`: `W = V diag(λ^-1/2) Vᵀ` and `W_inv = V diag(λ^1/2) Vᵀ`, with a pinv-style eigenvalue mask (`L229-242`).
- **Subspace basis.**
  ```python
  # leace_np.py:L257-262
  svd_input = W @ self.sigma_xz            # method == "leace"
  u, s, _ = np.linalg.svd(svd_input, full_matrices=False)
  u *= s > self.svd_tol                    # svd_tol = 1e-7 from projector.py:L323
  proj_left = W_inv @ u
  proj_right = u.conj().T @ W
  ```
- **Transform used everywhere downstream.**
  ```python
  # projector.py:L331-334, L344-346
  self.mean = eraser.bias                  # = mean_x  (affine=True default)
  self.proj_right = eraser.proj_right.conj().T   # = W u   (D × k)
  def transform(self, X, n_components=None):
      return (np.asarray(X) - self.mean) @ self.proj_right[:, :N]
  ```
  So `h_t = uᵀ W (x_t − μ)`, which is the projection of the whitened, centred frame onto the top-2 singular directions of `WΣ_xz`. This is the "whitened feature space" in the Fig 4 caption. CoNLL26 §3.1 describes LEACE's subspace as the image of the oblique projection `P = W⁻¹UUᵀW`.

**The `constrain_cov_trace` branch cannot fire for LEACE (resolved by algebra).** `constrain_cov_trace=True` is the default (`leace_np.py:L131`), and `LEACEProjection.fit` never overrides it. If `tr(PΣPᵀ) > tr(Σ)`, the code would rebuild `proj_left/proj_right` from an SVD of a mixed projector (`L264-298`). That condition is never met here:

- `W` and `W_inv` both come from `compute_whitening(self.sigma_xx)`, the same shrunk Σ that the trace check uses (`L248-249, L266-268`). Shrinkage makes Σ full rank, so `W = Σ^{-1/2}`, `W_inv = Σ^{1/2}` and `WΣ = W_inv`.
- `u` has orthonormal retained columns, and the columns zeroed by `svd_tol` contribute nothing.
- With `P = I − W_inv u uᵀ W`, this gives `PΣPᵀ = Σ − W_inv u uᵀ W_inv`, so `tr(PΣPᵀ) = tr Σ − ‖W_inv u‖²_F ≤ tr Σ`.

The branch therefore cannot trigger apart from floating-point noise, and **`h = uᵀ W (x − μ)` exactly**. `svd_tol` does not affect the 2-D subspace either, because the top-2 singular values sit far above both 1e-7 and upstream's 0.01.

**Recommendation:** vendor `leace_np.py` verbatim (MIT licence, `WTT:LICENSE`), or reimplement the five lines `Σ_shrunk → ZCA W → svd(WΣ_xz) → h = (x−μ) W u[:, :2]`. The reason is to reproduce the same shrinkage formula and dtype handling, not to avoid a code path.

**Dtype.** `LeaceFitterCCA.fit` uses `dtype=x.dtype` (`L191-195`), so whatever dtype the feature cache holds sets the precision of the statistics. Use float32 or float64, never float16.

## 2. Centring and the RoSeg origin

| Where | Origin / centring | Citation |
|---|---|---|
| LEACE transform | `μ` = mean of **all retained in-word training frames**: train-clean-100 10 h sample, silence removed, 5–35-frame words, groups with ≥ 50 tokens | `projector.py:L331, L344-346`; `leace_np.py:L206-208, L303` |
| CPCA (`Z_CPCA`) transform | `μ` = `PCA.mean_` = unweighted mean of the stacked centroid rows. Each duration group of length L contributes L rows, so long groups weigh more; group token counts play no role | `estimation.py:L55-58`; `projector.py:L167-188` |
| RoSeg `a_t` | No extra centring. The angle is taken on `subspace.transform(x_utt)`, so the origin is the estimator mean above. This is the paper's "centroid of the temporal variance" | `predict_boundaries.py:L62-67`; `segmentation.py:L32` |
| §II R² / completion | The transformed mean trajectory of each duration group is re-centred on **its own 2-D mean** (`group_centering_method="centroid"`), which is footnote 3 | `geometry.py:L149-174, L177-211`; `evaluate_temporal_metrics.py:L107, L318-319` |
| §II plots (Figs 1a, 2, 4) | 2-D trajectory plots show raw `transform` output. The polar-angle panel uses `"centroid"` | `plot_temporal_trajectories.py:L43-46, L99-104` |

## 3. CPCA details

There is **no contrastive PCA** anywhere in the code. The paper's "CPCA" is *Centroid-PCA* (Paper §II-A, footnote 1; CoNLL26 §3.1: "arranges the centroids … of the classes into a matrix … the concept subspace is the span of the principal components").

```python
# WTT:concept_analysis/trajectory/estimation.py:L55-58
elif estimator == "Z_CPCA":
    X_mean = get_mean_trajectories(X_group)          # {L: X_group[L].mean(axis=0)}  (L × D)
    X = np.concatenate([X_mean[k] for k in sorted(X_mean)])
    return PCAProjection(name=f"{estimator}_{subspace_name}").fit(X)   # sklearn PCA(whiten=False), all comps
```

- Every experiment script uses `Z_CPCA` ("CPCA" in the paper) and `LEACE`: `exp0:L54`, `exp3a:L51`, `exp3b:L28`, and the configs YAML. `CPCAProjection` (per-timestep centroids pooled over durations) exists at `projector.py:L240-289` but is not invoked.
- The centroid matrix stacks (duration, timestep) mean frames: about 31 groups for L = 5..35, giving about 620 rows. There is no per-row normalisation and no whitening.
- `transform` = `X @ components[:2].T − mean @ components[:2].T` (`projector.py:L178-188`).

**Word inclusion and grouping** (`WTT:scripts/fit_temporal_subspace.py`, `concept_analysis/annotations/*`, `trajectory/grouping.py`):

1. Silence rows (`<SIL>`, `<eps>`, `SIL`, `SPN`, …) are dropped (`alignment.py:L16, L166-168`).
2. Frame spans: `start = floor(round(start_s·50, 10))` and `end = start + ceil(round(dur_s·50, 10))` (`alignment.py:L20-41, L172-178`). Later rows overwrite earlier ones where spans overlap (`loading.py:L20-27`), so a word can lose its last frame to the next word.
3. Frames not covered by any word (silences) are **removed**, and each utterance becomes the concatenation of in-word frames (`loading.py:L29-30`).
4. Segments are split where the `token_pos` label changes (one id per alignment row), then grouped by length in frames (`grouping.py:L45-90`).
5. Groups with fewer than `low_freq_threshold=50` segments are dropped (`grouping.py:L92-97`; `exp0:L111`, `exp3a:L110`).
6. Groups outside `[floor(0.1·50), floor(0.7·50)] = [5, 35]` frames inclusive are dropped (`grouping.py:L34-38, L100-102`; `fit_temporal_subspace.py:L123-124`).

The same filtered groups feed LEACE, so LEACE also sees only 5–35-frame words.

**§II angle, regression, R², completion** (`WTT:concept_analysis/trajectory/metrics.py`, `geometry.py`):

```python
# geometry.py:L143-146, L200-209  (level="mean", default in exp2a:L21)
feat = subspace.transform(mean_trajectory, n_components=2)
feat = normalize_trajectory(feat, mean_trajectory=feat, centering_mode="centroid")  # subtract own mean
angles[seq_len] = np.arctan2(feat[:,1], feat[:,0]) % (2*np.pi)
# metrics.py:L35-42
elapsed = np.arange(seq_len) / max(seq_len - 1, 1)      # d_1 = 0 ... d_T = 1
unwp_angle = np.unwrap(sample_angle); relative_angle = unwp_angle - unwp_angle[0]
# metrics.py:L96-110
linear_model = LinearRegression(fit_intercept=False).fit(angle_train, elapsed_train)  # elapsed ~ angle
slope = coef; endpoint_completion = abs((1.0 - 0)/slope) / (2*np.pi)
"r2": linear_model.score(angle_train, elapsed_train)
```

- The regression pools all duration groups into one fit, with one row per (group, timestep) of the **mean** trajectory.
- R² is sklearn's `r2_score`. Its denominator is the variance about the mean of `elapsed`, even though there is no intercept.
- `force_ccw=False`, so clockwise rotation gives a negative slope. `abs()` in completion handles that, and R² is unaffected.
- Which data: `exp2a` fits the regression on **dev-clean** (`r2_train`) and also scores test-clean (`r2_test`) (`exp2a:L56-57, L68-78`). The subspaces in §II are estimated on dev-clean too (`exp0:L64, L76`). `plot_temporal_metrics.py` plots train metrics by default (`L360-366`), so **Fig 3 R² and completion% are in-sample on dev-clean**.
- The "54–79% of centroid variance" figure corresponds to `mean_explained_variance` with the ambient denominator (`metrics.py:L122-157`).

## 4. RoSeg peak detection

```python
# WTT:concept_analysis/trajectory/segmentation.py:L32-39
angles = get_sequence_relative_angles(projected_traj, relative_to=0, angle_mode="cosine_arccos")
smoothed = smoothing(angles, window=smooth_window_size)
smoothed = smoothed.max() - smoothed          # invert to find troughs
peaks, _ = find_peaks(smoothed, prominence=prominence)
if threshold_max_angle is not None:
    peaks = peaks[angles[peaks] <= threshold_max_angle]   # threshold on UNSMOOTHED angle
```

- **Signal.** `a_t = arccos(clip(cos(h_0, h_t)))` on the 2-D transform output (`geometry.py:L55-61, L95-109`). It is computed over the **whole utterance**, including leading, trailing and internal silences: `predict_boundaries` projects every frame in the cache (`L107-113, L62`). Fitting excluded silence.
- **Smoothing.** `smoothing(x, window)` is a mean over a sliding window with **zero padding** (`geometry.py:L40-48`). For an int window `w`, `left = w//2` and `right = w−left−1`. That gives w=1: none, w=2: `[t−1, t]` (asymmetric), w=3: `[t−1, t+1]`.
- **`find_peaks`.** Only `prominence` is passed. There is no `distance`, `height`, `threshold` or `width`. Prominence is measured in radians on the inverted smoothed signal.
- **Output.** Frame index `k` is written as `k / 50` seconds to `<utt>.list` (`predict_boundaries.py:L57-72`). No utterance start or end boundary is inserted (`segmentation.py:L19-20`).
- **Tuning grid.** These are the authors' "absolute and relative prominence thresholds and smoothing window sizes" (Paper §III-C-1).
  ```python
  # WTT:scripts/tune_peak_detection.py:L15-29
  for prominence in np.linspace(0, np.pi / 4, 10):            # 0, .087, .175, ..., .785 (rounded to 3 dp)
      for threshold_max_angle in np.linspace(np.pi / 4, np.pi, 10):  # π/4, π/3, ..., π  (π == off)
          for smooth_window_size in [1, 2, 3]:
              for n_components in [2]:
  ```
- **Protocol** (`run_all_exp.sh:L42-69`; `exp3a`, `exp3b`, `exp3c`, `exp3d`):
  1. Fit `Z_CPCA` and `LEACE` at layer 9 on the train-clean-100 10 h sample (`exp3a:L37-41`).
  2. Predict all 300 configs on dev-clean.
  3. Evaluate each with `boundary_eval.py --tolerance 1` (`exp3c:L55-56`).
  4. Run `gather_segmentation_results_v2.py --mode paired-threshold --paired-direction threshold-to-no-threshold --tuned-metrics R-value`.
- **How the paired mode selects.** It picks the best **R-value** among configs with θ_max < π (first one wins ties, `L102-114, L121-160`). The "no threshold" partner is then **the same prominence and window with θ_max = π**, not re-tuned. Table II's "−abs threshold" rows are this partner.
- **Test.** Test-clean is run once with the exported dev configs (`run_all_exp.sh:L64-81`).
- **Selected configs** (`WTT:experiments/temporal_trajectory/configs/*.yaml`):

  | Model | Estimator | prominence | window | θ_max (thresholded) | θ_max (ablation) |
  |---|---|---|---|---|---|
  | hubert-base-ls960 | LEACE | 0.175 | 3 | 0.785 (π/4) | π |
  | hubert-base-ls960 | Z_CPCA | 0.349 | 2 | 1.047 (π/3) | π |
  | hubert-base-random | Z_CPCA | 0.175 | 2 | π/4 | π |
  | hubert-base-random | LEACE | 0.698 | 3 | π/4 | π |

- **Grid edges.** The selected LEACE θ_max = π/4 is the **lowest** grid value, and untrained-LEACE prominence 0.698 is the second-highest grid value (max 0.785). The optimum may lie outside the grid.

## 5. Dissim. baseline (PWS)

- **Feature normalisation: corpus-level, not per-utterance.** `Features.normalize_features` fits one `StandardScaler.partial_fit` on `torch.cat` of **all** utterances in the batch, then transforms each utterance (`PWS:wordseg/utils.py:L99-123`). Batches are `--batch_size` (default 25000) utterances, i.e. all of dev-clean or test-clean (`PWS:extract_seg.py:L16-56, L129-133`). The paper's "(mean- and variance-normalized) frame representations" (§III-B) does not say over what.
- **Distance.** `scipy.spatial.distance.cosine(y_i, y_{i+1})` (or Euclidean norm of `diff`). The resulting T−1 distance curve is **z-scored per utterance** with a fresh `StandardScaler` (`PWS:wordseg/segment.py:L43-70`).
- **Smoothing.** `np.pad(dist, (w//2 + 1, 0), mode='edge')` (left padding only), then `np.convolve(·, ones(w)/w, 'valid')` (`segment.py:L72-85`).
- **Peaks.** `find_peaks(smoothed, prominence=p)` only (`segment.py:L87-113`).
- **Post-processing.** A boundary at `T` (the number of frames) is appended unless the last peak is already at T or T−1 (`extract_seg.py:L178-184`). Frames are converted to seconds as `k·20/1000` (`utils.py:L144-161`).
- **Layer.** The paper says "We then follow [19] and use HuBERT layer 9 features" (§III-C-1). PWS takes the layer from the CLI (`model_name/layer_#/…npy`, `README`).
- **Hyperparameters.** Neither PWS nor PSC ships values. The README shows an example JSON `{"distance": "euclidean", "window_size": 6, "prominence": 0.4}`, which is illustrative only. Malan24 §IV says:
  > "we extract features from the 9th HuBERT layer … For our prominence-based approach, we select a four-frame window with a 0.75 prominence threshold"

  Malan24 §II also says the distance is cosine. They tuned "by finding a balance between NED and R-value on the development data" (LibriSpeech dev-clean).
  - Their dev-clean result (Malan24 Table III: R-value 50.7, Token F1 15.6) is close to but **not equal to** the paper's Dissim. dev row (R 52.6, Tok F1 17.0). It is therefore unconfirmed whether the paper reused (L9, cosine, w=4, p=0.75) or re-tuned (see Unresolved).
- **Shipped bug.** The `--load_hyperparams` path dereferences `args.alignments_dir` (`PWS:extract_seg.py:L146`), which is not a defined argument, so loading hyperparameters from JSON raises `AttributeError` as shipped. The interactive-prompt path works. Re-implementing the ~30 lines is simpler than running PWS.

## 6. Evaluation

**Which evaluator version.**

- WTT's README and `exp3c` point to `s-malan/evaluation/boundary_eval.py` and invoke:
  ```bash
  # WTT:experiments/temporal_trajectory/exp3c_evaluate_segmentation.sh:L55-56
  "${python_bin}" "${evaluation_script}" "${config_dir}" "${reference_dir}" \
      --alignment_type="${alignment_type}" --tolerance 1      # alignment_type=word
  ```
- **No public EVAL version accepts `--alignment_type=word`.** The argument first appears at `7a6ed14` (2025-08-15), and from then to HEAD it has `choices=["words","syllables","phones"]`. The flattened TextGrids name their tier `words`. The authors therefore ran a locally edited copy.
- The parser in WTT's `gather_segmentation_results_v2.py:L56-70` expects the comma-separated `Name: value` lines and the `Token F1` label that EVAL has printed since `95fa8b5`. `95fa8b5` itself crashes in frame mode: `boundary_word.py:L390-391` reads `ref_utt` at module scope before any assignment, which raises a NameError. So the authors used logic from `6ddff70`(2025-11-18)..HEAD.
- Across that range, the only diffs that touch `split_utterances=False` scoring are numerically identical on non-degenerate data:
  - OS was `recall/precision − 1` and became `n_seg/n_ref − 1`. These are equal whenever `n_hit > 0`.
  - A redundant `len(ground_truth) > 0` guard was removed.
- **Use HEAD (`c9074a4`) logic.** The "deprecated, use language-speech-learning/unit-discovery-evaluation" notice (`EVAL:README.md`) was not followed up here.

**Semantics at HEAD with the authors' flags** (`--tolerance 1`; defaults `ms_per_frame=20`, `split_utterances=False`, `strict=True`):

- **Tolerance.** `1.0 % 1 == 0` puts the code in frame mode (`L363`). Both predictions and TextGrid times are converted with `get_frame_num(s) = floor(round(s/0.020, 1) + 0.5)` (round half up), and a hit needs `|Δframe| ≤ 1` (`L286-305, L393-401`). RoSeg's `k/50` times map back exactly to `k`.
- **Prediction parsing.** A leading `0.0` is dropped (`L374-381`). **An empty `.list` file crashes** on `seg_utt[0]` (IndexError), so every utterance needs at least one prediction. With prominence ≥ 0 and a non-trivial signal this is rare but possible.
- **File pairing.** Each `.list` must pair with a TextGrid with the same stem, sorted; the counts must match exactly (`L365-370`).
- **Reference boundaries.** These are `[interval.xmax for interval in tg]` over the **whole words tier, silence intervals included** (`L409-411`).
  - Every word end and every silence end is a boundary. A word onset after a pause appears as the silence's `xmax`.
  - The utterance end (last `xmax`) is included. The utterance start (0) is never included.
- **Last-boundary rule.** If the last prediction is within tolerance of the last reference, **both are deleted** from the counts (`L147-153`). Otherwise the final reference counts as a miss. RoSeg never emits the end boundary, while Dissim always appends one; see the discrepancies section.
- **Hits.** These are greedy and one-to-one:
  ```python
  # EVAL:boundary_eval.py:L162-168
  for i_ref in ground_truth:
      for i, i_seg in enumerate(prediction):
          if abs(i_ref - i_seg) <= tolerance:
              n_hit += 1
              prediction.pop(i)        # prediction consumed
              if strict: break         # each ref hit at most once
  ```
  References are processed in time order, and each takes the earliest unused prediction within ±1 frame. This is not optimal bipartite matching. P = hits/N_pred, R = hits/N_ref, F1 is harmonic (`L17-59`).
- **OS and R-value.**
  ```python
  # EVAL:boundary_eval.py:L83, L104-107
  os = n_seg/n_ref - 1
  r1 = np.sqrt((1 - recall)**2 + os**2); r2 = (-os + recall - 1)/np.sqrt(2)
  rvalue = 1 - (np.abs(r1) + np.abs(r2))/2
  ```
- **Token F1** (`L172-239`).
  - Segments are `pairwise([0] + boundaries)` for both hypothesis and reference. The first token always starts at frame 0, and the last hypothesis token ends at the last predicted boundary.
  - Reference silence intervals are reference tokens. A hypothesis token hits if its start and end are each within ±1 frame of an unused reference token's start and end. Each reference token can be used once.
  - Token scoring uses the untrimmed lists, so the last-boundary rule does not apply.
- **Silence handling.** There is none beyond the above. `split_utterances=True` would restrict scoring to non-silence stretches, but the authors did not set it.

**Per-syllable-count scores (Fig 6; Paper §III-C-2, footnote 7).** No code for this is in WTT. The paper describes grouping words by syllable count and scoring predictions only within the time spans of each group, with shared boundaries double-counted. See Unresolved.

## 7. Shipped data

**In WTT** (`alignments/`):

| File | Content | Format | Used for |
|---|---|---|---|
| `Librispeech/dev-clean-word.ali` | 2703 utterances, 65,089 rows (including 10,687 `<SIL>`) | space-separated with header `utt_id start_time word_dur word`; uppercase words; `<SIL>` rows; 10 ms resolution | §II fits and metrics (`exp0`, `exp2a`); dev-side LEACE/CPCA metrics |
| `Librispeech/test-clean-word.ali` | 2620 utterances | same | `exp2a` test R² |
| `Librispeech/train-clean-100sampled_10h_utt-word.ali` (+ symlink `train-clean-100-word.ali`) | **2834 utterances, 10.002 h** (sum of last end times), 251 speakers, 575 chapters, 98,648 word tokens (89% in 100–700 ms) | same | RoSeg subspace fitting (`exp3a`) |
| `Librispeech-Simon-flatten/{dev,test}-clean-flatten.tar.gz` | 2703 / 2620 Praat TextGrids, flat directory, tiers **`words`** (lowercase, `""` = silence), **`syllables`** (ARPAbet phones joined by `-`, e.g. `IH-G`, `Z-AE`, `M-AH-N-D`), **`phones`** (ARPAbet with stress, e.g. `IY1`) | ooTextFile | Evaluation reference (`exp3c`) |

- **Provenance.** The paper says word boundaries come from "Kaldi forced alignment [40]" (§II-C), which matches the `.ali` style. The TextGrids were "generated by Simon for reproducibility", flattened from the EVAL v0.1.0 release (`exp3c:L15-18`).
- **The two alignments differ.** For example, in `2277-149896-0015` "HE" is 0.28–0.50 s in the `.ali` but 0.25–0.49 s in the TextGrid.
- The TextGrids look like MFA output (lowercase words, empty-string silences, ARPAbet phones), but no file states this.
- The syllabification method is not documented anywhere I found. Syllables are phone groupings whose boundaries coincide with phone boundaries.
- **The 10 h sample.** The filename contains `sampled_10h_utt`, and it covers all 251 train-clean-100 speakers, which is consistent with random utterance-level sampling. The sampling code and seed are not shipped, but the file itself is, so **no re-sampling is needed**.

**In EVAL.**

- The repo itself has no alignments.
- Release **v0.1.0** (2025-12-04) ships `librispeech-alignments.tar.gz` (597,207,841 bytes): `https://github.com/s-malan/evaluation/releases/download/v0.1.0/librispeech-alignments.tar.gz`.
- Layout: `librispeech-alignments/<split>/<spk>/<chapter>/<utt>.TextGrid`, with tiers `words`, `phones`, `syllables` (checked on a streamed sample from `train-clean-360/7481/101276/`).
- `train-clean-360` is present. I did not download the full archive to enumerate all splits.

## 8. Feature extraction, layers, baselines (fog items)

- **WTT has no extractor.** It expects `<cache>/<model>/Librispeech/<split>/<layer_id>.npz` holding `{utt_id: (T, D)}` arrays, plus a `manifest.toml` with `feature_frame_rate = 50.0` and `feature_subsampling_factor = 320` (`WTT:README.md:L7-44`; `features/feature_reader.py`). The model names used are `hubert-base-ls960` and `hubert-base-random`, layers 0–12 (`exp0:L35-36`).
- **Layer convention.** The paper states "layer 0 indicates the inputs to the Transformer layers" (Fig 3 caption), so **L9 = output of the 9th transformer block**. In the sister repo CS, caches are written as `hidden_states[i]` from HF `AutoModel(..., output_hidden_states=True)` under file name `f"{layer_i}.npy"`, with `Wav2Vec2FeatureExtractor(do_normalize=True, sampling_rate=16000)` (`CS:concept_space/helper/inference_libri.py:L93-125, L172-195`; `CS:run_leace_train_clean460.py:L109-111`). HF `hidden_states[0]` for HuBERT-base is the post-positional-conv, post-LayerNorm input to block 1. **This is strong but indirect evidence** that the WTT caches used the same convention.
- **Frame rate.** 50 Hz, and the evaluator's 20 ms frames match it. With HF, the frame count is `floor((N−400)/320)+1`.
- **Untrained HuBERT.** The cache name is `hubert-base-random`. There is no code in WTT or CS for how it was initialised (for example `HubertModel(HubertConfig.from_pretrained(...))` with default init and a seed).
- **Log Mel.** The commented-out `"spectrogram"` model sits at `exp0:L39`, and the "log Mel features" plot name is commented out at `exp1:L48`. CS has `inference_melspectrogram` (`CS:inference_libri.py:L31-89`): torchaudio `MelSpectrogram(n_fft=512, win=25 ms, hop=20 ms, n_mels=80)` plus an optional `LayerNorm`. It applies **no log**, so it should not be read as the paper's log-Mel recipe.
- **Untrained Whisper and the other models.** These are commented out in the scripts. No extraction code exists.

---

## Code vs. paper discrepancies

1. **"CPCA" is `Z_CPCA`.** The paper's CPCA is the code's `Z_CPCA`, i.e. PCA on stacked (duration × timestep) centroids. The class literally named `CPCAProjection` (timestep-only centroids) is unused. The paper's footnote 1 acknowledges the duration grouping; the code naming is the trap.
2. **LEACE label specifics.** The paper says only "word-timing labels". The code uses one-hot absolute within-word frame index (0..34) with imbalanced classes. Proportion is not encoded, and long-word frames dominate the high-index classes.
3. **Unstated filters.** Duration groups with fewer than 50 tokens are dropped (`low_freq_threshold=50`), which the paper never mentions. The 100–700 ms filter is implemented as **5–35 frames inclusive** after floor/ceil frame conversion of Kaldi times.
4. **Silence removed before grouping.** Frames outside words are deleted, and span overlap resolution can shave frames. Not in the paper.
5. **Shrinkage.** LEACE whitens with an **RMT linear-shrinkage** estimate of Σ_xx, using ZCA whitening and `svd_tol=1e-7`. The paper just says "whitens". `constrain_cov_trace=True` is set but provably inactive for `method="leace"` (§1).
6. **Origin for a_t.** The paper's "centroid of the temporal variance" is concretely the **LEACE training-frame mean** (raw space, then whitened) or the **unweighted mean of CPCA centroid rows**. Both are fixed from training data, not per utterance.
7. **Silence at inference.** RoSeg runs on **whole utterances including silences**, while fitting used in-word frames only.
8. **Smoothing.** The paper says "time-averaged smoothing". The code uses a zero-padded box average with windows 1–3, and window 2 is asymmetric (`[t−1, t]`). The absolute threshold is applied to the **unsmoothed** angle at the smoothed-trough location.
9. **Tuning details.** The paper says it tunes "absolute and relative prominence thresholds and smoothing window sizes". The code uses a fixed 10×10×3 grid and selects by R-value among thresholded configs. The "−abs threshold" ablation is **not re-tuned**; it reuses the thresholded winner's prominence and window. The selected LEACE θ_max sits on the grid's lower edge.
10. **Train sample size.** The paper says 10 h, the `exp3a` header comment says "5h audios sampled from train-clean-100" (`exp3a:L5`), and the shipped file is 10.00 h. The README also says the training alignment is not supplied under the expected name, but the symlink exists. Both comments are stale.
11. **Evaluator CLI.** `exp3c` passes `--alignment_type=word`, which every public `boundary_eval.py` rejects (choices `words|syllables|phones`). The authors used a locally modified evaluator. The scoring logic is equivalent across `6ddff70..HEAD`.
12. **Two alignments.** The paper cites Kaldi alignments (§II-C). Evaluation uses different, MFA-style TextGrids from Simon Malan (§III-C-2, footnote 6). Boundaries differ by tens of ms, so the subspace is fit on one alignment and scored against another.
13. **Reference boundary set.** The paper describes boundary F1 with 20 ms tolerance. The code's reference set includes **silence-interval ends** (pause onsets and word onsets after pauses) and the utterance end, and token F1 treats silences as tokens. The paper does not say this.
14. **End-boundary asymmetry.** RoSeg writes no final boundary, while Dissim appends `T`. With EVAL's last-boundary rule, Dissim's appended end is deleted together with the reference end when they are within ±1 frame. RoSeg instead usually takes a miss on the utterance-final reference, unless a trough happens to fall there. Quantitatively, `find_peaks` never returns the last frame, and HF gives `T = floor((N−400)/320)+1`, so RoSeg's latest possible boundary is about `N/320 − 2` frames. The reference end (TextGrid `xmax` ≈ audio duration) rounds to about `N/320`. RoSeg therefore takes **about one guaranteed boundary miss per utterance** (about 1 in 24 references for LibriSpeech, roughly a 4-point recall ceiling) and loses the final reference token. **Decision:** to match Table II, do *not* append an end boundary to RoSeg output. Appending one would be a deliberate deviation that would raise scores.
15. **Dissim normalisation.** The paper's "mean- and variance-normalized frame representations" is corpus-level (whole split) per-dimension z-scoring in PWS, plus per-utterance z-scoring of the distance curve, which the paper does not mention.
16. **Layer indexing.** The paper's "layer 0 = Transformer input" implies L9 = HF `hidden_states[9]`. This is confirmed only via the sister repo CS.
17. **§II R² is in-sample.** Fig 3 and the "R² = 0.983" are computed on dev-clean, the same data used to estimate the subspace and fit the regression (`exp0`, `exp2a`, `plot_temporal_metrics` defaults). Test-clean R² is computed but not shown by default.
18. **Elapsed proportion.** The paper's `d_t ∈ [0,1]` is concretely `t/(L−1)`, so the last frame has d = 1. The regression is elapsed-on-angle (d ~ θ), as the paper states; it is not angle-on-elapsed.

## Unresolved (candidate open decisions)

1. *(Resolved; see §1.)* The `constrain_cov_trace` branch cannot fire for LEACE. The one remaining choice is whether to keep the shrinkage estimator (code) or use the plain sample covariance. The paper is silent; default to shrinkage to match the code.
2. **Dissim hyperparameters.** Did the paper reuse Malan24's (L9, cosine, w=4, p=0.75) or re-tune on dev-clean? Malan24's dev numbers (R 50.7 / Tok F1 15.6) differ from Table II (52.6 / 17.0). The difference could come from evaluator version or alignment, or from re-tuning.
3. **Fig 6 and gold-syllable code.** There is no code for the per-syllable-count evaluation (footnote 7's span-restricted scoring) or for the "Gold syllables" row. The TextGrids' `syllables` tier is the obvious source. How syllable counts per word are derived (by counting `syllables`-tier intervals inside each word interval?) and how predictions are restricted to spans must be designed.
4. **Syllabification method and aligner.** The method behind the TextGrid syllables, and confirmation that the TextGrids are MFA output, are undocumented.
5. **Feature extraction details** not shown in WTT:
   - HF vs fairseq checkpoint.
   - `do_normalize` (CS sets True).
   - Whether the whole utterance is fed at once (CS does; this is CPU-feasible for LibriSpeech lengths).
   - Feature dtype in the cache.
6. **Untrained HuBERT construction.** The init scheme and seed for `hubert-base-random` are unknown.
7. **Log-Mel baseline recipe.** It is unknown whether log is applied, plus the n_mels, hop and normalisation. CS's mel code has no log and an optional LayerNorm.
8. **10 h sampling procedure.** The script and seed are not shipped. This is moot if we use the shipped `.ali`.
9. **Exact evaluator copy.** The locally edited `boundary_eval.py` is not published. It is assumed equivalent to EVAL `6ddff70..HEAD` with `words` as the tier.
10. **Empty-prediction utterances.** EVAL crashes on empty `.list` files. How the authors handled this, if it ever occurred, is unknown. Options are to ensure at least one boundary or to patch the evaluator.
11. **Tie-breaking and grid edges.** Selection takes the first config on R-value ties in file-glob order. Whether to extend the grid below θ = π/4 or above p = π/4 is our choice.
12. **TextGrid-to-cache utterance coverage.** Prediction runs on every utterance in the cache, and EVAL asserts equal counts. The caches must cover exactly the 2703 / 2620 TextGrid utterances.
