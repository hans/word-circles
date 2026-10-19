# LibriSpeech alignment and syllable sources for the replication

Research ticket: where to get (1) word alignments comparable to the paper's Kaldi alignments, (2) gold syllable boundaries for the Table II "Gold syllables" row, and (3) per-word syllable counts for Fig 6, for dev-clean, test-clean and train-clean-100. This note builds on `docs/research/authors-code-and-evaluation.md` (section 7 and unresolved items 3, 4, 12) and does not repeat it.

## Sources and citation keys

| Key | Source | Version checked |
|---|---|---|
| **REL** | github.com/s-malan/evaluation release `v0.1.0`, asset `librispeech-alignments.tar.gz` (the "TextGrids") | published 2025-12-04; downloaded in full |
| **EVAL** | github.com/s-malan/evaluation (evaluator) | `c9074a4` (HEAD) |
| **ULE** | github.com/language-speech-learning/unit-discovery-evaluation (EVAL's successor) | `8b216a7`, release `v1.0.0` |
| **DP** | github.com/s-malan/data-process (Simon Malan's preprocessing repo) | `ccaf4bf` (HEAD, 2025-11-24) |
| **WTT** | github.com/burin-n/word_timing_trajectory | `e36e7e6` |
| **LUG** | Lugosch, "LibriSpeech Alignments", Zenodo 10.5281/zenodo.2619474 v1.0 (2019-03-31) | `librispeech_alignments.zip`, downloaded in full, md5 matches the record |
| **CJ** | github.com/CorentinJ/librispeech-alignments | `a34135e` (HEAD, 2019-03-25) |
| **SYL** | github.com/kylebgorman/syllabify (Python 3 port) | `d816db7` |
| **Paper** | `docs/ref.pdf` | text extracted with pypdf |

"Measured" means I ran it. All scratch work is in the session scratchpad (downloads, a venv with numpy/pypdf/praat-textgrids, plain-Python parsers). Nothing was installed or written in `/workdir` except this file. The scripts are not saved in the repo; each result below says what was computed.

## Summary (the key answers)

- **The TextGrids and the syllables tier come from Simon Malan's pipeline, and the paper itself points to it** (Paper footnote 6: "Evaluation scripts and alignments: github.com/s-malan/evaluation").
- **Recommendation, one line per need:**
  1. Word alignments for subspace fitting: WTT `train-clean-100sampled_10h_utt-word.ali` (2834 utts, MIT repo, 3.7 MB). It is the paper's Kaldi alignment.
  2. Word alignments for evaluation: the `words` tier of the REL TextGrids. For dev/test use WTT's flattened tarballs (about 3.5 MB each), which are byte-identical to REL (measured). Do not download the 597 MB tarball (ULE v1.0.0 ships the identical file).
  3. Gold syllable boundaries: the `syllables` tier of the same TextGrids. **Measured: it reproduces the paper's Table II gold row** (dev-clean F1 85.50 / OS 33.91 / Tok F1 65.92 / R 71.06 against 85.5 / 33.9 / 65.9 / 71.1).
  4. Per-word syllable counts: count non-silence `syllables`-tier intervals inside each `words`-tier interval of the same file. Measured to be exact (no syllable straddles a word boundary, and the count equals the stress-marked vowel count in every word on dev/test).
- **The REL TextGrids are not Lugosch's MFA output with syllables added.** They are a different alignment: `<unk>` words are resolved to spelled words, silences are `""` instead of `sil`/`spn`, and boundaries differ by tens of ms from LUG (measured). MFA version, acoustic model and dictionary are undocumented.
- **Syllabifier (verified by reproduction):** Gorman's `syllabify` applied to each word's stress-marked ARPAbet phones, then stress stripped and phones joined by `-`. It reproduces the tier interval for interval on 300 of 300 random dev-clean utterances (measured).
- **Kaldi vs TextGrid offset (measured, dev/test/train-sample, 2703/2620/2834 utts, about 105-197k boundaries each):**
  - Small bias, large random part. All boundaries: mean -5.7 ms (TextGrid earlier), median -10 ms, sd 30 ms.
  - Word starts: mean -2 ms. Word ends: mean -9 ms.
  - 84% of boundaries are within 20 ms and 94.6% within 40 ms.
  - Word-count mismatches: 0 in all three splits.
- **Coverage:** utterance IDs match exactly for dev (2703) and test (2620). All 2834 train-sample utterances are in the REL train-clean-100 TextGrids (28539 files).
- **Licences:** REL has no licence file or statement (GitHub reports none). WTT is MIT. LUG is CC BY 4.0. CJ has no licence. The audio is CC BY 4.0.
- **Audio sizes (HEAD on 4 hosts, all agree):** dev-clean 337,926,286 B, test-clean 346,663,984 B, train-clean-100 6,387,309,499 B. Total 7.07 GB.

---

## 1. How the TextGrids were produced

### 1.1 The alignment (words and phones tiers): undocumented, and not LUG's raw output

**What the documentation says.**
- REL release body, in full: "Boundary evaluation script with attached LibriSpeech alignment files." (GitHub API `repos/s-malan/evaluation/releases`).
- The EVAL README says nothing about how the alignments were made. It has no mention of MFA.
- The only pointer in the authors' code is `DP:README.md` (the feature-slicing section): the slicing script operates on alignments from `https://zenodo.org/records/2619474` (LUG), "Syllable alignments are created and added to the TextGrid files (requires … kylebgorman/syllabify)".
- Malan24 (arXiv:2409.14486v2) says only that NED "relies on phonemic transcriptions found by forced alignments" and cites no aligner. Malan's top-down paper (arXiv:2507.19204) has the same sentence.
- A later Kamper-group paper (ZeroSyl, arXiv:2602.15537, Sec. 4.1) says "phone-level forced alignments [27]" with [27] = McAuliffe et al. 2017 (MFA) and "a rule-based method [28]" with [28] = Gorman's syllabify.
- An independent toolkit paper (findsylls, arXiv:2603.26292, its datasets paragraph) says: "For LibriSpeech, we use word and phone alignments [Lugosch] … produced with the Montreal Forced Aligner", and syllables "via syllabify".
- The successor repo that EVAL's README points to, github.com/language-speech-learning/unit-discovery-evaluation (**ULE**, `8b216a7`, 2026-09-22; release `v1.0.0`), also documents nothing. Its README says only "If `.TextGrid` is used together with LibriSpeech, we include the alignment files in the release of this repository". Its release asset `librispeech-alignments.tar.gz` is **identical to REL** (sha256 `3eb5886ff0846d98ca0ca1a542f4036f989a1f407a6224d8205d3bfa874fefe8` for both downloads, measured). It has no LICENSE file (`git ls-files`; GitHub `license: null`), no MFA/syllabify mention in any `.md` or `.py`, and the accompanying paper (arXiv:2606.06183, "Revisiting Lexicon Evaluation in Unsupervised Word Discovery") says nothing about how the alignments were made (grep for align/MFA/syllab/forced found no hit). The org's other repos (`tdev2`, `zerospeech2020`) are unrelated.
- None of these gives an MFA version, acoustic model or dictionary for REL.

**What LUG says** (Zenodo record, API `zenodo.org/api/records/2619474`): "We obtained these alignments using the Montreal Forced Aligner, using their pre-trained LibriSpeech acoustic model." No version, no dictionary. License CC BY 4.0.

**What I measured: REL's words and phones tiers differ from LUG's.** I downloaded both zips and compared dev-clean utterance by utterance (same 2703 IDs).
- Tiers: LUG has `words, phones` (2 tiers). REL has `words, phones, syllables`.
- OOV words: LUG writes `<unk>` in the words tier and `spn` in the phones tier. REL writes the spelled word and real phones (e.g. `3536-8226-0001` "bozzle": LUG `spn` over 4.60-5.15 s, REL `B AA1 Z AH0 L`).
  - In the first-pass word-sequence comparison, 168 of 2703 utterances differ, and the most common differences are `<unk>` vs a spelled word.
  - REL contains no `<unk>` or `spn` in the words tier (0 in 2703 + 2620 + 3000 sampled utts).
- Silence: LUG phones use `sil`; REL uses `""` (empty) in all three tiers.
- Boundaries: where the word sequences agree (2535 utts), REL minus LUG gives:
  - starts: mean -0.4 ms, sd 30 ms, within 20 ms = 81%, exactly equal = 25%
  - ends: mean -5.7 ms, median -10 ms, sd 23 ms, within 20 ms = 85%
  - the number of silence intervals differs in 795 of 2535 utterances.
- So REL is a fresh alignment (or a heavily post-processed one), not a copy of LUG plus syllables. The phone labels are the same ARPAbet-with-stress inventory in both.
- **File format cannot identify the MFA version:** REL files use the `praat-textgrids` serialisation (two-tab interval indent, unpadded floats such as `0.38`), whereas LUG files keep MFA's native formatting (`0.360`, deeper indent). REL was therefore rewritten by a post-processing script, so the file layout says nothing about the aligner.
- **Inference only (not verified):** `""` for silence, resolved OOV pronunciations and ARPAbet phones with stress are consistent with a newer MFA release with an English ARPA model and G2P. I found no statement of this anywhere. The aligner, version, model and dictionary for REL are **undocumented**.

**Candidate hint for provenance of the Kaldi `.ali`.** The authors' Kaldi `.ali` is much closer to LUG's MFA (itself Kaldi-based) than to REL (measured, dev-clean, matching word sequences, LUG minus `.ali`):

| comparison (same 2534 dev-clean utterances, word sequences identical in all three) | mean | median | sd | within 20 ms | within 40 ms |
|---|---|---|---|---|---|
| LUG - Kaldi | -2.8 ms | 0 | 20.8 ms | 91.2% | 96.7% |
| REL - Kaldi | -5.9 ms | -10 | 26.2 ms | 84.1% | 94.7% |

(On all 2702 usable utterances, REL - Kaldi is sd 29.6 ms and 84.0% within 20 ms, as in Sec. 3; the 168 utterances where LUG has `<unk>` are harder, but the gap survives on the common subset.)

This suggests (does not prove) that the `.ali` was produced by a model similar to LUG's. I found no document on how the `.ali` was made.

### 1.2 The syllables tier: Gorman's `syllabify`, applied per word

**Evidence from code.** `DP:feature_slicing.py:L19, L104-119` (same logic since the first commit `704c7d6`, 2025-10-14) does `from syllabify import syllabify` (SYL) and calls `syllabify.syllabify(phone_transcription)` on the phones of one sub-utterance. It labels each syllable `' '.join(syl)`, **with stress digits and spaces**.

**That script is not what produced the REL tier.** I checked the data:
- REL labels are `B-AH-T` style (`-` joined, **stress stripped**).
- No REL syllable crosses a word boundary.

**Reproduction (measured).** I ran SYL's `syllabify()` over the phones inside each non-silence word interval of REL's `phones` tier (stress digits kept), stripped digits, joined with `-`, and compared with the `syllables` tier:
- Per word: 300 of 300 random dev-clean utterances equal interval for interval, including labels and times (9517 syllable intervals, 100% match).
- Utterance-level syllabification (all phones in one call, as `feature_slicing.py` does per sub-utterance): only 31 of 300 equal.

So the REL tier is **word-by-word Gorman syllabification of the MFA phone tier, with stress dropped**. The generating script is not published (the only syllabify use I found in s-malan's repos is `DP:feature_slicing.py`, which behaves differently). I checked the full commit history of `data-process` (`git log --all -S syllab`: only `704c7d6`, `ce96312`) and `evaluation` (only the `alignment_type` choice at `7a6ed14`).

The top-down paper (arXiv:2507.19204, text around its Table IV, footnote 2) says its syllable boundaries come from SYL and "are not perfect, as in some cases the rule-based system merges phones that span across words". That describes the utterance-level behaviour. The REL tier does not have this flaw (zero cross-word syllables, measured, below), so it is the better reference.

**Licence of SYL:** MIT-style permission notice in the header of `syllabify.py` (copyright Kyle Gorman 2012-2013). There is no separate LICENSE file, and GitHub reports `license: null`.

---

## 2. Verification on the real data (REL, dev-clean, test-clean, train-clean-100)

Setup (measured): I streamed REL once to disk (`curl`, 597,207,841 B, same size as the release asset), listed it with `tar tzf`, and extracted the three splits into the scratchpad.

**Layout and counts.**
- `librispeech-alignments/<split>/<spk>/<chapter>/<utt>.TextGrid`, 292,365 TextGrids over 7 splits (dev-clean, dev-other, test-clean, test-other, train-clean-100, train-clean-360, train-other-500).
- dev-clean 2703, test-clean 2620, train-clean-100 28539 files (all train-clean-100 utterances; LUG has 28538, REL has `7067-76048-0021` in addition; it is not in the 10 h sample).
- Tiers in file order: `words`, `phones`, `syllables`. Per-split size on disk is about 29 MB for dev and for test, 470 MB (block usage) for train-clean-100.

**Alignment between the tiers (all of dev-clean and test-clean; 3000 random train-clean-100 files for the syllable check).**

| check | dev-clean | test-clean | train-clean-100 (3000) |
|---|---|---|---|
| utterances | 2703 | 2620 | 3000 |
| non-silence words / syllables | 54,402 / 75,910 | 52,576 / 74,598 | 104,147 / 145,198 |
| silence intervals in `words` and in `syllables` | 11,735 / 11,735 | 12,083 / 12,083 | 20,813 / 20,813 |
| word boundary that is not a syllable boundary | 0 | 0 | 0 |
| syllable straddling a word boundary | 0 | 0 | 0 |
| words whose syllable count differs from their count of stress-marked vowel phones | 0 | 0 | 1 (`n't` in `6848-252322-0014`: 0 vowels, 1 syllable) |
| words with 0 syllables | 0 | 0 | 0 |

- Silence intervals in `words` and `syllables` are the same intervals (same count, same times). The `phones` tier also uses `""`; test-clean has 1 empty phone interval that is not a word-tier silence (not investigated).
- **How silence appears:** an interval with `text = ""` in all three tiers. There is no `<SIL>`, `sil` or `spn`.
  - Utterances start with a silence interval in 2683 of 2703 dev (2605 of 2620 test) files and end with one in all of them. So utterances can start directly with a word. This differs from CJ's txt format, which guarantees a (possibly zero-duration) leading and trailing silence.
- Syllable-per-word histogram (dev-clean, 54,402 words): 1: 38,923 · 2: 10,986 · 3: 3,229 · 4: 1,014 · 5: 228 · 6: 22. Test-clean has a tail to 8.
- Syllable labels contain no stress and no word boundary. The tier is usable as is.

**It reproduces the paper's gold-syllable row (measured).** I wrote one `.list` file per utterance containing the `xmax` of every `syllables`-tier interval (silence intervals included, since the evaluator's reference set also includes them), and ran the unmodified `EVAL:boundary_eval.py` (HEAD) with `--alignment_type=words --tolerance 1` against the flattened dev TextGrids (requires `praat-textgrids`, `tqdm`, `numpy`):

| dev-clean | F1 | OS | Token F1 | R-value |
|---|---|---|---|---|
| Paper Table II, Gold syllables | 85.5 | 33.9 | 65.9 | 71.1 |
| Measured, all syllable-interval ends | 85.50 | 33.91 | 65.92 | 71.06 |
| Measured, non-silence syllable ends only | 76.60 | 14.78 | 45.33 | 76.96 |

The first variant matches the paper to the last printed digit; the second does not. So the paper's gold-syllable predictions are the xmax of **all** syllables-tier intervals including silence ends. This is strong evidence that the paper used this tier (or an identical one). On test-clean the same procedure gives F1 84.93, OS 35.50, Token F1 65.26, R-value 69.70 (Table II has no gold row for test).

**Per-word syllable counts (Fig 6).** The paper groups words by syllable count and scores within word spans (Paper Sec. III-C-2, footnote 7). The counts for that grouping are available from the file itself: for each non-silence `words` interval, count non-silence `syllables` intervals with `xmin >= word.xmin` and `xmax <= word.xmax`. Because no syllable straddles a word boundary, this is exact. The independent cross-check (count of phones ending in a stress digit inside the word) agreed for every word on dev and test. I did not recompute Fig 6 itself. The footnote-7 span semantics (double-counted shared boundaries) still has to be implemented; no code exists for it (prior note, unresolved 3).

---

## 3. Kaldi `.ali` vs TextGrid word-boundary offset (measured)

Method: for every utterance present in both a WTT `.ali` and REL (all of them, not only 50-100), take the non-`<SIL>` words from the `.ali` (`start`, `start + dur`) and the non-empty `words` intervals from the TextGrid, require the same number of words, and compare word by word (TextGrid minus Kaldi, ms). The `.ali` has 10 ms resolution and so do the TextGrid word boundaries (utterance `xmax` values such as 12.485 do not).

| split | utts used | word-count mismatches | boundaries | mean | median | sd | p10 / p90 | within 20 ms | within 40 ms |
|---|---|---|---|---|---|---|---|---|---|
| dev-clean (all) | 2702 of 2703 | 0 (1 text mismatch: `bush'` vs `bush`, 2428-83705-0022, skipped) | 108,764 | -5.7 | -10 | 29.6 | -30 / +20 | 84.0% | 94.6% |
| test-clean (all) | 2620 of 2620 | 0 | 105,152 | -5.7 | -10 | 32.5 | -30 / +20 | 83.4% | 94.5% |
| train 10 h sample | 2834 of 2834 | 0 | 197,296 | -5.2 | -10 | 33.7 | -30 / +20 | 83.8% | 94.6% |

Dev-clean, split by boundary type (mean, within 20 ms):
- Word starts: -2.0 ms, 85.2%.
- Word ends: -9.4 ms, 82.7%.
- Word-to-word ends with no pause in either alignment: -5.7 ms, 88.0%.
- Median absolute difference 10 ms for all; p90 of the absolute difference 30 ms.
- Per-utterance mean offset: mean -6.0 ms, sd 9.8 ms (range -108 to +256 ms).

Dev-clean histogram of all boundary differences (10 ms bins, share): -30: 5.3% · -20: 14.8% · -10: 26.8% · 0: 24.3% · +10: 12.8% · +20: 5.3% · +30: 2.2%. Beyond ±60 ms: 2.5%.

**Systematic or random?** Both, but the random part is bigger.
- There is a small systematic bias: TextGrid words end about 9 ms earlier than Kaldi words and start about 2 ms earlier, so TextGrid words are about 7 ms shorter. Over all boundaries the mean is -5.7 ms, about a quarter of a 20 ms frame.
- The spread (sd about 30 ms, 16% of boundaries off by more than 20 ms) is much larger than the bias. This is aligner disagreement, not a fixed offset. The bias is consistent across the three splits, so it does not depend on the data.

**Oracle check (measured, own greedy implementation of the EVAL rules).** Scoring Kaldi boundaries (end of every `.ali` row, silence rows included) as if they were predictions against the TextGrid reference at ±1 frame gives:
- dev-clean: P 0.864, R 0.850, F1 0.857
- test-clean: P 0.862, R 0.845, F1 0.853

So a system that reproduced Kaldi boundaries exactly would score only about 85% F1 on the TextGrid reference; that is the size of the disagreement between the two aligners. It is not a ceiling for RoSeg: a predictor that tracks the TextGrid boundaries can recover those hits. RoSeg is fitted on Kaldi-derived labels, so some of this gap may carry through to its scores, but how much is unmeasured (Unresolved 6).

**Implication for fitting on Kaldi and scoring on TextGrids at ±1 frame.**
- The authors did exactly this, so replicating their setup carries the same mismatch. It is not an error to avoid, and it is part of what Table II reflects.
- During fitting, the 5-35 frame duration filter and the per-position labels use `floor(start·50)` and `ceil(dur·50)` on Kaldi times. A shift of one 10 ms step moves a label by half a frame, so label noise is about 1.5 frames (sd), and about 5% of boundaries move more than 2 frames. LEACE and CPCA average over tens of thousands of tokens, so random noise mostly blurs the mean trajectories slightly; the -9 ms end bias moves the end of each fitted word by about half a frame relative to the evaluation reference. I did not measure the effect on RoSeg; that would need the features.
- Scoring uses a ±1 frame tolerance, which absorbs the bias and about 84% of the Kaldi-vs-TextGrid disagreement; the other 16% of Kaldi boundaries lie more than 20 ms from the matching TextGrid boundary.
- An optional ablation for later: fit on TextGrid-derived labels for the same 2834 utterance IDs (all present in REL, so no new sampling). If the F1 gap is small, the Kaldi/TextGrid mismatch is not driving the result. This is a choice for the project, not something the paper did.

## 4. Coverage (measured)

- WTT `.ali` utterance sets vs REL TextGrids: dev-clean 2703 = 2703 (set equality, 0 extra either way); test-clean 2620 = 2620.
- WTT flattened tarballs (`Librispeech-Simon-flatten/{dev,test}-clean-flatten.tar.gz`) hold the same IDs as REL and are **byte-identical** to the REL files (md5 over all 2703 and 2620 files).
- 10 h train sample: all 2834 IDs are in REL train-clean-100 (and in LUG); word sequences agree for 2834 of 2834 (case-insensitive).
- Prior note's unresolved 12 (the feature caches must cover exactly 2703 / 2620 utterances) is therefore satisfiable: REL has no missing or extra utterances for dev/test. The remaining requirement is on our own feature extraction.

---

## 5. Candidates compared

| Candidate | Record format | Splits (of the 3) | Download size | Licence | How produced | Syllables |
|---|---|---|---|---|---|---|
| **WTT `.ali`** (github.com/burin-n/word_timing_trajectory `e36e7e6`, `alignments/Librispeech/`) | text: `utt_id start dur word`, `<SIL>` rows, 10 ms | dev, test, **10 h sample of train-clean-100** (2834 utts) | 2.1 MB, 2.0 MB, 3.7 MB | repo `LICENSE`: MIT, "Copyright (c) 2026 Burin Naowarat" (WTT:LICENSE:L1-3). Not stated whether this covers the data. | "Kaldi forced alignment" (Paper Sec. II-C); recipe/model undocumented | no |
| **WTT flattened TextGrids** (`Librispeech-Simon-flatten/`) | Praat TextGrid, flat dir, `words/phones/syllables` | dev, test | 3.5 MB, 3.4 MB (tar.gz) | as above (MIT repo) | byte-identical to REL | yes |
| **REL** (s-malan/evaluation `v0.1.0`) | nested Praat TextGrids, `words/phones/syllables` | all three (and 4 more) | 597,207,841 B (one tarball for all 7 splits; train-clean-360 is first in the stream, so partial extraction still reads most of it) | **none**: no LICENSE file in the repo (`git ls-files`: `.gitignore README.md boundary_eval.py clust_eval.py`), GitHub API `license: null`, release body has no terms. The repo README is marked deprecated; the same file is also released in ULE v1.0.0 (also no licence). | MFA-style realignment, details undocumented (Sec. 1.1); syllables = per-word Gorman syllabify (Sec. 1.2) | yes |
| **LUG** (Zenodo 2619474) | nested-by-split zip of TextGrids, `words/phones` only | all three (train-clean-100 has 28538 files) | 623,023,332 B (md5 `2bab567d0ace651a4ba254e813629f46`, matches record) | **CC BY 4.0** (Zenodo metadata) | MFA with "their pre-trained LibriSpeech acoustic model" (version unknown) | no (SYL could be run per word on its phone tier) |
| **CJ** (CorentinJ/librispeech-alignments `a34135e`) | (a) per-book `.alignment.txt`: words and per-word end times, silences as empty words, always starts and ends with silence; (b) raw MFA TextGrid per utterance | all 7 splits | (a) 70 MB, (b) 594 MB (Google Drive; not downloaded by me) | **none** (repo has `README.md` and `parser_example.py` only; GitHub `license: null`) | "generated using the Montreal Forced Aligner"; README names MFA only (I read it in full); no version, model or dictionary. 46 unaligned in (b), 127 in (a) | no |
| **HF `gilkeyio/librispeech-alignments`** | parquet with audio plus `words`/`phonemes` lists (silences removed) | all (train_clean_100 has 28538) | 62 GB for all splits (audio included) | cc-by-4.0 (dataset card) | LUG TextGrids parsed by the uploader (card says so) | no |
| **HF `Holger1997/LibriSpeech-Alignments`** | mirror of CJ's txt (a) | all | not measured (7378 files) | no licence field; card says the mirror "does not claim authorship … or grant additional rights" | CJ | no |
| **Kaldi recipes** (`egs/librispeech/s5`, e.g. `steps/align_fmllr.sh`, `ali-to-phones`/`ctm`) | whatever you generate | any | n/a (needs the recipe's acoustic models, not checked) | Apache-2.0 for Kaldi (not checked here) | you run it; I found no ready-made public dev/test/train word CTMs besides the authors' | no |

Notes:
- For CJ and the HF mirrors I read the READMEs/dataset cards and the HF API only; I did not download the Google Drive files. That CJ's raw TextGrid archive (594 MiB = about 623 MB) is the same file as LUG is plausible from the size but is **not verified**.
- No other credible source of ready-made gold syllable boundaries for LibriSpeech turned up. Every syllable reference I found (REL, ZeroSyl, findsylls, the top-down paper) is "MFA/LUG phone tier plus Gorman's `syllabify`".

## 6. Audio download sizes

Primary page: https://www.openslr.org/12 (fetched). Its license line: "CC BY 4.0". Sizes printed on the page: dev-clean 337M, test-clean 346M, train-clean-100 6.3G. HEAD requests (`curl -sIL`, `Content-Length`) on `www.openslr.org`, `openslr.trmal.net`, `openslr.elda.org` and `openslr.magicdatatech.com` (after a 301/302) all return the same bytes:

| file | bytes | decimal |
|---|---|---|
| `dev-clean.tar.gz` | 337,926,286 | 338 MB |
| `test-clean.tar.gz` | 346,663,984 | 347 MB |
| `train-clean-100.tar.gz` | 6,387,309,499 | 6.39 GB |
| total | 7,071,899,769 | 7.07 GB |

The paper's training sample uses 2834 train-clean-100 utterances (10 h), but the tarball cannot be fetched partially without reading it through; the `.ali` file lists the utterance IDs to keep.

## 7. Licence summary

| Item | Licence | Evidence |
|---|---|---|
| LibriSpeech audio and text | CC BY 4.0 | openslr.org/12 |
| REL (TextGrids, v0.1.0) | **none stated** | no LICENSE in repo at `c9074a4`; API `license: null`; release body. Derived from CC BY 4.0 material, but the redistributor has not said what applies. Safest course: do not vendor it into this repo; cite and download, or ask Simon Malan. |
| WTT repo (`.ali`, flattened TextGrids) | MIT | `LICENSE` at `e36e7e6`. The TextGrids inside are REL copies (byte-identical), so MIT coverage of them is uncertain. |
| LUG | CC BY 4.0 | Zenodo record; asks for citation of Lugosch et al. 2019 and McAuliffe et al. 2017 |
| CJ | none | repo contents; API `license: null` |
| SYL (`syllabify.py`) | MIT-style notice in the file header | no LICENSE file |
| EVAL (`boundary_eval.py`) | none | as for REL |

---

## Unresolved

1. **Aligner, version, acoustic model and dictionary behind REL's words and phones tiers.** Not documented in EVAL, ULE, DP, WTT, Malan24, or the follow-up papers I read. The measured differences from LUG (OOV words spelled out, `""` silences, boundary shifts of 10-30 ms) show it is a separate run. Asking Simon Malan or Burin Naowarat is the only way to settle it.
2. **The script that produced REL's syllables tier** is not public. I reproduced it exactly from the data (per-word Gorman syllabify, stress stripped, `-` joined, 300/300 utterances), so the behaviour is known even though the source is not. `DP:feature_slicing.py` is a different, utterance-level variant.
3. **How the Kaldi `.ali` files were made** (recipe, model, speaker adaptation). The `.ali` is closer to LUG than to REL (Sec. 1.1), which hints at an MFA/Kaldi-like model, nothing more.
4. **Which train-clean-100 TextGrid release the authors used for any train-side TextGrid analysis.** Irrelevant for fitting (they use the `.ali`), but unresolved.
5. **Fig 6 span semantics.** The per-word syllable counts are settled; the implementation of footnote 7 (restrict predictions to group spans, double-count shared boundaries) is not and no author code exists. Whether the Fig 6 gold-syllable curve scores syllable boundaries inside each word span in the same all-intervals way as Table II is also unknown.
6. **Effect of the Kaldi/TextGrid mismatch on RoSeg itself.** Quantified at the label level only (Sec. 3); needs the features to measure.
7. **CJ's Google Drive archives** were not downloaded, so the equality of its raw TextGrids with LUG is unverified.
8. **Minor:** one empty phone interval in test-clean that is not a word-tier silence, and 20 dev / 15 test utterances that begin without a leading silence interval; neither was investigated further.
