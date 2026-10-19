"""Fig 6: segmentation scored separately for words of each syllable count.

Locked by the gold-syllable oracle (#15). For each group (1-6 syllables) and
utterance, the references are the union of {start, end} over the group's
words; a prediction counts if it lies within one frame of one of those word
spans. Boundary hits use the evaluator's formulas without its last-boundary
drop. Token F1 uses the evaluator's `eval_token_boundaries` unchanged. `all`
pools groups 1-6, so boundaries shared across groups count once per group
(paper footnote 7). Silences are never spans; an utterance with no words in
a group is left out of that group.
"""

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import textgrids

from word_circles.evaluation import check_stems, load_evaluator, read_predictions

GROUPS = ["1", "2", "3", "4", "5", "6"]
TOLERANCE = 1
MS_PER_FRAME = 20


@dataclass
class Fig6Result:
    scores: dict
    word_counts: dict
    n_dropped_7plus: int


def words_by_syllable_count(textgrid_path):
    """Non-silence words of one utterance as (start, end, n_syllables).

    A syllable belongs to the word containing its midpoint; no syllable
    straddles a word boundary in the s-malan TextGrids.
    """
    tg = textgrids.TextGrid(textgrid_path)
    syllables = [(s.xmin + s.xmax) / 2 for s in tg["syllables"] if s.text != ""]
    return [
        (w.xmin, w.xmax, sum(1 for mid in syllables if w.xmin <= mid <= w.xmax))
        for w in tg["words"] if w.text != ""
    ]


def eval_boundaries_without_last_drop(seg, ref, tolerance):
    """The evaluator's strict `eval_boundaries`, minus deleting each utterance's final hit pair."""
    n_seg = n_ref = n_hit = 0
    for prediction, ground_truth in zip(seg, ref):
        prediction = list(prediction)
        n_seg += len(prediction)
        n_ref += len(ground_truth)
        for i_ref in ground_truth:
            for i, i_seg in enumerate(prediction):
                if abs(i_ref - i_seg) <= tolerance:
                    n_hit += 1
                    prediction.pop(i)
                    break
    return n_seg, n_ref, n_hit


def score_by_syllable_count(predictions, textgrid_dir, evaluator_dir):
    """Score boundary predictions (seconds) per syllable-count group and pooled `all`.

    Groups with no words in any utterance are absent from the result.
    """
    check_stems(predictions, textgrid_dir)
    be = load_evaluator(evaluator_dir)

    def frame(seconds):
        return int(be.get_frame_num(float(seconds), ms_per_frame=MS_PER_FRAME))

    seg = {g: [] for g in GROUPS}
    ref = {g: [] for g in GROUPS}
    word_counts = Counter()
    n_dropped = 0
    for stem in sorted(predictions):
        words = words_by_syllable_count(Path(textgrid_dir) / f"{stem}.TextGrid")
        n_dropped += sum(1 for *_, n in words if n > len(GROUPS))
        times = list(predictions[stem])
        if times and float(times[0]) == 0.0:  # the evaluator drops a leading 0.0
            times = times[1:]
        pred_frames = [frame(t) for t in times]
        for g in GROUPS:
            group_words = [(lo, hi) for lo, hi, n in words if n == int(g)]
            if not group_words:
                continue
            word_counts[g] += len(group_words)
            refs = sorted({t for lo, hi in group_words for t in (lo, hi) if t > 0})
            spans = [(frame(lo) - TOLERANCE, frame(hi) + TOLERANCE) for lo, hi in group_words]
            seg[g].append([p for p in pred_frames if any(a <= p <= b for a, b in spans)])
            ref[g].append([frame(t) for t in refs])

    seg["all"] = [s for g in GROUPS for s in seg[g]]
    ref["all"] = [r for g in GROUPS for r in ref[g]]
    scores = {}
    for g in ["all", *GROUPS]:
        if not ref[g]:  # no word of this group anywhere: nothing to score
            continue
        n_seg, n_ref, n_hit = eval_boundaries_without_last_drop(seg[g], ref[g], TOLERANCE)
        precision, recall, f1 = be.get_p_r_f1(n_seg, n_ref, n_hit)
        os_ = be.get_os(n_seg, n_ref)
        token_p, token_r, token_f1 = be.get_p_r_f1(*be.eval_token_boundaries(seg[g], ref[g], tolerance=TOLERANCE))
        scores[g] = {"precision": precision, "recall": recall, "F1": f1, "OS": os_, "R": be.get_rvalue(os_, recall),
                     "token_precision": token_p, "token_recall": token_r, "token_F1": token_f1}
    return Fig6Result(scores=scores, word_counts=dict(word_counts), n_dropped_7plus=n_dropped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("ref_dir", type=Path)
    parser.add_argument("evaluator_dir", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    result = score_by_syllable_count(read_predictions(args.predictions), args.ref_dir, args.evaluator_dir)
    with open(args.scores, "w", newline="") as fh:
        fields = ["method", "group", "n_words", *result.scores["all"]]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for group, scores in result.scores.items():
            n_words = sum(result.word_counts.values()) if group == "all" else result.word_counts[group]
            writer.writerow({"method": args.method, "group": group, "n_words": n_words, **scores})
    args.scores.with_suffix(".dropped.json").write_text(json.dumps({"n_dropped_7plus": result.n_dropped_7plus}))


if __name__ == "__main__":
    main()
