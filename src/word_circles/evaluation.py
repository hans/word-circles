"""Scoring through the pinned, unmodified s-malan evaluator.

Every boundary score in the pipeline goes through `score_boundaries`, which
writes one `.list` per utterance and runs `boundary_eval.py` as a subprocess
with `--alignment_type=words --tolerance 1`.
"""

import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# The evaluator crashes on an empty `.list`. A single boundary at frame 1 is a
# near-certain miss; a leading 0.0 would be dropped by the evaluator, and a
# last-frame placeholder would trigger its last-boundary rule.
EMPTY_PLACEHOLDER_S = 0.02

_OUTPUT_KEYS = {
    "Precision": "precision",
    "Recall": "recall",
    "F1": "F1",
    "Over-segmentation": "OS",
    "R-value": "R",
    "Token Precision": "token_precision",
    "Token Recall": "token_recall",
    "Token F1": "token_F1",
}
_NUMBER_RE = re.compile(r"([A-Za-z][\w -]*?): (-?[\d.]+(?:e[+-]?\d+)?|-?inf|nan)")


@dataclass
class EvalResult:
    scores: dict
    empty: list


def score_boundaries(predictions, ref_dir, evaluator_dir, work_dir):
    """Score predicted boundary times (seconds) against the `words` tier in `ref_dir`.

    `predictions` maps utterance stem to boundary times. The stems must equal
    the reference TextGrid stems exactly: the evaluator pairs files by sorted
    position and only checks counts. Returns the evaluator's metrics as
    fractions and the stems whose predictions were empty.
    """
    ref_stems = {p.stem for p in Path(ref_dir).glob("*.TextGrid")}
    pred_stems = set(predictions)
    if pred_stems != ref_stems:
        missing = sorted(ref_stems - pred_stems)
        extra = sorted(pred_stems - ref_stems)
        raise ValueError(f"prediction stems != reference stems; missing: {missing}; extra: {extra}")

    pred_dir = Path(work_dir)
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True)
    empty = []
    for stem in sorted(predictions):
        times = list(predictions[stem])
        if not times:
            empty.append(stem)
            times = [EMPTY_PLACEHOLDER_S]
        (pred_dir / f"{stem}.list").write_text("".join(f"{float(t)!r}\n" for t in times))

    out = subprocess.run(
        [sys.executable, str(Path(evaluator_dir) / "boundary_eval.py"), str(pred_dir), str(ref_dir),
         "--alignment_type=words", "--tolerance", "1"],
        capture_output=True, text=True, check=True,
    ).stdout
    parsed = dict(_NUMBER_RE.findall(out))
    return EvalResult(scores={key: float(parsed[name]) for name, key in _OUTPUT_KEYS.items()}, empty=empty)


def write_scores(result, scores_path, empty_path, method):
    with open(scores_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *result.scores])
        writer.writeheader()
        writer.writerow({"method": method, **result.scores})
    Path(empty_path).write_text(json.dumps({"n_empty": len(result.empty), "empty": result.empty}, indent=1))
