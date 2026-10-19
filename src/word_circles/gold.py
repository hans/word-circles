"""Gold-syllable baseline: the end of every `syllables`-tier interval is a boundary."""

import argparse
import tempfile
from pathlib import Path

import textgrids

from word_circles.evaluation import score_boundaries, write_scores


def gold_syllable_boundaries(textgrid_dir):
    """Map each utterance stem to the `xmax` of every syllable interval, silences included."""
    return {
        path.stem: [interval.xmax for interval in textgrids.TextGrid(path)["syllables"]]
        for path in sorted(Path(textgrid_dir).glob("*.TextGrid"))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref_dir", type=Path)
    parser.add_argument("evaluator_dir", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("empty", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as work:
        result = score_boundaries(gold_syllable_boundaries(args.ref_dir), args.ref_dir, args.evaluator_dir,
                                  Path(work) / "pred")
    write_scores(result, args.scores, args.empty, method="gold_syllables")


if __name__ == "__main__":
    main()
