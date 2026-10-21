"""Gold-syllable baseline: the end of every `syllables`-tier interval is a boundary."""

import argparse
from pathlib import Path

import textgrids

from word_circles.evaluation import write_predictions


def gold_syllable_boundaries(textgrid_dir):
    """Map each utterance stem to the `xmax` of every syllable interval, silences included."""
    return {
        path.stem: [interval.xmax for interval in textgrids.TextGrid(path)["syllables"]]
        for path in sorted(Path(textgrid_dir).glob("*.TextGrid"))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref_dir", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    write_predictions(gold_syllable_boundaries(args.ref_dir), args.predictions)


if __name__ == "__main__":
    main()
