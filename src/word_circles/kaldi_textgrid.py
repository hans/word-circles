"""Kaldi word alignments as `words`-tier TextGrids, for the Kaldi-reference diagnostic.

Final configs are rescored on test against these as well as the s-malan
TextGrids, to measure how much the alignment source moves the scores. They
are never used for tuning or selection.
"""

import argparse
from collections import defaultdict
from pathlib import Path

SILENCE = "<SIL>"


def write_words_textgrid(path, intervals):
    """Write a TextGrid with one `words` tier of contiguous (xmin, xmax, text) intervals."""
    end = intervals[-1][1]
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        "xmin = 0.0", f"xmax = {end}", "tiers? <exists>", "size = 1", "item []:",
        "\titem [1]:", '\t\tclass = "IntervalTier"', '\t\tname = "words"',
        "\t\txmin = 0.0", f"\t\txmax = {end}", f"\t\tintervals: size = {len(intervals)}",
    ]
    for i, (lo, hi, text) in enumerate(intervals, 1):
        lines += [f"\t\tintervals [{i}]:", f"\t\t\txmin = {lo}", f"\t\t\txmax = {hi}", f'\t\t\ttext = "{text}"']
    Path(path).write_text("\n".join(lines) + "\n")


def kaldi_to_textgrids(ali_path, stems, out_dir):
    """One TextGrid per selected utterance; `<SIL>` rows and gaps become empty-text intervals."""
    rows = defaultdict(list)
    for line in Path(ali_path).read_text().splitlines()[1:]:
        utt, start, dur, word = line.split()
        rows[utt].append((float(start), round(float(start) + float(dur), 6), "" if word == SILENCE else word))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        intervals = []
        cursor = 0.0
        for start, end, word in sorted(rows[stem]):
            if start > cursor:
                intervals.append((cursor, start, ""))
            intervals.append((start, end, word))
            cursor = end
        write_words_textgrid(out_dir / f"{stem}.TextGrid", intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ali", type=Path)
    parser.add_argument("utts", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    kaldi_to_textgrids(args.ali, args.utts.read_text().split(), args.out_dir)


if __name__ == "__main__":
    main()
