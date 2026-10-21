"""Kaldi word alignments for the selected utterances, checked against the feature frames.

The authors' loader silently clips word spans that run past an utterance's
last feature frame (numpy slicing). We keep that behaviour but fail when more
than 0.1% of words would be clipped, which would point to an alignment/feature
mismatch.
"""

import argparse
import json
import tomllib
from pathlib import Path

from concept_analysis.annotations import get_word_alignment

MAX_CLIPPED_FRACTION = 0.001


def subset_alignment(ali_path, stems, out_path):
    """Write the `.ali` rows (header kept) of the utterances in `stems`."""
    keep = set(stems)
    header, *rows = Path(ali_path).read_text().splitlines(keepends=True)
    kept = [row for row in rows if row.split(maxsplit=1)[0] in keep]
    found = {row.split(maxsplit=1)[0] for row in kept}
    if found != keep:
        raise ValueError(f"{ali_path} lacks {len(keep - found)} selected utterances, e.g. {sorted(keep - found)[:5]}")
    Path(out_path).write_text(header + "".join(kept))


def check_clipping(ali_path, frames, max_fraction=MAX_CLIPPED_FRACTION):
    """Count non-silence words whose frame span (authors' rounding) ends past the utterance's frames."""
    alignment = get_word_alignment(ali_path, drop_silence=True)
    n_frames = alignment["utt_id"].map(frames)
    n_clipped = int((alignment["end_time"] > n_frames).sum())
    n_words = len(alignment)
    if n_clipped > max_fraction * n_words:
        raise ValueError(f"{ali_path}: {n_clipped} of {n_words} words run past the last feature frame "
                         f"(limit {max_fraction:.1%})")
    return {"n_words": n_words, "n_clipped": n_clipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ali", type=Path)
    parser.add_argument("utts", type=Path)
    parser.add_argument("manifest", type=Path, help="feature cache manifest with per-utterance frame counts")
    parser.add_argument("out_ali", type=Path)
    parser.add_argument("stats", type=Path)
    args = parser.parse_args()
    args.out_ali.parent.mkdir(parents=True, exist_ok=True)
    subset_alignment(args.ali, args.utts.read_text().split(), args.out_ali)
    with args.manifest.open("rb") as f:
        frames = tomllib.load(f)["frames"]
    args.stats.write_text(json.dumps(check_clipping(args.out_ali, frames)) + "\n")


if __name__ == "__main__":
    main()
