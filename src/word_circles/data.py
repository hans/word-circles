"""Utterance selection and coverage checks for each split."""

import argparse
import os
from pathlib import Path


def select_utterances(textgrid_dir, n_utts=None, expected=None):
    """Return the sorted utterance stems used for a split.

    The full run (`n_utts=None`) must cover exactly `expected` utterances. It
    never subsets to pass. A smoke run takes the first `n_utts` by stem.
    """
    stems = sorted(p.stem for p in Path(textgrid_dir).glob("*.TextGrid"))
    return take(stems, n_utts, expected, source=str(textgrid_dir))


def ali_stems(ali_path):
    """Sorted utterance ids in a Kaldi word `.ali` file (header row skipped)."""
    lines = Path(ali_path).read_text().splitlines()[1:]
    return sorted({line.split(maxsplit=1)[0] for line in lines if line.strip()})


def take(stems, n_utts=None, expected=None, source=""):
    if n_utts is None:
        if expected is not None and len(stems) != expected:
            raise ValueError(f"{source}: {len(stems)} utterances, expected {expected}")
        return stems
    return stems[:n_utts]


def link_references(textgrid_dir, stems, ref_dir):
    """Populate `ref_dir` with symlinks to the selected TextGrids."""
    ref_dir = Path(ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = (Path(textgrid_dir) / f"{stem}.TextGrid").resolve()
        os.symlink(src, ref_dir / f"{stem}.TextGrid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sel = sub.add_parser("select-utterances")
    sel.add_argument("textgrid_dir", type=Path)
    sel.add_argument("utts", type=Path)
    sel.add_argument("ref_dir", type=Path)
    sel.add_argument("--n-utts", type=lambda s: None if s == "None" else int(s), default=None)
    sel.add_argument("--expected", type=int, default=None)
    ali = sub.add_parser("select-ali-utterances")
    ali.add_argument("ali", type=Path)
    ali.add_argument("utts", type=Path)
    ali.add_argument("--n-utts", type=lambda s: None if s == "None" else int(s), default=None)
    ali.add_argument("--expected", type=int, default=None)
    members = sub.add_parser("tar-members", help="LibriSpeech tarball member paths for the utterances in an .ali")
    members.add_argument("ali", type=Path)
    members.add_argument("prefix")
    args = parser.parse_args()

    if args.command == "tar-members":
        for stem in ali_stems(args.ali):
            speaker, chapter, _ = stem.split("-")
            print(f"{args.prefix}/{speaker}/{chapter}/{stem}.flac")
        return
    if args.command == "select-utterances":
        stems = select_utterances(args.textgrid_dir, args.n_utts, args.expected)
        link_references(args.textgrid_dir, stems, args.ref_dir)
    else:
        stems = take(ali_stems(args.ali), args.n_utts, args.expected, source=str(args.ali))
    args.utts.parent.mkdir(parents=True, exist_ok=True)
    args.utts.write_text("".join(f"{s}\n" for s in stems))


if __name__ == "__main__":
    main()
