from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def evaluator_dir():
    path = ROOT / "third_party" / "evaluation"
    if not (path / "boundary_eval.py").exists():
        pytest.fail("pinned evaluator missing: run `uv run snakemake -j1 third_party/evaluation`")
    return path


def write_words_textgrid(path, intervals):
    """Write a TextGrid with a `words` and a `syllables` tier.

    `intervals` maps tier name to a list of (xmin, xmax, text); a missing
    `syllables` tier copies `words`.
    """
    tiers = {"words": intervals["words"], "syllables": intervals.get("syllables", intervals["words"])}
    end = intervals["words"][-1][1]
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', "",
        "xmin = 0.0", f"xmax = {end}", "tiers? <exists>", f"size = {len(tiers)}", "item []:",
    ]
    for k, (name, ivs) in enumerate(tiers.items(), 1):
        lines += [
            f"\titem [{k}]:", '\t\tclass = "IntervalTier"', f'\t\tname = "{name}"',
            "\t\txmin = 0.0", f"\t\txmax = {end}", f"\t\tintervals: size = {len(ivs)}",
        ]
        for i, (lo, hi, text) in enumerate(ivs, 1):
            lines += [f"\t\tintervals [{i}]:", f"\t\t\txmin = {lo}", f"\t\t\txmax = {hi}", f'\t\t\ttext = "{text}"']
    Path(path).write_text("\n".join(lines) + "\n")
