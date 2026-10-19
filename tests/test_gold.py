import pytest

from conftest import ROOT, write_words_textgrid
from word_circles.evaluation import score_boundaries
from word_circles.gold import gold_syllable_boundaries

DEV_TEXTGRIDS = ROOT / "data" / "textgrids" / "dev"


def test_gold_boundaries_are_every_syllable_end_including_silences(tmp_path):
    write_words_textgrid(tmp_path / "u.TextGrid", {
        "words": [(0.0, 0.1, ""), (0.1, 0.5, "hello"), (0.5, 0.62, "")],
        "syllables": [(0.0, 0.1, ""), (0.1, 0.28, "hh-ah"), (0.28, 0.5, "l-ow"), (0.5, 0.62, "")],
    })
    assert gold_syllable_boundaries(tmp_path) == {"u": [0.1, 0.28, 0.5, 0.62]}


@pytest.mark.data
def test_gold_syllables_reproduce_table_ii_gold_row_on_dev_clean(evaluator_dir, tmp_path):
    if not DEV_TEXTGRIDS.exists():
        pytest.fail("dev TextGrids missing: run `uv run snakemake -j1 data/textgrids/dev`")
    result = score_boundaries(gold_syllable_boundaries(DEV_TEXTGRIDS), DEV_TEXTGRIDS, evaluator_dir, tmp_path)
    got = {k: round(100 * result.scores[k], 2) for k in ("F1", "OS", "token_F1", "R")}
    # Paper Table II gold row (85.5 / 33.9 / 65.9 / 71.1), to the digits measured in #3.
    assert got == {"F1": 85.50, "OS": 33.91, "token_F1": 65.92, "R": 71.06}
    assert result.empty == []
